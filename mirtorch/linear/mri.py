"""
Discrete-to-discrete system matrices for MRI.
2021-02. Guanhua Wang, University of Michigan
"""

import math
import sys
import warnings
from importlib.util import find_spec
from typing import cast

import torch
import torch.nn.functional as F
import torchkbnufft as tkbn
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.fft import fftn, ifftn

from ._finufft import (
    FinufftSenseBackend,
    finufft_sense_adjoint,
    finufft_sense_forward,
    finufft_type1,
    finufft_type2,
)
from .linearmaps import LinearMap
from .util import (
    adjoint_fft_norm,
    fftshift,
    ifftshift,
    nufft_trajectory_vjp,
    reduce_broadcast_batch_gradient,
)

_MAX_B0_KERNEL_BYTES = 2 * 1024**3
_MAX_B0_WORKSPACE_BYTES = 512 * 1024**2


def _resolve_batch_size(**sizes: int) -> int:
    """Resolve broadcast-compatible leading dimensions."""
    batch = max(sizes.values())
    if any(size not in (1, batch) for size in sizes.values()):
        details = ", ".join(f"{name}.shape[0]={size}" for name, size in sizes.items())
        raise ValueError(
            f"Incompatible batch sizes: {details}. Must be equal or one must be 1."
        )
    return batch


def _spatial_shape(smaps: Tensor, batchmode: bool = True) -> tuple[int, ...]:
    return tuple(smaps.shape[2:] if batchmode else smaps.shape[1:])


def _grid_shape(spatial_shape: tuple[int, ...], oversampling: float) -> tuple[int, ...]:
    return tuple(math.floor(size * oversampling) for size in spatial_shape)


def _resolve_b0_batch_size(smaps: Tensor, zmap: Tensor, traj: Tensor) -> int:
    if smaps.device != zmap.device or smaps.device != traj.device:
        raise ValueError(
            "sensitivity maps, field map, and trajectory must be on one device"
        )
    batch = _resolve_batch_size(smaps=smaps.shape[0], traj=traj.shape[0])
    if zmap.shape[0] not in (1, batch):
        raise ValueError(
            f"Incompatible zmap batch size: zmap.shape[0]={zmap.shape[0]}, "
            f"expected 1 or {batch}."
        )
    return batch


def readout_times(
    npoints: int,
    dt: float,
    *,
    template: Tensor,
    times: Tensor | None = None,
) -> Tensor:
    """Return readout sample times in milliseconds on ``template``'s device."""
    if npoints < 1:
        raise ValueError("npoints must be positive")
    if dt < 0:
        raise ValueError("dt must be non-negative")
    real_dtype = template.real.dtype
    if times is None:
        return torch.arange(npoints, dtype=real_dtype, device=template.device) * dt
    if times.ndim != 1 or times.numel() != npoints:
        raise ValueError(f"T must be one-dimensional with exactly {npoints} entries")
    if times.is_complex():
        raise TypeError("T must be real-valued")
    return times.to(device=template.device, dtype=real_dtype)


def _histogram_autocorrelation(histogram: Tensor) -> Tensor:
    values = histogram.reshape(1, 1, -1)
    padded = F.pad(values, (histogram.numel() - 1,) * 2)
    return F.conv1d(padded, values).reshape(-1)


def _uniform_histogram(values: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Return an equal-width histogram using device-portable tensor operations."""
    lower = values.amin()
    upper = values.amax()
    collapsed = lower == upper
    lower = torch.where(collapsed, lower - 0.5, lower)
    upper = torch.where(collapsed, upper + 0.5, upper)
    width = upper - lower
    detached_values = values.detach()
    indices = torch.floor((detached_values - lower.detach()) * bins / width.detach())
    indices = indices.to(torch.long).clamp_(0, bins - 1)
    histogram = torch.zeros(bins, dtype=values.dtype, device=values.device)
    histogram = histogram.scatter_add(
        0,
        indices,
        torch.ones_like(values),
    )
    fractions = torch.arange(
        bins + 1,
        dtype=values.dtype,
        device=values.device,
    )
    edges = lower + fractions * (width / bins)
    return histogram, edges


def mri_exp_approx(
    b0: Tensor,
    bins: int,
    lseg: int,
    t: Tensor,
    autocorrelation: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Approximate ``exp(-2j*pi*b0*t)`` using PyTorch time segmentation.

    Histogram membership and MIRT percentile segment placement are intentionally
    non-differentiable model-selection steps. The fitted target exponential
    retains gradients with respect to ``t``, and the spatial phase retains
    gradients with respect to ``b0``.
    """
    if not isinstance(b0, Tensor) or not isinstance(t, Tensor):
        raise TypeError("b0 and t must be torch tensors")
    if b0.is_complex():
        raise TypeError("b0 must be real-valued")
    if t.is_complex():
        raise TypeError("t must be real-valued")
    if bins < 1:
        raise ValueError("bins must be positive")
    if lseg < 1:
        raise ValueError("lseg must be positive")
    if t.ndim != 1 or t.numel() < 1:
        raise ValueError("t must be a non-empty one-dimensional tensor")
    if not torch.isfinite(t).all():
        raise ValueError("t must contain only finite values")
    if b0.device != t.device:
        raise ValueError("b0 and t must be on the same device")
    if not b0.is_floating_point() or not t.is_floating_point():
        raise TypeError("b0 and t must use floating-point dtypes")

    real_dtype = torch.promote_types(b0.dtype, t.dtype)
    b0 = b0.to(dtype=real_dtype)
    t = t.to(dtype=real_dtype)
    angular_frequency = 2 * math.pi * b0.reshape(-1)

    # Histogram membership has no useful derivative with respect to the field
    # values. The discrete assignments are detached, while bin edges, fitted
    # coefficients, and spatial phases retain their continuous gradient paths.
    hist_wt, bin_edges = _uniform_histogram(angular_frequency, bins)
    bin_width = bin_edges[1] - bin_edges[0]
    if autocorrelation:
        hist_wt = _histogram_autocorrelation(hist_wt)
        bin_centers = (
            torch.arange(
                1 - bins,
                bins,
                dtype=real_dtype,
                device=b0.device,
            )
            * bin_width
        )
    else:
        bin_centers = bin_edges[1:] - bin_width / 2

    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    # The temporal fit is a tiny, often ill-conditioned least-squares problem.
    # Running its pseudoinverse in complex64 gives platform-dependent zmap and
    # time gradients even when the fitted coefficients agree. CPU complex128
    # keeps those derivatives stable; the transferred tensors are only O(bins
    # + segments + samples), independent of the image size.
    fit_bin_centers = bin_centers.cpu().to(torch.float64)
    fit_histogram = hist_wt.cpu().to(torch.float64)
    fit_times = t.cpu().to(torch.float64)
    selection_times = fit_times.detach()
    if lseg == 1:
        # MIRT places a single time segment at the mean readout time.
        fit_tl = selection_times.mean().reshape(1)
    else:
        # Match MIRT's jf_prctile definition: sorted samples are located at
        # (j - 1/2) / M and the requested, uniformly spaced percentiles are
        # linearly interpolated between them. This differs slightly from the
        # default NumPy/PyTorch quantile convention, even for uniform samples.
        sorted_times = selection_times.sort().values
        fit_fractions = torch.linspace(0, 1, lseg, dtype=torch.float64)
        positions = (fit_fractions * selection_times.numel() - 0.5).clamp(
            0,
            selection_times.numel() - 1,
        )
        lower_positions = positions.floor()
        upper_positions = positions.ceil()
        weights = positions - lower_positions
        fit_tl = torch.lerp(
            sorted_times[lower_positions.to(torch.long)],
            sorted_times[upper_positions.to(torch.long)],
            weights,
        )
    # Segment placement is a basis-selection step, not a physical dependence
    # of the signal on the sample times. Differentiating through the selected
    # order statistics makes the near-singular pseudoinverse backward unstable;
    # holding the MIRT basis fixed preserves the physical target-exp gradient.
    fit_tl = fit_tl / 1000
    fit_frequencies = 1j * fit_bin_centers
    fit_basis = torch.exp(-fit_tl[:, None] * fit_frequencies[None, :]).transpose(0, 1)
    sqrt_histogram = fit_histogram.sqrt()
    fit_basis = sqrt_histogram[:, None] * fit_basis
    # Keep nearly redundant time segments out of the differentiated subspace.
    fit_dimension = max(fit_basis.shape)
    fit_rtol = max(
        fit_dimension * torch.finfo(real_dtype).eps,
        math.sqrt(fit_dimension * torch.finfo(torch.float64).eps),
    )
    interpolator = torch.linalg.pinv(fit_basis, rtol=fit_rtol)
    interpolator = interpolator * sqrt_histogram[None, :]
    target = torch.exp(-fit_frequencies[:, None] * fit_times[None, :] / 1000)
    temporal = (interpolator @ target).transpose(0, 1)
    temporal = temporal.to(dtype=complex_dtype).to(device=b0.device)
    if autocorrelation:
        temporal = temporal.real

    tl = fit_tl.to(dtype=real_dtype).to(device=b0.device)
    spatial_frequency = (2j * math.pi * b0.reshape(-1)).to(complex_dtype)
    spatial = torch.exp(-tl[:, None] * spatial_frequency[None, :]).transpose(0, 1)
    return temporal, spatial, tl


def time_segmentation_coefficients(
    zmap: Tensor,
    *,
    batch: int,
    bins: int,
    segments: int,
    times: Tensor,
    complex_dtype: torch.dtype,
    autocorrelation: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build batched temporal ``B`` and spatial ``C`` coefficients."""
    if zmap.ndim < 2:
        raise ValueError("zmap must include batch and spatial dimensions")
    if zmap.shape[0] not in (1, batch):
        raise ValueError(f"zmap batch size must be 1 or {batch}, not {zmap.shape[0]}")

    coefficient_batch = zmap.shape[0]
    approximations = [
        mri_exp_approx(
            zmap[index],
            bins,
            segments,
            times,
            autocorrelation=autocorrelation,
        )
        for index in range(coefficient_batch)
    ]
    temporal = torch.stack(
        [item[0].transpose(0, 1) for item in approximations],
        dim=1,
    ).to(dtype=complex_dtype)
    spatial = torch.stack(
        [item[1].transpose(0, 1) for item in approximations],
        dim=1,
    ).to(dtype=complex_dtype)

    if coefficient_batch == 1 and batch != 1:
        temporal = temporal.expand(-1, batch, -1)
        spatial = spatial.expand(-1, batch, -1)

    temporal = temporal.reshape(segments, batch, 1, 1, times.numel())
    spatial = spatial.reshape(
        segments,
        batch,
        1,
        *zmap.shape[1:],
    )
    return temporal, spatial, approximations[0][2]


def _resolve_nufft_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        if device.type == "cuda" and find_spec("cufinufft") is not None:
            return "finufft"
        if (
            device.type == "cpu"
            and sys.platform != "darwin"
            and find_spec("finufft") is not None
        ):
            return "finufft"
        return "torchkbnufft"
    if backend not in ("torchkbnufft", "finufft"):
        raise ValueError(
            "NUFFT backend must be 'auto', 'torchkbnufft', or 'finufft', "
            f"not {backend!r}."
        )
    return backend


class _KbNufftType2(torch.autograd.Function):
    """KbNufft with an efficient trajectory VJP."""

    @staticmethod
    def forward(ctx, modes, trajectory, forward_operator, adjoint_operator, norm):
        ctx.forward_operator = forward_operator
        ctx.adjoint_operator = adjoint_operator
        ctx.norm = norm
        ctx.save_for_backward(modes, trajectory)
        return forward_operator(modes, trajectory, norm=norm)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, grad_samples
    ):
        modes, trajectory = ctx.saved_tensors
        need_modes, need_trajectory, *_ = ctx.needs_input_grad

        grad_modes = None
        if need_modes:
            grad_modes = ctx.adjoint_operator(
                grad_samples,
                trajectory,
                norm=ctx.norm,
            )

        grad_trajectory = None
        if need_trajectory:
            grad_trajectory = nufft_trajectory_vjp(
                modes,
                trajectory,
                grad_samples,
                lambda weighted, traj: ctx.forward_operator(
                    weighted,
                    traj,
                    norm=ctx.norm,
                ),
            )
            grad_trajectory = reduce_broadcast_batch_gradient(
                grad_trajectory,
                trajectory.shape[0],
            )
        return grad_modes, grad_trajectory, None, None, None


class _KbNufftType1(torch.autograd.Function):
    """KbNufftAdjoint with an efficient trajectory VJP."""

    @staticmethod
    def forward(ctx, samples, trajectory, forward_operator, adjoint_operator, norm):
        ctx.forward_operator = forward_operator
        ctx.adjoint_operator = adjoint_operator
        ctx.norm = norm
        ctx.save_for_backward(samples, trajectory)
        return adjoint_operator(samples, trajectory, norm=norm)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, grad_modes
    ):
        samples, trajectory = ctx.saved_tensors
        need_samples, need_trajectory, *_ = ctx.needs_input_grad

        grad_samples = None
        if need_samples:
            grad_samples = ctx.forward_operator(
                grad_modes,
                trajectory,
                norm=ctx.norm,
            )

        grad_trajectory = None
        if need_trajectory:
            grad_trajectory = nufft_trajectory_vjp(
                grad_modes,
                trajectory,
                samples,
                lambda weighted, traj: ctx.forward_operator(
                    weighted,
                    traj,
                    norm=ctx.norm,
                ),
            )
            grad_trajectory = reduce_broadcast_batch_gradient(
                grad_trajectory,
                trajectory.shape[0],
            )
        return grad_samples, grad_trajectory, None, None, None


def _kb_type2(
    modes: Tensor,
    trajectory: Tensor,
    forward_operator: tkbn.KbNufft,
    adjoint_operator: tkbn.KbNufftAdjoint,
    norm: str | None,
) -> Tensor:
    if trajectory.requires_grad:
        return cast(
            Tensor,
            _KbNufftType2.apply(
                modes,
                trajectory,
                forward_operator,
                adjoint_operator,
                norm,
            ),
        )
    return forward_operator(modes, trajectory, norm=norm)


def _kb_type1(
    samples: Tensor,
    trajectory: Tensor,
    forward_operator: tkbn.KbNufft,
    adjoint_operator: tkbn.KbNufftAdjoint,
    norm: str | None,
) -> Tensor:
    if trajectory.requires_grad:
        return cast(
            Tensor,
            _KbNufftType1.apply(
                samples,
                trajectory,
                forward_operator,
                adjoint_operator,
                norm,
            ),
        )
    return adjoint_operator(samples, trajectory, norm=norm)


def _complex_dtype_like(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.complex64)


def _tensor_state_key(tensor: Tensor) -> tuple:
    """Identify a tensor and mutations that can invalidate cached MRI data."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tensor._version,
        tensor.device,
        tensor.dtype,
        tensor.requires_grad,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _coefficient_state_key(
    zmap: Tensor,
    times: Tensor | None,
    dt: float,
    bins: int,
    segments: int,
) -> tuple:
    time_state = None if times is None else _tensor_state_key(times)
    return _tensor_state_key(zmap), time_state, dt, bins, segments


class FFTCn(LinearMap):
    r"""
    FFT operators with FFTshift and iFFTshift for multidimensional data.
    Pytorch provides three modes in FFT: 'ortho', 'forward', 'backward'.
    Each pair of FFT and iFFT with same mode is the inverse, but not necessarily the adjoint to each other.

    Attributes:
        norm: normalization of the fft ('ortho', 'forward' or 'backward')
        dims: the dimensions to apply the fft
    """

    def __init__(
        self,
        size_in: list[int],
        size_out: list[int],
        dims: tuple[int, ...] | None = None,
        norm: str = "ortho",
    ):
        if list(size_in) != list(size_out):
            raise ValueError("FFTCn preserves shape, so size_in must equal size_out")
        if norm not in ("ortho", "forward", "backward", None):
            raise ValueError("norm must be None, 'ortho', 'forward', or 'backward'")
        super().__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims

    def _apply(self, x: Tensor) -> Tensor:
        x = ifftshift(x, self.dims)
        x = fftn(x, dim=self.dims, norm=self.norm)
        x = fftshift(x, self.dims)
        return x

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        x = ifftshift(x, self.dims)
        x = ifftn(x, dim=self.dims, norm=adjoint_fft_norm(self.norm))
        x = fftshift(x, self.dims)
        return x


class Sense(LinearMap):
    r"""
    Cartesian sense operator, following "SENSE: Sensitivity encoding for fast MRI".
    The input/output size depends on the sensitivity maps and masks.

    Broadcasting behavior:
    - If smaps or masks have shape [1, ...], they will be broadcast to the batch size
    - The batch size is determined by max(smaps.shape[0], masks.shape[0])
    - At least one of (smaps, masks) must have the desired batch size

    When both smaps and masks have batch size 1, the operator has batch size 1.

    If we use the batch dimension, the input dimension is [nbatch, 1, nx, ny, (nz)],
    and the output is [nbatch, ncoil, nx, ny, (nz)].
    Otherwise, the input dimension is [nx, ny, (nz)], and the output is [ncoil, nx, ny, (nz)].

    Attributes:
        smaps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. Sensitivity maps.
        masks: tensor with dimension [(batch), nx, ny, (nz)]. Sampling mask.
        norm: normalization of the fft ('ortho', 'forward' or 'backward')
        batchmode: bool, determining if there exist batch and channel dimension
    """

    def __init__(
        self, smaps: Tensor, masks: Tensor, norm: str = "ortho", batchmode: bool = True
    ):
        if not isinstance(batchmode, bool):
            raise TypeError("batchmode must be a bool")
        expected_rank = 4 if batchmode else 3
        if smaps.ndim not in (expected_rank, expected_rank + 1):
            raise ValueError(
                f"smaps must have rank {expected_rank} or {expected_rank + 1}"
            )
        if masks.ndim != smaps.ndim - 1:
            raise ValueError("masks must have one fewer dimension than smaps")
        if smaps.device != masks.device:
            raise ValueError("smaps and masks must be on the same device")
        if norm not in ("ortho", "forward", "backward", None):
            raise ValueError("norm must be None, 'ortho', 'forward', or 'backward'")
        ncoil = smaps.shape[1] if batchmode else smaps.shape[0]
        spatial_shape = _spatial_shape(smaps, batchmode)
        mask_shape = tuple(masks.shape[1:] if batchmode else masks.shape)
        if spatial_shape != mask_shape:
            raise ValueError(
                f"Spatial dimensions mismatch: smaps {spatial_shape}, "
                f"masks {mask_shape}"
            )

        if batchmode:
            nbatch = _resolve_batch_size(
                smaps=smaps.shape[0],
                masks=masks.shape[0],
            )
            size_in = (nbatch, 1) + spatial_shape
            size_out = (nbatch, ncoil) + spatial_shape
        else:
            size_in = spatial_shape
            size_out = tuple(smaps.shape)

        super().__init__(size_in, size_out)
        self.norm = norm
        self.dims = tuple(range(2 if batchmode else 1, smaps.ndim))
        self.smaps = smaps
        self.masks = masks.unsqueeze(1) if batchmode else masks
        self.batchmode = batchmode

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [batch, 1, nx, ny, (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            y:  tensor with dimension [batch, ncoil, nx, ny, (nz)] (batchmode=True) or [ncoil, nx, ny, nz]
        """
        x = x * self.smaps
        x = ifftshift(x, self.dims)
        k = fftn(x, dim=self.dims, norm=self.norm)
        k = fftshift(k, self.dims) * self.masks
        return k

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            k:  tensor with dimension [batch, ncoil, nx, ny, (nz)] (batchmode=True) or [ncoil, nx, ny, nz]
        Returns:
            x:  tensor with dimension [batch, 1, nx, ny, (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        k = x * self.masks.conj()
        k = ifftshift(k, self.dims)
        x = ifftn(k, dim=self.dims, norm=adjoint_fft_norm(self.norm))
        x = fftshift(x, self.dims)

        if self.batchmode:
            x = (x * torch.conj(self.smaps)).sum(1, keepdim=True)
        else:
            x = (x * torch.conj(self.smaps)).sum(0)

        return x


class NuSense(LinearMap):
    r"""
    Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    The automatic backend uses an installed FINUFFT or cuFINUFFT library on a
    supported device, and otherwise uses Matthew Muckley's torchkbnufft.
    Both backends use the coordinate-weighted NUFFT Jacobian for efficient
    first-order trajectory gradients.

    Trajectory coordinates are in radians per voxel. For an image stored as
    ``[..., ny, nx]``, the corresponding coordinate order is ``(ky, kx)``.

    Broadcasting behavior:
    - If smaps or traj have shape [1, ...], they will be broadcast to the batch size
    - The batch size is determined by max(smaps.shape[0], traj.shape[0])
    - At least one of (smaps, traj) must have the desired batch size

    When both smaps and traj have batch size 1, the operator has batch size 1.

    The input/output size depends on the sensitivity maps.
    If we use the batch dimension, the input dimension is [nbatch, 1, nx, ny, (nz)],
    and the output is [nbatch, ncoil, npoints].
    Otherwise, the input dimension is [nx, ny, (nz)], and the output is [ncoil, npoints].

    Attributes:
        smaps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. Sensitivity maps.
        traj: tensor with dimension [(batch), ndim, nshot*npoints]. k-space trajectory.
        norm: normalization of the fft ('ortho' or None)
        batchmode: bool, determining if there exist batch and channel dimension
        sequential: bool, memory saving mode
        numpoints: int, number of interpolation points in gridding
        grid_size: float, oversampling ratio (>1)
        backend: ``"auto"`` (default), ``"finufft"``, or ``"torchkbnufft"``
        eps: requested FINUFFT relative precision
    """

    def __init__(
        self,
        smaps: Tensor,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
        sequential: bool = False,
        backend: str = "auto",
        eps: float = 1e-6,
    ):
        self.smaps = smaps
        self.norm = norm
        self.traj = traj
        self.batchmode = batchmode
        self.sequential = sequential
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        backend = self.backend
        if grid_size < 1:
            raise ValueError("grid size must be at least 1")

        ncoil = smaps.shape[1] if batchmode else smaps.shape[0]
        spatial_shape = _spatial_shape(smaps, batchmode)
        self.grid_size = _grid_shape(spatial_shape, grid_size)

        if batchmode:
            nbatch = _resolve_batch_size(
                smaps=smaps.shape[0],
                traj=traj.shape[0],
            )
            size_in = (nbatch, 1) + spatial_shape
            size_out = (nbatch, ncoil, traj.shape[-1])
        else:
            size_in = spatial_shape
            size_out = (smaps.shape[0], traj.shape[-1])

        if backend == "torchkbnufft":
            self.A = tkbn.KbNufft(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
        else:
            self._finufft_backend = FinufftSenseBackend(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                norm=norm,
                batchmode=batchmode,
                sequential=sequential,
                eps=eps,
            )

        super().__init__(size_in, size_out)

    def to(self, device: torch.device | str):
        if self.backend == "finufft":
            self._finufft_backend.clear_plans()
        return super().to(device)

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            k： tensor with dimension [batch, ncoil, nshot*npoints] or [ncoil, nshot*npoints]
        """
        if self.backend == "finufft":
            return finufft_sense_forward(
                x,
                self.smaps,
                self.traj,
                self._finufft_backend,
            )

        image = x if self.batchmode else x.unsqueeze(0).unsqueeze(0)
        smaps = self.smaps if self.batchmode else self.smaps.unsqueeze(0)
        trajectory = self.traj if self.batchmode else self.traj.unsqueeze(0)
        if self.sequential:
            transformed = torch.cat(
                [
                    _kb_type2(
                        image * smaps[:, coil : coil + 1],
                        trajectory,
                        self.A,
                        self.AT,
                        self.norm,
                    )
                    for coil in range(smaps.shape[1])
                ],
                dim=1,
            )
        else:
            transformed = _kb_type2(
                image * smaps,
                trajectory,
                self.A,
                self.AT,
                self.norm,
            )
        return transformed if self.batchmode else transformed.squeeze(0)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            y： tensor with dimension [batch, ncoil, nshot*npoints] (batchmode=True) or [ncoil, nshot*npoints]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self.backend == "finufft":
            return finufft_sense_adjoint(
                x,
                self.smaps,
                self.traj,
                self._finufft_backend,
            )

        samples = x if self.batchmode else x.unsqueeze(0)
        smaps = self.smaps if self.batchmode else self.smaps.unsqueeze(0)
        trajectory = self.traj if self.batchmode else self.traj.unsqueeze(0)
        if self.sequential:
            image = torch.zeros(
                (samples.shape[0], 1, *self.size_in[-len(self.grid_size) :]),
                dtype=x.dtype,
                device=x.device,
            )
            for coil in range(smaps.shape[1]):
                coil_image = _kb_type1(
                    samples[:, coil : coil + 1],
                    trajectory,
                    self.A,
                    self.AT,
                    self.norm,
                )
                image = image + coil_image * smaps[:, coil : coil + 1].conj()
        else:
            coil_images = _kb_type1(
                samples,
                trajectory,
                self.A,
                self.AT,
                self.norm,
            )
            image = (coil_images * smaps.conj()).sum(dim=1, keepdim=True)
        return image if self.batchmode else image.squeeze(0).squeeze(0)


class NuSenseGram(LinearMap):
    r"""
    Gram operator (A'A) of the Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    With a compatible installed native library, the automatic backend builds a
    FINUFFT or cuFINUFFT Toeplitz embedding for a fixed trajectory. Otherwise
    it uses torchkbnufft. A trainable trajectory uses the direct
    ``NuSense.H * NuSense`` path so gradients are preserved.

    Broadcasting behavior:
    - If smaps or traj have shape [1, ...], they will be broadcast to the batch size
    - The batch size is determined by max(smaps.shape[0], traj.shape[0])
    - At least one of (smaps, traj) must have the desired batch size

    When both smaps and traj have batch size 1, the operator has batch size 1.

    The input/output size depends on the sensitivity maps.
    If we use the batch dimension, the input/output dimension is [nbatch, 1, nx, ny, (nz)].
    Otherwise, the input/output dimension is [nx, ny, (nz)].

    Attributes:
        smaps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. Sensitivity maps.
        traj: tensor with dimension [(batch), ndim, nshot*npoints]. k-space trajectory.
        norm: normalization of the fft ('ortho' or None)
        batchmode: bool, determining if there exist batch and channel dimension
        numpoints: int, number of interpolation points in gridding
        grid_size: float, oversampling ratio (>1)
        backend: ``"auto"`` (default), ``"finufft"``, or ``"torchkbnufft"``
        eps: requested FINUFFT relative precision for kernel construction
    """

    def __init__(
        self,
        smaps: Tensor,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
        backend: str = "auto",
        eps: float = 1e-6,
    ):
        self.smaps = smaps
        self.norm = norm
        self.traj = traj
        self.batchmode = batchmode
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        backend = self.backend

        spatial_shape = _spatial_shape(smaps, batchmode)
        self.grid_size = _grid_shape(spatial_shape, grid_size)
        if batchmode:
            nbatch = _resolve_batch_size(
                smaps=smaps.shape[0],
                traj=traj.shape[0],
            )
            size_in = (nbatch, 1) + spatial_shape
        else:
            size_in = spatial_shape

        self._uses_direct_gram = traj.requires_grad
        if self._uses_direct_gram:
            self._direct_operator = NuSense(
                smaps=smaps,
                traj=traj,
                norm=norm,
                batchmode=batchmode,
                numpoints=numpoints,
                grid_size=grid_size,
                backend=backend,
                eps=eps,
            )
            super().__init__(size_in, size_in)
            return

        if backend == "torchkbnufft":
            self.toep_op = tkbn.ToepNufft()

        if backend == "torchkbnufft":
            self.kernel = tkbn.calc_toeplitz_kernel(
                traj,
                list(spatial_shape),
                grid_size=self.grid_size,
                numpoints=numpoints,
                norm=self.norm,
            )
        else:
            if smaps.device != traj.device:
                raise ValueError(
                    "sensitivity maps and trajectory must be on one device"
                )
            self._finufft_backend = FinufftSenseBackend(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                norm=norm,
                batchmode=batchmode,
                sequential=False,
                eps=eps,
            )
            self.kernel = self._finufft_backend.toeplitz_kernel(
                traj,
                smaps.dtype,
            )

        self._trajectory_state = _tensor_state_key(self.traj)
        super().__init__(size_in, size_in)

    def _check_direct_aliases(self) -> None:
        for name in ("smaps", "traj"):
            if getattr(self, name) is not getattr(self._direct_operator, name):
                raise RuntimeError(
                    f"{name} was replaced on a direct NuSenseGram operator; "
                    "create a new operator instead."
                )

    def _sync_direct_aliases(self) -> None:
        self.smaps = self._direct_operator.smaps
        self.traj = self._direct_operator.traj

    def _check_fixed_trajectory(self) -> None:
        if _tensor_state_key(self.traj) != self._trajectory_state:
            raise RuntimeError(
                "The trajectory changed after the Toeplitz kernel "
                "was constructed; create a new NuSenseGram operator."
            )

    def to(self, device: torch.device | str):
        if self._uses_direct_gram:
            self._check_direct_aliases()
            self._direct_operator.to(device)
            self._sync_direct_aliases()
            return self

        self._check_fixed_trajectory()
        if self.backend == "finufft":
            self._finufft_backend.clear_plans()
        result = super().to(device)
        self._trajectory_state = _tensor_state_key(self.traj)
        return result

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self._uses_direct_gram:
            self._check_direct_aliases()
            samples = self._direct_operator(x)
            return self._direct_operator.adjoint(samples)

        self._check_fixed_trajectory()
        if self.backend == "finufft":
            return self._finufft_backend.sense_gram(
                x,
                self.smaps,
                self.kernel,
            )

        image = x if self.batchmode else x.unsqueeze(0).unsqueeze(0)
        smaps = self.smaps if self.batchmode else self.smaps.unsqueeze(0)
        result = self.toep_op(image, self.kernel, smaps=smaps, norm=self.norm)
        return result if self.batchmode else result.squeeze(0).squeeze(0)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return self._apply(x)


class Gmri(LinearMap):
    r"""
    B0-informed MRI encoding operator, following MIRT.

    The automatic backend uses an installed FINUFFT or cuFINUFFT library on a
    supported device, and otherwise uses torchkbnufft.
    Both backends use the coordinate-weighted NUFFT Jacobian for efficient
    first-order trajectory gradients. Image, sensitivity-map, field-map, and
    explicit-time gradients remain available.

    Trajectory coordinates are in radians per voxel. For an image stored as
    ``[..., ny, nx]``, the corresponding coordinate order is ``(ky, kx)``.

    Note that the data format is a little different from NuSENSE.
    The input/output size depends on the sensitivity maps.
    The input dimension is [nbatch, 1, nx, ny, (nz)], and the output is [nbatch, ncoil, nshot, nfe].

    Attributes:
        smaps: tensor with dimension [batch, ncoil, nx, ny, (nz)] (must have a batch dimension). Sensitivity maps.
        zmap: tensor with dimension [batch, nx, ny, (nz)]. Off-resonance effects in Hz. ref: DOI: 10.1109/TSP.2005.853152
        traj: tensor with dimension [nbatch (or 1), ndimension, nshot, nreadout]
        norm: normalization of the fft ('ortho' or None)
        L: int, number of segmentation
        dt: float, dwell time in ms
        nbins: int, granularity of exponential approximation
        numpoints: int, number of interpolation points in gridding
        grid_size: float, oversampling ratio (>1)
        T: readout times in ms with shape [nfe]. If omitted, sample ``j`` is
            acquired at ``j * dt``.
        backend: ``"auto"`` (default), ``"finufft"``, or ``"torchkbnufft"``
        eps: requested FINUFFT relative precision
    """

    def __init__(
        self,
        smaps: Tensor,
        zmap: Tensor,
        traj: Tensor,
        norm: str = "ortho",
        L: int = 6,
        nbins: int = 20,
        dt: float = 4e-3,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
        T: Tensor | None = None,
        backend: str = "auto",
        eps: float = 1e-6,
    ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        self.T = T
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        backend = self.backend
        if not isinstance(L, int) or L < 1:
            raise ValueError("L must be a positive integer")
        if not isinstance(nbins, int) or nbins < 1:
            raise ValueError("nbins must be a positive integer")
        if grid_size < 1:
            raise ValueError("grid size must be at least 1")

        self.nbatch = _resolve_b0_batch_size(smaps, zmap, traj)

        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        spatial_shape = _spatial_shape(smaps)
        transform_elements = math.prod(spatial_shape) + self.nshot * self.npoints
        bytes_per_segment = (
            4
            * self.nbatch
            * self.nc
            * transform_elements
            * torch.empty((), dtype=_complex_dtype_like(smaps)).element_size()
        )
        self._segment_chunk_size = min(
            self.L,
            max(1, _MAX_B0_WORKSPACE_BYTES // bytes_per_segment),
        )
        self.grid_size = _grid_shape(spatial_shape, grid_size)
        if backend == "torchkbnufft":
            self.A = tkbn.KbNufft(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
        else:
            self._finufft_backend = FinufftSenseBackend(
                im_size=spatial_shape,
                grid_size=self.grid_size,
                norm=norm,
                batchmode=True,
                sequential=False,
                eps=eps,
            )

        size_in = (self.nbatch, 1) + spatial_shape
        size_out = (self.nbatch, self.nc, self.nshot, self.npoints)

        self._coefficient_state = None
        self.B, self.C = self._refresh_coefficients(force=True)

        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        super().__init__(size_in, size_out)

    def to(self, device: torch.device | str):
        if self.backend == "finufft":
            self._finufft_backend.clear_plans()
        return super().to(device)

    def _coefficient_state_key(self) -> tuple:
        return _coefficient_state_key(
            self.zmap,
            self.T,
            self.dt,
            self.nbins,
            self.L,
        )

    def _refresh_coefficients(self, *, force: bool = False) -> tuple[Tensor, Tensor]:
        state = self._coefficient_state_key()
        trainable = self.zmap.requires_grad or (
            self.T is not None and self.T.requires_grad
        )
        if not force and not trainable and state == self._coefficient_state:
            return self.B, self.C

        times = readout_times(
            self.npoints,
            self.dt,
            template=self.zmap,
            times=self.T,
        )
        self.B, self.C, self.tl = time_segmentation_coefficients(
            self.zmap,
            batch=self.nbatch,
            bins=self.nbins,
            segments=self.L,
            times=times,
            complex_dtype=_complex_dtype_like(self.smaps),
        )
        self._coefficient_state = state
        return self.B, self.C

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            k: k-space data, [nbatch, ncoil, nshot, npoints]
        """
        B, C = self._refresh_coefficients()
        spatial_shape = _spatial_shape(self.smaps)
        output = torch.zeros(self.size_out, dtype=x.dtype, device=x.device)
        for start in range(0, self.L, self._segment_chunk_size):
            stop = min(start + self._segment_chunk_size, self.L)
            segment_count = stop - start
            segment_coefficients = C[start:stop]
            coil_images = (
                x.unsqueeze(0) * segment_coefficients
            ) * self.smaps.unsqueeze(0)
            modes = coil_images.permute(
                1,
                0,
                2,
                *range(3, coil_images.ndim),
            ).reshape(self.nbatch, segment_count * self.nc, *spatial_shape)
            if self.backend == "finufft":
                transformed = finufft_type2(
                    modes,
                    self.traj,
                    self._finufft_backend,
                )
            else:
                transformed = _kb_type2(
                    modes,
                    self.traj,
                    self.A,
                    self.AT,
                    self.norm,
                )
            segments = transformed.reshape(
                self.nbatch,
                segment_count,
                self.nc,
                self.nshot,
                self.npoints,
            ).permute(1, 0, 2, 3, 4)
            output = output + (B[start:stop] * segments).sum(dim=0)
        return output

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            k: k-space data, [nbatch, ncoil, nshot, npoints]
        Returns:
            x: [nbatch, 1, nx, ny (nz)]
        """
        B, C = self._refresh_coefficients()
        spatial_shape = _spatial_shape(self.smaps)
        y = x
        output = torch.zeros(self.size_in, dtype=y.dtype, device=y.device)
        for start in range(0, self.L, self._segment_chunk_size):
            stop = min(start + self._segment_chunk_size, self.L)
            segment_count = stop - start
            weighted = (y.unsqueeze(0) * B[start:stop].conj()).permute(1, 0, 2, 3, 4)
            samples = weighted.reshape(
                self.nbatch,
                segment_count * self.nc,
                self.nshot * self.npoints,
            )
            if self.backend == "finufft":
                modes = finufft_type1(
                    samples,
                    self.traj,
                    self._finufft_backend,
                )
            else:
                modes = _kb_type1(
                    samples,
                    self.traj,
                    self.A,
                    self.AT,
                    self.norm,
                )
            coil_images = modes.reshape(
                self.nbatch,
                segment_count,
                self.nc,
                *spatial_shape,
            )
            segment_images = (coil_images * self.smaps.unsqueeze(1).conj()).sum(
                dim=2, keepdim=True
            )
            coefficients = C[start:stop].permute(
                1,
                0,
                2,
                *range(3, C.ndim),
            )
            output = output + (coefficients.conj() * segment_images).sum(dim=1)
        return output


class GmriGram(LinearMap):
    r"""
    Toeplitz approximation to the B0-informed MRI normal operator.

    Autocorrelation time segmentation follows MIRT and Fessler et al.,
    IEEE TSP 2005, producing a Hermitian O(L) approximation.

    The automatic backend uses an installed FINUFFT or cuFINUFFT library on a
    supported device, and otherwise uses torchkbnufft. The input and output
    dimensions are both [nbatch, 1, nx, ny, (nz)]. Trainable ``zmap``, ``traj``,
    or ``T`` uses the exact composed ``Gmri.H * Gmri`` path so gradients are
    preserved.

    Attributes:
        smaps: tensor with dimension [batch, ncoil, nx, ny, (nz)] (must have a batch dimension). Sensitivity maps.
        zmap: tensor with dimension [batch, nx, ny, (nz)]. Off-resonance effects in Hz. ref: DOI: 10.1109/TSP.2005.853152
        traj: tensor with dimension [nbatch (or 1), ndimension, nshot, nreadout]
        norm: normalization of the fft ('ortho' or None)
        L: int, number of segmentation
        dt: float, dwell time in ms
        nbins: int, granularity of exponential approximation
        numpoints: int, number of interpolation points in gridding
        grid_size: float, oversampling ratio (>1)
        T: readout times in ms with shape [nfe]. If omitted, sample ``j`` is
            acquired at ``j * dt``.
        backend: ``"auto"`` (default), ``"finufft"``, or ``"torchkbnufft"``
        eps: requested FINUFFT relative precision for kernel construction
    """

    def __init__(
        self,
        smaps: Tensor,
        zmap: Tensor,
        traj: Tensor,
        norm: str = "ortho",
        L: int = 6,
        nbins: int = 20,
        dt: float = 4e-3,
        numpoints: int | list[int] = 6,
        grid_size: float = 2,
        T: Tensor | None = None,
        backend: str = "auto",
        eps: float = 1e-6,
    ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        self.T = T
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        if not isinstance(L, int) or L < 1:
            raise ValueError("L must be a positive integer")
        if not isinstance(nbins, int) or nbins < 1:
            raise ValueError("nbins must be a positive integer")
        if grid_size < 1:
            raise ValueError("grid size must be at least 1")
        backend = self.backend

        self.nbatch = _resolve_b0_batch_size(smaps, zmap, traj)

        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        spatial_shape = _spatial_shape(smaps)
        self.grid_size = _grid_shape(spatial_shape, grid_size)
        size_in = (self.nbatch, 1) + spatial_shape

        kernel_dtype = _complex_dtype_like(smaps)
        padded_elements = math.prod(2 * size for size in spatial_shape)
        # Allow room for padded modes, kernels, FFTs, products, and scratch.
        bytes_per_mode = (
            8
            * self.nbatch
            * padded_elements
            * torch.empty((), dtype=kernel_dtype).element_size()
        )
        modes_per_chunk = max(1, _MAX_B0_WORKSPACE_BYTES // bytes_per_mode)
        self._segment_chunk_size = min(self.L, modes_per_chunk)
        self._coil_chunk_size = min(
            self.nc,
            max(1, modes_per_chunk // self._segment_chunk_size),
        )
        estimated_kernel_bytes = (
            self.L
            * self.nbatch
            * padded_elements
            * torch.empty((), dtype=kernel_dtype).element_size()
        )
        memory_fallback = estimated_kernel_bytes > _MAX_B0_KERNEL_BYTES
        trainable_parameters = (
            zmap.requires_grad
            or traj.requires_grad
            or (T is not None and T.requires_grad)
        )
        self._uses_direct_gram = memory_fallback or trainable_parameters
        if self._uses_direct_gram:
            if memory_fallback:
                warnings.warn(
                    "The estimated B0 Toeplitz kernels require "
                    f"{estimated_kernel_bytes / 1024**3:.2f} GiB; "
                    "using direct Gmri.H*Gmri.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._direct_operator = Gmri(
                smaps=smaps,
                zmap=zmap,
                traj=traj,
                norm=norm,
                L=L,
                nbins=nbins,
                dt=dt,
                numpoints=numpoints,
                grid_size=grid_size,
                T=T,
                backend=backend,
                eps=eps,
            )
            self._sync_direct_aliases()
            super().__init__(size_in, size_in)
            return

        times = readout_times(
            self.npoints,
            self.dt,
            template=self.zmap,
            times=self.T,
        )
        self.B, self.C, self.tl = time_segmentation_coefficients(
            self.zmap,
            batch=self.nbatch,
            bins=self.nbins,
            segments=self.L,
            times=times,
            complex_dtype=_complex_dtype_like(self.smaps),
            autocorrelation=True,
        )
        self._coefficient_state = self._coefficient_state_key()

        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        self._build_toeplitz_kernels(spatial_shape, numpoints)
        self._trajectory_state = _tensor_state_key(self.traj)

        super().__init__(size_in, size_in)

    def _build_toeplitz_kernels(
        self,
        spatial_shape: tuple[int, ...],
        numpoints: int | list[int],
    ) -> None:
        if self.backend == "torchkbnufft":
            self.toep_op = tkbn.ToepNufft()
            self.kernel = [
                tkbn.calc_toeplitz_kernel(
                    self.traj,
                    list(spatial_shape),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                    norm=self.norm,
                    weights=self._segment_weights(segment),
                )
                for segment in range(self.L)
            ]
            return

        self._finufft_backend = FinufftSenseBackend(
            im_size=spatial_shape,
            grid_size=self.grid_size,
            norm=self.norm,
            batchmode=True,
            sequential=False,
            eps=self.eps,
        )
        self.kernel = [
            self._finufft_backend.toeplitz_kernel(
                self.traj,
                _complex_dtype_like(self.smaps),
                weights=self._segment_weights(segment),
            )
            for segment in range(self.L)
        ]

    def _sync_direct_aliases(self) -> None:
        direct = self._direct_operator
        for name in ("smaps", "zmap", "traj", "T", "B", "C", "tl"):
            setattr(self, name, getattr(direct, name))

    def _check_direct_aliases(self) -> None:
        direct = self._direct_operator
        for name in ("smaps", "zmap", "traj", "T"):
            if getattr(self, name) is not getattr(direct, name):
                raise RuntimeError(
                    f"{name} was replaced on a direct GmriGram operator; "
                    "create a new operator instead."
                )
        for name in ("dt", "nbins", "L"):
            if getattr(self, name) != getattr(direct, name):
                raise RuntimeError(
                    f"{name} changed on a direct GmriGram operator; "
                    "create a new operator instead."
                )

    def _coefficient_state_key(self) -> tuple:
        return _coefficient_state_key(
            self.zmap,
            self.T,
            self.dt,
            self.nbins,
            self.L,
        )

    def _check_fixed_coefficients(self) -> None:
        if self._coefficient_state_key() != self._coefficient_state:
            raise RuntimeError(
                "The field map or readout times changed after the Toeplitz "
                "kernels were constructed; create a new GmriGram operator."
            )

    def _segment_weights(self, segment: int) -> Tensor:
        return (
            self.B[segment]
            .expand(-1, -1, self.nshot, -1)
            .reshape(self.nbatch, 1, self.nshot * self.npoints)
        )

    def _check_fixed_trajectory(self) -> None:
        if _tensor_state_key(self.traj) != self._trajectory_state:
            raise RuntimeError(
                "The trajectory changed after the Toeplitz kernel "
                "was constructed; create a new GmriGram operator."
            )

    def to(self, device: torch.device | str):
        if self._uses_direct_gram:
            self._check_direct_aliases()
            self._direct_operator.to(device)
            self._sync_direct_aliases()
            return self

        self._check_fixed_coefficients()
        self._check_fixed_trajectory()
        if self.backend == "finufft":
            self._finufft_backend.clear_plans()
        result = super().to(device)
        self._coefficient_state = self._coefficient_state_key()
        self._trajectory_state = _tensor_state_key(self.traj)
        return result

    def _filter_toeplitz_modes(
        self,
        modes: Tensor,
        kernel_stack: Tensor,
    ) -> Tensor:
        """Apply one stack of Toeplitz kernels to batched segment/coil modes."""
        batch, segments, coils = modes.shape[:3]
        spatial_shape = tuple(modes.shape[3:])

        if self.backend == "finufft":
            kernels = kernel_stack.transpose(0, 1)
            kernels = kernels.expand(
                batch,
                segments,
                coils,
                *kernels.shape[3:],
            ).reshape(batch, segments * coils, *kernels.shape[3:])
            filtered = self._finufft_backend.toeplitz_filter(
                modes.reshape(batch, segments * coils, *spatial_shape),
                kernels,
            )
        else:
            kernels = kernel_stack.transpose(0, 1).reshape(
                batch * segments,
                1,
                *kernel_stack.shape[2:],
            )
            filtered = self.toep_op(
                modes.reshape(batch * segments, coils, *spatial_shape),
                kernels,
                norm=self.norm,
            )

        return filtered.reshape(batch, segments, coils, *spatial_shape)

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        if self._uses_direct_gram:
            self._check_direct_aliases()
            result = self._direct_operator.H(self._direct_operator(x))
            self._sync_direct_aliases()
            return result

        self._check_fixed_coefficients()
        self._check_fixed_trajectory()

        spatial_shape = _spatial_shape(self.smaps)
        output = torch.zeros_like(x)
        for segment_start in range(0, self.L, self._segment_chunk_size):
            segment_stop = min(
                segment_start + self._segment_chunk_size,
                self.L,
            )
            segment_count = segment_stop - segment_start
            coefficients = self.C[segment_start:segment_stop]
            segment_images = x.unsqueeze(0) * coefficients
            accumulated = torch.zeros(
                self.nbatch,
                segment_count,
                1,
                *spatial_shape,
                dtype=x.dtype,
                device=x.device,
            )

            kernels = torch.stack(self.kernel[segment_start:segment_stop])
            for coil_start in range(0, self.nc, self._coil_chunk_size):
                coil_stop = min(coil_start + self._coil_chunk_size, self.nc)
                smaps = self.smaps[:, coil_start:coil_stop]
                modes = (segment_images * smaps.unsqueeze(0)).transpose(0, 1)
                filtered = self._filter_toeplitz_modes(
                    modes,
                    kernels,
                )
                accumulated = accumulated + (smaps.unsqueeze(1).conj() * filtered).sum(
                    dim=2, keepdim=True
                )

            coefficients = coefficients.transpose(0, 1)
            output = output + (coefficients.conj() * accumulated).sum(dim=1)
        return output

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        return self._apply(x)
