"""
Discrete-to-discrete system matrices for MRI.
2021-02. Guanhua Wang, University of Michigan
"""

import sys

import numpy as np
import torch
import torchkbnufft as tkbn
from torch import Tensor
from torch.fft import fftn, ifftn

from ._finufft import (
    FinufftSenseBackend,
    finufft_sense_adjoint,
    finufft_sense_forward,
)
from .linearmaps import LinearMap
from .util import fftshift, ifftshift


def _resolve_nufft_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        if device.type == "cuda":
            return "finufft"
        if device.type == "cpu" and sys.platform != "darwin":
            return "finufft"
        return "torchkbnufft"
    if backend not in ("torchkbnufft", "finufft"):
        raise ValueError(
            "NUFFT backend must be 'auto', 'torchkbnufft', or 'finufft', "
            f"not {backend!r}."
        )
    return backend


def _complex_dtype_like(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.complex64)


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
        if self.norm == "ortho":
            x = ifftn(x, dim=self.dims, norm="ortho")
        elif self.norm == "forward":
            x = ifftn(x, dim=self.dims, norm="backward")
        else:
            x = ifftn(x, dim=self.dims, norm="forward")
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

    Important: When both smaps and masks have batch size 1, the operator will have
    size_in=[1, 1, nx, ny]. To use with larger batches, replicate one of the inputs.

    Example - Single sensitivity map for 10 time frames::

        smaps = torch.randn(1, 8, 128, 128)  # Single smap
        masks = torch.randn(1, 128, 128)     # Single mask

        # Replicate masks to desired batch size
        masks = masks.repeat(10, 1, 1)       # Now [10, 128, 128]
        sense = Sense(smaps, masks)          # size_in=[10, 1, 128, 128]

        x = torch.randn(10, 1, 128, 128)
        k = sense(x)                         # Works! Returns [10, 8, 128, 128]

        # Note: .repeat() is memory-efficient (uses views until modified)

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
        ncoil = smaps.shape[1] if batchmode else smaps.shape[0]

        if batchmode:
            # Determine batch size from inputs
            # Allow broadcasting: either can be size 1
            smaps_batch = smaps.shape[0]
            masks_batch = masks.shape[0]

            if smaps_batch == masks_batch:
                nbatch = smaps_batch
            elif smaps_batch == 1:
                nbatch = masks_batch
            elif masks_batch == 1:
                nbatch = smaps_batch
            else:
                raise ValueError(
                    f"Incompatible batch sizes: smaps.shape[0]={smaps_batch}, "
                    f"masks.shape[0]={masks_batch}. Must be equal or one must be 1."
                )

            # Set sizes
            size_in = (nbatch, 1) + tuple(smaps.shape[2:])
            size_out = (nbatch, ncoil) + tuple(smaps.shape[2:])
            dims = tuple(range(2, len(smaps.shape)))
            self.masks = masks.unsqueeze(1)  # Add channel dimension

            assert smaps.shape[2:] == masks.shape[1:], (
                f"Spatial dimensions mismatch: smaps {smaps.shape[2:]}, masks {masks.shape[1:]}"
            )
        else:
            size_in = tuple(smaps.shape[1:])
            size_out = tuple(smaps.shape)
            dims = tuple(range(1, len(smaps.shape)))
            self.masks = masks

            assert smaps.shape[1:] == masks.shape, (
                f"Spatial dimensions mismatch: smaps {smaps.shape[1:]}, masks {masks.shape}"
            )

        super().__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        self.smaps = smaps
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

    def _apply_adjoint(self, k: Tensor) -> Tensor:
        r"""
        Args:
            k:  tensor with dimension [batch, ncoil, nx, ny, (nz)] (batchmode=True) or [ncoil, nx, ny, nz]
        Returns:
            x:  tensor with dimension [batch, 1, nx, ny, (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        assert list(k.shape) == list(self.size_out), (
            f"Shape mismatch: expected {self.size_out}, got {k.shape}"
        )

        k = k * self.masks
        k = ifftshift(k, self.dims)

        if self.norm == "ortho":
            x = ifftn(k, dim=self.dims, norm="ortho")
        elif self.norm == "forward":
            x = ifftn(k, dim=self.dims, norm="backward")
        else:
            x = ifftn(k, dim=self.dims, norm="forward")

        x = fftshift(x, self.dims)

        if self.batchmode:
            x = (x * torch.conj(self.smaps)).sum(1, keepdim=True)
        else:
            x = (x * torch.conj(self.smaps)).sum(0)

        return x


class NuSense(LinearMap):
    r"""
    Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    The default implementation calls Matthew Muckley's Torchkbnufft toolbox:
    https://github.com/mmuckley/torchkbnufft. MIRTorch uses FINUFFT on
    supported CPUs, cuFINUFFT on CUDA, and torchkbnufft on Apple Metal.

    Broadcasting behavior:
    - If smaps or traj have shape [1, ...], they will be broadcast to the batch size
    - The batch size is determined by max(smaps.shape[0], traj.shape[0])
    - At least one of (smaps, traj) must have the desired batch size

    Important: When both smaps and traj have batch size 1, the operator will have
    size_in=[1, 1, nx, ny]. To use with larger batches, replicate one of the inputs.

    Example - Single sensitivity map for 10 time frames::

        smaps = torch.randn(1, 8, 128, 128)  # Single smap
        traj = torch.randn(1, 2, 1000)       # Single trajectory

        # Replicate trajectory to desired batch size
        traj = traj.repeat(10, 1, 1)         # Now [10, 2, 1000]
        nusense = NuSense(smaps, traj)       # size_in=[10, 1, 128, 128]

        x = torch.randn(10, 1, 128, 128)
        k = nusense(x)                       # Works! Returns [10, 8, 1000]

        # Note: .repeat() is memory-efficient (uses views until modified)

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
        assert grid_size >= 1, "grid size should be greater than 1"

        ncoil = smaps.shape[1] if batchmode else smaps.shape[0]

        if batchmode:
            # Determine batch size from inputs
            smaps_batch = smaps.shape[0]
            traj_batch = traj.shape[0]

            if smaps_batch == traj_batch:
                nbatch = smaps_batch
            elif smaps_batch == 1:
                nbatch = traj_batch
            elif traj_batch == 1:
                nbatch = smaps_batch
            else:
                raise ValueError(
                    f"Incompatible batch sizes: smaps.shape[0]={smaps_batch}, "
                    f"traj.shape[0]={traj_batch}. Must be equal or one must be 1."
                )

            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
            )
            if backend == "torchkbnufft":
                self.A = tkbn.KbNufft(
                    im_size=tuple(smaps.shape[2:]),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                ).to(smaps)
                self.AT = tkbn.KbNufftAdjoint(
                    im_size=tuple(smaps.shape[2:]),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                ).to(smaps)

            size_in = (
                nbatch,
                1,
            ) + tuple(smaps.shape[2:])
            size_out = (nbatch, ncoil, traj.shape[-1])
            super().__init__(size_in, size_out)
        else:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[1:]) * grid_size).astype(int)
            )
            if backend == "torchkbnufft":
                self.A = tkbn.KbNufft(
                    im_size=tuple(smaps.shape[1:]),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                ).to(smaps)
                self.AT = tkbn.KbNufftAdjoint(
                    im_size=tuple(smaps.shape[1:]),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                ).to(smaps)

            size_in = smaps.shape[1:]
            size_out = (smaps.shape[0], traj.shape[-1])
            super().__init__(size_in, size_out)

        if backend == "finufft":
            im_size = tuple(smaps.shape[2:] if batchmode else smaps.shape[1:])
            self._finufft_backend = FinufftSenseBackend(
                im_size=im_size,
                grid_size=self.grid_size,
                norm=norm,
                batchmode=batchmode,
                sequential=sequential,
                eps=eps,
            )

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

        if self.sequential:
            k = torch.zeros(self.size_out, dtype=x.dtype, device=x.device)
            if self.batchmode:
                for i in range(self.smaps.shape[1]):
                    k[:, i, ...] = self.A(
                        x,
                        self.traj,
                        smaps=self.smaps[:, i, ...].unsqueeze(1),
                        norm=self.norm,
                    ).squeeze(1)
                return k
            else:
                for i in range(self.smaps.shape[0]):
                    k[i, ...] = (
                        self.A(
                            x.unsqueeze(0).unsqueeze(0),
                            self.traj,
                            smaps=self.smaps[i, ...].unsqueeze(0).unsqueeze(0),
                            norm=self.norm,
                        )
                        .squeeze(0)
                        .squeeze(0)  # Remove batch and channel to get [npoints]
                    )
                return k
        else:
            if self.batchmode:
                return self.A(x, self.traj, smaps=self.smaps, norm=self.norm)
            else:
                return self.A(
                    x.unsqueeze(0).unsqueeze(0),
                    self.traj,
                    smaps=self.smaps.unsqueeze(0),
                    norm=self.norm,
                ).squeeze(0)  # Remove batch dimension only, keep [ncoil, npoints]

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        r"""
        Args:
            y： tensor with dimension [batch, ncoil, nshot*npoints] (batchmode=True) or [ncoil, nshot*npoints]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self.backend == "finufft":
            return finufft_sense_adjoint(
                y,
                self.smaps,
                self.traj,
                self._finufft_backend,
            )

        if self.sequential:
            x = torch.zeros(self.size_in, dtype=y.dtype, device=y.device)
            if self.batchmode:
                for i in range(self.smaps.shape[1]):
                    x += self.AT(
                        y[:, i, ...].unsqueeze(1),
                        self.traj,
                        smaps=self.smaps[:, i, ...].unsqueeze(1),
                        norm=self.norm,
                    )
                return x
            else:
                for i in range(self.smaps.shape[0]):
                    x += (
                        self.AT(
                            y[i, ...].unsqueeze(0).unsqueeze(0),
                            self.traj,
                            smaps=self.smaps[i, ...].unsqueeze(0).unsqueeze(0),
                            norm=self.norm,
                        )
                        .squeeze(0)
                        .squeeze(0)  # Remove batch and channel to get [H, W]
                    )
                return x
        else:
            if self.batchmode:
                return self.AT(y, self.traj, smaps=self.smaps, norm=self.norm)
            else:
                return (
                    self.AT(
                        y.unsqueeze(0),
                        self.traj,
                        smaps=self.smaps.unsqueeze(0),
                        norm=self.norm,
                    )
                    .squeeze(0)
                    .squeeze(0)  # Remove batch and channel dimensions to get [H, W]
                )


class NuSenseGram(LinearMap):
    r"""
    Gram operator (A'A) of the Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    The default implementation calls Matthew Muckley's Torchkbnufft toolbox.
    On CPU and CUDA, the default backend constructs a Toeplitz embedding for a
    fixed trajectory with FINUFFT or cuFINUFFT and applies it using PyTorch
    FFTs.

    Broadcasting behavior:
    - If smaps or traj have shape [1, ...], they will be broadcast to the batch size
    - The batch size is determined by max(smaps.shape[0], traj.shape[0])
    - At least one of (smaps, traj) must have the desired batch size

    Important: When both smaps and traj have batch size 1, the operator will have
    size_in=[1, 1, nx, ny]. To use with larger batches, replicate one of the inputs.

    Example - Single sensitivity map for 10 time frames::

        smaps = torch.randn(1, 8, 128, 128)  # Single smap
        traj = torch.randn(1, 2, 1000)       # Single trajectory

        # Replicate trajectory to desired batch size
        traj = traj.repeat(10, 1, 1)         # Now [10, 2, 1000]
        gram = NuSenseGram(smaps, traj)      # size_in=[10, 1, 128, 128]

        x = torch.randn(10, 1, 128, 128)
        y = gram(x)                          # Works! Returns [10, 1, 128, 128]

        # Note: .repeat() is memory-efficient (uses views until modified)

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
        if backend == "torchkbnufft":
            self.toep_op = tkbn.ToepNufft()

        if batchmode:
            # Determine batch size from inputs
            smaps_batch = smaps.shape[0]
            traj_batch = traj.shape[0]

            if smaps_batch == traj_batch:
                nbatch = smaps_batch
            elif smaps_batch == 1:
                nbatch = traj_batch
            elif traj_batch == 1:
                nbatch = smaps_batch
            else:
                raise ValueError(
                    f"Incompatible batch sizes: smaps.shape[0]={smaps_batch}, "
                    f"traj.shape[0]={traj_batch}. Must be equal or one must be 1."
                )

            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
            )
            im_size = tuple(smaps.shape[2:])

            size_in = (
                nbatch,
                1,
            ) + tuple(smaps.shape[2:])
            super().__init__(tuple(size_in), tuple(size_in))
        else:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[1:]) * grid_size).astype(int)
            )
            im_size = tuple(smaps.shape[1:])

            size_in = list(smaps.shape[1:])
            super().__init__(tuple(size_in), tuple(size_in))

        if backend == "torchkbnufft":
            self.kernel = tkbn.calc_toeplitz_kernel(
                traj,
                list(im_size),
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
                im_size=im_size,
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
            self._trajectory_version = traj._version

    def _check_fixed_trajectory(self) -> None:
        if self.traj.requires_grad:
            raise RuntimeError(
                "The FINUFFT Toeplitz backend currently supports fixed "
                "trajectories only."
            )
        if self.traj._version != self._trajectory_version:
            raise RuntimeError(
                "The trajectory changed after the FINUFFT Toeplitz kernel "
                "was constructed; create a new NuSenseGram operator."
            )

    def to(self, device: torch.device | str):
        if self.backend == "finufft":
            self._check_fixed_trajectory()
        result = super().to(device)
        if self.backend == "finufft":
            self._trajectory_version = self.traj._version
        return result

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self.backend == "finufft":
            self._check_fixed_trajectory()
            return self._finufft_backend.sense_gram(
                x,
                self.smaps,
                self.kernel,
            )

        if self.batchmode:
            return self.toep_op(x, self.kernel, smaps=self.smaps, norm=self.norm)
        else:
            return (
                self.toep_op(
                    x.unsqueeze(0).unsqueeze(0),
                    self.kernel,
                    smaps=self.smaps.unsqueeze(0),
                    norm=self.norm,
                )
                .squeeze(0)
                .squeeze(0)  # Remove batch and channel to get [H, W]
            )

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        if self.backend == "finufft":
            return self._apply(y)
        if self.batchmode:
            return self.toep_op(y, self.kernel, smaps=self.smaps, norm=self.norm)
        else:
            return (
                self.toep_op(
                    y.unsqueeze(0).unsqueeze(0),
                    self.kernel,
                    smaps=self.smaps.unsqueeze(0),
                    norm=self.norm,
                )
                .squeeze(0)
                .squeeze(0)  # Remove batch and channel to get [H, W]
            )


class Gmri(LinearMap):
    r"""
    B0-informed MRI encoding operator, following MIRT.

    FINUFFT is used by default on supported CPU systems and cuFINUFFT on
    CUDA. Apple Metal and macOS CPU use torchkbnufft.

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
        T: tensor with dimension [nfe]. Describe the time (in ms) of readout after excitation. When T is none,
           the readout is supposed to start immediately after the excitation.
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
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        backend = self.backend

        if smaps.device != zmap.device or smaps.device != traj.device:
            raise ValueError(
                "sensitivity maps, field map, and trajectory must be on one device"
            )

        # Determine batch size from inputs
        smaps_batch = smaps.shape[0]
        traj_batch = traj.shape[0]

        if smaps_batch == traj_batch:
            self.nbatch = smaps_batch
        elif smaps_batch == 1:
            self.nbatch = traj_batch
        elif traj_batch == 1:
            self.nbatch = smaps_batch
        else:
            raise ValueError(
                f"Incompatible batch sizes: smaps.shape[0]={smaps_batch}, "
                f"traj.shape[0]={traj_batch}. Must be equal or one must be 1."
            )
        if zmap.shape[0] not in (1, self.nbatch):
            raise ValueError(
                f"Incompatible zmap batch size: zmap.shape[0]={zmap.shape[0]}, "
                f"expected 1 or {self.nbatch}."
            )

        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        self.grid_size = tuple(
            np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
        )
        if backend == "torchkbnufft":
            self.A = tkbn.KbNufft(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
        else:
            self._finufft_backend = FinufftSenseBackend(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                norm=norm,
                batchmode=True,
                sequential=False,
                eps=eps,
            )

        size_in = (
            self.nbatch,
            1,
        ) + tuple(smaps.shape[2:])
        size_out = (self.nbatch, self.nc, self.nshot, self.npoints)

        complex_dtype = _complex_dtype_like(smaps)
        self.B = torch.zeros(
            self.L,
            self.nbatch,
            1,
            1,
            self.npoints,
            dtype=complex_dtype,
            device=smaps.device,
        )  # [L, batch, coil, shot, points]
        self.C = torch.zeros(
            (self.L, self.nbatch, 1) + tuple(self.smaps.shape[2:]),
            dtype=complex_dtype,
            device=smaps.device,
        )  # [L, batch, 1, nx, ny ...]

        for ib in range(self.nbatch):
            if T is None:
                t = np.linspace(0, dt * self.npoints, self.npoints)
            else:
                t = T.detach().cpu().numpy()

            # Handle broadcasting: zmap might be [1, ...] or [nbatch, ...]
            zmap_idx = 0 if zmap.shape[0] == 1 else ib
            b, c, _ = mri_exp_approx(
                zmap[zmap_idx].detach().cpu().numpy(),
                nbins,
                L,
                t,
            )

            self.B[:, ib, ...] = torch.as_tensor(
                np.transpose(b),
                dtype=complex_dtype,
                device=smaps.device,
            ).reshape(self.L, 1, 1, self.npoints)
            self.C[:, ib, 0, ...] = torch.as_tensor(
                np.transpose(c),
                dtype=complex_dtype,
                device=smaps.device,
            ).reshape((self.L,) + tuple(zmap.shape[1:]))

        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        super().__init__(size_in, size_out)

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            k: k-space data, [nbatch, ncoil, nshot, npoints]
        """
        y = torch.zeros(self.size_out, dtype=x.dtype, device=x.device)
        for il in range(self.L):
            if self.backend == "finufft":
                segment = finufft_sense_forward(
                    x * self.C[il],
                    self.smaps,
                    self.traj,
                    self._finufft_backend,
                )
            else:
                segment = self.A(
                    x * self.C[il],
                    self.traj,
                    smaps=self.smaps,
                    norm=self.norm,
                )
            y = y + self.B[il] * segment.reshape(self.size_out)
        return y

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        r"""
        Args:
            k: k-space data, [nbatch, ncoil, nshot, npoints]
        Returns:
            x: [nbatch, 1, nx, ny (nz)]
        """
        x = torch.zeros(self.size_in, dtype=y.dtype, device=y.device)
        for il in range(self.L):
            samples = (y * self.B[il].conj()).reshape(
                self.nbatch,
                self.nc,
                self.nshot * self.npoints,
            )
            if self.backend == "finufft":
                segment = finufft_sense_adjoint(
                    samples,
                    self.smaps,
                    self.traj,
                    self._finufft_backend,
                )
            else:
                segment = self.AT(
                    samples,
                    self.traj,
                    smaps=self.smaps,
                    norm=self.norm,
                )
            x = x + self.C[il].conj() * segment
        return x


class GmriGram(LinearMap):
    r"""
    Toeplitz approximation to the B0-informed MRI normal operator.

    The default autocorrelation time-segmentation method follows MIRT and
    Fessler et al., IEEE TSP 2005. It preserves a Hermitian O(L) Toeplitz
    approximation. ``gram_approximation="legacy"`` reproduces the previous
    MIRTorch coefficient construction for migration comparisons.

    FINUFFT is used by default on supported CPU systems and cuFINUFFT on
    CUDA. Apple Metal and macOS CPU use torchkbnufft. The input and output
    dimensions are both [nbatch, 1, nx, ny, (nz)].

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
        T: tensor with dimension [nfe]. Describe the time (in ms) of readout after excitation. When T is none,
           the readout is supposed to start immediately after the excitation.
        backend: ``"auto"`` (default), ``"finufft"``, or ``"torchkbnufft"``
        eps: requested FINUFFT relative precision for kernel construction
        gram_approximation: ``"autocorrelation"`` (default) or ``"legacy"``
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
        gram_approximation: str = "autocorrelation",
    ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        self.backend = _resolve_nufft_backend(backend, smaps.device)
        self.eps = eps
        if gram_approximation not in ("autocorrelation", "legacy"):
            raise ValueError("gram_approximation must be 'autocorrelation' or 'legacy'")
        self.gram_approximation = gram_approximation
        backend = self.backend

        if smaps.device != zmap.device or smaps.device != traj.device:
            raise ValueError(
                "sensitivity maps, field map, and trajectory must be on one device"
            )

        # Determine batch size from inputs
        smaps_batch = smaps.shape[0]
        traj_batch = traj.shape[0]

        if smaps_batch == traj_batch:
            self.nbatch = smaps_batch
        elif smaps_batch == 1:
            self.nbatch = traj_batch
        elif traj_batch == 1:
            self.nbatch = smaps_batch
        else:
            raise ValueError(
                f"Incompatible batch sizes: smaps.shape[0]={smaps_batch}, "
                f"traj.shape[0]={traj_batch}. Must be equal or one must be 1."
            )
        if zmap.shape[0] not in (1, self.nbatch):
            raise ValueError(
                f"Incompatible zmap batch size: zmap.shape[0]={zmap.shape[0]}, "
                f"expected 1 or {self.nbatch}."
            )

        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        self.grid_size = tuple(
            np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
        )

        size_in = (
            self.nbatch,
            1,
        ) + tuple(smaps.shape[2:])

        complex_dtype = _complex_dtype_like(smaps)
        self.B = torch.zeros(
            self.L,
            self.nbatch,
            1,
            1,
            self.npoints,
            dtype=complex_dtype,
            device=smaps.device,
        )  # [L, batch, coil, shot, points]
        self.C = torch.zeros(
            (self.L, self.nbatch, 1) + tuple(self.smaps.shape[2:]),
            dtype=complex_dtype,
            device=smaps.device,
        )  # [L, batch, 1, nx, ny ...]

        for ib in range(self.nbatch):
            if T is None:
                t = np.linspace(0, dt * self.npoints, self.npoints)
            else:
                t = T.detach().cpu().numpy()

            # Handle broadcasting: zmap might be [1, ...] or [nbatch, ...]
            zmap_idx = 0 if zmap.shape[0] == 1 else ib
            b, c, tl = mri_exp_approx(
                zmap[zmap_idx].detach().cpu().numpy(),
                nbins,
                L,
                t,
                autocorrelation=gram_approximation == "autocorrelation",
            )

            self.B[:, ib, ...] = torch.as_tensor(
                np.transpose(b),
                dtype=complex_dtype,
                device=smaps.device,
            ).reshape(self.L, 1, 1, self.npoints)
            self.C[:, ib, 0, ...] = torch.as_tensor(
                np.transpose(c),
                dtype=complex_dtype,
                device=smaps.device,
            ).reshape((self.L,) + tuple(zmap.shape[1:]))
            self.tl = torch.as_tensor(
                tl,
                dtype=zmap.real.dtype,
                device=smaps.device,
            )

        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        self.kernel = []

        if backend == "torchkbnufft":
            self.toep_op = tkbn.ToepNufft()
            for il in range(self.L):
                self.kernel.append(
                    tkbn.calc_toeplitz_kernel(
                        self.traj,
                        list(smaps.shape[2:]),
                        grid_size=self.grid_size,
                        numpoints=numpoints,
                        norm=self.norm,
                        weights=self._segment_weights(il),
                    )
                )
        else:
            self._finufft_backend = FinufftSenseBackend(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                norm=norm,
                batchmode=True,
                sequential=False,
                eps=eps,
            )
            for il in range(self.L):
                self.kernel.append(
                    self._finufft_backend.toeplitz_kernel(
                        self.traj,
                        smaps.dtype,
                        weights=self._segment_weights(il),
                    )
                )
            self._trajectory_version = self.traj._version

        super().__init__(tuple(size_in), tuple(size_in))

    def _segment_weights(self, segment: int) -> Tensor:
        return (
            self.B[segment]
            .expand(-1, -1, self.nshot, -1)
            .reshape(self.nbatch, 1, self.nshot * self.npoints)
        )

    def _check_fixed_trajectory(self) -> None:
        if self.traj.requires_grad:
            raise RuntimeError(
                "The FINUFFT Toeplitz backend currently supports fixed "
                "trajectories only."
            )
        if self.traj._version != self._trajectory_version:
            raise RuntimeError(
                "The trajectory changed after the FINUFFT Toeplitz kernel "
                "was constructed; create a new GmriGram operator."
            )

    def to(self, device: torch.device | str):
        if self.backend == "finufft":
            self._check_fixed_trajectory()
        result = super().to(device)
        if self.backend == "finufft":
            self._trajectory_version = self.traj._version
        return result

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        if self.backend == "finufft":
            self._check_fixed_trajectory()

        y = torch.zeros_like(x)
        for il in range(self.L):
            D = self.C[il]
            if self.backend == "finufft":
                segment = self._finufft_backend.sense_gram(
                    x * D,
                    self.smaps,
                    self.kernel[il],
                )
            else:
                segment = self.toep_op(
                    x * D,
                    self.kernel[il],
                    smaps=self.smaps,
                    norm=self.norm,
                )
            y = y + D.conj() * segment
        return y

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        return self._apply(x)


def mri_exp_approx(b0, bins, lseg, t, autocorrelation: bool = False):
    r"""
    From Sigpy: https://github.com/mikgroup/sigpy and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/
    Creates B [M*L] and Ct [L*N] matrices to approximate exp(-2i*pi*b0*t) [M*N]
    Args:
        b0: numpy array in dimension [nx, ny, nz], inhomogeneity matrix in Hz.
        bins: int, number of histogram bins to use.
        lseg: int, number of time segments.
        t: array, describing the readout times (ms).
        autocorrelation: use the field-map histogram autocorrelation for
            a Hermitian O(L) Gram approximation.
    Returns:
        3-element tuple containing:
            b: temporal interpolator [M, L]
            ct: off-resonance phase at each time segment center [L, N]
            tl: time segment centers [L]
    """

    # create time vector
    hist_wt, bin_edges = np.histogram(
        np.imag(2j * np.pi * np.ndarray.flatten(b0)), bins
    )

    # build B and Ct
    bin_width = bin_edges[1] - bin_edges[0]
    if autocorrelation:
        hist_wt = np.correlate(hist_wt, hist_wt, mode="full")
        bin_centers = np.arange(1 - bins, bins) * bin_width
    else:
        bin_centers = bin_edges[1:] - bin_width / 2
    zk = 0 + 1j * bin_centers
    tl = np.linspace(t[0], t[-1], lseg) / 1000  # time seg centers
    # calculate off-resonance phase @ each time seg, for histogram bins
    ch = np.exp(-np.expand_dims(tl, axis=1) @ np.expand_dims(zk, axis=0))
    w = np.diag(np.sqrt(hist_wt))
    p = np.linalg.pinv(w @ np.transpose(ch)) @ w
    b = p @ np.exp(-np.expand_dims(zk, axis=1) @ np.expand_dims(t, axis=0) / 1000)
    b = np.transpose(b)
    if autocorrelation:
        b = np.real(b)
    b0_v = np.expand_dims(2j * np.pi * np.ndarray.flatten(b0), axis=0)
    ct = np.transpose(np.exp(-np.expand_dims(tl, axis=1) @ b0_v))

    return b, ct, tl
