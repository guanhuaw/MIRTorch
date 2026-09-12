"""Matched SPECT forward and back projectors for a parallel-hole collimator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch import Tensor

from .linearmaps import LinearMap


def _positive_scalar(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def required_rotation_shape(
    size_in: Sequence[int], dy: float, *, dx: float | None = None
) -> tuple[int, int]:
    """Return a centered grid covering all in-plane rotations of an image.

    The result is ``(detector_width, depth_count)`` at spacings ``(dx, dy)``.
    It includes the bilinear interpolation support and preserves input parity,
    so a zero-degree rotation is an exact embedding. Additional detector width
    may be needed to collect PSF tails; detector truncation is not corrected.
    """
    if len(size_in) != 3 or any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in size_in
    ):
        raise ValueError("size_in must contain three positive integers")
    dy = _positive_scalar(dy, "dy")
    dx = dy if dx is None else _positive_scalar(dx, "dx")
    nx, ny = size_in[:2]
    diameter = math.hypot((nx + 1) * dx, (ny + 1) * dy)
    width = math.ceil(diameter / dx) + 1
    depth = math.ceil(diameter / dy) + 1
    return width + (width - nx) % 2, depth + (depth - ny) % 2


def parallel_hole_psfs(
    distances: Tensor,
    kernel_shape: Sequence[int],
    voxel_size: Sequence[float],
    *,
    hole_diameter: float | Tensor,
    hole_length: float | Tensor,
    intrinsic_fwhm: float | Tensor,
) -> Tensor:
    """Construct normalized Gaussian parallel-hole collimator responses.

    ``distances`` has shape ``(depth, view)`` and contains source-to-crystal
    distances, *not* source-to-collimator-face distances. All lengths use the
    same unit; ``voxel_size=(dx, dz)`` gives detector-bin spacing. The resolution
    approximation is ``FWHM**2 = (hole_diameter / hole_length * distance)**2
    + intrinsic_fwhm**2``. Scalar or broadcastable tensor calibration parameters
    and distances remain differentiable. The output has shape
    ``(*kernel_shape, depth, view)`` with its origin at ``kernel_shape // 2``.

    This normalized, truncated Gaussian models primary collimator/detector
    blur, not scatter, septal penetration, or absolute sensitivity. Choose a
    kernel spanning several standard deviations and calibrate real scanners.
    See Zhou and Gindi (2009), doi:10.1088/0031-9155/54/14/005.
    """
    if distances.ndim != 2 or 0 in distances.shape:
        raise ValueError("distances must have nonempty shape (depth, view)")
    if not distances.is_floating_point():
        raise TypeError("distances must have a real floating-point dtype")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    if len(kernel_shape) != 2 or any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in kernel_shape
    ):
        raise ValueError("kernel_shape must contain two positive integers")
    if len(voxel_size) != 2:
        raise ValueError("voxel_size must contain (dx, dz)")
    dx, dz = (_positive_scalar(value, "voxel_size") for value in voxel_size)
    calibration = []
    for name, value in (
        ("hole_diameter", hole_diameter),
        ("hole_length", hole_length),
        ("intrinsic_fwhm", intrinsic_fwhm),
    ):
        raw_value = torch.as_tensor(value)
        if raw_value.is_complex() or raw_value.dtype == torch.bool:
            raise TypeError(f"{name} must be real")
        # Convert the original value to retain Python-scalar float64 precision.
        if isinstance(value, Tensor):
            # Cast before transfer so backward never requests float64 on MPS.
            value = value.to(dtype=distances.dtype).to(device=distances.device)
        else:
            value = torch.as_tensor(
                value, dtype=distances.dtype, device=distances.device
            )
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError(f"{name} must be finite and positive")
        try:
            shape = torch.broadcast_shapes(value.shape, distances.shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to distances.shape") from error
        if shape != distances.shape:
            raise ValueError(f"{name} must broadcast to distances.shape")
        calibration.append(value)
    diameter, length, intrinsic = calibration
    variance = ((diameter / length * distances).square() + intrinsic.square()) / (
        8.0 * math.log(2.0)
    )
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("the PSF variance must be representable and positive")
    x = (
        torch.arange(kernel_shape[0], dtype=distances.dtype, device=distances.device)
        - kernel_shape[0] // 2
    ) * dx
    z = (
        torch.arange(kernel_shape[1], dtype=distances.dtype, device=distances.device)
        - kernel_shape[1] // 2
    ) * dz
    radius_squared = x[:, None].square() + z[None, :].square()
    psfs = torch.exp(-0.5 * radius_squared[..., None, None] / variance)
    return psfs / psfs.sum(dim=(0, 1), keepdim=True)


def _validate_model(
    size_in: Sequence[int],
    size_out: Sequence[int],
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
) -> tuple[int, int, int, int]:
    if len(size_in) != 3:
        raise ValueError(f"size_in must have three dimensions, got {tuple(size_in)}")
    if len(size_out) != 3:
        raise ValueError(f"size_out must have three dimensions, got {tuple(size_out)}")
    if any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in size_in
    ):
        raise ValueError(
            f"size_in must contain positive integers, got {tuple(size_in)}"
        )
    if any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in size_out
    ):
        raise ValueError(
            f"size_out must contain positive integers, got {tuple(size_out)}"
        )
    if mumap.ndim != 3:
        raise ValueError(f"mumap must be three-dimensional, got shape {mumap.shape}")
    if psfs.ndim != 4:
        raise ValueError(f"psfs must be four-dimensional, got shape {psfs.shape}")
    if mumap.device != psfs.device:
        raise ValueError("mumap and psfs must be on the same device")
    if mumap.dtype != psfs.dtype:
        raise TypeError("mumap and psfs must have the same dtype")
    if not mumap.is_floating_point() or not psfs.is_floating_point():
        raise TypeError("mumap and psfs must have real floating-point dtypes")
    if tuple(mumap.shape) != tuple(size_in):
        raise ValueError(
            f"mumap shape {tuple(mumap.shape)} does not match size_in {tuple(size_in)}"
        )

    nx, ny, nz = (int(size) for size in size_in)
    nview = int(psfs.shape[-1])
    expected_out = (size_out[0], nz, nview)
    if tuple(size_out) != expected_out:
        raise ValueError(f"size_out must be {expected_out}, got {tuple(size_out)}")
    if any(size <= 0 for size in psfs.shape):
        raise ValueError("psfs must have nonempty spatial, depth, and view dimensions")
    for name, values in (("mumap", mumap), ("psfs", psfs)):
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
    _positive_scalar(dy, "dy")
    return nx, ny, nz, nview


def _validate_chunk_size(view_chunk_size: int | None, nview: int) -> int:
    if view_chunk_size is None:
        return nview
    if not isinstance(view_chunk_size, Integral) or isinstance(view_chunk_size, bool):
        raise TypeError("view_chunk_size must be a positive integer or None")
    if view_chunk_size <= 0:
        raise ValueError("view_chunk_size must be positive")
    return min(int(view_chunk_size), nview)


def _validate_signal(x: Tensor, model: Tensor, name: str) -> None:
    if not (x.is_floating_point() or x.is_complex()):
        raise TypeError(f"{name} must have a floating-point or complex dtype")
    if x.device != model.device:
        raise ValueError(f"{name}, mumap, and psfs must be on the same device")


def _uniform_angles(nview: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return torch.arange(nview, device=device, dtype=calc_dtype) * (360.0 / float(nview))


def _model_angles(
    angles: float | Sequence[float] | Tensor | None,
    nview: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if angles is None:
        return _uniform_angles(nview, device=device, dtype=dtype)
    angles = torch.as_tensor(angles)
    if angles.is_complex() or angles.dtype == torch.bool:
        raise TypeError("angles must be real")
    if angles.ndim == 0:
        angles = angles[None]
    if angles.ndim != 1 or angles.numel() != nview:
        raise ValueError(f"angles must contain {nview} values")
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    angles = angles.to(dtype=calc_dtype).to(device=device)
    if not torch.isfinite(angles).all():
        raise ValueError("angles must be finite")
    return angles


def _rotation_plan(
    nx: int,
    ny: int,
    angles: Tensor,
    *,
    weight_dtype: torch.dtype,
    rotation_shape: tuple[int, int],
    dx: float,
    dy: float,
) -> tuple[Tensor, Tensor]:
    """Return bilinear source indices and weights for counter-clockwise rotations."""
    calc_dtype = torch.float64 if weight_dtype == torch.float64 else torch.float32
    angles = angles.to(dtype=calc_dtype)
    rows, columns = torch.meshgrid(
        torch.arange(rotation_shape[0], device=angles.device, dtype=calc_dtype),
        torch.arange(rotation_shape[1], device=angles.device, dtype=calc_dtype),
        indexing="ij",
    )
    row_center = (nx - 1.0) / 2.0
    column_center = (ny - 1.0) / 2.0
    y = (rows - (rotation_shape[0] - 1.0) / 2.0) * dx
    x = (columns - (rotation_shape[1] - 1.0) / 2.0) * dy

    radians = torch.deg2rad(angles).reshape(-1, 1, 1)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    source_x = (cosine * x - sine * y) / dy + column_center
    source_y = (sine * x + cosine * y) / dx + row_center

    x0 = torch.floor(source_x)
    y0 = torch.floor(source_y)
    x1 = x0 + 1
    y1 = y0 + 1
    neighbors_x = torch.stack((x0, x1, x0, x1), dim=1)
    neighbors_y = torch.stack((y0, y0, y1, y1), dim=1)
    weights = torch.stack(
        (
            (x1 - source_x) * (y1 - source_y),
            (source_x - x0) * (y1 - source_y),
            (x1 - source_x) * (source_y - y0),
            (source_x - x0) * (source_y - y0),
        ),
        dim=1,
    )
    valid = (
        (neighbors_x >= 0)
        & (neighbors_x < ny)
        & (neighbors_y >= 0)
        & (neighbors_y < nx)
    )
    weights = (weights * valid).to(dtype=weight_dtype)
    indices = neighbors_y.clamp(0, nx - 1).to(torch.long) * ny + neighbors_x.clamp(
        0, ny - 1
    ).to(torch.long)
    return indices, weights


def _rotate_many(
    volume: Tensor,
    indices: Tensor,
    weights: Tensor,
) -> Tensor:
    nx, ny, nz = volume.shape
    flat = volume.reshape(nx * ny, nz)
    gathered = flat[indices]
    rotated = torch.sum(gathered * weights[..., None], dim=1)
    return rotated


def _rotate_many_adjoint(
    rotated: Tensor,
    indices: Tensor,
    weights: Tensor,
    size_in: Sequence[int],
) -> Tensor:
    nx, ny, nz = size_in
    contributions = (rotated[:, None] * weights.conj()[..., None]).reshape(-1, nz)
    output = torch.zeros(
        nx * ny,
        nz,
        dtype=rotated.dtype,
        device=rotated.device,
    )
    output = output.index_add(0, indices.reshape(-1), contributions)
    return output.reshape(nx, ny, nz)


def _attenuation_factors(rotated_mumap: Tensor, dy: float, attenuation: str) -> Tensor:
    """Survival probability for midpoint or uniform within-voxel emission."""
    optical_depth = float(dy) * rotated_mumap
    prefix = torch.cumsum(optical_depth, dim=2) - optical_depth
    if attenuation == "midpoint":
        return torch.exp(-prefix - 0.5 * optical_depth)
    # The polynomial supplies the removable singularity and its derivatives.
    small = optical_depth.abs() < 0.05
    denominator = torch.where(small, torch.ones_like(optical_depth), optical_depth)
    average = torch.where(
        small,
        1
        - optical_depth / 2
        + optical_depth.square() / 6
        - optical_depth.pow(3) / 24
        + optical_depth.pow(4) / 120
        - optical_depth.pow(5) / 720
        + optical_depth.pow(6) / 5040,
        -torch.expm1(-optical_depth) / denominator,
    )
    return torch.exp(-prefix) * average


def _same_padding(kernel_shape: Sequence[int]) -> tuple[int, int, int, int]:
    pad_x = int(kernel_shape[0]) - 1
    pad_z = int(kernel_shape[1]) - 1
    top = pad_x // 2
    left = pad_z // 2
    return left, pad_z - left, top, pad_x - top


def _blur_depths(volumes: Tensor, psfs: Tensor) -> Tensor:
    """Apply one spatially invariant PSF per view and depth plane."""
    nview, nx, ny, nz = volumes.shape
    channels = nview * ny
    signal = volumes.permute(0, 2, 1, 3).reshape(1, channels, nx, nz)
    kernels = (
        psfs.flip((0, 1))
        .permute(3, 2, 0, 1)
        .reshape(channels, 1, psfs.shape[0], psfs.shape[1])
    )
    kernels = kernels.to(dtype=signal.dtype)
    signal = F.pad(signal, _same_padding(psfs.shape[:2]))
    blurred = F.conv2d(signal, kernels, groups=channels)
    return blurred.reshape(nview, ny, nx, nz).permute(0, 2, 1, 3)


def _blur_depths_adjoint(views: Tensor, psfs: Tensor) -> Tensor:
    """Apply the exact Hermitian transpose of :func:`_blur_depths`."""
    nview, nx, nz = views.shape
    ny = int(psfs.shape[2])
    channels = nview * ny
    signal = views[:, None, :, :].expand(nview, ny, nx, nz).reshape(1, channels, nx, nz)
    kernels = (
        psfs.flip((0, 1))
        .permute(3, 2, 0, 1)
        .reshape(channels, 1, psfs.shape[0], psfs.shape[1])
    )
    kernels = kernels.to(dtype=signal.dtype).conj()
    padded = F.conv_transpose2d(signal, kernels, groups=channels)
    left, _, top, _ = _same_padding(psfs.shape[:2])
    cropped = padded[..., top : top + nx, left : left + nz]
    return cropped.reshape(nview, ny, nx, nz).permute(0, 2, 1, 3)


def _project(
    image: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    angles: Tensor,
    view_chunk_size: int,
    rotation_shape: tuple[int, int],
    dx: float,
    attenuation: str,
) -> Tensor:
    nx, ny, _ = image.shape
    chunks = []
    for start in range(0, angles.numel(), view_chunk_size):
        stop = min(start + view_chunk_size, angles.numel())
        chunk_indices, chunk_weights = _rotation_plan(
            nx,
            ny,
            angles[start:stop],
            weight_dtype=mumap.dtype,
            rotation_shape=rotation_shape,
            dx=dx,
            dy=dy,
        )
        rotated_image = _rotate_many(image, chunk_indices, chunk_weights)
        rotated_mumap = _rotate_many(mumap, chunk_indices, chunk_weights)
        attenuated = rotated_image * _attenuation_factors(
            rotated_mumap, dy, attenuation
        )
        psf_chunk = psfs[..., start:stop]
        chunks.append(_blur_depths(attenuated, psf_chunk).sum(dim=2))
    return torch.cat(chunks, dim=0).permute(1, 2, 0)


def _backproject(
    views: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    angles: Tensor,
    view_chunk_size: int,
    rotation_shape: tuple[int, int],
    dx: float,
    attenuation: str,
) -> Tensor:
    nx, ny, _ = mumap.shape
    output = torch.zeros(
        mumap.shape,
        dtype=views.dtype,
        device=views.device,
    )
    views_by_angle = views.permute(2, 0, 1)
    for start in range(0, angles.numel(), view_chunk_size):
        stop = min(start + view_chunk_size, angles.numel())
        chunk_indices, chunk_weights = _rotation_plan(
            nx,
            ny,
            angles[start:stop],
            weight_dtype=mumap.dtype,
            rotation_shape=rotation_shape,
            dx=dx,
            dy=dy,
        )
        rotated_mumap = _rotate_many(mumap, chunk_indices, chunk_weights)
        factors = _attenuation_factors(rotated_mumap, dy, attenuation)
        blurred = _blur_depths_adjoint(
            views_by_angle[start:stop], psfs[..., start:stop]
        )
        output = output + _rotate_many_adjoint(
            blurred * factors.conj(),
            chunk_indices,
            chunk_weights,
            mumap.shape,
        )
    return output


class SPECT(LinearMap):
    """Parallel-hole SPECT model with attenuation and depth-dependent PSFs.

    The forward model rotates each axial plane with bilinear interpolation,
    applies an attenuation integral along detector depth, blurs each
    depth plane by its PSF, and sums over depth. Backprojection is the exact
    discrete Hermitian transpose of those operations.

    Args:
        size_in: Image shape ``(nx, ny, nz)``.
        size_out: Projection shape ``(detector_width, nz, nview)``.
        mumap: Nonnegative attenuation coefficients with shape ``size_in``, in
            inverse length units consistent with ``dy``. These are not CT HU.
        psfs: Nonnegative impulse responses ``(px, pz, depth_count, nview)``.
            Their origin is ``(px // 2, pz // 2)``. They may encode calibrated
            sensitivity and are not renormalized. Depth index zero is nearest
            the detector. Each view has a centered rotated grid of shape
            ``(detector_width, depth_count, nz)``; provide PSFs calibrated at
            those physical depths. No PSF extrapolation is performed.
        dy: Image voxel size and integration-grid spacing along depth.
        view_chunk_size: Number of views processed together. Smaller chunks
            reduce peak memory; ``None`` processes all views in one chunk.
        angles: Projection angles in degrees. By default, views are uniformly
            spaced over 360 degrees.
        dx: In-plane lateral voxel and detector-bin spacing. Defaults to ``dy``.
        attenuation: ``"midpoint"`` preserves the original half-voxel model;
            ``"voxel"`` averages attenuation exactly within each uniform voxel.

    The input image represents activity per voxel up to an external count
    calibration, not activity density requiring an additional ``dy`` factor.
    Use :func:`required_rotation_shape` for a grid covering the full rotated
    image. Legacy image-sized grids can clip activity *or attenuation* outside
    their rotated support. Finite detector boundaries also discard PSF tails.
    Bilinear resampling remains a discretization approximation, not an exactly
    count-conserving rotation. Scatter and septal penetration are not modeled;
    a known additive background belongs in the measurement likelihood.
    """

    def __init__(
        self,
        size_in: Sequence[int],
        size_out: Sequence[int],
        mumap: Tensor,
        psfs: Tensor,
        dy: float,
        view_chunk_size: int | None = 8,
        angles: float | Sequence[float] | Tensor | None = None,
        *,
        dx: float | None = None,
        attenuation: str = "midpoint",
    ):
        _, _, _, nview = _validate_model(size_in, size_out, mumap, psfs, dy)
        super().__init__(size_in, size_out)
        self.mumap = mumap
        self.psfs = psfs
        self.dy = float(dy)
        self.dx = self.dy if dx is None else _positive_scalar(dx, "dx")
        if attenuation not in ("midpoint", "voxel"):
            raise ValueError("attenuation must be 'midpoint' or 'voxel'")
        self.attenuation = attenuation
        self.rotation_shape = (int(size_out[0]), int(psfs.shape[2]))
        self.view_chunk_size = _validate_chunk_size(view_chunk_size, nview)
        self.angles = _model_angles(
            angles,
            nview,
            device=mumap.device,
            dtype=mumap.dtype,
        )

    def _apply(self, x: Tensor) -> Tensor:
        _validate_signal(x, self.mumap, "image")
        return _project(
            x,
            self.mumap,
            self.psfs,
            self.dy,
            self.angles,
            self.view_chunk_size,
            self.rotation_shape,
            self.dx,
            self.attenuation,
        )

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        _validate_signal(x, self.mumap, "views")
        return _backproject(
            x,
            self.mumap,
            self.psfs,
            self.dy,
            self.angles,
            self.view_chunk_size,
            self.rotation_shape,
            self.dx,
            self.attenuation,
        )


def _spect_model(
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None,
    angles: float | Sequence[float] | Tensor | None = None,
    *,
    detector_width: int | None = None,
    dx: float | None = None,
    attenuation: str = "midpoint",
) -> SPECT:
    nview = int(psfs.shape[-1]) if psfs.ndim == 4 else 0
    size_in = tuple(mumap.shape)
    width = (
        detector_width if detector_width is not None else (size_in[0] if size_in else 0)
    )
    size_out = (width, size_in[2], nview) if len(size_in) == 3 else ()
    return SPECT(
        size_in,
        size_out,
        mumap,
        psfs,
        dy,
        view_chunk_size,
        angles,
        dx=dx,
        attenuation=attenuation,
    )


def project(
    image: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None = 8,
    *,
    detector_width: int | None = None,
    dx: float | None = None,
    attenuation: str = "midpoint",
) -> Tensor:
    """Project an image at uniformly spaced angles over 360 degrees."""
    return _spect_model(
        mumap,
        psfs,
        dy,
        view_chunk_size,
        detector_width=detector_width,
        dx=dx,
        attenuation=attenuation,
    )(image)


def project_angle(
    image: Tensor,
    mumap: Tensor,
    psf: Tensor,
    dy: float,
    viewangle: float | Tensor,
    *,
    detector_width: int | None = None,
    dx: float | None = None,
    attenuation: str = "midpoint",
) -> Tensor:
    """Project one view at ``viewangle`` degrees."""
    return _spect_model(
        mumap,
        psf[..., None],
        dy,
        1,
        viewangle,
        detector_width=detector_width,
        dx=dx,
        attenuation=attenuation,
    )(image)[..., 0]


def backproject_angle(
    view: Tensor,
    mumap: Tensor,
    psf: Tensor,
    dy: float,
    viewangle: float | Tensor,
    *,
    dx: float | None = None,
    attenuation: str = "midpoint",
) -> Tensor:
    """Backproject one view with the exact adjoint of :func:`project_angle`."""
    model = _spect_model(
        mumap,
        psf[..., None],
        dy,
        1,
        viewangle,
        detector_width=view.shape[0],
        dx=dx,
        attenuation=attenuation,
    )
    return model.H(view[..., None])


def backproject(
    views: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None = 8,
    *,
    dx: float | None = None,
    attenuation: str = "midpoint",
) -> Tensor:
    """Backproject uniformly spaced views with the exact discrete adjoint."""
    return _spect_model(
        mumap,
        psfs,
        dy,
        view_chunk_size,
        detector_width=views.shape[0],
        dx=dx,
        attenuation=attenuation,
    ).H(views)
