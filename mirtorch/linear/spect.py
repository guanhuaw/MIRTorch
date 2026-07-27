"""Matched SPECT forward and back projectors for a parallel-hole collimator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch import Tensor

from .linearmaps import LinearMap


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
    if any(not isinstance(size, Integral) or size <= 0 for size in size_in):
        raise ValueError(
            f"size_in must contain positive integers, got {tuple(size_in)}"
        )
    if any(not isinstance(size, Integral) or size <= 0 for size in size_out):
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
    expected_out = (nx, nz, nview)
    if tuple(size_out) != expected_out:
        raise ValueError(f"size_out must be {expected_out}, got {tuple(size_out)}")
    if psfs.shape[2] != ny:
        raise ValueError(f"psfs depth dimension must be {ny}, got {int(psfs.shape[2])}")
    if psfs.shape[0] <= 0 or psfs.shape[1] <= 0:
        raise ValueError("psfs must have nonempty spatial dimensions")
    if not isinstance(dy, Real) or isinstance(dy, bool):
        raise TypeError("dy must be a real scalar")
    if not math.isfinite(float(dy)) or dy <= 0:
        raise ValueError("dy must be finite and positive")
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
    angles: Sequence[float] | Tensor | None,
    nview: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if angles is None:
        return _uniform_angles(nview, device=device, dtype=dtype)
    angles = torch.as_tensor(angles, device=device)
    if angles.ndim == 0:
        angles = angles[None]
    if angles.ndim != 1 or angles.numel() != nview:
        raise ValueError(f"angles must contain {nview} values")
    if not angles.is_floating_point():
        angles = angles.to(dtype=torch.float32)
    if not torch.isfinite(angles).all():
        raise ValueError("angles must be finite")
    return angles


def _rotation_plan(
    nx: int,
    ny: int,
    angles: Tensor,
    *,
    weight_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Return bilinear source indices and weights for counter-clockwise rotations."""
    calc_dtype = torch.float64 if weight_dtype == torch.float64 else torch.float32
    angles = angles.to(dtype=calc_dtype)
    rows, columns = torch.meshgrid(
        torch.arange(nx, device=angles.device, dtype=calc_dtype),
        torch.arange(ny, device=angles.device, dtype=calc_dtype),
        indexing="ij",
    )
    rows = rows.reshape(1, -1)
    columns = columns.reshape(1, -1)
    row_center = (nx - 1.0) / 2.0
    column_center = (ny - 1.0) / 2.0
    y = rows - row_center
    x = columns - column_center

    radians = torch.deg2rad(angles).reshape(-1, 1)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    source_x = cosine * x - sine * y + column_center
    source_y = sine * x + cosine * y + row_center

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
    return rotated.reshape(indices.shape[0], nx, ny, nz)


def _rotate_many_adjoint(
    rotated: Tensor,
    indices: Tensor,
    weights: Tensor,
) -> Tensor:
    nview, nx, ny, nz = rotated.shape
    values = rotated.reshape(nview, 1, nx * ny, nz)
    contributions = (values * weights.conj()[..., None]).reshape(-1, nz)
    output = torch.zeros(
        nx * ny,
        nz,
        dtype=rotated.dtype,
        device=rotated.device,
    )
    output = output.index_add(0, indices.reshape(-1), contributions)
    return output.reshape(nx, ny, nz)


def _attenuation_factors(rotated_mumap: Tensor, dy: float) -> Tensor:
    """Trapezoidal line integral from the detector-facing edge to each voxel."""
    path_integral = torch.cumsum(rotated_mumap, dim=2) - 0.5 * rotated_mumap
    return torch.exp(-float(dy) * path_integral)


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
    kernels = psfs.permute(3, 2, 0, 1).reshape(
        channels, 1, psfs.shape[0], psfs.shape[1]
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
    kernels = psfs.permute(3, 2, 0, 1).reshape(
        channels, 1, psfs.shape[0], psfs.shape[1]
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
        )
        rotated_image = _rotate_many(image, chunk_indices, chunk_weights)
        rotated_mumap = _rotate_many(mumap, chunk_indices, chunk_weights)
        attenuated = rotated_image * _attenuation_factors(rotated_mumap, dy)
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
        )
        rotated_mumap = _rotate_many(mumap, chunk_indices, chunk_weights)
        attenuation = _attenuation_factors(rotated_mumap, dy)
        blurred = _blur_depths_adjoint(
            views_by_angle[start:stop], psfs[..., start:stop]
        )
        output = output + _rotate_many_adjoint(
            blurred * attenuation.conj(),
            chunk_indices,
            chunk_weights,
        )
    return output


class SPECT(LinearMap):
    """Parallel-hole SPECT model with attenuation and depth-dependent PSFs.

    The forward model rotates each axial plane with bilinear interpolation,
    applies a trapezoidal attenuation integral along detector depth, blurs each
    depth plane by its PSF, and sums over depth. Backprojection is the exact
    discrete Hermitian transpose of those operations.

    Args:
        size_in: Image shape ``(nx, ny, nz)``.
        size_out: Projection shape ``(nx, nz, nview)``.
        mumap: Real attenuation map with shape ``size_in``.
        psfs: Real PSFs with shape ``(px, pz, ny, nview)``.
        dy: Voxel size along the attenuation-integration direction.
        view_chunk_size: Number of views processed together. Smaller chunks
            reduce peak memory; ``None`` processes all views in one chunk.
        angles: Projection angles in degrees. By default, views are uniformly
            spaced over 360 degrees.
    """

    def __init__(
        self,
        size_in: Sequence[int],
        size_out: Sequence[int],
        mumap: Tensor,
        psfs: Tensor,
        dy: float,
        view_chunk_size: int | None = 8,
        angles: Sequence[float] | Tensor | None = None,
    ):
        _, _, _, nview = _validate_model(size_in, size_out, mumap, psfs, dy)
        super().__init__(size_in, size_out)
        self.mumap = mumap
        self.psfs = psfs
        self.dy = float(dy)
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
        )


def _spect_model(
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None,
    angles: Sequence[float] | Tensor | None = None,
) -> SPECT:
    nview = int(psfs.shape[-1]) if psfs.ndim == 4 else 0
    size_in = tuple(mumap.shape)
    size_out = (size_in[0], size_in[2], nview) if len(size_in) == 3 else ()
    return SPECT(
        size_in,
        size_out,
        mumap,
        psfs,
        dy,
        view_chunk_size,
        angles,
    )


def project(
    image: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None = 8,
) -> Tensor:
    """Project an image at uniformly spaced angles over 360 degrees."""
    return _spect_model(mumap, psfs, dy, view_chunk_size)(image)


def project_angle(
    image: Tensor,
    mumap: Tensor,
    psf: Tensor,
    dy: float,
    viewangle: float | Tensor,
) -> Tensor:
    """Project one view at ``viewangle`` degrees."""
    return _spect_model(mumap, psf[..., None], dy, 1, viewangle)(image)[..., 0]


def backproject_angle(
    view: Tensor,
    mumap: Tensor,
    psf: Tensor,
    dy: float,
    viewangle: float | Tensor,
) -> Tensor:
    """Backproject one view with the exact adjoint of :func:`project_angle`."""
    model = _spect_model(mumap, psf[..., None], dy, 1, viewangle)
    return model.H(view[..., None])


def backproject(
    views: Tensor,
    mumap: Tensor,
    psfs: Tensor,
    dy: float,
    view_chunk_size: int | None = 8,
) -> Tensor:
    """Backproject uniformly spaced views with the exact discrete adjoint."""
    return _spect_model(mumap, psfs, dy, view_chunk_size).H(views)
