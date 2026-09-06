"""Portable, matched CT line projectors using Siddon's pixel intersections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch
from torch import Tensor

from .linearmaps import LinearMap


def _positive_length(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar in physical length units")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class CT(LinearMap):
    r"""Two-dimensional parallel-beam or flat-detector fan-beam CT.

    ``A(mu)`` computes pencil-ray line integrals of a piecewise-constant
    attenuation image using exact ray/pixel intersection lengths (Siddon,
    1985, DOI: 10.1118/1.595715). ``A.H`` is the matched Euclidean adjoint,
    not a filtered backprojection or inverse. Image derivatives, including
    higher derivatives, use ordinary PyTorch operations on CPU, CUDA, and MPS.

    Rows increase along physical +y and columns along +x; the image is centered
    at the origin. At angle zero, rays travel along +y and detector bins run
    along +x. Positive angles rotate both directions counter-clockwise.
    Distances and voxel sizes share one length unit, and ``mu`` is in its
    inverse (not Hounsfield units). Geometry is fixed at construction: gradient-
    requiring angles are rejected, and input angles are copied. Reconstruct
    the operator to change geometry; ``.to(device)`` moves its fixed plan.

    This is a monochromatic, zero-width-ray research model, without beam
    hardening, finite focal-spot/detector integration, or scatter transport.
    It is not a calibrated clinical scanner simulator.

    Args:
        image_shape: ``(ny, nx)`` image dimensions.
        angles: Nonempty 1D angles in radians. A floating tensor sets geometry
            precision and device; use float64 for double-precision validation.
        detector_count: Number of centered, equally spaced detector bins.
        voxel_size: Scalar or ``(dy, dx)`` pixel dimensions.
        detector_spacing: Physical spacing of detector bin centers.
        geometry: ``"parallel"`` or ``"fan"``.
        source_distance: Fan source-to-origin distance.
        detector_distance: Fan origin-to-detector-plane distance. Both fan
            distances must exceed the image half-diagonal, keeping the source
            and detector outside the whole image at every view.
        ray_chunk_size: Rays processed at once, bounding temporary workspace.
        max_cache_bytes: Maximum persistent intersection-plan storage. Larger
            plans are recomputed in chunks; zero disables caching. The default
            is 128 MiB. This limit excludes ray endpoints and autograd-saved
            tensors: differentiable runs may retain all chunk weights.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        angles: Sequence[float] | Tensor,
        detector_count: int,
        *,
        voxel_size: float | Sequence[float] = 1.0,
        detector_spacing: float = 1.0,
        geometry: str = "parallel",
        source_distance: float | None = None,
        detector_distance: float | None = None,
        ray_chunk_size: int = 1024,
        max_cache_bytes: int = 128 * 1024**2,
    ):
        if len(image_shape) != 2 or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in image_shape
        ):
            raise ValueError("image_shape must contain two positive integers")
        for name, value, minimum in (
            ("detector_count", detector_count, 1),
            ("ray_chunk_size", ray_chunk_size, 1),
            ("max_cache_bytes", max_cache_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if isinstance(voxel_size, Real):
            dy = dx = _positive_length(voxel_size, "voxel_size")
        else:
            if not isinstance(voxel_size, Sequence) or len(voxel_size) != 2:
                raise ValueError("voxel_size must be a scalar or (dy, dx)")
            dy, dx = (_positive_length(value, "voxel_size") for value in voxel_size)
        spacing = _positive_length(detector_spacing, "detector_spacing")
        if geometry not in ("parallel", "fan"):
            raise ValueError("geometry must be 'parallel' or 'fan'")
        angles = torch.as_tensor(angles)
        if angles.ndim != 1 or angles.numel() == 0:
            raise ValueError("angles must be a nonempty 1D sequence in radians")
        if angles.requires_grad:
            raise ValueError("CT geometry is fixed; angles must not require gradients")
        if angles.is_complex() or angles.dtype == torch.bool:
            raise TypeError("angles must be real")
        if not angles.is_floating_point():
            angles = angles.to(torch.get_default_dtype())
        if angles.dtype not in (torch.float32, torch.float64):
            raise TypeError("geometry requires float32 or float64 angles")
        if not torch.isfinite(angles).all():
            raise ValueError("angles must be finite")
        angles = angles.detach().clone()
        super().__init__(image_shape, (angles.numel(), detector_count))
        self._voxel_size = (dy, dx)
        self.geometry = geometry
        self._ray_chunk_size = int(ray_chunk_size)
        ny, nx = self.size_in
        radius = math.hypot(ny * dy, nx * dx) / 2
        source_radius = detector_radius = radius + max(dy, dx)
        if geometry == "fan":
            source_radius = _positive_length(source_distance, "source_distance")
            detector_radius = _positive_length(detector_distance, "detector_distance")
            if min(source_radius, detector_radius) <= radius:
                raise ValueError("fan distances must exceed the image half-diagonal")
        elif source_distance is not None or detector_distance is not None:
            raise ValueError("source/detector distances apply only to fan geometry")

        # Endpoints are (x, y); each detector samples its center pencil ray.
        axis = torch.stack((angles.cos(), angles.sin()), dim=-1)
        direction = torch.stack((-angles.sin(), angles.cos()), dim=-1)
        offsets = (
            torch.arange(detector_count, device=angles.device, dtype=angles.dtype)
            - (detector_count - 1) / 2
        ) * spacing
        detector = offsets[None, :, None] * axis[:, None, :]
        if geometry == "parallel":
            starts = detector - source_radius * direction[:, None, :]
            ends = detector + detector_radius * direction[:, None, :]
        else:
            starts = (-source_radius * direction[:, None, :]).expand_as(detector)
            ends = detector + detector_radius * direction[:, None, :]
        self._starts = starts.reshape(-1, 2)
        self._ends = ends.reshape(-1, 2)
        self._planes = (
            (torch.arange(nx + 1, device=angles.device, dtype=angles.dtype) - nx / 2)
            * dx,
            (torch.arange(ny + 1, device=angles.device, dtype=angles.dtype) - ny / 2)
            * dy,
        )
        estimated_bytes = (
            self._starts.shape[0] * (nx + ny + 3) * (8 + angles.element_size())
        )
        self._plan = (
            tuple(self._trace(start) for start in self._chunk_starts())
            if estimated_bytes <= max_cache_bytes
            else None
        )

    def _chunk_starts(self):
        return range(0, self._starts.shape[0], self._ray_chunk_size)

    def _trace(self, start: int) -> tuple[Tensor, Tensor]:
        """Intersect a ray chunk with all grid planes, then identify segments."""
        origin = self._starts[start : start + self._ray_chunk_size]
        direction = self._ends[start : start + self._ray_chunk_size] - origin
        crossings = [torch.zeros_like(origin[:, :1]), torch.ones_like(origin[:, :1])]
        for axis, planes in enumerate(self._planes):
            delta = direction[:, axis, None]
            parallel = delta == 0
            safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
            times = (planes[None, :] - origin[:, axis, None]) / safe_delta
            crossings.append(torch.where(parallel, torch.ones_like(times), times))
        times = torch.cat(crossings, dim=1).clamp(0, 1).sort(dim=1).values
        lengths = (times[:, 1:] - times[:, :-1]) * direction.square().sum(1).sqrt()[
            :, None
        ]
        midpoint = (
            origin[:, None, :]
            + ((times[:, 1:] + times[:, :-1]) / 2)[:, :, None] * direction[:, None, :]
        )
        ny, nx = self.size_in
        dy, dx = self._voxel_size
        ix = torch.floor(midpoint[..., 0] / dx + nx / 2).to(torch.long)
        iy = torch.floor(midpoint[..., 1] / dy + ny / 2).to(torch.long)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        indices = iy.clamp(0, ny - 1) * nx + ix.clamp(0, nx - 1)
        return indices, lengths * valid

    def _plans(self):
        return (
            self._plan
            if self._plan is not None
            else (self._trace(start) for start in self._chunk_starts())
        )

    def _validate_signal(self, value: Tensor):
        if value.device != self._starts.device:
            raise ValueError(
                "CT and its input must be on the same device; use A.to(device)"
            )
        if not (value.is_floating_point() or value.is_complex()):
            raise TypeError("CT input must have a real or complex floating dtype")

    def _apply(self, x: Tensor) -> Tensor:
        self._validate_signal(x)
        # MPS cannot scatter complex gradients from a gather operation.
        if x.device.type == "mps" and x.is_complex():
            return torch.complex(self._apply(x.real), self._apply(x.imag))
        flat = x.reshape(-1)
        projected = [
            (flat[indices] * weights.to(x.real.dtype)).sum(dim=1)
            for indices, weights in self._plans()
        ]
        return torch.cat(projected).reshape(self.size_out)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        self._validate_signal(x)
        if x.device.type == "mps" and x.is_complex():
            return torch.complex(
                self._apply_adjoint(x.real), self._apply_adjoint(x.imag)
            )
        flat = x.reshape(-1)
        output = torch.zeros(math.prod(self.size_in), device=x.device, dtype=x.dtype)
        for start, (indices, weights) in zip(self._chunk_starts(), self._plans()):
            values = flat[start : start + indices.shape[0], None] * weights.to(
                x.real.dtype
            )
            output = output.index_add(0, indices.reshape(-1), values.reshape(-1))
        return output.reshape(self.size_in)

    def expected_counts(
        self, mu: Tensor, incident: float | Tensor, background: float | Tensor = 0.0
    ) -> Tensor:
        r"""Return ``incident * exp(-A(mu)) + background`` (not a linear map).

        Inputs must be finite, real, and nonnegative. ``incident`` is the
        expected incident count at each detector bin, including source/detector
        calibration; it and the known additive background broadcast to
        ``size_out``. This method predicts the Poisson mean, not a noise draw.
        Gradients are supported for attenuation, incident counts, and background.
        """
        if not mu.is_floating_point():
            raise TypeError("attenuation must be real floating point")
        parameters = []
        for name, value in (
            ("attenuation", mu),
            ("incident", incident),
            ("background", background),
        ):
            raw = torch.as_tensor(value)
            if raw.is_complex() or raw.dtype == torch.bool:
                raise TypeError(f"{name} must be real")
            # Convert from the original scalar, preserving float64 precision;
            # validate after conversion so unrepresentable counts cannot pass.
            if isinstance(value, Tensor):
                # Cast on the source device so backward can restore CPU float64
                # without first asking Metal to construct a float64 tensor.
                value = value.to(dtype=mu.dtype).to(device=mu.device)
            else:
                value = torch.as_tensor(value, device=mu.device, dtype=mu.dtype)
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative")
            parameters.append(value)
        try:
            incident = torch.broadcast_to(parameters[1], self.size_out)
            background = torch.broadcast_to(parameters[2], self.size_out)
        except RuntimeError as error:
            raise ValueError(
                "incident/background must broadcast to the sinogram shape"
            ) from error
        return incident * torch.exp(-self(mu)) + background
