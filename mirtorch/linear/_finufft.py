"""Differentiable FINUFFT/cuFINUFFT primitives for non-Cartesian MRI."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .util import nufft_trajectory_vjp, reduce_broadcast_batch_gradient


def _as_batched_image(image: Tensor, batchmode: bool) -> Tensor:
    return image if batchmode else image.unsqueeze(0).unsqueeze(0)


def _as_batched_smaps(smaps: Tensor, batchmode: bool) -> Tensor:
    return smaps if batchmode else smaps.unsqueeze(0)


def _as_batched_traj(traj: Tensor, batchmode: bool) -> Tensor:
    return traj if batchmode else traj.unsqueeze(0)


def _as_batched_samples(samples: Tensor, batchmode: bool) -> Tensor:
    return samples if batchmode else samples.unsqueeze(0)


def _restore_image_batch(image: Tensor, batchmode: bool) -> Tensor:
    return image if batchmode else image[0, 0]


def _restore_samples_batch(samples: Tensor, batchmode: bool) -> Tensor:
    return samples if batchmode else samples[0]


def _expand_batch(tensor: Tensor, batch: int) -> Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(
        f"Cannot broadcast a tensor with batch size {tensor.shape[0]} to {batch}"
    )


def _complex_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.complex64:
        return "complex64"
    if dtype == torch.complex128:
        return "complex128"
    raise TypeError(
        "The FINUFFT backend supports torch.complex64 and torch.complex128, "
        f"not {dtype}."
    )


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError(
        "The FINUFFT backend supports torch.complex64 and torch.complex128, "
        f"not {dtype}."
    )


@dataclass
class FinufftSenseBackend:
    """Plan-caching FINUFFT backend for non-Cartesian MRI operators."""

    im_size: tuple[int, ...]
    grid_size: tuple[int, ...]
    norm: str | None
    batchmode: bool
    sequential: bool
    eps: float
    max_plans: int = 32
    _plans: OrderedDict[tuple[Any, ...], Any] = field(default_factory=OrderedDict)
    _plan_point_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = field(
        default_factory=dict
    )
    _coordinate_cache: OrderedDict[tuple[Any, ...], Tensor] = field(
        default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if not 1 <= len(self.im_size) <= 3:
            raise ValueError("FINUFFT supports one-, two-, and three-dimensional data")
        if self.norm not in (None, "ortho"):
            raise ValueError("FINUFFT norm must be None or 'ortho'")
        if self.eps <= 0:
            raise ValueError("FINUFFT eps must be positive")
        if self.max_plans < 1:
            raise ValueError("max_plans must be positive")

    @property
    def scale(self) -> float:
        if self.norm is None:
            return 1.0
        return 1.0 / math.sqrt(math.prod(self.grid_size))

    def _library(self, device: torch.device):
        if device.type == "cpu":
            module_name = "finufft"
        elif device.type == "cuda":
            module_name = "cufinufft"
        else:
            raise RuntimeError(
                "The FINUFFT backend supports CPU and CUDA tensors; "
                f"received device {device}."
            )

        if module_name == "finufft":
            self._check_macos_openmp_runtime()

        try:
            return importlib.import_module(module_name)
        except ImportError as error:
            install_target = (
                "finufft" if device.type == "cpu" else "MIRTorch[cufinufft]"
            )
            raise ImportError(
                f"{module_name} is required for FINUFFT on {device.type}. "
                f"Install it with `pip install {install_target}`."
            ) from error

    @staticmethod
    def _check_macos_openmp_runtime() -> None:
        if sys.platform != "darwin":
            return

        finufft_spec = importlib.util.find_spec("finufft")
        if finufft_spec is None or finufft_spec.origin is None:
            return

        torch_omp = Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"
        finufft_omp = (
            Path(finufft_spec.origin).resolve().parent / ".dylibs" / "libomp.dylib"
        )
        if torch_omp.exists() and finufft_omp.exists():
            raise RuntimeError(
                "The installed PyTorch and FINUFFT macOS wheels bundle separate "
                "OpenMP runtimes, which abort the process when FINUFFT executes. "
                "The CPU FINUFFT backend is disabled for this combination. Use "
                "the CUDA backend on Linux or a supported build where both "
                "packages share one OpenMP runtime."
            )

    def _plan(
        self,
        nufft_type: int,
        n_trans: int,
        dtype: torch.dtype,
        device: torch.device,
        mode_size: tuple[int, ...] | None = None,
        coordinate_slot: tuple[Any, ...] | None = None,
    ):
        mode_size = self.im_size if mode_size is None else mode_size
        dtype_name = _complex_dtype_name(dtype)
        device_index = device.index
        stream_pointer: int | None = None
        kwargs: dict[str, Any] = {"modeord": 0}
        if device.type == "cuda":
            stream_pointer = torch.cuda.current_stream(device).cuda_stream
            kwargs["gpu_stream"] = stream_pointer

        key = (
            nufft_type,
            n_trans,
            dtype_name,
            device.type,
            device_index,
            stream_pointer,
            mode_size,
            self.eps,
            coordinate_slot,
        )
        plan = self._plans.get(key)
        if plan is None:
            library = self._library(device)
            isign = -1 if nufft_type == 2 else 1
            plan = library.Plan(
                nufft_type,
                mode_size,
                n_trans=n_trans,
                eps=self.eps,
                isign=isign,
                dtype=dtype_name,
                **kwargs,
            )
            self._plans[key] = plan
            while len(self._plans) > self.max_plans:
                old_key, _ = self._plans.popitem(last=False)
                self._plan_point_signatures.pop(old_key, None)
        else:
            self._plans.move_to_end(key)
        return key, plan

    @staticmethod
    def _coordinate_identity(coordinates: Tensor) -> tuple[Any, ...]:
        storage = coordinates.untyped_storage()
        return (
            storage.data_ptr(),
            coordinates.storage_offset(),
            tuple(coordinates.shape),
            tuple(coordinates.stride()),
            coordinates.device.type,
            coordinates.device.index,
        )

    @classmethod
    def _coordinate_signature(cls, coordinates: Tensor) -> tuple[Any, ...]:
        return (
            *cls._coordinate_identity(coordinates),
            coordinates._version,
            coordinates.dtype,
        )

    def _converted_coordinates(
        self,
        coordinates: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, tuple[Any, ...]]:
        signature = (*self._coordinate_signature(coordinates), dtype)
        cached = self._coordinate_cache.get(signature)
        if cached is None:
            cached = coordinates.to(dtype=dtype).contiguous()
            self._coordinate_cache[signature] = cached
            while len(self._coordinate_cache) > self.max_plans:
                self._coordinate_cache.popitem(last=False)
        else:
            self._coordinate_cache.move_to_end(signature)
        return cached, signature

    def clear_plans(self) -> None:
        """Release cached native plans and converted trajectory coordinates."""
        self._plans.clear()
        self._plan_point_signatures.clear()
        self._coordinate_cache.clear()

    @staticmethod
    def _backend_array(tensor: Tensor):
        tensor = tensor.detach().contiguous()
        if tensor.device.type == "cpu":
            return tensor.numpy()
        return tensor

    @staticmethod
    def _torch_array(array: Any, template: Tensor) -> Tensor:
        if isinstance(array, Tensor):
            return array
        return torch.from_numpy(array).to(device=template.device)

    def _execute(
        self,
        nufft_type: int,
        data: Tensor,
        coordinates: Tensor,
        mode_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        n_trans = data.shape[0]
        coordinate_slot = self._coordinate_identity(coordinates)
        plan_key, plan = self._plan(
            nufft_type,
            n_trans,
            data.dtype,
            data.device,
            mode_size=mode_size,
            coordinate_slot=coordinate_slot,
        )
        coordinates, point_signature = self._converted_coordinates(
            coordinates,
            _real_dtype(data.dtype),
        )
        if self._plan_point_signatures.get(plan_key) != point_signature:
            backend_coordinates = [
                self._backend_array(coordinates[dimension])
                for dimension in range(coordinates.shape[0])
            ]
            plan.setpts(*backend_coordinates)
            self._plan_point_signatures[plan_key] = point_signature
        output = plan.execute(self._backend_array(data))
        return self._torch_array(output, data)

    def _transform(
        self,
        nufft_type: int,
        data: Tensor,
        traj: Tensor,
        mode_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        batch, transforms = data.shape[:2]
        if traj.shape[0] not in (1, batch):
            raise ValueError(
                f"Trajectory batch {traj.shape[0]} cannot broadcast to {batch}"
            )

        if traj.shape[0] == 1:
            flat = data.reshape(batch * transforms, *data.shape[2:])
            if self.sequential:
                output = [
                    self._execute(
                        nufft_type,
                        item.unsqueeze(0),
                        traj[0],
                        mode_size=mode_size,
                    )
                    for item in flat
                ]
                flat_output = torch.cat(output, dim=0)
            else:
                flat_output = self._execute(
                    nufft_type,
                    flat,
                    traj[0],
                    mode_size=mode_size,
                )
            return flat_output.reshape(batch, transforms, *flat_output.shape[1:])

        output_batches = []
        for batch_index in range(batch):
            batch_data = data[batch_index]
            if self.sequential:
                output = [
                    self._execute(
                        nufft_type,
                        item.unsqueeze(0),
                        traj[batch_index],
                        mode_size=mode_size,
                    )
                    for item in batch_data
                ]
                batch_output = torch.cat(output, dim=0)
            else:
                batch_output = self._execute(
                    nufft_type,
                    batch_data,
                    traj[batch_index],
                    mode_size=mode_size,
                )
            output_batches.append(batch_output)
        return torch.stack(output_batches)

    def type2(self, modes: Tensor, traj: Tensor) -> Tensor:
        return self._transform(2, modes, traj)

    def type1(
        self,
        samples: Tensor,
        traj: Tensor,
        mode_size: tuple[int, ...] | None = None,
    ) -> Tensor:
        return self._transform(1, samples, traj, mode_size=mode_size)

    def toeplitz_kernel(
        self,
        traj: Tensor,
        dtype: torch.dtype,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Build the FFT response for a fixed-trajectory weighted normal operator."""
        _complex_dtype_name(dtype)
        if traj.requires_grad:
            raise ValueError(
                "The FINUFFT Toeplitz backend currently supports fixed "
                "trajectories only; detach the trajectory or compose the "
                "forward and adjoint operators when trajectory gradients are "
                "required."
            )

        traj_b = _as_batched_traj(traj, self.batchmode)
        if traj_b.device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                "A FINUFFT Toeplitz kernel must be constructed on CPU or CUDA; "
                f"received device {traj_b.device}. The cached kernel may be "
                "moved to another device after construction."
            )

        difference_size = tuple(2 * size - 1 for size in self.im_size)
        if weights is None:
            strengths = torch.ones(
                traj_b.shape[0],
                1,
                traj_b.shape[-1],
                dtype=dtype,
                device=traj_b.device,
            )
        else:
            if weights.ndim != 3 or weights.shape[1] != 1:
                raise ValueError("Toeplitz weights must have shape [batch, 1, samples]")
            if weights.shape[-1] != traj_b.shape[-1]:
                raise ValueError(
                    "Toeplitz weights and trajectory must have the same "
                    "number of samples"
                )
            batch = max(weights.shape[0], traj_b.shape[0])
            if traj_b.shape[0] not in (1, batch):
                raise ValueError(
                    "Toeplitz weights and trajectory have incompatible batch sizes"
                )
            strengths = _expand_batch(
                weights.to(device=traj_b.device, dtype=dtype),
                batch,
            )
        kernel = self.type1(
            strengths,
            traj_b,
            mode_size=difference_size,
        )

        spatial_dims = tuple(range(2, kernel.ndim))
        reflected = kernel.conj().flip(spatial_dims)
        kernel = (kernel + reflected) / 2
        kernel = kernel * self.scale**2

        embedding_size = tuple(2 * size for size in self.im_size)
        embedded = torch.zeros(
            *kernel.shape[:2],
            *embedding_size,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        difference_slices = (slice(None), slice(None)) + tuple(
            slice(0, size) for size in difference_size
        )
        embedded[difference_slices] = kernel
        return torch.fft.fftn(embedded, dim=spatial_dims)

    def sense_gram(
        self,
        image: Tensor,
        smaps: Tensor,
        kernel: Tensor,
        transform_chunk_size: int | None = None,
    ) -> Tensor:
        """Apply a cached Toeplitz embedding of the fixed-trajectory Gram."""
        if image.dtype != kernel.dtype or smaps.dtype != kernel.dtype:
            raise TypeError("image, sensitivity maps, and kernel must have one dtype")
        if image.device != kernel.device or smaps.device != kernel.device:
            raise ValueError(
                "image, sensitivity maps, and kernel must be on one device"
            )

        image_b = _as_batched_image(image, self.batchmode)
        smaps_b = _as_batched_smaps(smaps, self.batchmode)
        smaps_b = _expand_batch(smaps_b, image_b.shape[0])
        coil_images = image_b * smaps_b
        filtered = self.toeplitz_filter(
            coil_images,
            kernel,
            transform_chunk_size=transform_chunk_size,
        )
        output = (smaps_b.conj() * filtered).sum(dim=1, keepdim=True)
        return _restore_image_batch(output, self.batchmode)

    def toeplitz_filter(
        self,
        modes: Tensor,
        kernel: Tensor,
        *,
        transform_chunk_size: int | None = None,
    ) -> Tensor:
        """Apply embedded convolution to arbitrary batched transforms."""
        if modes.ndim != len(self.im_size) + 2:
            raise ValueError("modes must have shape [batch, transforms, *im_size]")
        if tuple(modes.shape[2:]) != self.im_size:
            raise ValueError(
                f"modes spatial shape must be {self.im_size}, not {modes.shape[2:]}"
            )
        if modes.dtype != kernel.dtype:
            raise TypeError("modes and kernel must have one dtype")
        if modes.device != kernel.device:
            raise ValueError("modes and kernel must be on one device")

        batch, transforms = modes.shape[:2]
        if kernel.shape[0] not in (1, batch):
            raise ValueError("kernel batch cannot broadcast to modes")
        if kernel.shape[1] not in (1, transforms):
            raise ValueError("kernel transforms cannot broadcast to modes")
        if transform_chunk_size is None:
            transform_chunk_size = transforms
        if transform_chunk_size < 1:
            raise ValueError("transform_chunk_size must be positive")

        spatial_dims = tuple(range(2, modes.ndim))
        embedding_size = tuple(2 * size for size in self.im_size)
        image_slices = (slice(None), slice(None)) + tuple(
            slice(0, size) for size in self.im_size
        )
        crop = (slice(None), slice(None)) + tuple(
            slice(size - 1, 2 * size - 1) for size in self.im_size
        )
        outputs = []
        for start in range(0, transforms, transform_chunk_size):
            stop = min(start + transform_chunk_size, transforms)
            chunk = modes[:, start:stop]
            padded = torch.zeros(
                batch,
                stop - start,
                *embedding_size,
                dtype=modes.dtype,
                device=modes.device,
            )
            padded[image_slices] = chunk
            kernel_chunk = kernel
            if kernel.shape[1] != 1:
                kernel_chunk = kernel[:, start:stop]
            filtered = torch.fft.ifftn(
                torch.fft.fftn(padded, dim=spatial_dims) * kernel_chunk,
                dim=spatial_dims,
            )
            outputs.append(filtered[crop])
        return torch.cat(outputs, dim=1)


class _FinufftType2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        modes: Tensor,
        traj: Tensor,
        backend: FinufftSenseBackend,
    ) -> Tensor:
        output = backend.type2(modes, traj) * backend.scale
        ctx.backend = backend
        ctx.save_for_backward(modes, traj)
        return output

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, *grad_outputs
    ):
        grad_samples: Tensor = grad_outputs[0]
        modes, traj = ctx.saved_tensors
        backend: FinufftSenseBackend = ctx.backend
        need_modes, need_traj, _ = ctx.needs_input_grad

        grad_modes = None
        if need_modes:
            grad_modes = backend.type1(grad_samples, traj) * backend.scale

        grad_traj = None
        if need_traj:
            grad_traj = (
                nufft_trajectory_vjp(
                    modes,
                    traj,
                    grad_samples,
                    backend.type2,
                )
                * backend.scale
            )
            grad_traj = reduce_broadcast_batch_gradient(grad_traj, traj.shape[0])
        return grad_modes, grad_traj, None


class _FinufftType1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        samples: Tensor,
        traj: Tensor,
        backend: FinufftSenseBackend,
    ) -> Tensor:
        output = backend.type1(samples, traj) * backend.scale
        ctx.backend = backend
        ctx.save_for_backward(samples, traj)
        return output

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, *grad_outputs
    ):
        grad_modes: Tensor = grad_outputs[0]
        samples, traj = ctx.saved_tensors
        backend: FinufftSenseBackend = ctx.backend
        need_samples, need_traj, _ = ctx.needs_input_grad

        grad_samples = None
        if need_samples:
            grad_samples = backend.type2(grad_modes, traj) * backend.scale

        grad_traj = None
        if need_traj:
            grad_traj = (
                nufft_trajectory_vjp(
                    grad_modes,
                    traj,
                    samples,
                    backend.type2,
                )
                * backend.scale
            )
            grad_traj = reduce_broadcast_batch_gradient(grad_traj, traj.shape[0])
        return grad_samples, grad_traj, None


def finufft_type2(
    modes: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    """Apply a batched, differentiable type-2 transform with backend scaling."""
    return cast(Tensor, _FinufftType2.apply(modes, traj, backend))


def finufft_type1(
    samples: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    """Apply a batched, differentiable type-1 transform with backend scaling."""
    return cast(Tensor, _FinufftType1.apply(samples, traj, backend))


def finufft_sense_forward(
    image: Tensor,
    smaps: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    """Apply SENSE encoding using the differentiable type-2 transform."""
    image_b = _as_batched_image(image, backend.batchmode)
    smaps_b = _as_batched_smaps(smaps, backend.batchmode)
    traj_b = _as_batched_traj(traj, backend.batchmode)
    smaps_b = _expand_batch(smaps_b, image_b.shape[0])
    samples = finufft_type2(image_b * smaps_b, traj_b, backend)
    return _restore_samples_batch(samples, backend.batchmode)


def finufft_sense_adjoint(
    samples: Tensor,
    smaps: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    """Apply the SENSE adjoint using the differentiable type-1 transform."""
    samples_b = _as_batched_samples(samples, backend.batchmode)
    smaps_b = _as_batched_smaps(smaps, backend.batchmode)
    traj_b = _as_batched_traj(traj, backend.batchmode)
    smaps_b = _expand_batch(smaps_b, samples_b.shape[0])
    coil_images = finufft_type1(samples_b, traj_b, backend)
    image = (smaps_b.conj() * coil_images).sum(dim=1, keepdim=True)
    return _restore_image_batch(image, backend.batchmode)
