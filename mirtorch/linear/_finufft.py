"""FINUFFT/cuFINUFFT implementation for non-Cartesian SENSE."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


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


def _restore_smaps_batch(smaps: Tensor, batchmode: bool) -> Tensor:
    return smaps if batchmode else smaps[0]


def _restore_traj_batch(traj: Tensor, batchmode: bool) -> Tensor:
    return traj if batchmode else traj[0]


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


def _reduce_broadcast_gradient(gradient: Tensor, original_batch: int) -> Tensor:
    if original_batch == 1 and gradient.shape[0] != 1:
        return gradient.sum(dim=0, keepdim=True)
    return gradient


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
    _complex_dtype_name(dtype)
    raise AssertionError("unreachable")


@dataclass
class FinufftSenseBackend:
    """Plan-caching FINUFFT implementation for non-Cartesian SENSE operators."""

    im_size: tuple[int, ...]
    grid_size: tuple[int, ...]
    norm: str | None
    batchmode: bool
    sequential: bool
    eps: float
    _plans: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= len(self.im_size) <= 3:
            raise ValueError("FINUFFT supports one-, two-, and three-dimensional data")
        if self.norm not in (None, "ortho"):
            raise ValueError("FINUFFT norm must be None or 'ortho'")
        if self.eps <= 0:
            raise ValueError("FINUFFT eps must be positive")

    @property
    def scale(self) -> float:
        if self.norm is None:
            return 1.0
        return 1.0 / math.sqrt(math.prod(self.grid_size))

    def _library(self, device: torch.device):
        if device.type == "cpu":
            module_name = "finufft"
            extra = "finufft"
        elif device.type == "cuda":
            module_name = "cufinufft"
            extra = "cufinufft"
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
            raise ImportError(
                f"{module_name} is required for FINUFFT on {device.type}. "
                f"Install it with `pip install MIRTorch[{extra}]`."
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
        return plan

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
        plan = self._plan(
            nufft_type,
            n_trans,
            data.dtype,
            data.device,
            mode_size=mode_size,
        )
        coordinates = coordinates.to(dtype=_real_dtype(data.dtype)).contiguous()
        backend_coordinates = [
            self._backend_array(coordinates[dimension])
            for dimension in range(coordinates.shape[0])
        ]
        plan.setpts(*backend_coordinates)
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

        spatial_dims = tuple(range(2, coil_images.ndim))
        embedding_size = tuple(2 * size for size in self.im_size)
        padded = torch.zeros(
            *coil_images.shape[:2],
            *embedding_size,
            dtype=coil_images.dtype,
            device=coil_images.device,
        )
        image_slices = (slice(None), slice(None)) + tuple(
            slice(0, size) for size in self.im_size
        )
        padded[image_slices] = coil_images
        filtered = torch.fft.ifftn(
            torch.fft.fftn(padded, dim=spatial_dims) * kernel,
            dim=spatial_dims,
        )
        crop = (slice(None), slice(None)) + tuple(
            slice(size - 1, 2 * size - 1) for size in self.im_size
        )
        filtered = filtered[crop]
        output = (smaps_b.conj() * filtered).sum(dim=1, keepdim=True)
        return _restore_image_batch(output, self.batchmode)

    def sense_forward(
        self,
        image: Tensor,
        smaps: Tensor,
        traj: Tensor,
    ) -> Tensor:
        smaps = _expand_batch(smaps, image.shape[0])
        return self.type2(image * smaps, traj)

    def sense_adjoint(
        self,
        samples: Tensor,
        smaps: Tensor,
        traj: Tensor,
    ) -> Tensor:
        smaps = _expand_batch(smaps, samples.shape[0])
        coil_images = self.type1(samples, traj)
        return (smaps.conj() * coil_images).sum(dim=1, keepdim=True)

    def trajectory_vjp(
        self,
        coil_images: Tensor,
        traj: Tensor,
        grad_samples: Tensor,
    ) -> Tensor:
        real_dtype = _real_dtype(coil_images.dtype)
        weighted_images = []
        for dimension, size in enumerate(self.im_size):
            modes = torch.arange(
                -(size // 2),
                (size - 1) // 2 + 1,
                dtype=real_dtype,
                device=coil_images.device,
            )
            shape = [1] * len(self.im_size)
            shape[dimension] = size
            weighted_images.append(coil_images * modes.reshape(shape))

        batch, coils = coil_images.shape[:2]
        weighted = torch.stack(weighted_images, dim=1)
        weighted = weighted.reshape(
            batch,
            len(self.im_size) * coils,
            *self.im_size,
        )
        derivatives = -1j * self.type2(weighted, traj)
        derivatives = derivatives.reshape(
            batch,
            len(self.im_size),
            coils,
            grad_samples.shape[-1],
        )
        return (grad_samples.conj().unsqueeze(1) * derivatives).real.sum(dim=2)


class _FinufftSenseForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image: Tensor,
        smaps: Tensor,
        traj: Tensor,
        backend: FinufftSenseBackend,
    ) -> Tensor:
        image_b = _as_batched_image(image, backend.batchmode)
        smaps_b = _as_batched_smaps(smaps, backend.batchmode)
        traj_b = _as_batched_traj(traj, backend.batchmode)
        output = backend.sense_forward(image_b, smaps_b, traj_b) * backend.scale

        ctx.backend = backend
        ctx.save_for_backward(image, smaps, traj)
        return _restore_samples_batch(output, backend.batchmode)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        grad_samples: Tensor,
    ):
        image, smaps, traj = ctx.saved_tensors
        backend: FinufftSenseBackend = ctx.backend
        need_image, need_smaps, need_traj, _ = ctx.needs_input_grad

        image_b = _as_batched_image(image, backend.batchmode)
        smaps_b = _as_batched_smaps(smaps, backend.batchmode)
        traj_b = _as_batched_traj(traj, backend.batchmode)
        grad_samples_b = _as_batched_samples(grad_samples, backend.batchmode)
        smaps_expanded = _expand_batch(smaps_b, image_b.shape[0])

        grad_image = None
        grad_smaps = None
        if need_image or need_smaps:
            grad_coil_images = backend.type1(grad_samples_b, traj_b) * backend.scale
            if need_image:
                grad_image_b = (smaps_expanded.conj() * grad_coil_images).sum(
                    dim=1, keepdim=True
                )
                grad_image = _restore_image_batch(
                    grad_image_b,
                    backend.batchmode,
                )
            if need_smaps:
                grad_smaps_b = image_b.conj() * grad_coil_images
                grad_smaps_b = _reduce_broadcast_gradient(
                    grad_smaps_b,
                    smaps_b.shape[0],
                )
                grad_smaps = _restore_smaps_batch(
                    grad_smaps_b,
                    backend.batchmode,
                )

        grad_traj = None
        if need_traj:
            coil_images = image_b * smaps_expanded
            grad_traj_b = (
                backend.trajectory_vjp(
                    coil_images,
                    traj_b,
                    grad_samples_b,
                )
                * backend.scale
            )
            grad_traj_b = _reduce_broadcast_gradient(
                grad_traj_b,
                traj_b.shape[0],
            )
            grad_traj = _restore_traj_batch(grad_traj_b, backend.batchmode)

        return grad_image, grad_smaps, grad_traj, None


class _FinufftSenseAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        samples: Tensor,
        smaps: Tensor,
        traj: Tensor,
        backend: FinufftSenseBackend,
    ) -> Tensor:
        samples_b = _as_batched_samples(samples, backend.batchmode)
        smaps_b = _as_batched_smaps(smaps, backend.batchmode)
        traj_b = _as_batched_traj(traj, backend.batchmode)
        output = backend.sense_adjoint(samples_b, smaps_b, traj_b) * backend.scale

        ctx.backend = backend
        ctx.save_for_backward(samples, smaps, traj)
        return _restore_image_batch(output, backend.batchmode)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        grad_image: Tensor,
    ):
        samples, smaps, traj = ctx.saved_tensors
        backend: FinufftSenseBackend = ctx.backend
        need_samples, need_smaps, need_traj, _ = ctx.needs_input_grad

        samples_b = _as_batched_samples(samples, backend.batchmode)
        smaps_b = _as_batched_smaps(smaps, backend.batchmode)
        traj_b = _as_batched_traj(traj, backend.batchmode)
        grad_image_b = _as_batched_image(grad_image, backend.batchmode)
        smaps_expanded = _expand_batch(smaps_b, samples_b.shape[0])

        grad_samples = None
        if need_samples:
            grad_samples_b = (
                backend.sense_forward(
                    grad_image_b,
                    smaps_b,
                    traj_b,
                )
                * backend.scale
            )
            grad_samples = _restore_samples_batch(
                grad_samples_b,
                backend.batchmode,
            )

        grad_smaps = None
        if need_smaps:
            coil_images = backend.type1(samples_b, traj_b)
            grad_smaps_b = coil_images * grad_image_b.conj() * backend.scale
            grad_smaps_b = _reduce_broadcast_gradient(
                grad_smaps_b,
                smaps_b.shape[0],
            )
            grad_smaps = _restore_smaps_batch(
                grad_smaps_b,
                backend.batchmode,
            )

        grad_traj = None
        if need_traj:
            coil_inputs = grad_image_b * smaps_expanded
            grad_traj_b = (
                backend.trajectory_vjp(
                    coil_inputs,
                    traj_b,
                    samples_b,
                )
                * backend.scale
            )
            grad_traj_b = _reduce_broadcast_gradient(
                grad_traj_b,
                traj_b.shape[0],
            )
            grad_traj = _restore_traj_batch(grad_traj_b, backend.batchmode)

        return grad_samples, grad_smaps, grad_traj, None


def finufft_sense_forward(
    image: Tensor,
    smaps: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    return cast(Tensor, _FinufftSenseForward.apply(image, smaps, traj, backend))


def finufft_sense_adjoint(
    samples: Tensor,
    smaps: Tensor,
    traj: Tensor,
    backend: FinufftSenseBackend,
) -> Tensor:
    return cast(Tensor, _FinufftSenseAdjoint.apply(samples, smaps, traj, backend))
