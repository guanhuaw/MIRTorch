"""
Basic linear operators, including diagonal matrix, convolution and first-order finite difference.
More on the way ...
2021-02. Guanhua Wang and Keyue Zhu, University of Michigan
"""

import copy
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mirtorch.util import compile_callable, should_compile

from .linearmaps import LinearMap
from .util import DiffFunc, DiffFunc_adj, dim_conv


class Diff1d(LinearMap):
    """
    A 1st order finite difference operator.

    Attributes:
        dim: assign the dimension to apply operation
    """

    def __init__(self, size_in: Sequence[int], dim: int, mode="reflexive"):
        if not isinstance(dim, (int, np.integer)):
            raise TypeError("dim must be an integer")
        dim = int(dim)
        if not -len(size_in) <= dim < len(size_in):
            raise ValueError(f"dim={dim} is out of range for shape {tuple(size_in)}")
        if mode not in ("reflexive", "periodic"):
            raise ValueError("mode must be either 'reflexive' or 'periodic'")

        size_out = list(size_in)
        if mode == "reflexive":
            size_out[dim] -= 1
            if size_out[dim] < 1:
                raise ValueError(
                    "reflexive finite differences require at least two samples"
                )
        super().__init__(size_in, size_out)
        self.dim = dim
        self.mode = mode

    def _apply(self, x):
        return DiffFunc.apply(x, self.dim, self.mode)

    def _apply_adjoint(self, y):
        return DiffFunc_adj.apply(y, self.dim, self.mode)


class Diffnd(LinearMap):
    """
    A multidimensional finite difference operator, with the periodic boundary condition.

    Attributes:
        dims: assign the dimension to apply operation
    """

    def __init__(self, size_in: Sequence[int], dims: Sequence[int]):
        self.dims = sorted(dims)
        size_out = copy.copy(list(size_in))
        size_out[self.dims[0]] = size_out[self.dims[0]] * len(dims)
        super().__init__(size_in, size_out)

    def _apply(self, x):
        diff = []
        for i in range(len(self.dims)):
            diff.append(DiffFunc.apply(x, self.dims[i], "periodic"))
        return torch.cat(diff, dim=self.dims[0])

    def _apply_adjoint(self, y):
        x = torch.zeros(self.size_in).to(y)
        for i in range(len(self.dims)):
            x += DiffFunc_adj.apply(
                torch.narrow(
                    y,
                    self.dims[0],
                    i * self.size_in[self.dims[0]],
                    self.size_in[self.dims[0]],
                ),
                self.dims[i],
                "periodic",
            )
        return x


class Diff2dgram(LinearMap):
    """
    A little more efficient way to implement the gram operator for the Gram (A'A) of finite difference.
    Apply to last two dimensions, with the reflexive boundary condition.
    Real-valued CUDA inputs are compiled automatically by default when the
    installed PyTorch supports ``torch.compile``. Complex inputs use eager
    execution because current Inductor releases do not improve this path.
    """

    def __init__(self, size_in: Sequence[int], compile: bool = True):
        super().__init__(size_in, size_in)
        self.compile = compile
        self._compiled_rtr = None

    def RtR(self, x):
        return torch.cat(
            (
                (x[..., 0, :] - x[..., 1, :]).unsqueeze(-2),
                (2 * x[..., 1:-1, :] - x[..., :-2, :] - x[..., 2:, :]),
                (x[..., -1, :] - x[..., -2, :]).unsqueeze(-2),
            ),
            dim=-2,
        ) + torch.cat(
            (
                (x[..., 0] - x[..., 1]).unsqueeze(-1),
                (2 * x[..., 1:-1] - x[..., :-2] - x[..., 2:]),
                (x[..., -1] - x[..., -2]).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _apply(self, x):
        if should_compile(self.compile, x):
            if self._compiled_rtr is None:
                self._compiled_rtr = compile_callable(self.RtR)
            return self._compiled_rtr(x)
        return self.RtR(x)

    def _apply_adjoint(self, x):
        return self._apply(x)


class Diff3dgram(LinearMap):
    """
    A little more efficient way to implement the gram operator for the Gram of finite difference, with the reflexive boundary condition.
    Apply to last three dimensions.
    """

    def __init__(self, size_in: Sequence[int]):
        super().__init__(size_in, size_in)

    def RtR(self, x):
        return (
            torch.cat(
                (
                    (x[..., 0, :, :] - x[..., 1, :, :]).unsqueeze(-3),
                    (2 * x[..., 1:-1, :, :] - x[..., :-2, :, :] - x[..., 2:, :, :]),
                    (x[..., -1, :, :] - x[..., -2, :, :]).unsqueeze(-3),
                ),
                dim=-3,
            )
            + torch.cat(
                (
                    (x[..., 0, :] - x[..., 1, :]).unsqueeze(-2),
                    (2 * x[..., 1:-1, :] - x[..., :-2, :] - x[..., 2:, :]),
                    (x[..., -1, :] - x[..., -2, :]).unsqueeze(-2),
                ),
                dim=-2,
            )
            + torch.cat(
                (
                    (x[..., 0] - x[..., 1]).unsqueeze(-1),
                    (2 * x[..., 1:-1] - x[..., :-2] - x[..., 2:]),
                    (x[..., -1] - x[..., -2]).unsqueeze(-1),
                ),
                dim=-1,
            )
        )

    def _apply(self, x):
        return self.RtR(x)

    def _apply_adjoint(self, x):
        return self.RtR(x)


class Diag(LinearMap):
    """
    Expand an input vector into a diagonal matrix.
    For example, x is an 5*5 image.
    So P should be also a 5*5 weight vector.
    P*x (pytorch multiplication here) = Diag{vec(P)}*vec(x)

    Attributes:
        P: the diagonal matrix
    """

    def __init__(self, P: Tensor):
        super().__init__(list(P.shape), list(P.shape))
        self.P = P

    def _apply(self, x):
        return self.P * x

    def _apply_adjoint(self, x):
        # conjugate here
        return torch.conj(self.P) * x


class Identity(LinearMap):
    """
    Identity mapping.
    """

    def __init__(self, size_in):
        size_out = size_in
        super().__init__(size_in, size_out)

    def _apply(self, x):
        return x

    def _apply_adjoint(self, x):
        return x


def _spatial_tuple(
    value: int | Sequence[int],
    ndim: int,
    name: str,
    *,
    allow_zero: bool,
) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        result = (int(value),) * ndim
    else:
        result = tuple(value)
        if len(result) != ndim:
            raise ValueError(f"{name} must contain exactly {ndim} values")
        if not all(isinstance(item, (int, np.integer)) for item in result):
            raise TypeError(f"{name} values must be integers")
        result = tuple(int(item) for item in result)

    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {comparison}")
    return result


def _convolution_output_padding(
    size_in: Sequence[int],
    size_out: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> tuple[int, ...]:
    output_padding = tuple(
        input_size - ((output_size - 1) * step - 2 * pad + dilate * (kernel - 1) + 1)
        for input_size, output_size, kernel, step, pad, dilate in zip(
            size_in,
            size_out,
            kernel_size,
            stride,
            padding,
            dilation,
            strict=True,
        )
    )
    if any(
        value < 0 or value >= step
        for value, step in zip(output_padding, stride, strict=True)
    ):
        raise ValueError(
            "the convolution parameters do not admit an adjoint with the "
            "declared input shape"
        )
    return output_padding


def _convolution_context(data: Tensor, weight: Tensor):
    """Disable lossy cuDNN TF32 for complex Hermitian-adjoint calculations."""
    if data.device.type != "cuda" or not (data.is_complex() or weight.is_complex()):
        return nullcontext()
    return torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    )


class _ConvolveNd(LinearMap):
    """Shared implementation for linear cross-correlation maps."""

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        ndim: int,
        bias,
        stride: int | Sequence[int],
        padding: int | Sequence[int],
        dilation: int | Sequence[int],
    ):
        operator_name = type(self).__name__
        if len(size_in) != ndim + 2:
            raise ValueError(
                f"input to {operator_name} must have {ndim + 2} dimensions"
            )
        if weight.ndim != ndim + 2:
            raise ValueError(
                f"weight for {operator_name} must have {ndim + 2} dimensions"
            )
        if bias is not None:
            raise ValueError(
                f"{operator_name} is a LinearMap and does not support bias; "
                "add an offset outside the operator for an affine convolution"
            )

        batch, in_channels, *input_shape = size_in
        out_channels, weight_in_channels, *kernel_shape = weight.shape
        if in_channels != weight_in_channels:
            raise ValueError(
                "input and weight must contain the same number of input channels"
            )

        self.stride = _spatial_tuple(stride, ndim, "stride", allow_zero=False)
        self.padding = _spatial_tuple(padding, ndim, "padding", allow_zero=True)
        self.dilation = _spatial_tuple(dilation, ndim, "dilation", allow_zero=False)
        output_shape = tuple(
            (input_size + 2 * pad - dilate * (kernel_size - 1) - 1) // step + 1
            for input_size, kernel_size, step, pad, dilate in zip(
                input_shape,
                kernel_shape,
                self.stride,
                self.padding,
                self.dilation,
                strict=True,
            )
        )
        if any(size < 1 for size in output_shape):
            raise ValueError("kernel and convolution parameters exceed the input size")

        super().__init__(size_in, (batch, out_channels, *output_shape))
        self.weight = weight
        self.bias = None
        self.output_padding = _convolution_output_padding(
            input_shape,
            output_shape,
            kernel_shape,
            self.stride,
            self.padding,
            self.dilation,
        )
        self._conv = (F.conv1d, F.conv2d, F.conv3d)[ndim - 1]
        self._conv_transpose = (
            F.conv_transpose1d,
            F.conv_transpose2d,
            F.conv_transpose3d,
        )[ndim - 1]

    def _apply(self, x):
        with _convolution_context(x, self.weight):
            return self._conv(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

    def _apply_adjoint(self, x):
        with _convolution_context(x, self.weight):
            return self._conv_transpose(
                x,
                self.weight.conj(),
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
            )


class Convolve1d(_ConvolveNd):
    """1D cross-correlation with its exact Hermitian transpose."""

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ):
        super().__init__(size_in, weight, 1, bias, stride, padding, dilation)


class Convolve2d(_ConvolveNd):
    """2D cross-correlation with its exact Hermitian transpose."""

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ):
        super().__init__(size_in, weight, 2, bias, stride, padding, dilation)


class Convolve3d(_ConvolveNd):
    """3D cross-correlation with its exact Hermitian transpose."""

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ):
        super().__init__(size_in, weight, 3, bias, stride, padding, dilation)


class Patch2D(LinearMap):
    """
    Patch operator to decompose image into blocks

    Attributes:
        kernel_size: int, isotropic kernel size
        stride: int, size of stride
    """

    def __init__(
        self,
        size_in: Sequence[int],
        size_kernel: int,
        stride: int = 1,
        padded: bool = False,
    ):
        self.size_in = size_in
        self.size_kernel = size_kernel
        self.stride = stride
        self.npatchx = dim_conv(size_in[2], size_kernel, stride)
        self.npatchy = dim_conv(size_in[3], size_kernel, stride)
        self.padded = padded
        if padded:
            self.size_out = (
                size_in[0],
                size_in[1],
                size_kernel * size_kernel,
                self.npatchx * self.npatchy,
            )
        else:
            self.size_out = (
                size_in[0],
                size_in[1],
                self.npatchx,
                self.npatchy,
                size_kernel,
                size_kernel,
            )
        super().__init__(self.size_in, self.size_out)

    def _apply(self, x) -> Tensor:
        """
        Args:
            x: [nbatch, nchannel, nx, ny]

        Returns:
            y: [nbatch, nchannel, npatchx, npatchy, kernel_size, kernel_size] (normal)
                [nbatch, nchannel, kernel_size*kernel_size, npatchx*npatchy] (padded)
        """
        x = (
            x.unfold(2, self.size_kernel, self.stride)
            .unfold(3, self.size_kernel, self.stride)
            .contiguous()
        )
        if self.padded:
            x = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4] * x.shape[5]
            ).permute(0, 1, 3, 2)
        return x

    def _apply_adjoint(self, x) -> Tensor:
        if self.padded:
            # to [nbatch, nchannel*kernel_size*kernel_size, npatchx*npatchy]
            x = x.reshape(
                self.size_out[0], self.size_out[1] * self.size_out[2], self.size_out[3]
            )
        else:
            # Permute to [nbatch, nchannel, kernel_size, kernel_size, npatchx, npatchy]
            x = x.permute(0, 1, 4, 5, 2, 3)
            # reshape
            x = x.reshape(
                self.size_in[0],
                self.size_in[1] * self.size_kernel * self.size_kernel,
                self.npatchx * self.npatchy,
            )
        return F.fold(
            x,
            output_size=self.size_in[2:],
            kernel_size=self.size_kernel,
            stride=self.stride,
        )


class Patch3D(LinearMap):
    """
    Patch operator to decompose 3D image into patches.
    Attributes:
        kernel_size: isotropic kernel size
        stride: size of stride
    """

    def __init__(
        self,
        size_in: Sequence[int],
        size_kernel: int,
        stride: int = 1,
        padded: bool = False,
    ):
        self.size_in = size_in
        self.size_kernel = size_kernel
        self.stride = stride
        self.npatchx = dim_conv(size_in[2], size_kernel, stride)
        self.npatchy = dim_conv(size_in[3], size_kernel, stride)
        self.npatchz = dim_conv(size_in[4], size_kernel, stride)
        self.padded = padded
        if padded:
            self.size_out = (
                size_in[0],
                size_in[1],
                size_kernel**3,
                self.npatchx * self.npatchy * self.npatchz,
            )
        else:
            self.size_out = (
                size_in[0],
                size_in[1],
                self.npatchx,
                self.npatchy,
                self.npatchz,
                size_kernel,
                size_kernel,
                size_kernel,
            )
        super().__init__(self.size_in, self.size_out)

    def _apply(self, x) -> Tensor:
        """
        Args:
            x: [nbatch, nchannel, nx, ny, nz]

        Returns:
            y: [nbatch, nchannel, npatchx, npatchy, npatchz, kernel_size, kernel_size, kernel_size] (normal)
             : [nbatch, nchannel,kernel_size**3, npatchx, npatchy, npatchz] (padded)


        """
        x = (
            x.unfold(2, self.size_kernel, self.stride)
            .unfold(3, self.size_kernel, self.stride)
            .unfold(4, self.size_kernel, self.stride)
            .contiguous()
        )
        if self.padded:
            return x.reshape(
                x.shape[0],
                x.shape[1],
                x.shape[2] * x.shape[3] * x.shape[4],
                x.shape[5] * x.shape[6] * x.shape[7],
            ).permute(0, 1, 3, 2)
        else:
            return x

    def _apply_adjoint(self, x) -> Tensor:
        # This code is following https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/71
        # Pytorch's fold only supports 2d, though it actually has vol2im function ...
        # First, do the fold on the last two dimensions
        # Permute to [nbatch, nchannel, kernel_size, npatchx, kernel_size, kernel_size, npatchy, npatchz]
        if self.padded:
            x = x.permute(0, 1, 3, 2).reshape(
                self.size_in[0],
                self.size_in[1],
                self.npatchx,
                self.npatchy,
                self.npatchz,
                self.size_kernel,
                self.size_kernel,
                self.size_kernel,
            )
        x = x.permute(0, 1, 5, 2, 6, 7, 3, 4).reshape(
            self.size_in[0],
            self.size_in[1] * self.npatchx * self.size_kernel**3,
            self.npatchy * self.npatchz,
        )
        x = F.fold(
            x,
            output_size=self.size_in[3:],
            kernel_size=self.size_kernel,
            stride=self.stride,
        )
        # New shape: [nbatch, nchannel. kernel_size*npatchx, ny, nz]
        # Now let's move on to the first dimension
        x = x.reshape(self.size_in[0], self.size_in[1] * self.size_kernel, -1)
        # [nbatch, nchannel*kernel_size, npatchx*ny*nz]
        x = F.fold(
            x,
            output_size=(self.size_in[2], self.size_in[3] * self.size_in[4]),
            kernel_size=(self.size_kernel, 1),
            stride=(self.stride, 1),
        )
        x = x.reshape(self.size_in)
        return x
