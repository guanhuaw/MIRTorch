"""
Basic linear operators, including diagonal matrix, convolution and first-order finite difference.
More on the way ...
2021-02. Guanhua Wang and Keyue Zhu, University of Michigan
"""

import torch
import torch.nn.functional as F
import copy
import numpy as np
from .linearmaps import LinearMap
from typing import Sequence
from torch import Tensor
from .util import DiffFunc, DiffFunc_adj, dim_conv


class Diff1d(LinearMap):
    """
    A 1st order finite difference operator.

    Attributes:
        dim: assign the dimension to apply operation
    """

    def __init__(self, size_in: Sequence[int], dim: int, mode="reflexive"):
        # TODO: determine size_out by size in
        size_out = copy.copy(size_in)
        size_out[dim] -= 1
        super(Diff1d, self).__init__(size_in, size_out)
        self.dim = dim
        self.mode = mode
        assert np.isscalar(
            dim
        ), "Please denote 1 dimension for a 1D finite difference operator"

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
        super(Diffnd, self).__init__(size_in, size_out)

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
    """

    def __init__(self, size_in: Sequence[int]):
        super(Diff2dgram, self).__init__(size_in, size_in)

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
        return self.RtR(x)

    def _apply_adjoint(self, x):
        return self.RtR(x)


class Diff3dgram(LinearMap):
    """
    A little more efficient way to implement the gram operator for the Gram of finite difference, with the reflexive boundary condition.
    Apply to last three dimensions.
    """

    def __init__(self, size_in: Sequence[int]):
        super(Diff3dgram, self).__init__(size_in, size_in)

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
        super(Diag, self).__init__(list(P.shape), list(P.shape))
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
        super(Identity, self).__init__(size_in, size_out)

    def _apply(self, x):
        return x

    def _apply_adjoint(self, x):
        return x


class Convolve1d(LinearMap):
    """
    1D cross-correlation linear map.
    The attributes follow the definition of torch.nn.Functional.conv1d
    """

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        # only weight and input size
        assert (
            len(list(size_in)) == 3
        ), "input must have the shape (minibatch, in_channels, iW)"
        assert (
            len(list(weight.shape)) == 3
        ), "weight must have the shape (out_channels, in_channels, kW)"
        minimatch, _, iW = size_in
        out_channel, _, kW = weight.shape
        assert iW >= kW, "Kernel size can't be greater than actual input size"
        Lout = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
        size_out = (minimatch, out_channel, Lout)
        super(Convolve1d, self).__init__(size_in, size_out)
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _apply(self, x):
        return F.conv1d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _apply_adjoint(self, x):
        return F.conv_transpose1d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class Convolve2d(LinearMap):
    """
    2D cross-correlation linear map.
    The attributes follow the definition of torch.nn.Functional.conv2d
    """

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        assert (
            len(list(size_in)) == 4
        ), "input must have the shape (minibatch, in_channels, iH, iW)"
        assert (
            len(list(weight.shape)) == 4
        ), "weight must have the shape (out_channels, in_channels, kH, kW)"
        minimatch, _, iH, iW = size_in
        out_channel, _, kH, kW = weight.shape
        assert (
            iH >= kH and iW >= kW
        ), "Kernel size can't be greater than actual input size"

        if isinstance(stride, int):
            stride = tuple([stride] * 2)
        if isinstance(padding, int):
            padding = tuple([padding] * 2)
        if isinstance(dilation, int):
            dilation = tuple([dilation] * 2)

        Hout = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        Wout = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
        size_out = (minimatch, out_channel, Hout, Wout)

        super(Convolve2d, self).__init__(size_in, size_out)
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _apply(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _apply_adjoint(self, x):
        return F.conv_transpose2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class Convolve3d(LinearMap):
    """
    3D cross-correlation linear map.
    The attributes follow the definition of torch.nn.Functional.conv3d
    """

    def __init__(
        self,
        size_in: Sequence[int],
        weight: Tensor,
        bias=None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        assert (
            len(list(size_in)) == 5
        ), "input must have the shape (minibatch, in_channels, iD, iH, iW)"
        assert (
            len(list(weight.shape)) == 5
        ), "weight must have the shape (out_channels, in_channels, kD, kH, kW)"
        minimatch, _, iD, iH, iW = size_in
        out_channel, _, kD, kH, kW = weight.shape
        assert (
            iD >= kD and iH >= kH and iW >= kW
        ), "Kernel size can't be greater than actual input size"

        if isinstance(stride, int):
            stride = tuple([stride] * 3)
        if isinstance(padding, int):
            padding = tuple([padding] * 3)
        if isinstance(dilation, int):
            dilation = tuple([dilation] * 3)

        Dout = (iD + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0] + 1
        Hout = (iH + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1] + 1
        Wout = (iW + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2] + 1
        size_out = (minimatch, out_channel, Dout, Hout, Wout)

        super(Convolve3d, self).__init__(size_in, size_out)
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _apply(self, x):
        return F.conv3d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _apply_adjoint(self, x):
        return F.conv_transpose3d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


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
        super(Patch2D, self).__init__(self.size_in, self.size_out)

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
        super(Patch3D, self).__init__(self.size_in, self.size_out)

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
