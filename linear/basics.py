"""
Basic linear operators, including diagonal matrix, convolution and first-order finite difference.
More on the way ...
2021-02. Guanhua Wang and Keyue Zhu, University of Michigan
"""

import torch
import torch.nn.functional as F
import copy
import numpy as np
from .linearmaps import LinearMap, check_device
from typing import Union, Sequence, TypeVar
from torch import Tensor




class Diff1d(LinearMap):
    '''
        1st order finite difference.
        Params:
            size_in: size of the input x
            dim: assign the dimension to apply operation
    '''

    def __init__(self,
                 size_in: Sequence[int],
                 dim: int):
        # TODO: determine size_out by size in
        size_out = copy.copy(size_in)
        size_out[dim] -= 1
        super(Diff1d, self).__init__(size_in, size_out)
        self.dim = dim
        assert np.isscalar(dim), "Please denote 1 dimension for a 1D finite difference operator"

    def _apply(self, x):
        return DiffFunc.apply(x, self.dim)

    def _apply_adjoint(self, y):
        return DiffFunc_adj.apply(y, self.dim)


class Diff2d(LinearMap):
    def __init__(self,
                 size_in: Sequence[int],
                 dim: Sequence[int]):
        size_out = copy.copy(size_in)
        size_out[dim[1]] -= 1
        size_out[dim[0]] -= 1
        super(Diff2d, self).__init__(size_in, size_out)
        # TODO: determine size_out by size in
        self.dim = dim
        assert len(self.dim) == 2, "Please denote two dimension for a 2D finite difference operator"

    def _apply(self, x):
        return DiffFunc.apply(DiffFunc.apply(x, self.dim[0]), self.dim[1])

    def _apply_adjoint(self, y):
        return DiffFunc_adj.apply(DiffFunc_adj.apply(y, self.dim[0]), self.dim[1])


class Diff2dframe(LinearMap):
    def __init__(self,
                 size_in: Sequence[int],
                 dim: Sequence[int]):
        super(Diff2dframe, self).__init__(size_in, size_in)

    def RtR(self, x):
        return torch.cat(((x[..., 0, :] - x[..., 1, :]).unsqueeze(-2),
                          (2 * x[..., 1:-1, :] - x[..., :-2, :] - x[..., 2:, :]),
                          (x[..., -1, :] - x[..., -2, :]).unsqueeze(-2)), dim=-2) + torch.cat(((x[..., 0] - x[
            ..., 1]).unsqueeze(-1), (2 * x[..., 1:-1] - x[..., :-2] - x[..., 2:]), (x[..., -1] - x[..., -2]).unsqueeze(
            -1)), dim=-1)

    def _apply(self, x):
        return self.RtR(x)

    def _apply_adjoint(self, x):
        return self.RtR(x)


class Diag(LinearMap):
    '''
        Expand an input vector into a diagonal matrix.
        For example, x is an 5*5 image.
        So P should be also a 5*5 weight vector.
        P*x (pytorch multiplication here) = Diag{vec(P)}*vec(x)
    '''

    def __init__(self,
                 P: Tensor):
        super(Diag, self).__init__(list(P.shape), list(P.shape))
        self.P = P

    def _apply(self, x):
        return self.P * x

    def _apply_adjoint(self, x):
        # conjugate here
        return torch.conj(self.P) * x


class Identity(LinearMap):
    def __init__(self, size_in, size_out):
        super(Identity, self).__init__(size_in, size_out)

    def _apply(self, x):
        return x

    def _apply_adjoint(self, x):
        return x


class Convolve1d(LinearMap):
    def __init__(self, size_in, weight, bias=None, stride=1, padding=0, dilation=1, device='cuda:0'):
        # only weight and input size
        assert len(list(size_in)) == 3, "input must have the shape (minibatch, in_channels, iW)"
        assert len(list(weight.shape)) == 3, "weight must have the shape (out_channels, in_channels, kW)"
        assert device == weight.device, "Tensors should be on the same device"
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
        self.device = device

    def _apply(self, x):
        check_device(self, x)
        return F.conv1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation)

    def _apply_adjoint(self, x):
        check_device(self, x)
        return F.conv_transpose1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation)


class Convolve2d(LinearMap):
    def __init__(self, size_in, weight, bias=None, stride=1, padding=0, dilation=1, device='cuda:0'):
        # TODO: unsqueeze check batch
        assert len(list(size_in)) == 4, "input must have the shape (minibatch, in_channels, iH, iW)"
        assert len(list(weight.shape)) == 4, "weight must have the shape (out_channels, in_channels, kH, kW)"
        assert device == weight.device, "Tensors should be on the same device"
        minimatch, _, iH, iW = size_in
        out_channel, _, kH, kW = weight.shape
        assert iH >= kH and iW >= kW, "Kernel size can't be greater than actual input size"

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
        self.device = device

    def _apply(self, x):
        check_device(self, x)
        return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation)

    def _apply_adjoint(self, x):
        check_device(self, x)
        return F.conv_transpose2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation)


class Convolve3d(LinearMap):
    def __init__(self, size_in, weight, bias=None, stride=1, padding=0, dilation=1, device='cuda:0'):
        # TODO: unsqueeze check batch
        assert len(list(size_in)) == 5, "input must have the shape (minibatch, in_channels, iD, iH, iW)"
        assert len(list(weight.shape)) == 5, "weight must have the shape (out_channels, in_channels, kD, kH, kW)"
        assert device == weight.device, "Tensors should be on the same device"
        minimatch, _, iD, iH, iW = size_in
        out_channel, _, kD, kH, kW = weight.shape
        assert iD >= kD and iH >= kH and iW >= kW, "Kernel size can't be greater than actual input size"

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
        self.device = device

    def _apply(self, x):
        check_device(self, x)
        return F.conv3d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation)

    def _apply_adjoint(self, x):
        check_device(self, x)
        return F.conv_transpose3d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation)


class DiffFunc(torch.autograd.Function):
    '''
        autograd.Function for the 1st-order finite difference operators
    '''

    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        return finitediff(x, dim)

    @staticmethod
    def backward(ctx, dx):
        return finitediff_adj(dx, ctx.dim), None


class DiffFunc_adj(torch.autograd.Function):
    '''
        autograd.Function for the 1st-order finite difference operators
    '''

    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        return finitediff_adj(x, dim)

    @staticmethod
    def backward(ctx, dx):
        return finitediff(dx, ctx.dim), None


def finitediff(x, dim=-1):
    """
    Apply finite difference operator on a certain dim
    Args:
        x: input data
        dim: dimension to apply the operator
    Returns:
        Diff(x)
    """
    len_dim = x.shape[dim]
    return torch.narrow(x, dim, 1, len_dim - 1) - torch.narrow(x, dim, 0, len_dim - 1)


def finitediff_adj(y, dim=-1):
    """
    Apply finite difference operator on a certain dim
    Args:
        y: input data
        dim: dimension to apply the operator
    Returns:
        Diff'(x)
    """
    len_dim = y.shape[dim]
    return torch.cat(
        (-torch.narrow(y, dim, 0, 1), (torch.narrow(y, dim, 0, len_dim - 1) - torch.narrow(y, dim, 1, len_dim - 1)),
         torch.narrow(y, dim, len_dim - 1, 1)),
        dim=dim)

# class Smoothing1d(LinearMap):
#     def __init__(self, size_in, size_out, device):
#         super(Smoothing1d, self).__init__(size_in, size_out)

#     def _apply(self, x):
#         pass

#     def _apply_adjoint(self, x):
#         pass

# class Smoothing2d(LinearMap):
#     def __init__(self, size_in, size_out, device):
#         super(Smoothing2d, self).__init__(size_in, size_out)

#     def _apply(self, x):
#         pass

#     def _apply_adjoint(self, x):
#         pass
