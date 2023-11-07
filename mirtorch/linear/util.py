import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Sequence
import torchvision
import numpy as np


def finitediff(x: Tensor, dim: int = -1, mode='reflexive'):
    """
    Apply finite difference operator on a certain dim
    Args:
        x: tensor, input data
        dim: int, dimension to apply the operator
        mode: 'reflexive' or 'periodic'
    Returns:
        y: the first-order finite difference of x
    """
    if mode == 'reflexive':
        len_dim = x.shape[dim]
        return torch.narrow(x, dim, 1, len_dim - 1) - torch.narrow(x, dim, 0, len_dim - 1)
    elif mode == 'periodic':
        return torch.roll(x, 1, dims=dim) - x
    else:
        raise ValueError("mode should be either 'reflexive' or 'periodic'")


def finitediff_adj(y: Tensor, dim: int = -1, mode='reflexive'):
    """
    Apply finite difference operator on a certain dim
    Args:
        y: tensor, input data
        dim: int, dimension to apply the operator
        mode: 'reflexive' or 'periodic'

    Returns:
        y: the first-order finite difference of x
    """
    if mode == 'reflexibe':
        len_dim = y.shape[dim]
        return torch.cat(
            (-torch.narrow(y, dim, 0, 1), (torch.narrow(y, dim, 0, len_dim - 1) - torch.narrow(y, dim, 1, len_dim - 1)),
             torch.narrow(y, dim, len_dim - 1, 1)),
            dim=dim)
    elif mode == 'periodic':
        return torch.roll(y, -1, dims=dim) - y
    else:
        raise ValueError("mode should be either 'reflexive' or 'periodic'")


class DiffFunc(torch.autograd.Function):
    """
    autograd.Function for the 1st-order finite difference operators
    """

    @staticmethod
    def forward(ctx, x, dim, mode):
        ctx.dim = dim
        ctx.mode = mode
        return finitediff(x, dim, mode)

    @staticmethod
    def backward(ctx, dx):
        return finitediff_adj(dx, ctx.dim, ctx.mode), None, None


class DiffFunc_adj(torch.autograd.Function):
    """
    autograd.Function for the 1st-order finite difference operators
    """

    @staticmethod
    def forward(ctx, x, dim, mode):
        ctx.dim = dim
        ctx.mode = mode
        return finitediff_adj(x, dim, mode)

    @staticmethod
    def backward(ctx, dx):
        return finitediff(dx, ctx.dim, ctx.mode), None, None


def fftshift(x: Tensor, dims: Union[int, Sequence[int]] = None):
    """
    Similar to np.fft.fftshift but applies to PyTorch tensors. From fastMRI code.
    """
    if dims is None:
        dims = tuple(range(x.dim()))
        shifts = [dim // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = x.shape[dims] // 2
    else:
        shifts = [x.shape[i] // 2 for i in dims]
    return torch.roll(x, shifts, dims)


def ifftshift(x: Tensor, dims: Union[int, Sequence[int]] = None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch tensors. From fastMRI code.
    """
    if dims is None:
        dims = tuple(range(x.dims()))
        shifts = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = (x.shape[dims] + 1) // 2
    else:
        shifts = [(x.shape[i] + 1) // 2 for i in dims]
    return torch.roll(x, shifts, dims)
