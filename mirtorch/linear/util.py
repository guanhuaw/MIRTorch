import torch
from torch import Tensor
from typing import Union, Sequence

def finitediff(x: Tensor, dim: int = -1):
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


def finitediff_adj(y: Tensor, dim: int = -1):
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

def dim_conv(dim_in, dim_kernel_size, dim_stride = 1, dim_padding = 0, dim_dilation = 1):
    dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
    return dim_out

