import numpy as np
import torch
from torch import Tensor
from torch.fft import fftn, ifftn
from linear import LinearMap

def fftshift(x: Tensor, dims=None):
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


def ifftshift(x: Tensor, dims=None):
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
