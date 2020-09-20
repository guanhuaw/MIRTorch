"""
Linear Operator implementations.
"""
import torch
from .LinearMap import Linearmap
class FiniteDiff(Linearmap):


def Finitediff(x, dim = -1):
    """
    Apply finite difference operator on a certain dim
    Args:
        x: input data
        dim: dimension to apply the operator

    Returns:
        Diff(x)
    """
    len_dim = x.shape[dim]
    return torch.narrow(x, dim, 1, len_dim-1) - torch.narrow(x, dim, 0, len_dim-1)