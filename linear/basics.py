import torch
from .linearmaps import LinearMap, check_device


class Diff1d(LinearMap):
    def __init__(self, size_in, size_out, dim, device='cuda:0'):
        super(Diff1d, self).__init__(size_in, size_out)
        self.dim = dim

    def _apply(self, x):
        return finitediff(x, self.dim)

    def _apply_adjoint(self, y):
        return finitediff_adj(y, self.dim)

class Diff2d(LinearMap):
    def __init__(self, size_in, size_out, dim, device='cuda:0'):
        super(Diff2d, self).__init__(size_in, size_out)
        self.dim = dim
        assert len(self.dim) == 2, "Please denote two dimension for a 2d finite difference operator"

    def _apply(self, x):
        return finitediff(finitediff(x, self.dim[1]), self.dim[2])

    def _apply_adjoint(self, y):
        return finitediff_adj(finitediff_adj(y, self.dim[1]), self.dim[2])

class Diag(LinearMap):
    def __init__(self, size_in, size_out, P):
        assert size_in == size_out
        assert P.shape == size_in
        super(Diag, self).__init__(size_in, size_out)
        self.P = P

    def _apply(self, x):
        return self.P*x

    def _apply_adjoint(self, x):
        # conjugate here
        return torch.conj(self.P)*x
class Identity(LinearMap):
    pass

class Smoothing1d(LinearMap):
    pass

class Smoothing2d(LinearMap):
    pass

class convolve(LinearMap):
    pass

class interp_li_1d(LinearMap):
    pass


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
