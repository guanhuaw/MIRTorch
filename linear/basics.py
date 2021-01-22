import torch
import torch.nn.functional as F
from .linearmaps import LinearMap, check_device


class Diff1d(LinearMap):
    def __init__(self, size_in, size_out, dim, device='cuda:0'):
        # TODO: determine size_out by size in
        super(Diff1d, self).__init__(size_in, size_out)
        self.dim = dim
        assert len(self.dim) == 1, "Please denote two dimension for a 1D finite difference operator"

    def _apply(self, x):
        return finitediff(x, self.dim)

    def _apply_adjoint(self, y):
        return finitediff_adj(y, self.dim)

class Diff2d(LinearMap):
    def __init__(self, size_in, size_out, dim, device='cuda:0'):
        super(Diff2d, self).__init__(size_in, size_out)
        # TODO: determine size_out by size in
        self.dim = dim
        assert len(self.dim) == 2, "Please denote two dimension for a 2D finite difference operator"

    def _apply(self, x):
        return finitediff(finitediff(x, self.dim[1]), self.dim[2])

    def _apply_adjoint(self, y):
        return finitediff_adj(finitediff_adj(y, self.dim[1]), self.dim[2])

class Diag(LinearMap):
    '''
        Expand an input vetor into a diagonal matrix
    '''
    def __init__(self, P):
        super(Diag, self).__init__(list(P.shape), list(P.shape))
        self.P = P

    def _apply(self, x):
        return self.P*x

    def _apply_adjoint(self, x):
        # conjugate here
        return torch.conj(self.P)*x

class Identity(LinearMap):
    def __init__(self, size_in, size_out):
        super(Identity, self).__init__(size_in, size_out)

    def _apply(self, x):
        pass

    def _apply_adjoint(self, x):
        pass

class Smoothing1d(LinearMap):
    def __init__(self, size_in, size_out, device):
        super(Smoothing1d, self).__init__(size_in, size_out)

    def _apply(self, x):
        pass

    def _apply_adjoint(self, x):
        pass

class Smoothing2d(LinearMap):
    def __init__(self, size_in, size_out, device):
        super(Smoothing2d, self).__init__(size_in, size_out)

    def _apply(self, x):
        pass

    def _apply_adjoint(self, x):
        pass

class convolve(LinearMap):
    def __init__(self, size_in, size_out, device, h):
        super(convolve, self).__init__(size_in, size_out, device=device)
        self.h = h
        
    def _apply(self, x):
        return F.conv2d(self.x, self.h)

    def _apply_adjoint(self, x):
        # ?
        tconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
        tconv.weight.data = self.h
        return tconv(x)


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
