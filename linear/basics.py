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
        Expand an input vector into a diagonal matrix.
        For example, x is an 5*5 image.
        So P should be also a 5*5 weight vector.
        P*x (pytorch multiplication here) = Diag{vec(P)}*vec(x)
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

class Convolve1d(LinearMap):
    def __init__(self, size_in, weight, device):
        # only weight and input size
        assert len(list(size_in)) == 3, "input must have the shape (minibatch, in_channels, iW)"
        assert len(list(weight.shape)) == 3, "weight must have the shape (out_channels, in_channels, kW)"
        minimatch, _, iW = size_in
        out_channel, _, kW = weight.shape
        assert iW >= kW, "Kernel size can't be greater than actual input size"
        size_out = (minimatch, out_channel, iW - kW + 1)
        # TODO: bias, padding, stride ....
        super(Convolve1d, self).__init__(size_in, size_out)
        self.weight = weight
        
    def _apply(self, x):
        return F.conv1d(x, self.weight)

    def _apply_adjoint(self, x):
        return F.conv_transpose1d(x, self.weight)

class Convolve2d(LinearMap):
    def __init__(self, size_in, weight, device):
        assert len(list(size_in)) == 4, "input must have the shape (minibatch, in_channels, iH, iW)"
        assert len(list(weight.shape)) == 4, "weight must have the shape (out_channels, in_channels, kH, kW)"
        minimatch, _, iH, iW = size_in
        out_channel, _, kH, kW = weight.shape
        assert iH >= kH and iW >= kW, "Kernel size can't be greater than actual input size"
        size_out = (minimatch, out_channel, iH - kH + 1, iW - kW + 1)
        # TODO: bias, padding, stride ....
        super(Convolve2d, self).__init__(size_in, size_out)
        self.weight = weight
        
    def _apply(self, x):
        return F.conv2d(x, self.weight)

    def _apply_adjoint(self, x):
        return F.conv_transpose2d(x, self.weight)

class Convolve3d(LinearMap):
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
