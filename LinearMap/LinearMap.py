"""
Linear Operator implementations.
"""
import torch

class Add(torch.autograd.Function):
    """
    Test class for addition.
    """
    @staticmethod
    def forward(ctx, a, b):
        result = torch.add(a, b)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result



class Linearmap():
    '''
        Our major difference with Sigpy is that:
         In sigpy, for each linear op, they IMPLEMENTED it. So each instance inherit Linop()
         In out, we directly call the forward and backward ops. Which is provided by 3rd package.

         Alternative: you can try using the nn.module as the base class. It also support manual forward() and backward()
    '''
    def __init__(self, size_in, size_out, forward_op, adjoint_op):
        '''
        For initilization, this can be provided:
        size_in: the size of the input of the linear map
        size_out:
        '''
        self. size_in = size_in
        self.size_out = size_out
        self. .....

        class forward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return self.forward_op(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return self.adjoint_op(grad_data_in)
        self._apply = forward.apply # THis may look wired to you. But the torch.autograd.Function requires .apply
        class adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return self.adjoint_op(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return self.forward_op(grad_data_in)
        self._adjoint_linop = adjoint.apply
    def check_device(self): # Make sure that the input and output and forwardop, adjop are on the same divice (CPU/CPU)
    def __call__(self):
    def __add__(self, other): # you can try the sigpy approach or your own one
    def __matmul__(self, other):
    def __sub__(self, other):
    def __neg__(self):
    @property
    def H(self):
    def _combine_compose_linops(linops):


# Below for Reference
def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def FiniteDiff(x, dim = -1):
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