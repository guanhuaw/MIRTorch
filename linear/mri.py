from .linearmaps import LinearMap
import torch
from torch.fft import fft, ifft
from .linearmaps import LinearMap, check_device


def fftshift(x, dims=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dims is None:
        dims = tuple(range(x.dim()))
        shifts = [dims // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = x.shape[dims] // 2
    else:
        shifts = [x.shape[i] // 2 for i in dims]
    return torch.roll(x, shifts, dims)


def ifftshift(x, dims=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dims is None:
        dims = tuple(range(x.dims()))
        shifts = [(dims + 1) // 2 for dims in x.shape]
    elif isinstance(dims, int):
        shifts = (x.shape[dims] + 1) // 2
    else:
        shifts = [(x.shape[i] + 1) // 2 for i in dims]
    return torch.roll(x, shifts, dims)

def fft2(data, dim, norm = 'ortho'):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, dim = dim, norm='norm')
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data, norm = 'ortho'):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

class FFTn(LinearMap):
    def __init__(self, size_in, size_out, dims, norm = 'ortho'):
        super(FFTn, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
    def _apply(self, x):
        x = ifftshift(x, self.dims)
        x = fft(x, dim=self.dims, norm=self.norm)
        x = fftshift(x, self.dims)
        return x
    def _apply_adjoint(self, x):
        x = ifftshift(x, self.dims)
        if self.norm == 'ortho':
            x = ifft(x, dim=self.dims, norm='ortho')
        elif self.norm == 'forward':
            x = ifft(x, dim=self.dims, norm='backward')
        else:
            x = ifft(x, dim=self.dims, norm='forward')
        x = fftshift(x, self.dims)
        return x

class Sense(LinearMap):
    def __init__(self, size_in, size_out, dims, smaps, masks, norm = 'ortho'):
        super(Sense, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        self.smaps = smaps
        self.masks = masks
    def _apply(self, x):
        x = ifftshift(x, self.dims)
        x = fft(x, dim=self.dims, norm=self.norm)
        x = fftshift(x, self.dims)
        return x
    def _apply_adjoint(self, x):
        x = ifftshift(x, self.dims)
        if self.norm == 'ortho':
            x = ifft(x, dim=self.dims, norm='ortho')
        elif self.norm == 'forward':
            x = ifft(x, dim=self.dims, norm='backward')
        else:
            x = ifft(x, dim=self.dims, norm='forward')
        x = fftshift(x, self.dims)
        return x
class NuFFT

class NuSense

