from .linearmaps import LinearMap
import torch
from torch.fft import fft, ifft, fftn, ifftn
from .linearmaps import LinearMap, check_device


# To Do: toeplitz embedding, field inhomogeneity

def fftshift(x, dims=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dims is None:
        dims = tuple(range(x.dim()))
        shifts = [dim // 2 for dim in x.shape]
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
        shifts = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = (x.shape[dims] + 1) // 2
    else:
        shifts = [(x.shape[i] + 1) // 2 for i in dims]
    return torch.roll(x, shifts, dims)


class FFTn(LinearMap):
    '''
    FFT operators with FFTshift and iFFTshift
    Pytorch provides three modes in FFT: 'ortho', 'forward', 'backward'.
    Each pair of FFT and iFFT with same mode is inverse, but not necessarily adjoint to each other.
    '''

    def __init__(self, size_in, size_out, dims, norm='ortho'):
        super(FFTn, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims

    def _apply(self, x):
        x = ifftshift(x, self.dims)
        x = fftn(x, dim=self.dims, norm=self.norm)
        x = fftshift(x, self.dims)
        return x

    def _apply_adjoint(self, x):
        x = ifftshift(x, self.dims)
        if self.norm == 'ortho':
            x = ifftn(x, dim=self.dims, norm='ortho')
        elif self.norm == 'forward':
            x = ifftn(x, dim=self.dims, norm='backward')
        else:
            x = ifftn(x, dim=self.dims, norm='forward')
        x = fftshift(x, self.dims)
        return x


class Sense(LinearMap):
    '''
    Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    Input:
        mask: [batch, nx, ny, (nz)]
        sensitivity maps: [batch, ncoil, nx, ny, (nz)]
    '''

    def __init__(self, size_in, size_out, dims, smaps, masks, norm='ortho'):
        super(Sense, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        assert smaps.shape[0] + smaps.shape[2:] == masks.shape, "mask and sensitivity map mismatch"
        self.smaps = smaps
        self.masks = masks

    def _apply(self, x):
        assert x.shape == self.masks.shape, "mask and image's shape mismatch"
        x = x.unsqueeze(1)
        masks = self.masks.unsqueeze(1)
        x = x * self.smaps
        x = ifftshift(x, self.dims)
        k = fftn(x, dim=self.dims, norm=self.norm)
        k = fftshift(k, self.dims) * masks
        return k

    def _apply_adjoint(self, k):
        assert k.shape == self.smaps.shape, "sensitivity maps and signal's shape mismatch"
        masks = self.masks.unsqueeze(1)
        k = k * masks
        k = ifftshift(k, self.dims)
        if self.norm == 'ortho':
            x = ifftn(k, dim=self.dims, norm='ortho')
        elif self.norm == 'forward':
            x = ifftn(k, dim=self.dims, norm='backward')
        else:
            x = ifftn(k, dim=self.dims, norm='forward')
        x = fftshift(x, self.dims)
        x = (x * torch.conj(self.smaps)).sum(1)
        return x

#
# class NuFFT:
#     def __init__(self, size_in, size_out, dims, smaps, traj, norm='ortho'):
#
#
# class NuSense:
#     pass
#
# class MRI:
#     def __init__(self, size_in, size_out, dims, smaps, masks, zmap, norm='ortho'):
#         pass