from .linearmaps import LinearMap
import torch
from torch.fft import fft, ifft, fftn, ifftn
from .linearmaps import LinearMap, check_device
from torchkbnufft import AdjMriSenseNufft, MriSenseNufft, KbNufft, AdjKbNufft, ToepSenseNufft

# To Do: toeplitz embedding, field inhomogeneity
def fftshift(x, dims=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch tensors
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
    Similar to np.fft.ifftshift but applies to PyTorch tensors
    """
    if dims is None:
        dims = tuple(range(x.dims()))
        shifts = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = (x.shape[dims] + 1) // 2
    else:
        shifts = [(x.shape[i] + 1) // 2 for i in dims]
    return torch.roll(x, shifts, dims)


class FFTCn(LinearMap):
    '''
    FFT operators with FFTshift and iFFTshift for multidimensional data.
    Pytorch provides three modes in FFT: 'ortho', 'forward', 'backward'.
    Each pair of FFT and iFFT with same mode is inverse, but not necessarily adjoint to each other.
    '''

    def __init__(self, size_in, size_out, dims, norm='ortho'):
        super(FFTCn, self).__init__(size_in, size_out)
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
    Parameters:
        mask: [batch, nx, ny, (nz)]
        sensitivity maps: [batch, ncoil, nx, ny, (nz)]
    '''

    def __init__(self, size_in, size_out, dims, smaps, masks, norm='ortho', batchmode=True):
        super(Sense, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        # TODO: check the size match between smaps and masks. (try-catch)
        self.smaps = smaps
        self.masks = masks
        self.batchmode = batchmode

    def _apply(self, x):
        '''
            x in size [batch, nx, ny, (nz)]
        '''
        assert x.shape == self.masks.shape, "mask and image's shape mismatch"
        if self.batchmode:
            x = x.unsqueeze(1)
            masks = self.masks.unsqueeze(1)
        else:
            x = x.unsqueeze(0)
            masks = self.masks.unsqueeze(0)
        dims = tuple([x + 1 for x in self.dims])
        x = x * self.smaps
        x = ifftshift(x, dims)
        k = fftn(x, dim=dims, norm=self.norm)
        k = fftshift(k, dims) * masks
        return k

    def _apply_adjoint(self, k):
        assert k.shape == self.smaps.shape, "sensitivity maps and signal's shape mismatch"
        if self.batchmode:
            masks = self.masks.unsqueeze(1)
        else:
            masks = self.masks.unsqueeze(0)
        dims = tuple([x + 1 for x in self.dims])
        k = k * masks
        k = ifftshift(k, dims)
        if self.norm == 'ortho':
            x = ifftn(k, dim=dims, norm='ortho')
        elif self.norm == 'forward':
            x = ifftn(k, dim=dims, norm='backward')
        else:
            x = ifftn(k, dim=dims, norm='forward')
        x = fftshift(x, dims)
        if self.batchmode:
            x = (x * torch.conj(self.smaps)).sum(1)
        else:
            x = (x * torch.conj(self.smaps)).sum(0)
        return x


class NuSense:
    '''
    Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    Parameters:
        traj: [ndim, nshot*npoints]
        sensitivity maps: [batch, ncoil, nx, ny, (nz)]
    Input/Output:
        x(complex-valued images): [batch, nx, ny, (nz)]
        k(k-space data): [batch, ncoil, nshot*npoints]
    '''
    def __init__(self, smaps, traj, norm='ortho', batchmode=True):
        if batchmode:
            self.A = MriSenseNufft(smap=smaps, im_size=tuple(smaps.shape[2:]), norm=norm).to(device=smaps.device,
                                                                                           dtype=smaps.dtype)
            self.AT = AdjMriSenseNufft(smap=smaps, im_size=tuple(smaps.shape[2:]), norm=norm).to(device=smaps.device,
                                                                                               dtype=smaps.dtype)
            super(NuSense, self).__init__([smaps.shape[0]]+list(smaps.shape[2:]), list(smaps.shape[0:2])+[traj.shape[1]], device = smaps.device)
        else:
            self.A = MriSenseNufft(smap=smaps, im_size=tuple(smaps.shape[1:]), norm=norm).to(device=smaps.device,
                                                                                       dtype=smaps.dtype)
            self.AT = AdjMriSenseNufft(smap=smaps, im_size=tuple(smaps.shape[1:]), norm=norm).to(device=smaps.device,
                                                                                           dtype=smaps.dtype)
            super(NuSense, self).__init__(list(smaps.shape[0:2])+[traj.shape[1]], [smaps.shape[0]]+list(smaps.shape[2:]), device=smaps.device)
        self.traj = traj

    def _apply(self,x):
        if self.batchmode:
            # export complex x to 2-channel
            torch.view_as_real(x)
            self.A(x, self.traj)

        else:
            # export x to complex
            # unsqueeze x
            self.A(x, self.traj)

    def _apply_adjoint(self,y):
        if self.batchmode:
            # export commplex y to 2-channel

            self.A(y, self.traj)

        else:
            # export y to complex
            # unsqueeze y
            self.A(y, self.traj)
class MRI:
    def __init__(self, size_in, size_out, dims, smaps, masks, zmap, norm='ortho'):
        pass
