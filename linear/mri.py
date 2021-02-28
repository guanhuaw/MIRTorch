"""
Discrete-to-discreate system matrices for MRI.
2021-02. Guanhua Wang, University of Michigan
To Do: toeplitz embedding, field inhomogeneity, frame operator
"""

import numpy as np
from .linearmaps import LinearMap
import torch
from torch import Tensor
from torch.fft import fft, ifft, fftn, ifftn
from .linearmaps import LinearMap, check_device
import torchkbnufft as tkbn
from typing import Union, Sequence, TypeVar


class FFTCn(LinearMap):
    '''
    FFT operators with FFTshift and iFFTshift for multidimensional data.
    Pytorch provides three modes in FFT: 'ortho', 'forward', 'backward'.
    Each pair of FFT and iFFT with same mode is the inverse, but not necessarily the adjoint to each other.
    '''

    def __init__(self,
                 size_in: Sequence[int],
                 size_out: Sequence[int],
                 dims: Union[int, Sequence[int]],
                 norm='ortho'):
        super(FFTCn, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims

    def _apply(self, x: Tensor) -> Tensor:
        x = ifftshift(x, self.dims)
        x = fftn(x, dim=self.dims, norm=self.norm)
        x = fftshift(x, self.dims)
        return x

    def _apply_adjoint(self, x: Tensor) -> Tensor:
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

    def __init__(self,
                 size_in: Sequence[int],
                 size_out: Sequence[int],
                 dims: Union[int, Sequence[int]],
                 smaps: Tensor,
                 masks: Tensor,
                 norm='ortho',
                 batchmode=True):
        super(Sense, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        # TODO: check the size match between smaps and masks. (try-catch)
        self.smaps = smaps
        self.masks = masks
        self.batchmode = batchmode

    def _apply(self, x: Tensor) -> Tensor:
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

    def _apply_adjoint(self, k: Tensor) -> Tensor:
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


class NuSense(LinearMap):
    '''
    Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    Parameters:
        traj: [ndim, nshot*npoints]
        sensitivity maps: [batch, ncoil, nx, ny, (nz)]
    Input/Output:
        x(complex-valued images): [batch, nx, ny, (nz)]
        k(k-space data): [batch, ncoil, nshot*npoints]

    The device follow smaps. So make sure that smaps and image stay on the same device.
    '''

    def __init__(self,
                 smaps: Tensor,
                 traj: Tensor,
                 norm='ortho',
                 batchmode=True,
                 numpoints: Union[int, Sequence[int]] = 6):
        self.smaps = smaps
        self.norm = norm
        self.traj = traj
        self.batchmode = batchmode
        if batchmode:
            self.A = tkbn.KbNufft(im_size=tuple(smaps.shape[2:]), grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                  numpoints=numpoints).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[2:]), grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                          numpoints=numpoints).to(smaps)
            size_in = [smaps.shape[0]] + list(smaps.shape[2:])
            size_out = list(smaps.shape[0:2]) + [traj.shape[1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out), device=smaps.device)
        else:
            self.A = tkbn.KbNufft(im_size=tuple(smaps.shape[2:]), grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                  numpoints=numpoints).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[2:]), grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                          numpoints=numpoints).to(smaps)
            size_in = smaps.shape[2:]
            size_out = [smaps.shape[0]] + [traj.shape[1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out), device=smaps.device)

    def _apply(self, x: Tensor) -> Tensor:
        if self.batchmode:
            return self.A(x, self.traj, smaps=self.smaps, norm = self.norm)
        else:
            return self.A(x.unsqueeze(0), self.traj, smaps=self.smaps, norm = self.norm).squeeze(0)

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        if self.batchmode:
            return self.AT(y, self.traj, smaps=self.smaps, norm = self.norm).squeeze(1)
        else:
            return self.AT(y.unsqueeze(0), self.traj, smaps=self.smaps, norm = self.norm).squeeze(0).squeeze(0)


class MRI:
    def __init__(self, size_in, size_out, dims, norm='ortho', smaps=None, masks=None, zmap=None):
        pass



def fftshift(x: Tensor, dims=None):
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


def ifftshift(x: Tensor, dims=None):
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