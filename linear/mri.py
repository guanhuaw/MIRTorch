"""
Discrete-to-discreate system matrices for MRI.
2021-02. Guanhua Wang, University of Michigan
To Do: toeplitz embedding, field inhomogeneity, frame operator
"""

import numpy as np
import torch
from torch import Tensor
from torch.fft import fftn, ifftn
from .linearmaps import LinearMap, check_device
import torchkbnufft as tkbn
from typing import Union, Sequence, TypeVar


# To Do: frame operator

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
        mask: [(batch), nx, ny, (nz)]
        sensitivity maps: [(batch), ncoil, nx, ny, (nz)]
    '''

    def __init__(self,
                 smaps: Tensor,
                 masks: Tensor,
                 norm: str = 'ortho',
                 batchmode: bool = True):
        if batchmode:
            size_in = [smaps.shape[0]] + list(smaps.shape[2:])
            size_out = list(smaps.shape)
            dims = tuple(np.arange(2, len(smaps.shape)))
        else:
            size_in = list(smaps.shape[2:])
            size_out = list(smaps.shape[1:])
            dims = tuple(np.arange(1, len(smaps.shape)))
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
    The implementation calls Matthew Muckley's Torchkbnufft toolbox:
    https://github.com/mmuckley/torchkbnufft

    Parameters:
        traj: [(batch), ndim, nshot*npoints]
        # Attention: Traj can have no batch dimension even x have. ref: https://github.com/mmuckley/torchkbnufft/pull/24
        sensitivity maps: [(batch), ncoil, nx, ny, (nz)]
    Input/Output:
        x(complex-valued images): [(batch), nx, ny, (nz)]
        k(k-space data): [(batch), ncoil, nshot*npoints]

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
            self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[2:]),
                                          grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                          numpoints=numpoints).to(smaps)
            size_in = [smaps.shape[0]] + list(smaps.shape[2:])
            size_out = list(smaps.shape[0:2]) + [traj.shape[-1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out), device=smaps.device)
        else:
            self.A = tkbn.KbNufft(im_size=tuple(smaps.shape[1:]), grid_size=tuple(np.array(smaps.shape[1:]) * 2),
                                  numpoints=numpoints).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[1:]),
                                          grid_size=tuple(np.array(smaps.shape[1:]) * 2),
                                          numpoints=numpoints).to(smaps)
            size_in = smaps.shape[1:]
            size_out = [smaps.shape[0]] + [traj.shape[-1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out), device=smaps.device)

    def _apply(self, x: Tensor) -> Tensor:
        if self.batchmode:
            return self.A(x, self.traj, smaps=self.smaps, norm=self.norm)
        else:
            return self.A(x.unsqueeze(0), self.traj, smaps=self.smaps, norm=self.norm).squeeze(0)

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        if self.batchmode:
            return self.AT(y, self.traj, smaps=self.smaps, norm=self.norm).squeeze(1)
        else:
            return self.AT(y.unsqueeze(0), self.traj, smaps=self.smaps, norm=self.norm).squeeze(0).squeeze(0)


class Gmri(LinearMap):
    '''
    B0-informed mri reconstruction
    Parameters:
        Norm: like the Sense cases
        smaps: sensitivity maps: [batch, nx, ny, (nz)] (must have a batch dimension)
        zmap: relaxation and off-resonance effects in Hz: [batch, nx, ny, (nz)]
                ref: DOI: 10.1109/TSP.2005.853152
        traj: [(batch), ndim, nshot, npoints]
        or mask: [(batch), nx, ny, (nz)] (name sure that ny is the phase-encoding direction)
        L: number of segmentation
        ti: time points, in ms
        numpoints: length of the Nufft interpolation kernels
    Input/Output:
        x(complex-valued images): [(batch), nx, ny, (nz)]
        k(k-space data): [(batch), ncoil, nshot*npoints] or [(batch), ncoil, nx, ny, (nz)]
    '''

    def __init__(self,
                 norm='ortho',
                 smaps: Tensor = None,
                 zmap: Tensor = None,
                 traj: Tensor = None,
                 mask: Tensor = None,
                 L: int = 6,
                 nbins: int = 20,
                 dt: int = 4e-3,
                 numpoints: Union[int, Sequence[int]] = 6
                 ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        assert (mask is None) != (
                    traj is None), "Please provide either mask (for cartesian sampling) or Non-cartesian trajectory"
        if mask is not None:
            self.mask = mask
            if batchmode:
                size_in = [smaps.shape[0]] + list(smaps.shape[2:])
                size_out = list(smaps.shape)
                dims = tuple(np.arange(2, len(smaps.shape)))
            else:
                size_in = list(smaps.shape[2:])
                size_out = list(smaps.shape[1:])
                dims = tuple(np.arange(1, len(smaps.shape)))
            self.A = FFTCn(size_in, size_out, dims=dims)
            self.AT = self.A.H
        elif traj is not None:
            self.traj = traj
            if batchmode:
                self.batch, self.ndim, self.nshot, self.npoints = self.zmap.shape
                self.A = tkbn.KbNufft(im_size=tuple(smaps.shape[2:]), grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                      numpoints=numpoints).to(smaps)
                self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[2:]),
                                              grid_size=tuple(np.array(smaps.shape[2:]) * 2),
                                              numpoints=numpoints).to(smaps)
                size_in = [smaps.shape[0]] + list(smaps.shape[2:])
                size_out = list(smaps.shape[0:2]) + [traj.shape[-1]]
            else:
                self.A = tkbn.KbNufft(im_size=tuple(smaps.shape[1:]), grid_size=tuple(np.array(smaps.shape[1:]) * 2),
                                      numpoints=numpoints).to(smaps)
                self.AT = tkbn.KbNufftAdjoint(im_size=tuple(smaps.shape[1:]),
                                              grid_size=tuple(np.array(smaps.shape[1:]) * 2),
                                              numpoints=numpoints).to(smaps)
                size_in = smaps.shape[1:]
                size_out = (smaps.shape[0], traj.shape[-1])

        super(Gmri, self).__init__(tuple(size_in), tuple(size_out), device=smaps.device)
        if batchmode:
            B ,C = mri_exp_approx(zmap, nbins, L, dt, dt*self.npoints)
            self.B = torch.tensor(B).to(smaps).reshape()
            self.C = torch.tensor(C).to(smaps)
        else:
            B ,C = mri_exp_approx(zmap, nbins, L, dt, dt*self.npoints)
            self.B = torch.tensor(B).to(smaps)
            self.C = torch.tensor(C).to(smaps)
    def _apply(self, x: Tensor) -> Tensor:
        if self.batchmode:
            y = torch.zeros(self.size_out)
            for il in range(self.L):
                y = y + self.B*self.A*(x*self.C)

        else:
            return self.A(x.unsqueeze(0), self.traj, smaps=self.smaps, norm=self.norm).squeeze(0)

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        if self.batchmode:
            return self.AT(y, self.traj, smaps=self.smaps, norm=self.norm).squeeze(1)
        else:
            return self.AT(y.unsqueeze(0), self.traj, smaps=self.smaps, norm=self.norm).squeeze(0).squeeze(0)


def mri_exp_approx(b0, bins, lseg, dt, T):
    """
    From Sigpy: https://github.com/mikgroup/sigpy
    and MIRT: https://web.eecs.umich.edu/~fessler/code/
    Creates B [M*L] and Ct [L*N] matrices to approximate exp(-2*pi*zmap) [M*N]
    Params:
        b0 (array): inhomogeneity matrix. in Hz.
        bins (int): number of histogram bins to use.
        lseg (int): number of time segments.
        dt (float): hardware dwell time (ms).
        T (float): length of pulse (ms).
    Input/output:
        2-element tuple containing
        - **B** (*array*): temporal interpolator.
        - **Ct** (*array*): off-resonance phase at each time segment center.
    TODO: The SVD approach.
    """

    # create time vector
    t = np.linspace(0, T, np.int(T / dt))
    hist_wt, bin_edges = np.histogram(np.imag(2j * np.pi * np.concatenate(b0)),
                                      bins)

    # Build B and Ct
    bin_centers = bin_edges[1:] - bin_edges[1] / 2
    zk = 0 + 1j * bin_centers
    tl = np.linspace(0, lseg, lseg) / lseg * T / 1000  # time seg centers
    # calculate off-resonance phase @ each time seg, for hist bins
    ch = np.exp(-np.expand_dims(tl, axis=1) @ np.expand_dims(zk, axis=0))
    w = np.diag(np.sqrt(hist_wt))
    p = np.linalg.pinv(w @ np.transpose(ch)) @ w
    b = p @ np.exp(-np.expand_dims(zk, axis=1)
                   @ np.expand_dims(t, axis=0) / 1000)
    b = np.transpose(b)
    b0_v = np.expand_dims(2j * np.pi * np.concatenate(b0), axis=0)
    ct = np.transpose(np.exp(-np.expand_dims(tl, axis=1) @ b0_v))

    return b, ct
