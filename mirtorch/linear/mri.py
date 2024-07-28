"""
Discrete-to-discreate system matrices for MRI.
2021-02. Guanhua Wang, University of Michigan
"""

import math
from typing import Union, List, Tuple

import numpy as np
import torch
import torchkbnufft as tkbn
from torch import Tensor
from torch.fft import fftn, ifftn

from .linearmaps import LinearMap
from .util import fftshift, ifftshift


class FFTCn(LinearMap):
    r"""
    FFT operators with FFTshift and iFFTshift for multidimensional data.
    Pytorch provides three modes in FFT: 'ortho', 'forward', 'backward'.
    Each pair of FFT and iFFT with same mode is the inverse, but not necessarily the adjoint to each other.

    Attributes:
        norm: normalization of the fft ('ortho', 'forward' or 'backward')
        dims: the dimensions to apply the fft
    """

    def __init__(
        self,
        size_in: List[int],
        size_out: List[int],
        dims: Tuple[int] | None = None,
        norm: str = "ortho",
    ):
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
        if self.norm == "ortho":
            x = ifftn(x, dim=self.dims, norm="ortho")
        elif self.norm == "forward":
            x = ifftn(x, dim=self.dims, norm="backward")
        else:
            x = ifftn(x, dim=self.dims, norm="forward")
        x = fftshift(x, self.dims)
        return x


class Sense(LinearMap):
    r"""
    Cartesian sense operator, following "SENSE: Sensitivity encoding for fast MRI".
    The input/ourput size depends on the sensitivity maps.
    If we use the batch dimension, the input dimension is [nbatch, 1, nx, ny, (nz)], and the output is [nbatch, ncoil, nx, ny, (nz)].
    Otherwise, the input dimension is [nx, ny, (nz)], and the output is [ncoil, nx, ny, (nz)].

    Attributes:
        masks: tensor with dimension [(batch), nx, ny, (nz)]
        sensitivity maps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. On the same device as masks
        batchmode: bool, determining if there exist batch and channel dimension (should always be 1).
        norm: normalization of the fft ('ortho', 'forward' or 'backward')
    """

    def __init__(
        self, smaps: Tensor, masks: Tensor, norm: str = "ortho", batchmode: bool = True
    ):
        if batchmode:
            # comform to [nbatch, 1, nx, ny, nz]
            size_in = [smaps.shape[0]] + [1] + list(smaps.shape[2:])
            size_out = list(smaps.shape)
            dims = tuple(np.arange(2, len(smaps.shape)))
            self.masks = masks.unsqueeze(1)
            assert (
                smaps.shape[2:] == masks.shape[1:]
            ), "size of sensitivity maps and mask not matched!"
        else:
            size_in = list(smaps.shape[1:])
            size_out = list(smaps.shape)
            dims = tuple(np.arange(1, len(smaps.shape)))
            self.masks = masks
            assert (
                smaps.shape[1:] == masks.shape
            ), "size of sensitivity maps and mask not matched!"
        super(Sense, self).__init__(size_in, size_out)
        self.norm = norm
        self.dims = dims
        self.smaps = smaps
        self.batchmode = batchmode

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [batch, 1, nx, ny, (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            y:  tensor with dimension [batch, ncoil, nx, ny, (nz)] (batchmode=True) or [ncoil, nx, ny, nz]
        """
        x = x * self.smaps
        x = ifftshift(x, self.dims)
        k = fftn(x, dim=self.dims, norm=self.norm)
        k = fftshift(k, self.dims) * self.masks
        return k

    def _apply_adjoint(self, k: Tensor) -> Tensor:
        r"""
        Args:
            y:  tensor with dimension [batch, ncoil, nx, ny, (nz)] (batchmode=True) or [ncoil, nx, ny, nz]
        Returns:
            x:  tensor with dimension [batch, 1, nx, ny, (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        assert (
            k.shape == self.smaps.shape
        ), "sensitivity maps and signal's shape mismatch"
        k = k * self.masks
        k = ifftshift(k, self.dims)
        if self.norm == "ortho":
            x = ifftn(k, dim=self.dims, norm="ortho")
        elif self.norm == "forward":
            x = ifftn(k, dim=self.dims, norm="backward")
        else:
            x = ifftn(k, dim=self.dims, norm="forward")
        x = fftshift(x, self.dims)
        if self.batchmode:
            x = (x * torch.conj(self.smaps)).sum(1).unsqueeze(1)
        else:
            x = (x * torch.conj(self.smaps)).sum(0)
        return x


class NuSense(LinearMap):
    r"""
    Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    The implementation calls Matthew Muckley's Torchkbnufft toolbox:
    https://github.com/mmuckley/torchkbnufft
    The input/ourput size depends on the sensitivity maps.
    If we use the batch dimension, the input dimension is [nbatch, 1, nx, ny, (nz)], and the output is [nbatch, ncoil, npoints].
    Otherwise, the input dimension is [nx, ny, (nz)], and the output is [ncoil, npoints].


    Attributes:
        traj: tensor with dimension [(batch), ndim, nshot*npoints]. Note that traj can have no batch dimension even x have. ref: https://github.com/mmuckley/torchkbnufft/pull/24
        sensitivity maps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. On the same device as traj.
        sequential: bool, memory saving mode
        batchmode: bool, determining if there exist batch and channel dimension (should always be 1).
        norm: normalization of the fft ('ortho' or None)
        numpoints: int, number of interpolation points in gridding.
        grid_size: float, oversampling ratio (>1)
    """

    def __init__(
        self,
        smaps: Tensor,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        numpoints: Union[int, List[int]] = 6,
        grid_size: float = 2,
        sequential: bool = False,
    ):
        self.smaps = smaps
        self.norm = norm
        self.traj = traj
        self.batchmode = batchmode
        self.sequential = sequential
        assert grid_size >= 1, "grid size should be greater than 1"
        if batchmode:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
            )
            self.A = tkbn.KbNufft(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(
                im_size=tuple(smaps.shape[2:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            size_in = [smaps.shape[0]] + [1] + list(smaps.shape[2:])
            size_out = list(smaps.shape[0:2]) + [traj.shape[-1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out))
        else:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[1:]) * grid_size).astype(int)
            )
            self.A = tkbn.KbNufft(
                im_size=tuple(smaps.shape[1:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            self.AT = tkbn.KbNufftAdjoint(
                im_size=tuple(smaps.shape[1:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
            ).to(smaps)
            size_in = smaps.shape[1:]
            size_out = [smaps.shape[0]] + [traj.shape[-1]]
            super(NuSense, self).__init__(tuple(size_in), tuple(size_out))

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            k： tensor with dimension [batch, ncoil, nshot*npoints] or [ncoil, nshot*npoints]
        """
        if self.sequential:
            k = torch.zeros(self.size_out).to(self.smaps)
            if self.batchmode:
                for i in range(self.smaps.shape[1]):
                    k[:, i, ...] = self.A(
                        x,
                        self.traj,
                        smaps=self.smaps[:, i, ...].unsqueeze(1),
                        norm=self.norm,
                    ).squeeze(1)
                return k
            else:
                for i in range(self.smaps.shape[0]):
                    k[i, ...] = (
                        self.A(
                            x.unsqueeze(0).unsqueeze(0),
                            self.traj,
                            smaps=self.smaps[i, ...].unsqueeze(0).unsqueeze(0),
                            norm=self.norm,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                return k
        else:
            if self.batchmode:
                return self.A(x, self.traj, smaps=self.smaps, norm=self.norm)
            else:
                return (
                    self.A(
                        x.unsqueeze(0).unsqueeze(0),
                        self.traj,
                        smaps=self.smaps.unsqueeze(0),
                        norm=self.norm,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        r"""
        Args:
            y： tensor with dimension [batch, ncoil, nshot*npoints] (batchmode=True)  or [ncoil, nshot*npoints]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self.sequential:
            x = torch.zeros(self.size_in).to(self.smaps)
            if self.batchmode:
                for i in range(self.smaps.shape[1]):
                    x += self.AT(
                        y[:, i, ...].unsqueeze(1),
                        self.traj,
                        smaps=self.smaps[:, i, ...].unsqueeze(1),
                        norm=self.norm,
                    )
                return x
            else:
                for i in range(self.smaps.shape[0]):
                    x += (
                        self.AT(
                            y[i, ...].unsqueeze(0).unsqueeze(0),
                            self.traj,
                            smaps=self.smaps[i, ...].unsqueeze(0).unsqueeze(0),
                            norm=self.norm,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                return x
        else:
            if self.batchmode:
                return self.AT(y, self.traj, smaps=self.smaps, norm=self.norm)
            else:
                return (
                    self.AT(
                        y.unsqueeze(0),
                        self.traj,
                        smaps=self.smaps.unsqueeze(0),
                        norm=self.norm,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )


class NuSenseGram(LinearMap):
    r"""
    Gram operator (A'A) of the Non-Cartesian sense operator: "SENSE: Sensitivity encoding for fast MRI"
    The implementation calls Matthew Muckley's Torchkbnufft toolbox:
    https://github.com/mmuckley/torchkbnufft
    The input/ourput size depends on the sensitivity maps.
    If we use the batch dimension, the input/output dimension is [nbatch, 1, nx, ny, (nz)].
    Otherwise, the input/output dimension is [nx, ny, (nz)].


    Attributes:
        traj: tensor with dimension [(batch), ndim, nshot*npoints]. Note that traj can have no batch dimension even x have. ref: https://github.com/mmuckley/torchkbnufft/pull/24
        sensitivity maps: tensor with dimension [(batch), ncoil, nx, ny, (nz)]. On the same device with traj.
        norm: normalization of the fft ('ortho' or None)
        numpoints: int, number of interpolation points in gridding.
        grid_size: float, oversampling ratio (>1)
        batchmode: bool, determining if there exist batch and channel dimension (should always be 1).
    """

    def __init__(
        self,
        smaps: Tensor,
        traj: Tensor,
        norm="ortho",
        batchmode=True,
        numpoints: Union[int, List[int]] = 6,
        grid_size: float = 2,
    ):
        self.smaps = smaps
        self.norm = norm
        self.traj = traj
        self.batchmode = batchmode
        self.toep_op = tkbn.ToepNufft()

        if batchmode:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
            )
            self.kernel = tkbn.calc_toeplitz_kernel(
                traj,
                list(smaps.shape[2:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
                norm=self.norm,
            )
            size_in = [smaps.shape[0]] + [1] + list(smaps.shape[2:])
            super(NuSenseGram, self).__init__(tuple(size_in), tuple(size_in))
        else:
            self.grid_size = tuple(
                np.floor(np.array(smaps.shape[1:]) * grid_size).astype(int)
            )
            self.kernel = tkbn.calc_toeplitz_kernel(
                traj,
                list(smaps.shape[1:]),
                grid_size=self.grid_size,
                numpoints=numpoints,
                norm=self.norm,
            )
            size_in = list(smaps.shape[1:])
            super(NuSenseGram, self).__init__(tuple(size_in), tuple(size_in))

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        Returns:
            x:  tensor with dimension [nbatch, 1, nx, ny (nz)] (batchmode=True) or [nx, ny, (nz)]
        """
        if self.batchmode:
            return self.toep_op(x, self.kernel, smaps=self.smaps, norm=self.norm)
        else:
            return (
                self.toep_op(
                    x.unsqueeze(0).unsqueeze(0),
                    self.kernel,
                    smaps=self.smaps.unsqueeze(0),
                    norm=self.norm,
                )
                .squeeze(0)
                .squeeze(0)
            )

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        if self.batchmode:
            return self.toep_op(y, self.kernel, smaps=self.smaps, norm=self.norm)
        else:
            return (
                self.toep_op(
                    y.unsqueeze(0).unsqueeze(0),
                    self.kernel,
                    smaps=self.smaps.unsqueeze(0),
                    norm=self.norm,
                )
                .squeeze(0)
                .squeeze(0)
            )


class Gmri(LinearMap):
    r"""
    B0-informed mri reconstruction, the name follows MIRT.
    Note that the data format is a little different from NuSENSE.
    The input/ourput size depends on the sensitivity maps.
    The input dimension is [nbatch, 1, nx, ny, (nz)], and the output is [nbatch, ncoil, nshot, nfe].

    Attributes:
        norm: normalization of the fft ('ortho' or None)
        smaps: tensor with dimension [batch, ncoil, nx, ny, (nz)] (must have a batch dimension). Sensitivity maps.
        zmap: tensor with dimension [batch, nx, ny, (nz)]. Off-resonance effects in Hz. ref: DOI: 10.1109/TSP.2005.853152
        traj: tensor with dimension [nbatch (or 1), ndimension, nshot, nreadout]
        numpoints: int, number of interpolation points in gridding.
        grid_size: float, oversampling ratio (>1)
        L: int, number of segmentation
        dt: float, dwell time in ms
        nbins: int, granularity of exponential approximation.
        T: tensor with dimension [nfe]. Descrbe the time (in ms) of readout out after excitation. When T is none,
        the readout is supposed to start immediately after the excitation.

    TODO: add DataParallel
    """

    def __init__(
        self,
        smaps: Tensor,
        zmap: Tensor,
        traj: Tensor,
        norm: str = "ortho",
        L: int = 6,
        nbins: int = 20,
        dt: int = 4e-3,
        numpoints: Union[int, List[int]] = 6,
        grid_size: float = 2,
        T: Tensor = None,
    ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        self.nbatch = self.smaps.shape[0]
        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        self.grid_size = tuple(
            np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
        )
        self.A = tkbn.KbNufft(
            im_size=tuple(smaps.shape[2:]),
            grid_size=self.grid_size,
            numpoints=numpoints,
        ).to(smaps)
        self.AT = tkbn.KbNufftAdjoint(
            im_size=tuple(smaps.shape[2:]),
            grid_size=self.grid_size,
            numpoints=numpoints,
        ).to(smaps)
        size_in = [self.nbatch] + [1] + list(smaps.shape[2:])
        size_out = (self.nbatch, self.nc, self.nshot, self.npoints)
        self.B = (
            torch.zeros(self.L, self.nbatch, 1, 1, self.npoints).to(self.smaps.device)
            * 1j
        )  # [L, batch, coil, shot, points]
        self.C = (
            torch.zeros((self.L, self.nbatch, 1) + tuple(self.smaps.shape[2:])).to(
                self.smaps.device
            )
            * 1j
        )  # [L, batch, 1, nx, ny ...]
        for ib in range(self.nbatch):
            if T is None:
                t = np.linspace(0, dt * self.npoints, self.npoints)
            else:
                t = T.cpu().numpy()
            b, c, _ = mri_exp_approx(zmap[ib].cpu().data.numpy(), nbins, L, t)
            self.B[:, ib, ...] = (
                torch.tensor(np.transpose(b))
                .to(smaps.device)
                .reshape(self.L, 1, 1, self.npoints)
            )
            self.C[:, ib, 0, ...] = (
                torch.tensor(np.transpose(c))
                .to(smaps.device)
                .reshape((self.L,) + tuple(zmap.shape[1:]))
            )
        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        super(Gmri, self).__init__(tuple(size_in), tuple(size_out))

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            k: k-space data, [(batch), ncoil, nshot, npoints]
        """
        y = torch.zeros(self.size_out).to(self.smaps)
        for il in range(self.L):
            y = y + self.B[il] * self.A(
                x * self.C[il], self.traj, smaps=self.smaps, norm=self.norm
            ).reshape(self.size_out)
        return y

    def _apply_adjoint(self, y: Tensor) -> Tensor:
        r"""
        Args:
            k: k-space data, [(batch), ncoil, nshot, npoints]
        Returns:
            x: [nbatch, 1, nx, ny (nz)]
        """
        x = torch.zeros(self.size_in).to(self.smaps)
        for il in range(self.L):
            x = x + self.C[il].conj() * self.AT(
                (y * self.B[il].conj()).reshape(
                    self.nbatch, self.nc, self.nshot * self.npoints
                ),
                self.traj,
                smaps=self.smaps,
                norm=self.norm,
            )
        return x


class GmriGram(LinearMap):
    r"""
    B0-informed mri reconstruction, the name follows MIRT.
    Note that the data format is a little different from NuSENSE.
    The input/ourput size depends on the sensitivity maps.
    The input dimension is [nbatch, 1, nx, ny, (nz)], and the output is [nbatch, ncoil, nshot, nfe].

    Attributes:
        norm: normalization of the fft ('ortho' or None)
        smaps: tensor with dimension [batch, ncoil, nx, ny, (nz)] (must have a batch dimension). Sensitivity maps.
        zmap: tensor with dimension [batch, nx, ny, (nz)]. Off-resonance effects in Hz. ref: DOI: 10.1109/TSP.2005.853152
        traj: tensor with dimension [nbatch (or 1), ndimension, nshot, nreadout]
        numpoints: int, number of interpolation points in gridding.
        grid_size: float, oversampling ratio (>1)
        L: int, number of segmentation
        dt: float, dwell time in ms
        nbins: int, granularity of exponential approximation.
        T: tensor with dimension [nfe]. Descrbe the time (in ms) of readout out after excitation. When T is none,
        the readout is supposed to start immediately after the excitation.

    TODO: add DataParallel
    """

    def __init__(
        self,
        smaps: Tensor,
        zmap: Tensor,
        traj: Tensor,
        norm: str = "ortho",
        L: int = 6,
        nbins: int = 20,
        dt: int = 4e-3,
        numpoints: Union[int, List[int]] = 6,
        grid_size: float = 2,
        T: Tensor = None,
    ):
        self.norm = norm
        self.smaps = smaps
        self.zmap = zmap
        self.L = L
        self.nbins = nbins
        self.dt = dt
        self.nbatch = self.smaps.shape[0]
        self.nc = self.smaps.shape[1]
        self.traj = traj
        _, self.ndim, self.nshot, self.npoints = self.traj.shape
        self.grid_size = tuple(
            np.floor(np.array(smaps.shape[2:]) * grid_size).astype(int)
        )
        size_in = [self.nbatch] + [1] + list(smaps.shape[2:])
        self.B = (
            torch.zeros(self.L, self.nbatch, 1, 1, self.npoints).to(self.smaps.device)
            * 1j
        )  # [L, batch, coil, shot, points]
        self.C = (
            torch.zeros((self.L, self.nbatch, 1) + tuple(self.smaps.shape[2:])).to(
                self.smaps.device
            )
            * 1j
        )  # [L, batch, 1, nx, ny ...]
        for ib in range(self.nbatch):
            if T is None:
                t = np.linspace(0, dt * self.npoints, self.npoints)
            else:
                t = T.cpu().numpy()
            b, c, tl = mri_exp_approx(zmap[ib].cpu().data.numpy(), nbins, L, t)
            self.B[:, ib, ...] = (
                torch.tensor(np.transpose(b))
                .to(smaps.device)
                .reshape(self.L, 1, 1, self.npoints)
            )
            self.C[:, ib, 0, ...] = (
                torch.tensor(np.transpose(c))
                .to(smaps.device)
                .reshape((self.L,) + tuple(zmap.shape[1:]))
            )
            self.tl = torch.tensor(tl).to(smaps.device)
        self.traj = self.traj.reshape(
            (self.traj.shape[0], self.ndim, self.nshot * self.npoints)
        )
        self.toep_op = tkbn.ToepNufft()
        self.kernel = []
        for il in range(self.L):
            self.kernel.append(
                tkbn.calc_toeplitz_kernel(
                    self.traj,
                    list(smaps.shape[2:]),
                    grid_size=self.grid_size,
                    numpoints=numpoints,
                    norm=self.norm,
                    weights=self.B[il]
                    .repeat(1, 1, self.nshot, 1)
                    .reshape(self.nbatch, 1, self.nshot * self.npoints),
                )
            )
        super(GmriGram, self).__init__(tuple(size_in), tuple(size_in))

    def _apply(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        y = torch.zeros_like(x).to(self.smaps)
        for il in range(self.L):
            D = torch.exp(-2 * math.pi * 1j * self.zmap.unsqueeze(1) * self.tl[il])
            y = y + D.conj() * self.toep_op(
                x * D, self.kernel[il], smaps=self.smaps, norm=self.norm
            )
        return y

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: [nbatch, 1, nx, ny (nz)]

        Returns:
            y: [nbatch, 1, nx, ny (nz)]
        """
        y = torch.zeros_like(x).to(self.smaps)
        for il in range(self.L):
            D = torch.exp(-2 * math.pi * 1j * self.zmap.unsqueeze(1) * self.tl[il])
            y = y + D.conj() * self.toep_op(
                x * D, self.kernel[il], smaps=self.smaps, norm=self.norm
            )
        return y


def mri_exp_approx(b0, bins, lseg, t):
    r"""
    From Sigpy: https://github.com/mikgroup/sigpy and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/
    Creates B [M*L] and Ct [L*N] matrices to approximate exp(-2i*pi*b0*t) [M*N]
    Args:
        b0: numpy array in dimension [nx, ny, nz], inhomogeneity matrix in Hz.
        bins: int, number of histogram bins to use.
        lseg: int, number of time segments.
        t: float, describing the readout time (ms).
    Returns:
        2-element tuple containing b: temporal interpolator; ct: off-resonance phase at each time segment center.
    TODO(guahuaw@umich.edu): The SVD approach and pure pytorch implementation.
    """

    # create time vector
    hist_wt, bin_edges = np.histogram(
        np.imag(2j * np.pi * np.ndarray.flatten(b0)), bins
    )

    # build B and Ct
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2
    zk = 0 + 1j * bin_centers
    tl = np.linspace(t[0], t[-1], lseg) / 1000  # time seg centers
    # calculate off-resonance phase @ each time seg, for histogram bins
    ch = np.exp(-np.expand_dims(tl, axis=1) @ np.expand_dims(zk, axis=0))
    w = np.diag(np.sqrt(hist_wt))
    p = np.linalg.pinv(w @ np.transpose(ch)) @ w
    b = p @ np.exp(-np.expand_dims(zk, axis=1) @ np.expand_dims(t, axis=0) / 1000)
    b = np.transpose(b)
    b0_v = np.expand_dims(2j * np.pi * np.ndarray.flatten(b0), axis=0)
    ct = np.transpose(np.exp(-np.expand_dims(tl, axis=1) @ b0_v))

    return b, ct, tl
