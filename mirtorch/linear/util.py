from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import Tensor


def finitediff(x: Tensor, dim: int = -1, mode="reflexive"):
    """
    Apply finite difference operator on a certain dimension.
    Args:
        x: tensor, input data
        dim: int, dimension to apply the operator
        mode: 'reflexive' or 'periodic'
    Returns:
        y: the first-order finite difference of x
    """
    if mode == "reflexive":
        len_dim = x.shape[dim]
        return torch.narrow(x, dim, 1, len_dim - 1) - torch.narrow(
            x, dim, 0, len_dim - 1
        )
    elif mode == "periodic":
        return torch.roll(x, 1, dims=dim) - x
    else:
        raise ValueError("mode should be either 'reflexive' or 'periodic'")


def finitediff_adj(y: Tensor, dim: int = -1, mode="reflexive"):
    """
    Apply finite difference operator on a certain dimension. Adjoint operator.
    Args:
        y: tensor, input data
        dim: int, dimension to apply the operator
        mode: 'reflexive' or 'periodic'

    Returns:
        y: the first-order finite difference of x
    """
    if mode == "reflexive":
        len_dim = y.shape[dim]
        return torch.cat(
            (
                -torch.narrow(y, dim, 0, 1),
                (
                    torch.narrow(y, dim, 0, len_dim - 1)
                    - torch.narrow(y, dim, 1, len_dim - 1)
                ),
                torch.narrow(y, dim, len_dim - 1, 1),
            ),
            dim=dim,
        )
    elif mode == "periodic":
        return torch.roll(y, -1, dims=dim) - y
    else:
        raise ValueError("mode should be either 'reflexive' or 'periodic'")


class DiffFunc(torch.autograd.Function):
    """
    autograd.Function for the 1st-order finite difference operators, on top of auto-diff.
    """

    @staticmethod
    def forward(ctx, x, dim, mode):
        ctx.dim = dim
        ctx.mode = mode
        return finitediff(x, dim, mode)

    @staticmethod
    def backward(ctx, dx):
        return finitediff_adj(dx, ctx.dim, ctx.mode), None, None


class DiffFunc_adj(torch.autograd.Function):
    """
    autograd.Function for the 1st-order finite difference operators, on top of auto-diff.
    """

    @staticmethod
    def forward(ctx, x, dim, mode):
        ctx.dim = dim
        ctx.mode = mode
        return finitediff_adj(x, dim, mode)

    @staticmethod
    def backward(ctx, dx):
        return finitediff(dx, ctx.dim, ctx.mode), None, None


def fftshift(x: Tensor, dims: Union[int, List[int]] | None = None):
    """
    Similar to np.fft.fftshift but applies to PyTorch tensors. From fastMRI code.
    """
    if dims is None:
        dims = list(range(x.dim()))
        shifts = [dim // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = [x.shape[dims] // 2]
        dims = [dims]
    else:
        shifts = [x.shape[i] // 2 for i in dims]
    return torch.roll(x, shifts, dims)


def ifftshift(x: Tensor, dims: Union[int, Tuple[int]] | None = None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch tensors. From fastMRI code.
    """
    if dims is None:
        dims = list(range(x.dim()))
        shifts = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dims, int):
        shifts = [(x.shape[dims] + 1) // 2]
        dims = [dims]
    else:
        shifts = [(x.shape[i] + 1) // 2 for i in dims]
    return torch.roll(x, shifts, dims)


def dim_conv(dim_in, dim_kernel_size, dim_stride=1, dim_padding=0, dim_dilation=1):
    dim_out = (
        dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
    ) // dim_stride + 1
    return dim_out


def imrotate(img, angle):
    """
    Args:
        img: N * C * H * W tensor
        angle: in degree
    Returns:
        rotated img
    """
    return torchvision.transforms.functional.rotate(
        img,
        angle,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        fill=0,
    )


def fft2(img):
    """
    Args:
        img: H * W tensor
    Returns:
         2D FFT of the img
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))


def ifft2(img):
    """
    Args:
        img: H * W tensor
    Returns:
        2D iFFT of the img
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(img)))


def power2(x):
    """
    Args:
        x: floating point
    Returns:

    """
    return (2 ** (np.ceil(np.log2(x)))).astype(int)


def _padup(nx, px):
    """
    Args:
        nx: floating point
        px: floating point
    Returns:

    """
    return np.ceil((power2(nx + px - 1) - nx) / 2).astype(int)


def _paddown(nx, px):
    """
    Args:
        nx: floating point
        px: floating point
    Returns:

    """
    return np.floor((power2(nx + px - 1) - nx) / 2).astype(int)


def _padleft(nz, pz):
    """
    Args:
        nz: floating point
        pz: floating point
    Returns:
    """
    return np.ceil((power2(nz + pz - 1) - nz) / 2).astype(int)


def _padright(nz, pz):
    """
    Args:
        nz: floating point
        pz: floating point
    Returns:
    """
    return np.floor((power2(nz + pz - 1) - nz) / 2).astype(int)


def pad2sizezero(img, padx, padz):
    """
    Args:
        mg: H * W tensor
        padx: floating point
        padz: floating point
    Returns:
    """
    px, pz = img.shape
    pad_img = torch.zeros(padx, padz).to(img.device).to(img.dtype)
    padx_dims = np.ceil((padx - px) / 2).astype(int)
    padz_dims = np.ceil((padz - pz) / 2).astype(int)
    pad_img[padx_dims : padx_dims + px, padz_dims : padz_dims + pz] = img
    return pad_img


def fft_conv(img, ker):
    """
    Args:
        img: nx * nz
        ker: px * pz
    Returns:
        xout: nx * nz
    """
    nx, nz = img.shape[0], img.shape[1]
    px, pz = ker.shape[0], ker.shape[1]
    padup = _padup(nx, px)
    paddown = _paddown(nx, px)
    padleft = _padleft(nz, pz)
    padright = _padright(nz, pz)
    m = nn.ReplicationPad2d((padleft, padright, padup, paddown))
    pad_img = m(img.unsqueeze(0).unsqueeze(0)).squeeze()

    padx, padz = pad_img.shape[0], pad_img.shape[1]

    pad_ker = pad2sizezero(ker, padx, padz)
    pad_img_fft = fft2(pad_img)
    pad_ker_fft = fft2(pad_ker)
    freq = torch.mul(pad_img_fft, pad_ker_fft)
    xout = torch.real(ifft2(freq))
    return xout[padup : padup + nx, padleft : padleft + nz]


def fft_conv_adj(img, ker):
    """
    Args:
        img: nx * nz
        ker: px * pz
    Returns:
        xout: nx * nz
    """
    nx, nz = img.shape[0], img.shape[1]
    px, pz = ker.shape[0], ker.shape[1]
    padup = _padup(nx, px)
    paddown = _paddown(nx, px)
    padleft = _padleft(nz, pz)
    padright = _padright(nz, pz)
    m = nn.ZeroPad2d((padleft, padright, padup, paddown))
    pad_img = m(img.unsqueeze(0).unsqueeze(0)).squeeze()

    padx, padz = pad_img.shape[0], pad_img.shape[1]

    pad_ker = pad2sizezero(ker, padx, padz)
    pad_img_fft = fft2(pad_img)
    pad_ker_fft = fft2(pad_ker)
    freq = torch.mul(pad_img_fft, pad_ker_fft)
    xout = torch.real(ifft2(freq))
    xout[padup, :] += torch.sum(xout[0:padup, :], dim=0)
    xout[nx + padup - 1, :] += torch.sum(xout[nx + padup :, :], dim=0)
    xout[:, padleft] += torch.sum(xout[:, 0:padleft], dim=1)
    xout[:, nz + padleft - 1] += torch.sum(xout[:, nz + padleft :], dim=1)
    return xout[padup : padup + nx, padleft : padleft + nz]


def map2x(x1, y1, x2, y2):
    """
    Args:
        x1: floating point
        x2: nx * nz
        y1: floating point
        y2: nx * nz
    Return:
        x: nx * nz
    """
    x = x1 - (y1.mul(x1 - x2)).div(y1 - y2)
    return x


def map2y(x1, y1, x2, y2):
    """
    Args:
        x1: floating point
        x2: nx * nz
        y1: floating point
        y2: nx * nz
    Returns:
        y: nx * nz
    """
    y = y1 - (x1.mul(y1 - y2)).div(x1 - x2)
    return y


def integrate1D(p_v, pixelSize):
    """
    Args:
        p_v: nx * nz
        pixelSize: floating point
    Returns:
        Ppj: (nx+1) * nz
    """
    n_pixel = len(p_v)

    P_x = 0
    Ppj = torch.zeros(n_pixel + 1).to(p_v.device)

    for pj in range(n_pixel):
        P_x += p_v[pj] * pixelSize[pj]

        Ppj[pj + 1] = P_x

    return Ppj
