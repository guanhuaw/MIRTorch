from .linearmaps import LinearMap
import torch
from torch import Tensor
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Union, Sequence


def coeffs_to_tensor(yl: Tensor,
                     yh: Sequence[Tensor]):
    """
    Array 2D DFT array into a tensor
    Args:
        yl:
        yh:
    Returns:

    """
    nlevel = len(yh)
    dic_size = []
    size_x = yl.shape[-2]
    size_y = yl.shape[-1]
    dic_size.append(list(yl.shape[-2:]))
    for ilevel in range(nlevel - 1, -1, -1):
        size_x += yh[ilevel].shape[-2]
        size_y += yh[ilevel].shape[-1]
        dic_size.append(list(yh[ilevel].shape[-2:]))
    wl_cat = torch.zeros(list(yl.shape[:-2]) + [size_x, size_y])
    wl_cat[..., :yl.shape[-2], :yl.shape[-1]] = yl
    for ilevel in range(nlevel):
        start_x = 0
        start_y = 0
        y = yh[nlevel - ilevel - 1]
        print(len(dic_size))
        print(ilevel)
        for i in range(ilevel + 1): start_x += dic_size[i][0]
        for i in range(ilevel + 1): start_y += dic_size[i][1]
        wl_cat[..., start_x:start_x + y.shape[-2], start_y:start_y + y.shape[-1]] = y[..., 2, :, :]
        wl_cat[..., :y.shape[-2], start_y:start_y + y.shape[-1]] = y[..., 1, :, :]
        wl_cat[..., start_x:start_x + y.shape[-2], :y.shape[-1]] = y[..., 0, :, :]
    return wl_cat, dic_size


def tensor_to_coeffs(wl_cat: Tensor,
                     dic_size: Sequence[int]):
    yl = wl_cat[..., :dic_size[0][0], :dic_size[0][1]]
    yh = []
    for ilevel in range(len(dic_size) - 1):
        start_x = 0
        start_y = 0
        for i in range(ilevel + 1): start_x += dic_size[i][0]
        for i in range(ilevel + 1): start_y += dic_size[i][1]
        aa = wl_cat[..., start_x:start_x + dic_size[ilevel + 1][0], start_y:start_y + dic_size[ilevel + 1][1]]
        ad = wl_cat[..., :dic_size[ilevel + 1][0], start_y:start_y + dic_size[ilevel + 1][1]]
        da = wl_cat[..., start_x:start_x + dic_size[ilevel + 1][0], :dic_size[ilevel + 1][1]]
        yh.insert(0, torch.stack((da, ad, aa), dim=-3))

    return yl, yh


class Wavelet2D(LinearMap):
    """
    A very preliminary implementation of 2D DWT.
    Implementation based on Pytorch_wavelets toolboxes:
    https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html
    It should support all wave types available in PyWavelets
    Inputs:
        size_in: Input size.
                 If batchmode: [nbatch, nchannel, nx, ny],
                 else: [nx, ny]
        wave_type: all that pywt supports
        padding: 'zero', 'symmetric', 'reflect' or 'periodization'

    """

    # TODO: consider complex
    # TODO: compare autograd
    def __init__(self,
                 size_in: Sequence[int],
                 wave_type: str = 'db4',
                 padding: str = 'zero',
                 J: int = 3,
                 batchmode=True):
        self.J = J
        self.wave_type = wave_type
        self.padding = padding
        self.batchmode = batchmode
        self.Fop = DWTForward(J=self.J, mode=self.padding, wave=self.wave_type)
        self.Aop = DWTInverse(mode=self.padding, wave=self.wave_type)
        if batchmode:
            Yl, Yh = self.Fop(torch.zeros(size_in))
            wl_cat, self.dic_size = coeffs_to_tensor(Yl, Yh)
            size_out = wl_cat.shape
        else:
            Yl, Yh = self.Fop(torch.zeros(size_in).unsqueeze(0).unsqueeze(0))
            wl_cat, self.dic_size = coeffs_to_tensor(Yl, Yh)
            size_out = wl_cat.shape[2:]
        super(Wavelet2D, self).__init__(size_in, size_out)

    def _apply(self, x: Tensor) -> Tensor:
        if self.batchmode:
            Yl, Yh = self.Fop(x)
            wl_cat, _ = coeffs_to_tensor(Yl, Yh)
            return wl_cat
        else:
            Yl, Yh = self.Fop(x.unsqueeze(0).unsqueeze(0))
            wl_cat, _ = coeffs_to_tensor(Yl, Yh)
            return wl_cat.squeeze(0).squeeze(0)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        if self.batchmode:
            Yl, Yh = tensor_to_coeffs(x, self.dic_size)
            print(Yl.shape)
            print(Yh[0].shape,Yh[1].shape,Yh[2].shape)

            return self.Aop((Yl, Yh))
        else:
            Yl, Yh = tensor_to_coeffs(x.unsqueeze(0).unsqueeze(0), self.dic_size)
            return self.Aop((Yl, Yh)).squeeze(0).squeeze(0)
