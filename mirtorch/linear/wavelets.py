import sys
from typing import Sequence, Tuple, List

import torch
from mirtorch.vendors.pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor

from .linearmaps import LinearMap

# TODO: 3d wavelets


def coeffs_to_tensor(yl: Tensor, yh: Sequence[Tensor]) -> Tuple[Tensor, List[int]]:
    """
    Assemble 2D DWT array into a tensor
    Args:
        yl: coefficients of the lowest level
        yh: multi-level coefficients
    Returns:
        wl_cat: pieced-together tensors
        dic_size: recorded size of the dictionary
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
    wl_cat = torch.zeros(list(yl.shape[:-2]) + [size_x, size_y]).to(yl.device, yl.dtype)
    wl_cat[..., : yl.shape[-2], : yl.shape[-1]] = yl
    for ilevel in range(nlevel):
        start_x = 0
        start_y = 0
        y = yh[nlevel - ilevel - 1]
        for i in range(ilevel + 1):
            start_x += dic_size[i][0]
        for i in range(ilevel + 1):
            start_y += dic_size[i][1]
        wl_cat[
            ..., start_x : start_x + y.shape[-2], start_y : start_y + y.shape[-1]
        ] = y[..., 2, :, :]
        wl_cat[..., : y.shape[-2], start_y : start_y + y.shape[-1]] = y[..., 1, :, :]
        wl_cat[..., start_x : start_x + y.shape[-2], : y.shape[-1]] = y[..., 0, :, :]
    return wl_cat, dic_size


def tensor_to_coeffs(
    wl_cat: Tensor, dic_size: Sequence[int]
) -> Tuple[Tensor, List[int]]:
    """
    Args:
        wl_cat:
        dic_size:
    Returns:

    """
    yl = wl_cat[..., : dic_size[0][0], : dic_size[0][1]]
    yh = []
    for ilevel in range(len(dic_size) - 1):
        start_x = 0
        start_y = 0
        for i in range(ilevel + 1):
            start_x += dic_size[i][0]
        for i in range(ilevel + 1):
            start_y += dic_size[i][1]
        aa = wl_cat[
            ...,
            start_x : start_x + dic_size[ilevel + 1][0],
            start_y : start_y + dic_size[ilevel + 1][1],
        ]
        ad = wl_cat[
            ..., : dic_size[ilevel + 1][0], start_y : start_y + dic_size[ilevel + 1][1]
        ]
        da = wl_cat[
            ..., start_x : start_x + dic_size[ilevel + 1][0], : dic_size[ilevel + 1][1]
        ]
        yh.insert(0, torch.stack((da, ad, aa), dim=-3))

    return yl, yh


class Wavelet2D(LinearMap):
    """
    A very preliminary implementation of 2D DWT.
    Implementation based on Pytorch_wavelets toolboxes:
    https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html
    It should support all wave types available in PyWavelets
    Attributes:
        size_in: Input size. If batchmode: [nbatch, nchannel, nx, ny]; else [nx, ny] (real)
        wave_type: all that pywt supports
        padding: 'zero', 'symmetric', 'reflect' or 'periodization'
        When using periodization, it should be a unitary transform
    NB: x should be single precision float ...
    TODO: 3D version of it
    """

    def __init__(
        self,
        size_in: Sequence[int],
        wave_type: str = "db4",
        padding: str = "zero",
        J: int = 3,
        device="cpu",
    ):
        self.J = J
        self.wave_type = wave_type
        self.padding = padding
        if len(size_in) == 4:
            self.batchmode = True
        elif len(size_in) == 2:
            self.batchmode = False
        else:
            sys.exit(
                "Input size should be of 2D wavelets should be [nbatch, nchannel, nx, ny] or [nx, ny]"
            )

        self.Fop = DWTForward(J=self.J, mode=self.padding, wave=self.wave_type).to(
            device
        )
        self.Aop = DWTInverse(mode=self.padding, wave=self.wave_type).to(device)
        if self.batchmode:
            Yl, Yh = self.Fop(torch.zeros(size_in).to(device))
            wl_cat, self.dic_size = coeffs_to_tensor(Yl, Yh)
            size_out = wl_cat.shape
        else:
            Yl, Yh = self.Fop(torch.zeros(size_in).to(device).unsqueeze(0).unsqueeze(0))
            wl_cat, self.dic_size = coeffs_to_tensor(Yl, Yh)
            size_out = wl_cat.shape[2:]
        super(Wavelet2D, self).__init__(size_in, size_out)

    def _apply(self, x: Tensor) -> Tensor:
        if x.is_complex():
            if self.batchmode:
                x = (
                    torch.view_as_real(x)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(
                        self.size_in[0],
                        self.size_in[1] * 2,
                        self.size_in[2],
                        self.size_in[3],
                    )
                    .contiguous()
                )
                Yl, Yh = self.Fop(x)
                wl_cat, _ = coeffs_to_tensor(Yl, Yh)
                wl_cat = torch.view_as_complex(
                    wl_cat.reshape(
                        wl_cat.shape[0],
                        wl_cat.shape[1] // 2,
                        2,
                        wl_cat.shape[2],
                        wl_cat.shape[3],
                    )
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                return wl_cat
            else:
                x = torch.view_as_real(x).permute(2, 0, 1).contiguous()
                Yl, Yh = self.Fop(x.unsqueeze(0))
                wl_cat, _ = coeffs_to_tensor(Yl, Yh)
                return torch.view_as_complex(
                    wl_cat.squeeze(0).permute(1, 2, 0).contiguous()
                )
        else:
            if self.batchmode:
                Yl, Yh = self.Fop(x)
                wl_cat, _ = coeffs_to_tensor(Yl, Yh)
                return wl_cat
            else:
                Yl, Yh = self.Fop(x.unsqueeze(0).unsqueeze(0))
                wl_cat, _ = coeffs_to_tensor(Yl, Yh)
                return wl_cat.squeeze(0).squeeze(0)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        if x.is_complex():
            if self.batchmode:
                x = (
                    torch.view_as_real(x)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(
                        self.size_out[0],
                        self.size_out[1] * 2,
                        self.size_out[2],
                        self.size_out[3],
                    )
                    .contiguous()
                )
                Yl, Yh = tensor_to_coeffs(x, self.dic_size)
                y = self.Aop((Yl, Yh))
                return torch.view_as_complex(
                    y.reshape(y.shape[0], y.shape[1] // 2, 2, y.shape[2], y.shape[3])
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
            else:
                x = torch.view_as_real(x).permute(2, 0, 1).contiguous()
                Yl, Yh = tensor_to_coeffs(x.unsqueeze(0).unsqueeze(0), self.dic_size)
                return torch.view_as_complex(
                    (self.Aop((Yl, Yh)).squeeze(0)).permute(1, 2, 0).contiguous()
                )
        else:
            if self.batchmode:
                Yl, Yh = tensor_to_coeffs(x, self.dic_size)

                return self.Aop((Yl, Yh))
            else:
                Yl, Yh = tensor_to_coeffs(x.unsqueeze(0).unsqueeze(0), self.dic_size)
                return self.Aop((Yl, Yh)).squeeze(0).squeeze(0)
