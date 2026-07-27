from collections.abc import Sequence

import pywt
import torch
from torch import Tensor

from mirtorch.vendors.pytorch_wavelets import DWTForward
from mirtorch.vendors.pytorch_wavelets.dwt import lowlevel

from .linearmaps import LinearMap

# TODO: 3d wavelets


def _coeffs_to_tensor(yl: Tensor, yh: Sequence[Tensor]) -> Tensor:
    """Pack a multilevel 2D DWT into one tensor."""
    nlevel = len(yh)
    band_sizes = []
    size_x = yl.shape[-2]
    size_y = yl.shape[-1]
    band_sizes.append(list(yl.shape[-2:]))
    for ilevel in range(nlevel - 1, -1, -1):
        size_x += yh[ilevel].shape[-2]
        size_y += yh[ilevel].shape[-1]
        band_sizes.append(list(yh[ilevel].shape[-2:]))
    wl_cat = yl.new_zeros((*yl.shape[:-2], size_x, size_y))
    wl_cat[..., : yl.shape[-2], : yl.shape[-1]] = yl
    for ilevel in range(nlevel):
        y = yh[nlevel - ilevel - 1]
        start_x = sum(size[0] for size in band_sizes[: ilevel + 1])
        start_y = sum(size[1] for size in band_sizes[: ilevel + 1])
        wl_cat[
            ..., start_x : start_x + y.shape[-2], start_y : start_y + y.shape[-1]
        ] = y[..., 2, :, :]
        wl_cat[..., : y.shape[-2], start_y : start_y + y.shape[-1]] = y[..., 1, :, :]
        wl_cat[..., start_x : start_x + y.shape[-2], : y.shape[-1]] = y[..., 0, :, :]
    return wl_cat


class Wavelet2D(LinearMap):
    """Packed multilevel two-dimensional discrete wavelet transform.

    ``A.H`` is the exact discrete Hermitian adjoint, including boundary
    padding. It equals the inverse for orthogonal wavelets with periodization,
    but not for general biorthogonal wavelets or padding modes.

    Attributes:
        size_in: ``[batch, channel, nx, ny]`` or ``[nx, ny]``.
        wave_type: Any wavelet name supported by PyWavelets.
        padding: ``"zero"``, ``"symmetric"``, ``"reflect"``, or
            ``"periodization"``.
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
        if not isinstance(J, int) or J < 1:
            raise ValueError("J must be a positive integer")
        if padding not in ("zero", "symmetric", "reflect", "periodization"):
            raise ValueError(
                "padding must be 'zero', 'symmetric', 'reflect', or 'periodization'"
            )
        if len(size_in) == 4:
            self.batchmode = True
            spatial_shape = tuple(size_in[-2:])
        elif len(size_in) == 2:
            self.batchmode = False
            spatial_shape = tuple(size_in)
        else:
            raise ValueError(
                "Input size should be of 2D wavelets should be [nbatch, nchannel, nx, ny] or [nx, ny]"
            )
        if any(not isinstance(size, int) or size < 1 for size in size_in):
            raise ValueError("size_in must contain positive integers")
        try:
            pywt.Wavelet(wave_type)
        except ValueError as error:
            raise ValueError(f"unknown wavelet {wave_type!r}") from error

        self.Fop = DWTForward(J=self.J, mode=self.padding, wave=self.wave_type).to(
            device
        )
        prototype = torch.zeros((1, 1, *spatial_shape), device=device)
        Yl, Yh = self._analysis(prototype)
        wl_cat = _coeffs_to_tensor(Yl, Yh)
        size_out = (*size_in[:-2], *wl_cat.shape[-2:])
        super().__init__(size_in, size_out)

    def _analysis(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Apply the DWT using native operations with an exact autograd VJP."""
        details = []
        low = x
        for _ in range(self.J):
            bands = lowlevel.afb1d(
                low,
                self.Fop.h0_row,
                self.Fop.h1_row,
                mode=self.padding,
                dim=3,
            )
            bands = lowlevel.afb1d(
                bands,
                self.Fop.h0_col,
                self.Fop.h1_col,
                mode=self.padding,
                dim=2,
            )
            shape = bands.shape
            bands = bands.reshape(
                shape[0],
                -1,
                4,
                shape[-2],
                shape[-1],
            )
            low = bands[:, :, 0].contiguous()
            details.append(bands[:, :, 1:].contiguous())
        return low, details

    def _as_real_channels(self, x: Tensor) -> tuple[Tensor, bool]:
        if not self.batchmode:
            x = x[None, None]
        is_complex = x.is_complex()
        if is_complex:
            batch, channels, height, width = x.shape
            x = (
                torch.view_as_real(x)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch, 2 * channels, height, width)
            )
        return x, is_complex

    def _restore_layout(self, x: Tensor, is_complex: bool) -> Tensor:
        if is_complex:
            batch, real_channels, height, width = x.shape
            x = torch.view_as_complex(
                x.reshape(batch, real_channels // 2, 2, height, width)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        if not self.batchmode:
            x = x[0, 0]
        return x

    def _apply(self, x: Tensor) -> Tensor:
        x, is_complex = self._as_real_channels(x)
        Yl, Yh = self._analysis(x)
        coefficients = _coeffs_to_tensor(Yl, Yh)
        return self._restore_layout(coefficients, is_complex)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        x, is_complex = self._as_real_channels(x)
        create_graph = torch.is_grad_enabled() and x.requires_grad
        with torch.enable_grad():
            prototype = torch.zeros(
                (*x.shape[:2], *self.size_in[-2:]),
                dtype=x.dtype,
                device=x.device,
                requires_grad=True,
            )
            Yl, Yh = self._analysis(prototype)
            coefficients = _coeffs_to_tensor(Yl, Yh)
            (image,) = torch.autograd.grad(
                coefficients,
                prototype,
                grad_outputs=x,
                create_graph=create_graph,
            )
        return self._restore_layout(image, is_complex)
