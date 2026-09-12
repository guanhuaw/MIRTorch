from collections.abc import Sequence

import pywt
import torch
import torch.nn.functional as F
from torch import Tensor

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
    padding. It equals the inverse for orthogonal wavelets with periodization
    when both spatial dimensions are divisible by ``2**J``, but not for
    general biorthogonal wavelets, boundary modes, or odd-length extensions.

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
            wavelet = pywt.Wavelet(wave_type)
        except ValueError as error:
            raise ValueError(f"unknown wavelet {wave_type!r}") from error

        # Retain the original coefficients for double-precision input, rather
        # than promoting filters that have already been rounded to float32.
        self._filter_values = (wavelet.dec_lo[::-1], wavelet.dec_hi[::-1])
        self._filters = torch.tensor(self._filter_values, device=device)
        self._level_shapes = [spatial_shape]
        self._extensions = []
        for _ in range(J):
            shape = self._level_shapes[-1]
            self._extensions.append(
                tuple(self._extension(n, wavelet.dec_len, device) for n in shape)
            )
            self._level_shapes.append(
                tuple(pywt.dwt_coeff_len(n, wavelet.dec_len, padding) for n in shape)
            )
        packed_shape = tuple(
            self._level_shapes[-1][d]
            + sum(shape[d] for shape in self._level_shapes[1:])
            for d in range(2)
        )
        size_out = (*size_in[:-2], *packed_shape)
        super().__init__(size_in, size_out)

    def _extension(self, length: int, filter_length: int, device):
        """Describe boundary extension before a stride-two convolution."""
        if self.padding == "periodization":
            even_length = length + length % 2
            indices = torch.arange(
                1 - filter_length // 2,
                even_length + filter_length // 2 - 1,
                device=device,
            )
            # Odd-length periodization repeats the last sample before wrapping.
            return indices.remainder(even_length).clamp_max(length - 1)

        output_length = pywt.dwt_coeff_len(length, filter_length, self.padding)
        total = 2 * (output_length - 1) - length + filter_length
        before, after = total // 2, (total + 1) // 2
        if self.padding == "zero":
            return before, after
        indices = torch.arange(-before, length + after, device=device)
        if self.padding == "symmetric":
            indices = indices.remainder(2 * length)
            return torch.minimum(indices, 2 * length - 1 - indices)
        if length == 1:
            return torch.zeros_like(indices)
        indices = indices.remainder(2 * (length - 1))
        return torch.minimum(indices, 2 * (length - 1) - indices)

    def _filter_bank(self, x: Tensor, dim: int, channels: int) -> Tensor:
        if x.dtype == self._filters.dtype:
            filters = self._filters.to(device=x.device)
        else:
            filters = x.new_tensor(self._filter_values)
        shape = (2, 1, -1, 1) if dim == 2 else (2, 1, 1, -1)
        return filters.reshape(shape).repeat(channels, 1, 1, 1)

    def _analysis_axis(self, x: Tensor, dim: int, extension) -> Tensor:
        filters = self._filter_bank(x, dim, x.shape[1])
        if isinstance(extension, tuple):
            pad = (0, 0, *extension) if dim == 2 else (*extension, 0, 0)
            x = F.pad(x, pad)
        else:
            x = x.index_select(dim, extension.to(x.device))
        stride = (2, 1) if dim == 2 else (1, 2)
        return F.conv2d(x, filters, stride=stride, groups=x.shape[1])

    def _adjoint_axis(self, x: Tensor, dim: int, extension, length: int) -> Tensor:
        """Transpose filtering and sum every extended sample into its source."""
        channels = x.shape[1] // 2
        filters = self._filter_bank(x, dim, channels)
        stride = (2, 1) if dim == 2 else (1, 2)
        extended = F.conv_transpose2d(x, filters, stride=stride, groups=channels)
        if isinstance(extension, tuple):
            return extended.narrow(dim, extension[0], length)
        shape = list(extended.shape)
        shape[dim] = length
        return extended.new_zeros(shape).index_add(
            dim, extension.to(x.device), extended
        )

    def _analysis(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Apply the DWT using native operations with an exact autograd VJP."""
        details = []
        low = x
        for rows, columns in self._extensions:
            bands = self._analysis_axis(low, 3, columns)
            bands = self._analysis_axis(bands, 2, rows)
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
                torch.view_as_real(x.resolve_conj())
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
        height, width = self._level_shapes[-1]
        low = x[..., :height, :width]
        row, column = height, width
        for level in range(self.J - 1, -1, -1):
            height, width = self._level_shapes[level + 1]
            bands = torch.stack(
                (
                    low,
                    x[..., row : row + height, :width],
                    x[..., :height, column : column + width],
                    x[..., row : row + height, column : column + width],
                ),
                dim=2,
            ).flatten(1, 2)
            rows, columns = self._extensions[level]
            input_height, input_width = self._level_shapes[level]
            bands = self._adjoint_axis(bands, 2, rows, input_height)
            low = self._adjoint_axis(bands, 3, columns, input_width)
            row += height
            column += width
        return self._restore_layout(low, is_complex)
