import pytest
import torch

from mirtorch.linear import Wavelet2D
from mirtorch.vendors.pytorch_wavelets import (
    DTCWTForward,
    DTCWTInverse,
    DWT1DForward,
    DWT1DInverse,
    __version__,
)
from mirtorch.vendors.pytorch_wavelets.dwt.lowlevel import prep_filt_afb2d


@pytest.mark.parametrize("shape", [(32, 32), (1, 2, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="Apple Metal is unavailable",
            ),
        ),
    ],
)
def test_wavelet2d_periodization_round_trip(shape, dtype, device):
    image = torch.randn(shape, dtype=dtype, device=device)
    operator = Wavelet2D(
        image.shape,
        wave_type="db4",
        padding="periodization",
        J=2,
        device=device,
    )
    reconstructed = operator.H(operator(image))
    assert torch.allclose(reconstructed, image, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("shape", "wave_type", "padding"),
    [
        ((17, 17), "db4", "zero"),
        ((32, 32), "db4", "symmetric"),
        ((1, 2, 32, 32), "bior2.2", "periodization"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="Apple Metal is unavailable",
            ),
        ),
    ],
)
def test_wavelet2d_has_exact_adjoint(
    shape,
    wave_type,
    padding,
    dtype,
    device,
):
    image = torch.randn(shape, dtype=dtype, device=device)
    operator = Wavelet2D(
        shape,
        wave_type=wave_type,
        padding=padding,
        J=1 if shape[-1] == 17 else 2,
        device=device,
    )
    coefficients = torch.randn(operator.size_out, dtype=dtype, device=device)

    adjoint = operator.H(coefficients)
    lhs = (operator(image).conj() * coefficients).sum()
    rhs = (image.conj() * adjoint).sum()

    assert tuple(adjoint.shape) == shape
    torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="Apple Metal is unavailable",
            ),
        ),
    ],
)
def test_wavelet2d_forward_and_adjoint_gradients(dtype, device):
    shape = (1, 2, 32, 32)
    operator = Wavelet2D(
        shape,
        wave_type="bior2.2",
        padding="symmetric",
        J=2,
        device=device,
    )
    image = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    coefficients = torch.randn(
        operator.size_out,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    forward_objective = (operator(image).conj() * coefficients.detach()).sum().real
    (image_gradient,) = torch.autograd.grad(forward_objective, image)
    torch.testing.assert_close(
        image_gradient,
        operator.H(coefficients.detach()),
        rtol=3e-5,
        atol=3e-5,
    )

    adjoint_objective = (operator.H(coefficients).conj() * image.detach()).sum().real
    (coefficient_gradient,) = torch.autograd.grad(adjoint_objective, coefficients)
    torch.testing.assert_close(
        coefficient_gradient,
        operator(image.detach()),
        rtol=3e-5,
        atol=3e-5,
    )


def test_wavelet2d_rejects_invalid_rank_and_level():
    with pytest.raises(ValueError, match="Input size"):
        Wavelet2D((1, 2, 3))
    with pytest.raises(ValueError, match="positive integer"):
        Wavelet2D((8, 8), wave_type="db4", J=0)


def test_wavelet2d_preserves_deep_padded_decompositions():
    image = torch.randn(32, 32)
    operator = Wavelet2D(image.shape)
    coefficients = torch.randn(operator.size_out)

    lhs = (operator(image) * coefficients).sum()
    rhs = (image * operator.H(coefficients)).sum()

    torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-5)


def test_vendored_dtcwt_coefficients_ship_and_round_trip():
    assert __version__ == "1.3.0"
    image = torch.randn(1, 1, 16, 16)
    coefficients = DTCWTForward(J=2)(image)
    reconstructed = DTCWTInverse()(coefficients)
    assert torch.allclose(reconstructed, image, rtol=2e-5, atol=2e-5)


def test_vendored_dwt1d_round_trip():
    signal = torch.randn(2, 3, 32)
    coefficients = DWT1DForward(J=2, wave="db2", mode="periodization")(signal)
    reconstructed = DWT1DInverse(wave="db2", mode="periodization")(coefficients)
    torch.testing.assert_close(reconstructed, signal, rtol=2e-5, atol=2e-5)


def test_vendored_filter_preparation_defaults_row_filters():
    prepared = prep_filt_afb2d([0.5, 0.5], [-0.5, 0.5])
    assert len(prepared) == 4
    assert all(isinstance(item, torch.Tensor) for item in prepared)
