import subprocess
import sys
import textwrap

import pytest
import torch

from mirtorch.linear.util import (
    dim_conv,
    fft2,
    fft_conv,
    fft_conv_adj,
    fftshift,
    finitediff,
    finitediff_adj,
    ifft2,
    ifftshift,
    imrotate,
    integrate1D,
    map2x,
    map2y,
    nufft_trajectory_vjp,
    pad2sizezero,
)


@pytest.fixture
def tensor_2d():
    return torch.rand((4, 4))


@pytest.fixture
def tensor_3d():
    return torch.rand((3, 4, 4))


@pytest.fixture
def tensor_4d():
    return torch.rand((2, 3, 4, 4))


def test_finitediff(tensor_2d):
    result = finitediff(tensor_2d, dim=1, mode="reflexive")
    assert result.shape == (4, 3)


def test_finitediff_periodic(tensor_2d):
    result = finitediff(tensor_2d, dim=1, mode="periodic")
    assert result.shape == (4, 4)


def test_finitediff_adj(tensor_2d):
    result = finitediff_adj(tensor_2d, dim=1, mode="reflexive")
    assert result.shape == (4, 5)


def test_finitediff_adj_periodic(tensor_2d):
    result = finitediff_adj(tensor_2d, dim=1, mode="periodic")
    assert result.shape == (4, 4)


def test_fftshift(tensor_2d):
    result = fftshift(tensor_2d)
    assert result.shape == tensor_2d.shape


def test_ifftshift(tensor_2d):
    result = ifftshift(tensor_2d)
    assert result.shape == tensor_2d.shape


def test_dim_conv():
    result = dim_conv(32, 3, dim_stride=2, dim_padding=1)
    assert result == 16


def test_imrotate(tensor_4d):
    pytest.importorskip("torchvision")
    angle = 45
    result = imrotate(tensor_4d, angle)
    assert result.shape == tensor_4d.shape


def test_core_import_and_reconstruction_work_without_torchvision(tmp_path):
    script = textwrap.dedent(
        """\
        import sys

        # A fresh interpreter cannot reuse torchvision imported by other tests.
        sys.modules["torchvision"] = None
        import torch
        import mirtorch
        from mirtorch.alg import CG
        from mirtorch.linear import Diag
        from mirtorch.linear.util import imrotate

        rhs = torch.tensor([2.0, 8.0], requires_grad=True)
        result = CG(Diag(torch.tensor([2.0, 4.0]))).run(torch.zeros_like(rhs), rhs)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
        result.sum().backward()
        torch.testing.assert_close(rhs.grad, torch.tensor([0.5, 0.25]))

        try:
            imrotate(torch.zeros(1, 1, 4, 4), 30)
        except ImportError as error:
            assert "MIRTorch[examples]" in str(error), str(error)
        else:
            raise AssertionError("imrotate must explain its optional dependency")
        assert sys.modules["torchvision"] is None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fft2(tensor_2d):
    result = fft2(tensor_2d)
    assert result.shape == tensor_2d.shape


def test_ifft2(tensor_2d):
    result = ifft2(tensor_2d)
    assert result.shape == tensor_2d.shape


def test_pad2sizezero(tensor_2d):
    result = pad2sizezero(tensor_2d, 6, 6)
    assert result.shape == (6, 6)


def test_fft_conv(tensor_2d):
    ker = torch.rand((3, 3))
    result = fft_conv(tensor_2d, ker)
    assert result.shape == tensor_2d.shape


def test_fft_conv_adj(tensor_2d):
    ker = torch.rand((3, 3))
    result = fft_conv_adj(tensor_2d, ker)
    assert result.shape == tensor_2d.shape


_CONV_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _CONV_DEVICES.append("cuda")
elif torch.backends.mps.is_available():
    _CONV_DEVICES.append("mps")


def _spatial_replicate_convolution(image, kernel):
    """Independent centered convolution, including even/asymmetric kernels."""
    result = torch.zeros_like(image)
    for i in range(kernel.shape[0]):
        rows = (
            torch.arange(image.shape[0], device=image.device) - i + kernel.shape[0] // 2
        ).clamp(0, image.shape[0] - 1)
        for j in range(kernel.shape[1]):
            columns = (
                torch.arange(image.shape[1], device=image.device)
                - j
                + kernel.shape[1] // 2
            ).clamp(0, image.shape[1] - 1)
            result = result + kernel[i, j] * image[rows[:, None], columns[None, :]]
    return result


@pytest.mark.parametrize("device", _CONV_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize(
    ("shape", "kernel_shape"),
    [((5, 6), (3, 3)), ((6, 7), (2, 4)), ((1, 7), (1, 3)), ((1, 1), (1, 1))],
)
def test_fft_convolution_matches_spatial_reference_and_adjoint(
    device, dtype, shape, kernel_shape
):
    torch.manual_seed(29)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    kernel = torch.randn(kernel_shape, dtype=dtype, device=device)
    y = torch.randn_like(x)
    forward = fft_conv(x, kernel)
    adjoint = fft_conv_adj(y, kernel)
    assert forward.dtype == adjoint.dtype == dtype
    assert forward.device == adjoint.device == x.device
    torch.testing.assert_close(
        forward, _spatial_replicate_convolution(x, kernel), rtol=3e-5, atol=3e-5
    )
    torch.testing.assert_close(
        (forward.conj() * y).sum(),
        (x.conj() * adjoint).sum(),
        rtol=3e-5,
        atol=3e-5,
    )
    vjp = torch.autograd.grad(forward, x, y)[0]
    torch.testing.assert_close(adjoint, vjp, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("function", [fft_conv, fft_conv_adj])
def test_fft_convolution_image_and_kernel_gradients(dtype, function):
    image = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    kernel = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(function, (image, kernel), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (image, kernel), fast_mode=True)


def test_map2x():
    x1 = torch.tensor(1.0)
    y1 = torch.rand((4, 4))
    x2 = torch.tensor(0.0)
    y2 = torch.rand((4, 4))
    result = map2x(x1, y1, x2, y2)
    assert result.shape == y1.shape


def test_map2y():
    x1 = torch.tensor(1.0)
    y1 = torch.rand((4, 4))
    x2 = torch.tensor(0.0)
    y2 = torch.rand((4, 4))
    result = map2y(x1, y1, x2, y2)
    assert result.shape == y1.shape


def test_integrate1D():
    p_v = torch.rand((4,))
    pixelSize = torch.tensor([1.0, 1.0, 1.0, 1.0])
    result = integrate1D(p_v, pixelSize)
    assert result.shape == (5,)


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize("trajectory_batch", [1, 2])
def test_batched_trajectory_vjp_matches_exact_dft_gradient(shape, trajectory_batch):
    torch.manual_seed(97)
    modes = torch.randn(2, 3, *shape, dtype=torch.complex128)
    trajectory = torch.randn(
        trajectory_batch, len(shape), 9, dtype=torch.float64, requires_grad=True
    )
    probe = torch.randn(2, 3, 9, dtype=torch.complex128)
    coordinates = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(-(size // 2), (size - 1) // 2 + 1, dtype=torch.float64)
                for size in shape
            ],
            indexing="ij",
        )
    ).reshape(len(shape), -1)

    def exact_transform(values, points):
        phase = torch.exp(-1j * torch.einsum("dn,bdm->bnm", coordinates, points))
        return values.flatten(2) @ phase

    expected = torch.autograd.grad(
        exact_transform(modes, trajectory), trajectory, probe
    )[0]
    actual = nufft_trajectory_vjp(modes, trajectory, probe, exact_transform)
    if trajectory_batch == 1:
        actual = actual.sum(0, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
