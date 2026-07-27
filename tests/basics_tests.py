import pytest
import torch
import torch.nn.functional as F

from mirtorch.linear import basics

_TEST_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _TEST_DEVICES.append("cuda")
elif torch.backends.mps.is_available():
    _TEST_DEVICES.append("mps")


@pytest.fixture(params=_TEST_DEVICES)
def device(request):
    """Run basic operators on CPU and the available accelerator."""
    return torch.device(request.param)


# Individual tests using the fixture for device
def test_diag():
    x = torch.randn(5, 5)
    P = torch.randn(5, 5)
    diag = basics.Diag(P)
    out = diag.apply(x)
    exp = P * x
    assert torch.allclose(out, exp, rtol=1e-3)


def test_conv1d_apply_simple(device):
    x = torch.randn(1, 16, 50, device=device)
    weight = torch.randn(33, 16, 3, device=device)
    conv = basics.Convolve1d(x.shape, weight)
    out = conv.apply(x)
    exp = F.conv1d(x, weight)
    assert torch.allclose(out, exp, rtol=1.5e-3)


def test_conv2d_apply_simple(device):
    x = torch.randn(1, 4, 5, 5, device=device)
    weight = torch.randn(8, 4, 3, 3, device=device)
    conv = basics.Convolve2d(x.shape, weight)
    out = conv.apply(x)
    exp = F.conv2d(x, weight)
    assert torch.allclose(out, exp, rtol=1e-3)


def test_conv3d_apply_simple(device):
    x = torch.randn(20, 16, 50, 10, 20, device=device)
    weight = torch.randn(33, 16, 3, 3, 3, device=device)
    conv = basics.Convolve3d(x.shape, weight)
    out = conv.apply(x)
    exp = F.conv3d(x, weight)
    assert torch.allclose(out, exp, rtol=1e-3)


@pytest.mark.parametrize(
    ("operator", "shape", "weight_shape"),
    [
        (basics.Convolve1d, (1, 2, 8), (3, 2, 3)),
        (basics.Convolve2d, (1, 2, 8, 7), (3, 2, 3, 2)),
        (basics.Convolve3d, (1, 2, 8, 7, 6), (3, 2, 3, 2, 2)),
    ],
)
def test_convolution_rejects_affine_bias(operator, shape, weight_shape, device):
    weight = torch.randn(*weight_shape, device=device)
    bias = torch.randn(weight_shape[0], device=device)
    with pytest.raises(ValueError, match="does not support bias"):
        operator(shape, weight, bias=bias)


def test_conv1d_adjoint_simple(device):
    x = torch.randn(20, 16, 50, device=device)
    weight = torch.randn(33, 16, 3, device=device)
    Ax = F.conv1d(x, weight)
    conv = basics.Convolve1d(x.shape, weight)
    out = conv.adjoint(Ax)
    exp = F.conv_transpose1d(Ax, weight)
    assert torch.allclose(out, exp, rtol=1e-3)


def test_conv2d_adjoint_simple(device):
    x = torch.randn(1, 4, 5, 5, device=device)
    weight = torch.randn(8, 4, 3, 3, device=device)
    Ax = F.conv2d(x, weight)
    conv = basics.Convolve2d(x.shape, weight)
    out = conv.adjoint(Ax)
    exp = F.conv_transpose2d(Ax, weight)
    assert torch.allclose(out, exp, rtol=1e-3)


def test_conv3d_adjoint_simple(device):
    x = torch.randn(20, 16, 50, 10, 20, device=device)
    weight = torch.randn(33, 16, 3, 3, 3, device=device)
    Ax = F.conv3d(x, weight)
    conv = basics.Convolve3d(x.shape, weight)
    out = conv.adjoint(Ax)
    exp = F.conv_transpose3d(Ax, weight)
    assert torch.allclose(out, exp, rtol=1e-3)


@pytest.mark.parametrize(
    ("operator", "shape", "weight_shape", "stride", "padding", "dilation"),
    [
        (basics.Convolve1d, (2, 2, 6), (3, 2, 3), 2, 1, 1),
        (
            basics.Convolve2d,
            (1, 2, 6, 7),
            (3, 2, 3, 2),
            (2, 2),
            (1, 0),
            (1, 1),
        ),
        (
            basics.Convolve3d,
            (1, 2, 6, 7, 8),
            (3, 2, 3, 2, 2),
            (2, 2, 3),
            (1, 0, 0),
            (1, 1, 1),
        ),
    ],
)
def test_complex_strided_convolution_is_a_true_adjoint(
    operator,
    shape,
    weight_shape,
    stride,
    padding,
    dilation,
    device,
):
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    weight = torch.randn(*weight_shape, dtype=torch.complex64, device=device)
    linear_map = operator(
        shape,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    y = torch.randn(*linear_map.size_out, dtype=torch.complex64, device=device)

    adjoint_output = linear_map.H(y)
    lhs = (linear_map(x).conj() * y).sum()
    rhs = (x.conj() * adjoint_output).sum()

    assert tuple(adjoint_output.shape) == shape
    assert torch.allclose(lhs, rhs, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("mode", "expected_length"),
    [("reflexive", 4), ("periodic", 5)],
)
@pytest.mark.parametrize("shape_type", [list, tuple])
def test_diff1d_shape_and_adjoint_identity(mode, expected_length, shape_type):
    shape = shape_type((2, 5))
    linear_map = basics.Diff1d(shape, dim=-1, mode=mode)
    x = torch.randn(2, 5, dtype=torch.float64)
    y = torch.randn(2, expected_length, dtype=torch.float64)

    lhs = torch.vdot(linear_map(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), linear_map.H(y).reshape(-1))

    assert linear_map.size_out == [2, expected_length]
    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_patch2d_forward(device):
    x = torch.randn(2, 3, 10, 10, device=device)
    kernel_size = 2
    stride = 1
    exp = torch.zeros(2, 3, 9, 9, 2, 2, device=device)
    for ix in range(9):
        for iy in range(9):
            exp[:, :, ix, iy, :, :] = x[:, :, ix : ix + 2, iy : iy + 2]
    P = basics.Patch2D(x.shape, kernel_size, stride)
    out = P * x
    assert torch.allclose(out, exp, rtol=1e-3)


def test_patch2d_adjoint(device):
    x = torch.randn(2, 3, 9, 9, 2, 2, device=device)
    kernel_size = 2
    stride = 1
    exp = torch.zeros(2, 3, 10, 10, device=device)
    for ix in range(9):
        for iy in range(9):
            exp[:, :, ix : ix + 2, iy : iy + 2] += x[:, :, ix, iy, :, :]
    P = basics.Patch2D(exp.shape, kernel_size, stride)
    out = P.H * x
    assert torch.allclose(out, exp, rtol=1e-3)


def test_patch3d_forward(device):
    x = torch.randn(2, 3, 10, 10, 10, device=device)
    kernel_size = 2
    stride = 1
    exp = torch.zeros(2, 3, 9, 9, 9, 2, 2, 2, device=device)
    for ix in range(9):
        for iy in range(9):
            for iz in range(9):
                exp[:, :, ix, iy, iz, :, :, :] = x[
                    :, :, ix : ix + 2, iy : iy + 2, iz : iz + 2
                ]
    P = basics.Patch3D(x.shape, kernel_size, stride)
    out = P * x
    assert torch.allclose(out, exp, rtol=1e-3)


def test_patch3d_adjoint(device):
    x = torch.randn(2, 3, 9, 9, 9, 2, 2, 2, device=device)
    kernel_size = 2
    stride = 1
    exp = torch.zeros(2, 3, 10, 10, 10, device=device)
    for ix in range(9):
        for iy in range(9):
            for iz in range(9):
                exp[:, :, ix : ix + 2, iy : iy + 2, iz : iz + 2] += x[
                    :, :, ix, iy, iz, :, :, :
                ]
    P = basics.Patch3D(exp.shape, kernel_size, stride)
    out = P.H * x
    # CUDA may accumulate overlapping patches in a different order.
    assert torch.allclose(out, exp, rtol=1e-3, atol=1e-5)
