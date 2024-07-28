import pytest
import torch
import torch.nn.functional as F
from mirtorch.linear import basics
from mirtorch.linear import wavelets

@pytest.fixture
def device():
    """Fixture to handle the allocation of tensors to devices."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Here we use a fixture to initialize data used across multiple tests
@pytest.fixture
def setup_conv_data(device):
    """Setup data for convolution tests."""
    x = torch.randn(20, 16, 50, device=device)
    weight = torch.randn(33, 16, 3, device=device)
    bias = torch.randn(33, device=device)
    return x, weight, bias

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

def test_conv1d_apply_hard(setup_conv_data):
    x, weight, bias = setup_conv_data
    conv = basics.Convolve1d(x.shape, weight, bias=bias, stride=2, padding=1, dilation=2)
    out = conv.apply(x)
    exp = F.conv1d(x, weight, bias, stride=2, padding=1, dilation=2)
    assert torch.allclose(out, exp, rtol=1e-3)

def test_conv2d_apply_hard(device):
    x = torch.randn(1, 4, 5, 5, device=device)
    weight = torch.randn(8, 4, 3, 3, device=device)
    bias = torch.randn(8, device=device)
    conv = basics.Convolve2d(x.shape, weight, bias=bias, stride=3, padding=2, dilation=2)
    out = conv.apply(x)
    exp = F.conv2d(x, weight, bias, stride=3, padding=2, dilation=2)
    assert torch.allclose(out, exp, rtol=1e-3)

def test_conv3d_apply_hard(device):
    x = torch.randn(20, 16, 50, 10, 20, device=device)
    weight = torch.randn(33, 16, 3, 3, 3, device=device)
    bias = torch.randn(33, device=device)
    conv = basics.Convolve3d(x.shape, weight, bias=bias, stride=3, padding=3, dilation=4)
    out = conv.apply(x)
    exp = F.conv3d(x, weight, bias, stride=3, padding=3, dilation=4)
    assert torch.allclose(out, exp, rtol=1e-3)

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

def test_patch2d_forward(device):
    x = torch.randn(2, 3, 10, 10, device=device)
    kernel_size = 2
    stride = 1
    exp = torch.zeros(2, 3, 9, 9, 2, 2, device=device)
    for ix in range(9):
        for iy in range(9):
            exp[:, :, ix, iy, :, :] = x[:, :, ix:ix+2, iy:iy+2]
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
            exp[:, :, ix:ix+2, iy:iy+2] += x[:, :, ix, iy, :, :]
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
                exp[:, :, ix, iy, iz, :, :, :] = x[:, :, ix:ix+2, iy:iy+2, iz:iz+2]
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
                exp[:, :, ix:ix+2, iy:iy+2, iz:iz+2] += x[:, :, ix, iy, iz, :, :, :]
    P = basics.Patch3D(exp.shape, kernel_size, stride)
    out = P.H * x
    assert torch.allclose(out, exp, rtol=1e-3)
