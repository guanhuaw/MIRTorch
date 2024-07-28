import pytest
import torch
import torchvision.transforms.functional as F

from mirtorch.linear.util import (
    finitediff,
    finitediff_adj,
    fftshift,
    ifftshift,
    dim_conv,
    imrotate,
    fft2,
    ifft2,
    pad2sizezero,
    fft_conv,
    fft_conv_adj,
    map2x,
    map2y,
    integrate1D
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
    result = finitediff(tensor_2d, dim=1, mode='reflexive')
    assert result.shape == (4, 3)

def test_finitediff_periodic(tensor_2d):
    result = finitediff(tensor_2d, dim=1, mode="periodic")
    assert result.shape == (4, 4)

def test_finitediff_adj(tensor_2d):
    result = finitediff_adj(tensor_2d, dim=1, mode='reflexive')
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
    angle = 45
    result = imrotate(tensor_4d, angle)
    assert result.shape == tensor_4d.shape

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
