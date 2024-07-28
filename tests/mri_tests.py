import pytest
import torch
import numpy as np
from mirtorch.linear import FFTCn, Sense, NuSense, NuSenseGram

@pytest.fixture
def complex_tensor():
    return torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))

@pytest.fixture
def smaps():
    return torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))

@pytest.fixture
def masks():
    return torch.randint(0, 2, (2, 16, 16)).float()

@pytest.fixture
def traj():
    return torch.rand(2, 2, 1000) * 2 - 1

def test_fftcn_forward_backward(complex_tensor):
    fftcn = FFTCn([2, 1, 16, 16], [2, 1, 16, 16], dims=(2, 3))
    k_space = fftcn(complex_tensor)
    image = fftcn.H(k_space)
    assert torch.allclose(complex_tensor, image, atol=1e-6)

def test_fftcn_adjoint_property(complex_tensor):
    fftcn = FFTCn([2, 1, 16, 16], [2, 1, 16, 16], dims=(2, 3))
    k_space = torch.randn_like(complex_tensor)
    lhs = torch.sum(fftcn(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * fftcn.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)

def test_sense_forward_backward(complex_tensor, smaps, masks):
    sense = Sense(smaps, masks)
    k_space = sense(complex_tensor)
    image = sense.H(k_space)
    assert k_space.shape == (2, 4, 16, 16)
    assert image.shape == (2, 1, 16, 16)
    assert not torch.allclose(complex_tensor, image, atol=1e-6)  # Due to undersampling

def test_sense_adjoint_property(complex_tensor, smaps, masks):
    sense = Sense(smaps, masks)
    k_space = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    lhs = torch.sum(sense(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * sense.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)

def test_nusense_forward_backward(complex_tensor, smaps, traj):
    nusense = NuSense(smaps, traj)
    k_space = nusense(complex_tensor)
    image = nusense.H(k_space)
    assert k_space.shape == (2, 4, 1000)
    assert image.shape == (2, 1, 16, 16)
    assert not torch.allclose(complex_tensor, image, atol=1e-6)  # Due to non-Cartesian sampling

def test_nusense_adjoint_property(complex_tensor, smaps, traj):
    nusense = NuSense(smaps, traj)
    k_space = torch.randn(2, 4, 1000, dtype=torch.complex64)
    lhs = torch.sum(nusense(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * nusense.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)

def test_nusense_gram_forward(complex_tensor, smaps, traj):
    nusense_gram = NuSenseGram(smaps, traj)
    output = nusense_gram(complex_tensor)
    assert output.shape == complex_tensor.shape
    assert not torch.allclose(complex_tensor, output, atol=1e-6)

def test_nusense_gram_adjoint_property(complex_tensor, smaps, traj):
    nusense_gram = NuSenseGram(smaps, traj)
    y = torch.randn_like(complex_tensor)
    lhs = torch.sum(nusense_gram(complex_tensor).conj() * y)
    rhs = torch.sum(complex_tensor.conj() * nusense_gram.H(y))
    assert torch.allclose(lhs, rhs, atol=1e-6)
