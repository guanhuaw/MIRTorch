import pytest
import torch
import numpy as np
import numpy.testing as npt
from mirtorch.prox import (
    Prox,
    L1Regularizer,
    L2Regularizer,
    SquaredL2Regularizer,
    BoxConstraint,
    L0Regularizer,
    Conj,
    Const,
)

# Fixtures for common test data
@pytest.fixture
def random_tensor():
    return torch.rand((5, 4, 8), dtype=torch.float)

@pytest.fixture
def random_lambda():
    return np.abs(np.random.random())

@pytest.fixture
def random_tensor_complex():
    return torch.rand(2, 2, dtype=torch.float, requires_grad=True)

# Test cases
def test_l1_regularizer(random_tensor, random_lambda):
    prox = L1Regularizer(random_lambda)
    out = prox(random_tensor, 0.1)

    lambd = 0.1 * random_lambda
    a = random_tensor.numpy().flatten()
    exp = np.zeros_like(a)

    for i in range(a.shape[0]):
        if a[i] > lambd:
            exp[i] = a[i] - lambd
        elif a[i] < -lambd:
            exp[i] = a[i] + lambd
        else:
            exp[i] = 0

    exp = exp.reshape(random_tensor.shape)
    npt.assert_allclose(out, exp, rtol=1e-3)

def test_l2_regularizer(random_tensor, random_lambda):
    prox = L2Regularizer(random_lambda)
    out = prox(random_tensor, 0.1)

    exp = 1.0 - random_lambda * 0.1 / max(np.linalg.norm(random_tensor.numpy()), random_lambda * 0.1)
    npt.assert_allclose(out, exp * random_tensor.numpy(), rtol=1e-3)

def test_squaredl2_regularizer(random_tensor, random_lambda):
    prox = SquaredL2Regularizer(random_lambda)
    out = prox(random_tensor, 0.1)

    exp = random_tensor.numpy() / (1.0 + 2 * random_lambda * 0.1)
    npt.assert_allclose(out, exp, rtol=1e-3)

def test_boxconstraint(random_tensor, random_lambda):
    lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
    prox = BoxConstraint(random_lambda, lower, upper)
    out = prox(random_tensor, 0.1)

    exp = np.clip(random_tensor.numpy(), lower, upper)
    npt.assert_allclose(out, exp, rtol=1e-3)

def test_l0_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L0Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    torch.sum(out).backward()

    random_tensor_complex.requires_grad = False
    an = random_tensor_complex.numpy()
    exp = torch.from_numpy(an * (np.abs(an) > (random_lambda * 0.1))).to(out)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)

def test_l1_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L1Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    torch.sum(out).backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(random_tensor_complex.abs(), 0.1)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)

def test_l2_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L2Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    torch.sum(out).backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(random_tensor_complex.abs(), 0.1)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)

def test_squaredl2_regularizer_complex(random_tensor_complex, random_lambda):
    prox = SquaredL2Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    torch.sum(out).backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(random_tensor_complex.abs(), 0.1)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)

def test_angle():
    a = torch.complex(torch.Tensor([1]), torch.Tensor([-1]))
    npt.assert_allclose(a.angle(), torch.atan2(a.imag, a.real))

def test_boxconstraint_complex(random_tensor_complex, random_lambda):
    lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
    prox = BoxConstraint(random_lambda, lower, upper)
    out = prox(random_tensor_complex, 0.1)
    torch.sum(out).backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(random_tensor_complex.abs(), 0.1)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)

def test_complex_edge_cases():
    a = torch.complex(torch.Tensor([1]), torch.Tensor([0]))
    npt.assert_allclose(a.angle(), torch.atan2(a.imag, a.real))

def test_complex_edge_cases2():
    a = torch.complex(torch.Tensor([0]), torch.Tensor([1]))
    npt.assert_allclose(a.angle(), torch.atan2(a.imag, a.real))

def test_complex_edge_cases3():
    a = torch.complex(torch.Tensor([0]), torch.Tensor([0]))
    npt.assert_allclose(a.angle(), torch.atan2(a.imag, a.real))
