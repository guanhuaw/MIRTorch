import numpy as np
import numpy.testing as npt
import pytest
import torch

from mirtorch.linear import Diag
from mirtorch.prox import (
    BoxConstraint,
    Conj,
    Const,
    L0Regularizer,
    L1Regularizer,
    L2Regularizer,
    SquaredL2Regularizer,
    Stack,
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
    return torch.randn(2, 2, dtype=torch.complex64, requires_grad=True)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    exp = 1.0 - random_lambda * 0.1 / max(
        np.linalg.norm(random_tensor.numpy()), random_lambda * 0.1
    )
    npt.assert_allclose(out, exp * random_tensor.numpy(), rtol=1e-3)


def test_l2_regularizer_preserves_device(device):
    value = torch.rand(8, device=device)
    out = L2Regularizer(0.5)(value, 0.1)
    assert out.device == value.device


def test_squaredl2_regularizer(random_tensor, random_lambda):
    prox = SquaredL2Regularizer(random_lambda)
    out = prox(random_tensor, 0.1)

    exp = random_tensor.numpy() / (1.0 + 2 * random_lambda * 0.1)
    npt.assert_allclose(out, exp, rtol=1e-3)


def test_squaredl2_regularizer_preserves_device(device):
    value = torch.rand(8, device=device)
    out = SquaredL2Regularizer(0.5)(value, 0.1)
    assert out.device == value.device


@pytest.mark.parametrize(
    "prox",
    [
        L0Regularizer(1.0),
        L1Regularizer(1.0),
        L2Regularizer(1.0),
        SquaredL2Regularizer(1.0),
    ],
)
@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="Apple Metal is unavailable",
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_regularizers_reject_negative_tensor_step_on_every_device(prox, device_name):
    value = torch.ones(3, device=device_name)
    alpha = torch.tensor(-1.0, device=device_name)

    with pytest.raises(ValueError, match="non-negative"):
        prox(value, alpha)


def test_boxconstraint(random_tensor, random_lambda):
    lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
    prox = BoxConstraint(random_lambda, lower, upper)
    out = prox(random_tensor, 0.1)

    exp = np.clip(random_tensor.numpy(), lower, upper)
    npt.assert_allclose(out, exp, rtol=1e-3)


def test_weighted_boxconstraint_projects_between_bounds(device):
    value = torch.tensor([0.0, 0.5, 1.0], device=device)
    weights = torch.tensor([1.0, 2.0, 1.0], device=device)
    prox = BoxConstraint(1.0, 0.25, 0.75, P=Diag(weights))
    expected = torch.tensor([0.25, 0.375, 0.75], device=device)
    assert torch.equal(prox(value, 1.0), expected)


def test_prox_to_moves_nested_operators(device):
    prox = L1Regularizer(0.5, P=Diag(torch.ones(3)))
    assert prox.to(device) is prox
    assert prox.P.P.device.type == device.type


def test_l0_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L0Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    out.abs().sum().backward()

    random_tensor_complex.requires_grad = False
    an = random_tensor_complex.numpy()
    threshold = np.sqrt(2 * random_lambda * 0.1)
    exp = torch.from_numpy(an * (np.abs(an) > threshold)).to(out)
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)


def test_l1_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L1Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    out.abs().sum().backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(
        random_tensor_complex.abs(), 0.1
    )
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)


def test_l2_regularizer_complex(random_tensor_complex, random_lambda):
    prox = L2Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    out.abs().sum().backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(
        random_tensor_complex.abs(), 0.1
    )
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)


def test_squaredl2_regularizer_complex(random_tensor_complex, random_lambda):
    prox = SquaredL2Regularizer(random_lambda)
    out = prox(random_tensor_complex, 0.1)
    out.abs().sum().backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(
        random_tensor_complex.abs(), 0.1
    )
    npt.assert_allclose(out.detach(), exp, rtol=1e-3)


def test_angle():
    a = torch.complex(torch.Tensor([1]), torch.Tensor([-1]))
    npt.assert_allclose(a.angle(), torch.atan2(a.imag, a.real))


def test_boxconstraint_complex(random_tensor_complex, random_lambda):
    lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
    prox = BoxConstraint(random_lambda, lower, upper)
    out = prox(random_tensor_complex, 0.1)
    out.abs().sum().backward()

    random_tensor_complex.requires_grad = False
    exp = torch.exp(1j * random_tensor_complex.angle()) * prox(
        random_tensor_complex.abs(), 0.1
    )
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


def test_l0_regularizer_minimizes_documented_objective():
    value = torch.tensor([0.75])
    alpha = 0.5
    regularizer = L0Regularizer(1.0)

    result = regularizer(value, alpha)

    def objective(candidate):
        return 0.5 * (candidate - value).square().sum() + alpha * (candidate != 0).sum()

    assert torch.equal(result, torch.zeros_like(value))
    assert objective(result) < objective(value)


def test_l0_diagonal_weights_follow_cardinality_semantics():
    value = torch.tensor([0.75, 0.75])
    weights = torch.tensor([0.0, 100.0])
    result = L0Regularizer(1.0, P=Diag(weights))(value, 0.5)
    assert torch.equal(result, torch.tensor([0.75, 0.0]))


def test_weighted_l1_uses_absolute_diagonal_weights():
    value = torch.tensor([2.0, 2.0])
    weights = torch.tensor([-0.5, 2.0])
    result = L1Regularizer(1.0, P=Diag(weights))(value, 0.5)
    assert torch.allclose(result, torch.tensor([1.75, 1.0]))


def test_weighted_squared_l2_matches_closed_form_minimizer():
    value = torch.tensor([1.0, -2.0], dtype=torch.float64)
    weights = torch.tensor([0.5, 2.0], dtype=torch.float64)
    alpha = 0.3
    result = SquaredL2Regularizer(0.7, P=Diag(weights))(value, alpha)
    expected = value / (1 + 2 * alpha * 0.7 * weights.square())
    assert torch.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_weighted_l2_satisfies_optimality_condition():
    value = torch.tensor([1.2, -0.7], dtype=torch.float64)
    weights = torch.tensor([0.5, 2.0], dtype=torch.float64)
    strength = 0.3
    result = L2Regularizer(1.0, P=Diag(weights))(value, strength)
    weighted_norm = torch.linalg.vector_norm(weights * result)
    expected = value / (1 + strength * weights.square() / weighted_norm)
    assert torch.allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_stack_uses_one_equal_section_per_prox():
    value = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    prox = Stack([L1Regularizer(1.0), Const()])
    expected = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert torch.equal(prox(value, [1.0, 1.0]), expected)


def test_conjugate_prox_rejects_zero_step():
    with pytest.raises(ValueError, match="positive"):
        Conj(Const())(torch.ones(3), 0.0)
