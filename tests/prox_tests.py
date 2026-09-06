import numpy as np
import numpy.testing as npt
import pytest
import torch

from mirtorch.linear import Diag, FFTCn
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
from mirtorch.util import l2_norm, squared_l2_norm


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


@pytest.mark.parametrize("regularizer", [L1Regularizer, SquaredL2Regularizer])
def test_unweighted_elementwise_prox_does_not_materialize_identity_weights(
    regularizer, monkeypatch
):
    prox = regularizer(0.3)

    def unexpected_identity_weights(_value):
        pytest.fail("An unweighted prox must use scalar strength without dense weights")

    monkeypatch.setattr(prox, "_diagonal_weights", unexpected_identity_weights)
    value = torch.ones(16, 32, dtype=torch.complex128)
    expected = value * 0.85 if regularizer is L1Regularizer else value / 1.3
    torch.testing.assert_close(prox(value, 0.5), expected)


@pytest.mark.parametrize("regularizer", [L1Regularizer, SquaredL2Regularizer])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("step", [0.0, 0.3])
def test_scalar_prox_matches_explicit_identity_values_and_gradients(
    regularizer, dtype, step
):
    data = torch.tensor([0.0, 1.2, -0.7], dtype=dtype)
    probe = torch.tensor([0.5, -0.2, 0.7], dtype=dtype)
    if dtype.is_complex:
        data += torch.tensor([0.0, 0.3j, 0.8j], dtype=dtype)
        probe += torch.tensor([0.2j, -0.7j, 0.4j], dtype=dtype)
    results = []
    for weights in (None, Diag(torch.ones(3, dtype=torch.float64))):
        value = data.clone().requires_grad_()
        alpha = torch.tensor(step, dtype=torch.float64, requires_grad=True)
        result = regularizer(0.7, P=weights)(value, alpha)
        gradients = torch.autograd.grad(result, (value, alpha), grad_outputs=probe)
        results.append((result, *gradients))

    for actual, expected in zip(*results, strict=True):
        assert torch.equal(actual, expected)


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


@pytest.mark.parametrize(
    "prox",
    [
        Const(),
        L0Regularizer(0.7),
        L1Regularizer(0.7),
        L2Regularizer(0.7),
        SquaredL2Regularizer(0.7),
        BoxConstraint(1, 0, 2),
        Conj(SquaredL2Regularizer(0.7)),
    ],
)
def test_complex_prox_gradients_at_smooth_zero_components(prox):
    value = torch.tensor(
        [0, 1.2 + 0.3j, -0.7 + 0.8j], dtype=torch.complex128, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda v: prox(v, 0.3), (value,))


@pytest.mark.parametrize(
    "regularizer", [L0Regularizer, L1Regularizer, L2Regularizer, SquaredL2Regularizer]
)
@pytest.mark.parametrize("zero_parameter", ["lambda", "alpha", "tensor_alpha"])
def test_complex_zero_strength_is_identity_in_values_and_gradients(
    regularizer, zero_parameter
):
    value = torch.tensor(
        [0, 1e-20 + 2e-20j, 1.2 - 0.7j], dtype=torch.complex128, requires_grad=True
    )
    strength = 0 if zero_parameter == "lambda" else 0.7
    alpha = 0.3 if zero_parameter == "lambda" else 0.0
    if zero_parameter == "tensor_alpha":
        alpha = torch.tensor(alpha, dtype=torch.float64)
    result = regularizer(strength)(value, alpha)
    torch.testing.assert_close(result, value, atol=0, rtol=0)
    probe = torch.tensor([1 + 2j, -0.3 + 0.7j, 0.9 - 0.2j], dtype=value.dtype)
    gradient = torch.autograd.grad(result, value, grad_outputs=probe)[0]
    torch.testing.assert_close(gradient, probe, atol=0, rtol=0)


@pytest.mark.parametrize("prox", [Const(), SquaredL2Regularizer(0.7)])
def test_complex_linear_prox_preserves_tiny_values_and_zero_derivatives(prox):
    value = torch.tensor(
        [0, 1e-20 + 2e-20j], dtype=torch.complex128, requires_grad=True
    )
    scale = 1 if isinstance(prox, Const) else 1 / (1 + 2 * 0.3 * 0.7)
    result = prox(value, 0.3)
    torch.testing.assert_close(result, value * scale, atol=0, rtol=1e-15)
    gradient = torch.autograd.grad(result, value, grad_outputs=torch.ones_like(value))[
        0
    ]
    torch.testing.assert_close(gradient, torch.full_like(value, scale))


def test_complex_soft_threshold_preserves_sub_epsilon_values():
    value = torch.tensor([1e-20 + 2e-20j], dtype=torch.complex128, requires_grad=True)
    threshold = 1e-22
    expected = value * (1 - threshold / value.abs())
    actual = L1Regularizer(1)(value, threshold)
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-15)
    actual_gradient = torch.autograd.grad(actual.real.sum(), value, retain_graph=True)[
        0
    ]
    expected_gradient = torch.autograd.grad(expected.real.sum(), value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=1e-15)


@pytest.mark.parametrize(
    "regularizer", [L0Regularizer, L1Regularizer, L2Regularizer, SquaredL2Regularizer]
)
def test_complex_diagonal_regularizers_preserve_unpenalized_zero_gradient(regularizer):
    value = torch.tensor(
        [0, 1.2 + 0.3j, -0.7 + 0.8j], dtype=torch.complex128, requires_grad=True
    )
    weights = torch.tensor([0, 0.5j, -2], dtype=torch.complex128)
    prox = regularizer(0.7, P=Diag(weights))
    assert torch.autograd.gradcheck(lambda v: prox(v, 0.3), (value,))
    result = prox(value, 0.3)
    probe = torch.tensor(1 + 2j, dtype=value.dtype)
    gradient = torch.autograd.grad(result[0], value, grad_outputs=probe)[0]
    torch.testing.assert_close(gradient[0], probe)


def test_weighted_complex_l2_satisfies_optimality_and_gradcheck():
    value = torch.tensor(
        [0, 1.2 + 0.3j, -0.7 + 0.8j], dtype=torch.complex128, requires_grad=True
    )
    weights = torch.tensor([0.5, 2, 0], dtype=torch.float64)
    prox = L2Regularizer(1, P=Diag(weights))
    result = prox(value, 0.3)
    weighted_norm = torch.linalg.vector_norm(weights * result)
    residual = result - value + 0.3 * weights.square() * result / weighted_norm
    torch.testing.assert_close(residual, torch.zeros_like(value), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda v: prox(v, 0.3), (value,))


@pytest.mark.parametrize("weighted", [False, True])
def test_complex_l2_zero_solution_has_finite_zero_gradient(weighted):
    value = torch.zeros(3, dtype=torch.complex128, requires_grad=True)
    weights = Diag(torch.tensor([0.5, 1, 2], dtype=torch.float64)) if weighted else None
    prox = L2Regularizer(1, P=weights)
    assert torch.autograd.gradcheck(lambda v: prox(v, 0.3), (value,))


def test_complex_prox_preserves_unitary_transform_gradient_at_zero():
    transform = FFTCn([4], [4])
    value = torch.zeros(4, dtype=torch.complex128, requires_grad=True)
    prox = SquaredL2Regularizer(0.7, T=transform)
    assert torch.autograd.gradcheck(lambda v: prox(v, 0.3), (value,))
    probe = torch.tensor([1 + 2j, 2 - 1j, 0.7j, -0.3], dtype=value.dtype)
    gradient = torch.autograd.grad(prox(value, 0.3), value, grad_outputs=probe)[0]
    torch.testing.assert_close(gradient, probe / (1 + 2 * 0.3 * 0.7))


def test_complex_weighted_box_projection_retains_phase_and_origin_gradient():
    value = torch.tensor([0, 2j, -2, 0], dtype=torch.complex128, requires_grad=True)
    weights = torch.tensor([1, 2, 1, 0], dtype=torch.float64)
    prox = BoxConstraint(1, 0, 1, P=Diag(weights))
    expected = torch.tensor([0, 0.5j, -1, 0], dtype=value.dtype)
    torch.testing.assert_close(prox(value, 1), expected)
    assert torch.autograd.gradcheck(lambda v: prox(v, 1), (value,))


def test_complex_box_zero_has_a_feasible_projection_when_lower_bound_is_positive():
    value = torch.zeros(1, dtype=torch.complex128)
    result = BoxConstraint(1, 0.5, 1)(value, 1)
    torch.testing.assert_close(result, torch.tensor([0.5 + 0j], dtype=value.dtype))


@pytest.mark.parametrize(
    "regularizer",
    [Const, L0Regularizer, L1Regularizer, L2Regularizer, SquaredL2Regularizer],
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
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_complex_prox_values_and_gradients_match_cpu_reference(
    regularizer, device_name
):
    reference = torch.tensor(
        [0, 1.2 + 0.3j, -0.7 + 0.8j], dtype=torch.complex128, requires_grad=True
    )
    value = (
        reference.detach()
        .to(device=device_name, dtype=torch.complex64)
        .requires_grad_()
    )
    prox = regularizer(0.7)
    expected = prox(reference, 0.3)
    actual = prox(value, 0.3)
    expected_gradient = torch.autograd.grad(
        expected, reference, torch.ones_like(expected)
    )[0]
    actual_gradient = torch.autograd.grad(actual, value, torch.ones_like(actual))[0]
    torch.testing.assert_close(
        actual.cpu().to(expected.dtype), expected, rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(
        actual_gradient.cpu().to(expected.dtype),
        expected_gradient,
        rtol=2e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "regularizer", [L1Regularizer, L2Regularizer, SquaredL2Regularizer]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_complex_prox_zero_tensor_step_retains_right_step_derivative(
    regularizer, weighted
):
    value = torch.tensor([0, 1 + 2j, 3 - 4j], dtype=torch.complex128)
    weights = (
        torch.tensor([0.5, 2, 0], dtype=torch.float64)
        if weighted
        else torch.ones(3, dtype=torch.float64)
    )
    prox = regularizer(0.7, P=Diag(weights) if weighted else None)
    alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    result = prox(value, alpha)
    if regularizer is L1Regularizer:
        safe_magnitude = torch.where(
            value.abs() > 0, value.abs(), torch.ones_like(value.real)
        )
        derivative = -0.7 * weights * value / safe_magnitude
    elif regularizer is L2Regularizer:
        derivative = (
            -0.7 * weights.square() * value / torch.linalg.vector_norm(weights * value)
        )
    else:
        derivative = -1.4 * weights.square() * value
    probe = torch.tensor([1 + 0.3j, -0.5 + 1j, 0.8 - 0.2j], dtype=value.dtype)
    actual = torch.autograd.grad(result, alpha, grad_outputs=probe)[0]
    expected = (probe.conj() * derivative).sum().real
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    step = 1e-7
    finite_difference = (probe.conj() * (prox(value, step) - result) / step).sum().real
    torch.testing.assert_close(actual, finite_difference, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("zero", [False, True])
def test_squared_norm_has_exact_hessian_including_zero_components(dtype, zero):
    value = torch.tensor([0, 1.2, -0.7], dtype=dtype)
    if value.is_complex():
        value = value + torch.tensor([0, 0.3j, 0.8j], dtype=dtype)
    value = (torch.zeros_like(value) if zero else value).requires_grad_()
    assert torch.autograd.gradcheck(squared_l2_norm, (value,))
    assert torch.autograd.gradgradcheck(squared_l2_norm, (value,))
    gradient = torch.autograd.grad(squared_l2_norm(value), value, create_graph=True)[0]
    probe = torch.ones_like(value) * (1 + 2j if value.is_complex() else 2)
    hessian_product = torch.autograd.grad(gradient, value, probe)[0]
    torch.testing.assert_close(hessian_product, 2 * probe)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_norm_has_smooth_nonzero_hessian_and_finite_origin_convention(dtype):
    value = torch.tensor([0, 1.2, -0.7], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradgradcheck(l2_norm, (value,))
    origin = torch.zeros_like(value, requires_grad=True)
    gradient = torch.autograd.grad(l2_norm(origin), origin, create_graph=True)[0]
    second = torch.autograd.grad(gradient, origin, torch.ones_like(origin))[0]
    torch.testing.assert_close(gradient, torch.zeros_like(origin))
    torch.testing.assert_close(second, torch.zeros_like(origin))


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("step", [0.0, 0.3])
def test_l2_prox_first_and_second_input_derivatives(dtype, weighted, zero, step):
    value = torch.tensor([0, 1.2, -0.7], dtype=dtype)
    if value.is_complex():
        value = value + torch.tensor([0, 0.3j, 0.8j], dtype=dtype)
    value = (torch.zeros_like(value) if zero else value).requires_grad_()
    weights = Diag(torch.tensor([0.5, 2, 1], dtype=torch.float64)) if weighted else None
    prox = L2Regularizer(0.7, P=weights)
    function = lambda value: prox(value, torch.tensor(step, dtype=torch.float64))
    assert torch.autograd.gradcheck(function, (value,))
    assert torch.autograd.gradgradcheck(function, (value,))


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("weight_dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("regularizer", [L2Regularizer, SquaredL2Regularizer])
def test_weighted_l2_joint_second_derivatives(dtype, weight_dtype, regularizer):
    value = torch.tensor([0, 1.2, -0.7, 0.9], dtype=dtype)
    weights = torch.tensor([0.5, 2, 0.7, 0], dtype=weight_dtype)
    if dtype.is_complex:
        value = value + torch.tensor([0, 0.3j, 0.8j, -0.4j], dtype=dtype)
    if weight_dtype.is_complex:
        weights = weights + torch.tensor([0.2j, -0.4j, 0.5j, 0], dtype=weight_dtype)
    value.requires_grad_()
    weights.requires_grad_()
    alpha = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    def function(value, alpha, weights):
        return regularizer(0.7, P=Diag(weights))(value, alpha)

    assert torch.autograd.gradcheck(function, (value, alpha, weights))
    assert torch.autograd.gradgradcheck(function, (value, alpha, weights))


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_weighted_l2_zero_step_has_correct_joint_second_derivatives(dtype):
    value = torch.tensor([0, 1.2, -0.7, 0.9], dtype=dtype)
    weights = torch.tensor([0.5, 2, 0.7, 0], dtype=dtype)
    if dtype.is_complex:
        value = value + torch.tensor([0, 0.3j, 0.8j, -0.4j], dtype=dtype)
        weights = weights + torch.tensor([0.2j, -0.4j, 0.5j, 0], dtype=dtype)
    value.requires_grad_()
    weights.requires_grad_()
    alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    probe = torch.tensor([0.2, -0.7, 1.1, 0.4], dtype=dtype)

    def actual(value, alpha, weights):
        return (
            probe.conj() * L2Regularizer(0.7, P=Diag(weights))(value, alpha)
        ).real.sum()

    def second_order_expansion(value, alpha, weights):
        # Differentiate x-v+s*q(x)=0 twice at s=0, q(x)=D*x/||w*x||.
        diagonal = weights.real.square()
        value_squared = value.real.square()
        if dtype.is_complex:
            diagonal = diagonal + weights.imag.square()
            value_squared = value_squared + value.imag.square()
        norm_squared = (diagonal * value_squared).sum()
        curvature = diagonal.square() * value / norm_squared - (
            diagonal
            * value
            * (diagonal.square() * value_squared).sum()
            / norm_squared.square()
        )
        strength = 0.7 * alpha
        result = value - strength * diagonal * value / norm_squared.sqrt()
        result = result + strength.square() * curvature
        return (probe.conj() * result).real.sum()

    parameters = (value, alpha, weights)
    actual_gradients = torch.autograd.grad(
        actual(*parameters), parameters, create_graph=True
    )
    expected_gradients = torch.autograd.grad(
        second_order_expansion(*parameters), parameters, create_graph=True
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
        )
        for imaginary in (False, True) if actual_gradient.is_complex() else (False,):
            direction = torch.ones_like(actual_gradient) * (1j if imaginary else 1)
            actual_second = torch.autograd.grad(
                actual_gradient, parameters, direction, retain_graph=True
            )
            expected_second = torch.autograd.grad(
                expected_gradient, parameters, direction, retain_graph=True
            )
            for actual_part, expected_part in zip(actual_second, expected_second):
                torch.testing.assert_close(
                    actual_part, expected_part, atol=1e-11, rtol=1e-11
                )
