import importlib

import pytest
import torch

from mirtorch.alg import CG, FBPD, FISTA, POGM, power_iter
from mirtorch.linear import Diag, Identity
from mirtorch.prox import Const


def test_cg_solves_diagonal_system_and_backpropagates():
    diagonal = torch.tensor([2.0, 4.0])
    operator = Diag(diagonal)
    rhs = torch.tensor([6.0, 8.0], requires_grad=True)

    solution = CG(operator, max_iter=4, tol=1e-12).run(
        x0=torch.zeros_like(rhs),
        b=rhs,
    )
    assert torch.allclose(solution, torch.tensor([3.0, 2.0]))

    solution.sum().backward()
    assert torch.allclose(rhs.grad, diagonal.reciprocal())


def test_cg_with_evaluation_history_backpropagates():
    rhs = torch.ones(2, requires_grad=True)
    solution, saved = CG(
        Identity([2]),
        max_iter=2,
        tol=1e-12,
        eval_func=torch.linalg.vector_norm,
    ).run(torch.zeros_like(rhs), rhs)

    solution.sum().backward()
    assert len(saved) == 1
    assert torch.allclose(rhs.grad, torch.ones_like(rhs))


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_gradient_solvers_reach_quadratic_minimum(solver):
    target = torch.tensor([1.0, -2.0, 3.0])
    initial = torch.zeros_like(target)
    algorithm = solver(
        f_grad=lambda value: value - target,
        f_L=1.0,
        g_prox=Const(),
        max_iter=3,
        eval_func=lambda value: torch.linalg.vector_norm(value - target),
    )

    result, saved = algorithm.run(initial)
    assert torch.linalg.vector_norm(result - target) < torch.linalg.vector_norm(
        initial - target
    )
    assert len(saved) == 3


def test_fbpd_reaches_quadratic_minimum():
    target = torch.tensor([1.0, -2.0, 3.0])
    evaluations = []
    algorithm = FBPD(
        g_grad=lambda value: value - target,
        f_prox=Const(),
        h_prox=Const(),
        g_L=1.0,
        G_norm=1.0,
        G=Identity(target.shape),
        max_iter=15,
        eval_func=lambda value: evaluations.append(value.detach().clone()) or 0.0,
    )

    result, saved = algorithm.run(torch.zeros_like(target))
    assert torch.allclose(result, target, atol=1e-6)
    assert len(saved) == len(evaluations) == 15


def test_power_iteration_finds_largest_singular_value():
    operator = Diag(torch.tensor([1.0, 3.0, 2.0]))
    _, singular_value = power_iter(
        operator,
        torch.ones(3),
        max_iter=100,
        tol=1e-7,
        alert=False,
    )
    assert torch.allclose(singular_value, torch.tensor(3.0), atol=1e-5)


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_gradient_solvers_validate_lipschitz_constant(solver):
    with pytest.raises(ValueError, match="f_L"):
        solver(lambda value: value, 0.0, Const())


def test_cg_validates_initial_shape():
    solver = CG(Identity([3]))
    with pytest.raises(ValueError, match="x0"):
        solver.run(torch.zeros(2), torch.ones(3))


def test_power_iteration_rejects_zero_initialization():
    with pytest.raises(ValueError, match="nonzero"):
        power_iter(Identity([3]), torch.zeros(3))


@pytest.mark.parametrize(
    ("solver", "module_name"),
    [
        (FISTA, "mirtorch.alg.fista"),
        (POGM, "mirtorch.alg.pogm"),
    ],
)
def test_proximal_gradient_solvers_compile_once_by_default(
    monkeypatch,
    solver,
    module_name,
):
    module = importlib.import_module(module_name)
    compiled = []
    monkeypatch.setattr(
        module,
        "should_compile",
        lambda enabled, _tensor: enabled,
    )
    monkeypatch.setattr(
        module,
        "compile_callable",
        lambda function: compiled.append(function) or function,
    )
    target = torch.tensor([1.0, -2.0, 3.0])
    algorithm = solver(
        f_grad=lambda value: value - target,
        f_L=1.0,
        g_prox=Const(),
        max_iter=3,
    )

    first = algorithm.run(torch.zeros_like(target))
    second = algorithm.run(torch.zeros_like(target))

    assert len(compiled) == 1
    assert torch.allclose(first, second)

    eager = solver(
        f_grad=lambda value: value - target,
        f_L=1.0,
        g_prox=Const(),
        max_iter=3,
        compile=False,
    )
    eager.run(torch.zeros_like(target))
    assert len(compiled) == 1
