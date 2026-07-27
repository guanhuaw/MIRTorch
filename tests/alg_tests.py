import importlib

import pytest
import torch

from mirtorch.alg import CG, FBPD, FISTA, POGM, power_iter
from mirtorch.linear import Diag, Identity, LinearMap
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


@pytest.mark.parametrize("backward_mode", ["implicit", "unrolled"])
def test_cg_tol_zero_handles_exact_convergence(backward_mode):
    rhs = torch.ones(4)
    result = CG(
        Identity(rhs.shape),
        max_iter=3,
        tol=0,
        backward_mode=backward_mode,
    ).run(torch.zeros_like(rhs), rhs)

    assert torch.equal(result, rhs)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Metal is unavailable",
)
@pytest.mark.parametrize("preconditioned", [False, True])
def test_complex_cg_and_power_iteration_run_on_mps(preconditioned):
    device = torch.device("mps")
    diagonal = torch.tensor([2.0, 4.0], device=device)
    operator = Diag(diagonal)
    rhs = torch.tensor([6.0 + 2.0j, 8.0 - 4.0j], device=device, requires_grad=True)
    preconditioner = Identity([2]) if preconditioned else None

    solution = CG(
        operator,
        max_iter=4,
        tol=1e-12,
        P=preconditioner,
    ).run(torch.zeros_like(rhs), rhs)

    expected = rhs.detach() / diagonal
    assert torch.allclose(solution.cpu(), expected.cpu(), atol=1e-5)
    solution.abs().square().sum().backward()
    assert rhs.grad is not None
    assert torch.isfinite(rhs.grad).all().item()

    _, singular_value = power_iter(
        operator,
        torch.ones(2, dtype=torch.complex64, device=device),
        max_iter=50,
        tol=1e-6,
        alert=False,
    )
    assert torch.allclose(singular_value.cpu(), torch.tensor(4.0), atol=1e-4)


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


def test_cg_caches_operator_application_once_per_iteration():
    class CountingDiagonal(LinearMap):
        def __init__(self):
            super().__init__([6], [6])
            self.diagonal = torch.arange(1, 7, dtype=torch.float64)
            self.calls = 0

        def _apply(self, value):
            self.calls += 1
            return self.diagonal * value

        def _apply_adjoint(self, value):
            return self.diagonal * value

    operator = CountingDiagonal()
    solver = CG(operator, max_iter=5, tol=0)
    solver.run(torch.zeros(6, dtype=torch.float64), torch.ones(6, dtype=torch.float64))
    assert operator.calls == 6  # initial residual plus one A*p per iteration


def test_cg_unrolled_mode_differentiates_truncated_initialization():
    initial = torch.tensor([2.0, -1.0], requires_grad=True)
    result = CG(
        Identity([2]),
        max_iter=0,
        backward_mode="unrolled",
    ).run(initial, torch.ones(2))
    result.sum().backward()
    assert torch.equal(initial.grad, torch.ones_like(initial))


def test_cg_unrolled_mode_differentiates_operator_parameters():
    diagonal = torch.tensor([2.0], requires_grad=True)
    result = CG(
        Diag(diagonal),
        max_iter=1,
        tol=0,
        backward_mode="unrolled",
    ).run(torch.zeros(1), torch.tensor([4.0]))
    result.sum().backward()
    assert torch.allclose(result, torch.tensor([2.0]))
    assert torch.allclose(diagonal.grad, torch.tensor([-1.0]))


def test_cg_rejects_non_positive_operator_breakdown():
    with pytest.raises(RuntimeError, match="positive definite"):
        CG(Diag(torch.zeros(2)), max_iter=2).run(
            torch.zeros(2),
            torch.ones(2),
        )


def test_cg_accepts_small_positive_denominator():
    diagonal = torch.tensor([1e-8], dtype=torch.float32)
    right_hand_side = torch.tensor([1e-4], dtype=torch.float32)
    result = CG(Diag(diagonal), max_iter=1, tol=0).run(
        torch.zeros_like(right_hand_side),
        right_hand_side,
    )
    torch.testing.assert_close(result, right_hand_side / diagonal)


def test_power_iteration_rejects_zero_initialization():
    with pytest.raises(ValueError, match="nonzero"):
        power_iter(Identity([3]), torch.zeros(3))


def test_power_iteration_returns_zero_for_zero_operator():
    vector, singular_value = power_iter(
        0 * Identity([3]),
        torch.ones(3),
        alert=False,
    )
    assert torch.isfinite(vector).all()
    assert singular_value == 0


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
