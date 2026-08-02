import importlib

import pytest
import torch

from mirtorch.alg import CG, FBPD, FISTA, POGM, SolverResult, power_iter
from mirtorch.linear import Diag, Identity, LinearMap
from mirtorch.prox import Const, L1Regularizer


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
    class CountingIdentity(LinearMap):
        def __init__(self):
            super().__init__([4], [4])
            self.calls = 0

        def _apply(self, value):
            self.calls += 1
            return value

        def _apply_adjoint(self, value):
            return value

    operator = CountingIdentity()
    rhs = torch.ones(4, requires_grad=True)
    result = CG(
        operator,
        max_iter=3,
        tol=0,
        backward_mode=backward_mode,
    ).run(torch.zeros_like(rhs), rhs)

    assert torch.equal(result, rhs)
    assert operator.calls == 4  # A*x0 followed by exactly max_iter A*p calls.
    result.sum().backward()
    torch.testing.assert_close(rhs.grad, torch.ones_like(rhs))


def test_cg_fixed_iterations_do_not_check_convergence_each_iteration(monkeypatch):
    original_item = torch.Tensor.item
    calls = 0

    def counted_item(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_item(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    CG(Identity([3]), max_iter=8, tol=0).run(torch.zeros(3), torch.ones(3))

    # The only synchronization is the final positive-definiteness check.
    assert calls == 1


def test_cg_relative_tolerance_is_scale_invariant():
    diagonal = torch.tensor([1.0, 4.0])

    def solve(scale):
        return CG(
            Diag(diagonal),
            max_iter=5,
            rtol=0.7,
            atol=0,
            eval_func=lambda residual: residual.clone(),
        ).run(torch.zeros(2), scale * torch.ones(2))

    small, small_history = solve(1.0)
    large, large_history = solve(1.0e6)

    assert len(small_history) == len(large_history) == 1
    torch.testing.assert_close(large, 1.0e6 * small)


def test_cg_absolute_and_legacy_squared_tolerances_are_distinct():
    initial = torch.zeros(2)
    right_hand_side = torch.ones(2)

    # ||b|| = sqrt(2), while historical tol compares against ||b||^2 = 2.
    absolute_result = CG(Identity([2]), max_iter=2, atol=1.5, rtol=0).run(
        initial,
        right_hand_side,
    )
    legacy_result = CG(Identity([2]), max_iter=2, tol=1.5).run(
        initial,
        right_hand_side,
    )

    torch.testing.assert_close(absolute_result, initial)
    torch.testing.assert_close(legacy_result, right_hand_side)


def test_cg_returns_optional_diagnostics_without_changing_default_return():
    solver = CG(
        Diag(torch.tensor([1.0, 4.0])),
        max_iter=5,
        rtol=0.7,
        atol=0,
        eval_func=lambda residual: residual.clone(),
    )
    default_solution, default_history = solver.run(torch.zeros(2), torch.ones(2))
    result = solver.run(torch.zeros(2), torch.ones(2), return_info=True)

    assert isinstance(result, SolverResult)
    torch.testing.assert_close(result.solution, default_solution)
    assert result.iterations == 1
    assert result.converged
    assert len(result.history) == len(default_history) == 1
    torch.testing.assert_close(result.residual_norm, torch.tensor(0.6 * 2**0.5))


@pytest.mark.parametrize(("name", "value"), [("rtol", -1.0), ("atol", -1.0)])
def test_cg_rejects_negative_modern_tolerances(name, value):
    with pytest.raises(ValueError, match=name):
        CG(Identity([1]), **{name: value})


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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Metal is unavailable",
)
def test_complex_mature_solver_paths_match_cpu_on_mps():
    def solve(solver_type, device):
        target = torch.tensor(
            [1 + 0.5j, -2 + 0.25j],
            dtype=torch.complex64,
            device=device,
            requires_grad=True,
        )
        if solver_type is FBPD:
            result = solver_type(
                lambda value: value - target,
                Const(),
                Const(),
                1.0,
                G=Identity([2]),
                G_norm_squared=1.0,
                max_iter=30,
                rtol=1e-5,
                atol=1e-7,
            ).run(torch.zeros_like(target), return_info=True)
        else:
            result = solver_type(
                lambda value: value - target,
                2.0,
                Const(),
                max_iter=12,
                restart=True,
                rtol=1e-5,
                atol=1e-7,
                compile=False,
            ).run(torch.zeros_like(target), return_info=True)
        (gradient,) = torch.autograd.grad(
            result.solution.abs().square().sum(),
            target,
        )
        return (
            result.solution.detach().cpu(),
            gradient.detach().cpu(),
            result.iterations,
            result.converged,
        )

    for solver_type in (FISTA, POGM, FBPD):
        cpu = solve(solver_type, "cpu")
        mps = solve(solver_type, "mps")
        torch.testing.assert_close(mps[0], cpu[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(mps[1], cpu[1], rtol=2e-5, atol=2e-5)
        assert mps[2:] == cpu[2:]


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


@pytest.mark.parametrize(
    ("solver", "iterations", "expected"),
    [
        (FISTA, 3, 0.08978080935933488),
        # One iteration exercises POGM's special fixed-horizon final coefficient.
        (POGM, 1, 0.25),
    ],
)
def test_default_proximal_recurrences_match_legacy_results(
    solver,
    iterations,
    expected,
):
    result = solver(
        lambda value: value,
        2.0,
        Const(),
        max_iter=iterations,
        compile=False,
    ).run(torch.ones(1, dtype=torch.float64))

    torch.testing.assert_close(result, torch.tensor([expected], dtype=torch.float64))


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_solver_can_stop_and_return_diagnostics(solver):
    initial = torch.tensor([1 + 2j, -0.5j], dtype=torch.complex128)
    algorithm = solver(
        f_grad=torch.zeros_like,
        f_L=1.0,
        g_prox=Const(),
        max_iter=8,
        eval_func=lambda value: value.abs().square().sum(),
        atol=1e-14,
        compile=False,
    )

    result = algorithm.run(initial, return_info=True)

    assert isinstance(result, SolverResult)
    torch.testing.assert_close(result.solution, initial)
    assert result.iterations == 1
    assert result.converged
    assert result.residual_norm == 0
    assert len(result.history) == 1


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_solver_stopping_preserves_legacy_return(solver):
    initial = torch.ones(2)
    result, history = solver(
        f_grad=torch.zeros_like,
        f_L=1.0,
        g_prox=Const(),
        max_iter=4,
        eval_func=lambda value: value.sum(),
        atol=1e-12,
        compile=False,
    ).run(initial)

    torch.testing.assert_close(result, initial)
    assert len(history) == 1


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_adaptive_restart_accelerates_strongly_convex_quadratic(solver):
    diagonal = torch.tensor([1.0, 10.0], dtype=torch.float64)
    initial = torch.full((2,), 10.0, dtype=torch.float64)

    def gradient(value):
        return diagonal * value

    plain = solver(
        gradient,
        10.0,
        Const(),
        max_iter=40,
        compile=False,
    ).run(initial)
    restarted = solver(
        gradient,
        10.0,
        Const(),
        max_iter=40,
        restart=True,
        compile=False,
    ).run(initial)

    assert torch.linalg.vector_norm(restarted) < 0.01 * torch.linalg.vector_norm(plain)


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_restarted_proximal_solver_matches_nonsmooth_reference(solver):
    diagonal = torch.tensor([0.5, 2.0, 5.0], dtype=torch.float64)
    target = torch.tensor([2.0, -0.7, 0.15], dtype=torch.float64)
    regularization = 0.2
    expected = torch.sign(target) * torch.clamp(
        target.abs() - regularization / diagonal,
        min=0,
    )

    result = solver(
        lambda value: diagonal * (value - target),
        f_L=5.0,
        g_prox=L1Regularizer(regularization),
        max_iter=100,
        restart=True,
        rtol=1e-9,
        atol=1e-14,
        compile=False,
    ).run(torch.zeros_like(target), return_info=True)

    assert result.converged
    torch.testing.assert_close(result.solution, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_restarted_proximal_solver_remains_differentiable(solver):
    def solve(target):
        return solver(
            lambda value: value - target,
            f_L=2.0,
            g_prox=Const(),
            max_iter=10,
            restart=True,
            compile=False,
        ).run(torch.zeros_like(target))

    target = torch.tensor(
        [1.2 + 0.3j, -0.7 + 0.2j],
        dtype=torch.complex128,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(solve, (target,))


@pytest.mark.parametrize("solver", [FISTA, POGM])
@pytest.mark.parametrize(
    ("keyword", "value"),
    [("rtol", -1.0), ("rtol", float("nan")), ("atol", -1.0)],
)
def test_proximal_solver_validates_stopping_tolerances(solver, keyword, value):
    with pytest.raises(ValueError, match=keyword):
        solver(torch.zeros_like, 1.0, Const(), **{keyword: value})


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_solver_validates_restart_type(solver):
    with pytest.raises(TypeError, match="restart"):
        solver(torch.zeros_like, 1.0, Const(), restart="gradient")


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_zero_iterations_have_empty_proximal_solver_diagnostics(solver):
    initial = torch.ones(2)
    result = solver(
        torch.zeros_like,
        1.0,
        Const(),
        max_iter=0,
    ).run(initial, return_info=True)

    torch.testing.assert_close(result.solution, initial)
    assert result.iterations == 0
    assert not result.converged
    assert result.residual_norm is None
    assert result.history == []


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


def test_fbpd_default_recurrence_matches_legacy_result():
    result = FBPD(
        lambda value: value - 1,
        Const(),
        Const(),
        1.0,
        1.0,
        G=Identity([1]),
        max_iter=2,
    ).run(torch.zeros(1, dtype=torch.float64))

    torch.testing.assert_close(result, torch.tensor([8 / 9], dtype=torch.float64))


def test_fbpd_matches_lasso_solution_and_primal_dual_condition():
    target = torch.tensor([2.0, -0.7, 0.15], dtype=torch.float64)
    regularization = 0.2
    expected = torch.sign(target) * torch.clamp(
        target.abs() - regularization,
        min=0,
    )
    result = FBPD(
        lambda value: value - target,
        Const(),
        L1Regularizer(regularization),
        1.0,
        G=Identity(target.shape),
        G_norm_squared=1.0,
        max_iter=100,
        rtol=1e-10,
        atol=1e-12,
    ).run(torch.zeros_like(target), return_info=True)

    assert result.converged
    torch.testing.assert_close(result.solution, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(result.state["dual"], target - result.solution)


def test_fbpd_complex_gradient_matches_finite_differences():
    def solve(target):
        return FBPD(
            lambda value: value - target,
            Const(),
            Const(),
            1.0,
            G=Identity(target.shape),
            G_norm_squared=1.0,
            max_iter=5,
        ).run(torch.zeros_like(target))

    target = torch.tensor(
        [1.2 + 0.3j, -0.7 + 0.2j],
        dtype=torch.complex128,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(solve, (target,))


def test_fbpd_returns_diagnostics_and_reusable_dual_state():
    target = torch.tensor([1.0, -2.0, 3.0])
    solver = FBPD(
        g_grad=lambda value: value - target,
        f_prox=Const(),
        h_prox=Const(),
        g_L=1.0,
        G_norm_squared=1.0,
        G=Identity(target.shape),
        max_iter=100,
        rtol=1e-6,
    )

    result = solver.run(torch.zeros_like(target), return_info=True)
    assert isinstance(result, SolverResult)
    assert result.converged
    assert result.iterations < solver.max_iter
    assert result.residual_norm is not None
    assert set(result.state) == {"dual"}
    torch.testing.assert_close(result.solution, target, rtol=1e-5, atol=1e-6)

    warm_result = solver.run(
        result.solution,
        result.state["dual"],
        return_info=True,
    )
    assert warm_result.converged
    assert warm_result.iterations == 1


def test_fbpd_keeps_legacy_norm_argument_and_rejects_ambiguous_norms():
    legacy = FBPD(
        lambda value: value,
        Const(),
        Const(),
        1.0,
        4.0,
        G=Identity([1]),
    )
    assert legacy.G_norm == legacy.G_norm_squared == 4.0

    with pytest.raises(ValueError, match="only one"):
        FBPD(
            lambda value: value,
            Const(),
            Const(),
            1.0,
            4.0,
            G=Identity([1]),
            G_norm_squared=4.0,
        )


def test_fbpd_validates_dual_initialization_and_explicit_step_size():
    with pytest.raises(ValueError, match="sigma"):
        FBPD(
            lambda value: value,
            Const(),
            Const(),
            1.0,
            G=Identity([2]),
            G_norm_squared=1.0,
            sigma=2.0,
        )

    solver = FBPD(
        lambda value: value,
        Const(),
        Const(),
        1.0,
        G=Identity([2]),
        G_norm_squared=1.0,
    )
    with pytest.raises(ValueError, match="u0"):
        solver.run(torch.zeros(2), torch.zeros(3))


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


def test_power_iteration_stops_on_a_small_eigen_residual():
    operator = Diag(torch.tensor([1.0, 0.8], dtype=torch.float64))
    vector, singular_value = power_iter(
        operator,
        torch.ones(2, dtype=torch.float64),
        max_iter=200,
        tol=1e-8,
        alert=False,
    )

    normal_vector = operator.H(operator(vector))
    relative_residual = torch.linalg.vector_norm(
        normal_vector - singular_value.square() * vector
    ) / torch.linalg.vector_norm(normal_vector)
    assert relative_residual < 1e-8


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


@pytest.mark.parametrize("backward_mode", ["implicit", "unrolled"])
def test_cg_rejects_non_positive_preconditioner_breakdown(backward_mode):
    with pytest.raises(RuntimeError, match="positive definite"):
        CG(
            Identity([2]),
            max_iter=2,
            tol=0,
            P=Diag(torch.tensor([1.0, -1.0])),
            backward_mode=backward_mode,
        ).run(torch.zeros(2), torch.ones(2))


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


@pytest.mark.parametrize("tol", [float("nan"), float("inf")])
def test_power_iteration_rejects_nonfinite_tolerance(tol):
    with pytest.raises(ValueError, match="finite"):
        power_iter(Identity([3]), torch.ones(3), tol=tol)


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
    eager_result = eager.run(torch.zeros_like(target))
    assert len(compiled) == 1
    torch.testing.assert_close(first, eager_result)
