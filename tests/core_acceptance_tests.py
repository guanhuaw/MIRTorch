"""Numerical acceptance tests for core linear maps, proximal maps, and solvers."""

import math

import pytest
import torch

from mirtorch.alg import CG, FISTA, POGM
from mirtorch.linear import (
    Convolve2d,
    Diag,
    Diff1d,
    FFTCn,
    Identity,
    LinearMap,
)
from mirtorch.prox import BoxConstraint, L2Regularizer, SquaredL2Regularizer


def _real_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.conj() * right).sum().real


def _l2_norm(value: torch.Tensor) -> torch.Tensor:
    return value.abs().square().sum().sqrt()


def _dense_matrix(operator: LinearMap, dtype: torch.dtype) -> torch.Tensor:
    columns = []
    input_elements = math.prod(operator.size_in)
    for column in range(input_elements):
        basis = torch.zeros(input_elements, dtype=dtype)
        basis[column] = 1
        columns.append(operator(basis.reshape(operator.size_in)).reshape(-1))
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize(
    "operator",
    [
        Diff1d((2, 4), dim=-1, mode="periodic"),
        FFTCn([2, 3], [2, 3], dims=(-2, -1), norm="forward"),
        Convolve2d(
            (1, 1, 4, 5),
            torch.tensor(
                [
                    [
                        [
                            [1.0 + 0.5j, -0.25 + 0.75j, 0.5 - 0.1j],
                            [0.2 - 0.3j, 0.4 + 0.1j, -0.7 + 0.2j],
                        ]
                    ],
                    [
                        [
                            [0.1 - 0.4j, 0.3 + 0.2j, 0.6 + 0.3j],
                            [-0.2 + 0.5j, 0.9 - 0.1j, 0.2 + 0.4j],
                        ]
                    ],
                ],
                dtype=torch.complex128,
            ),
            stride=(2, 2),
            padding=(0, 1),
        ),
    ],
    ids=["periodic-difference", "centered-fft", "strided-complex-convolution"],
)
def test_complex_linear_maps_match_explicit_matrix_adjoint_and_gradient(operator):
    torch.manual_seed(101)
    matrix = _dense_matrix(operator, torch.complex128)
    value = torch.randn(*operator.size_in, dtype=torch.complex128, requires_grad=True)
    probe = torch.randn(*operator.size_out, dtype=torch.complex128)

    output = operator(value)
    adjoint_output = operator.H(probe)
    explicit_output = matrix @ value.reshape(-1)
    explicit_adjoint = matrix.mH @ probe.reshape(-1)

    torch.testing.assert_close(output.reshape(-1), explicit_output)
    torch.testing.assert_close(adjoint_output.reshape(-1), explicit_adjoint)

    loss = _real_inner(probe, output)
    (gradient,) = torch.autograd.grad(loss, value)
    torch.testing.assert_close(gradient, adjoint_output)


def test_weighted_l2_prox_satisfies_stationarity_and_directional_gradient():
    value = torch.tensor(
        [1.2 + 0.7j, -0.7 + 1.1j, 2.1 - 0.4j],
        dtype=torch.complex128,
        requires_grad=True,
    )
    weights = torch.tensor(
        [0.5, 2.0, 1.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    alpha = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    direction_value = torch.tensor(
        [0.3 - 0.2j, -0.4 + 0.1j, 0.25 + 0.35j],
        dtype=torch.complex128,
    )
    direction_weights = torch.tensor([0.2, -0.1, 0.15], dtype=torch.float64)
    direction_alpha = torch.tensor(-0.05, dtype=torch.float64)
    upstream = torch.tensor(
        [-0.2 + 0.3j, 0.5 - 0.1j, 0.4 + 0.2j],
        dtype=torch.complex128,
    )
    regularization = 0.3

    def apply_prox(current_value, current_weights, current_alpha):
        return L2Regularizer(
            regularization,
            P=Diag(current_weights),
        )(current_value, current_alpha)

    result = apply_prox(value, weights, alpha)
    weighted_norm = _l2_norm(weights * result)
    stationarity = (
        result
        - value
        + alpha * regularization * weights.square() * result / weighted_norm
    )
    assert _l2_norm(stationarity) < 1e-12

    loss = _real_inner(upstream, result)
    gradients = torch.autograd.grad(loss, (value, weights, alpha))
    analytical = sum(
        (
            _real_inner(gradients[0], direction_value),
            _real_inner(gradients[1], direction_weights),
            _real_inner(gradients[2], direction_alpha),
        )
    )

    epsilon = 1e-6
    with torch.no_grad():
        plus = apply_prox(
            value + epsilon * direction_value,
            weights + epsilon * direction_weights,
            alpha + epsilon * direction_alpha,
        )
        minus = apply_prox(
            value - epsilon * direction_value,
            weights - epsilon * direction_weights,
            alpha - epsilon * direction_alpha,
        )
        numerical = (_real_inner(upstream, plus) - _real_inner(upstream, minus)) / (
            2 * epsilon
        )

    torch.testing.assert_close(analytical, numerical, rtol=2e-6, atol=2e-8)


class _DenseMap(LinearMap):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        super().__init__((matrix.shape[1],), (matrix.shape[0],))

    def _apply(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix @ value

    def _apply_adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix.mH @ value


def test_complex_cg_solution_and_implicit_gradient_match_direct_solve():
    torch.manual_seed(202)
    factor = torch.randn(4, 4, dtype=torch.complex128)
    matrix = factor.mH @ factor + 0.7 * torch.eye(4)
    operator = _DenseMap(matrix)
    right_hand_side = torch.randn(4, dtype=torch.complex128, requires_grad=True)
    upstream = torch.randn(4, dtype=torch.complex128)

    result = CG(operator, max_iter=8, tol=1e-28).run(
        torch.zeros_like(right_hand_side),
        right_hand_side,
    )
    expected = torch.linalg.solve(matrix, right_hand_side.detach())
    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-11)

    loss = _real_inner(upstream, result)
    (gradient,) = torch.autograd.grad(loss, right_hand_side)
    expected_gradient = torch.linalg.solve(matrix, upstream)
    torch.testing.assert_close(
        gradient,
        expected_gradient,
        rtol=1e-11,
        atol=1e-11,
    )


def test_unrolled_cg_combined_directional_gradient_matches_finite_difference():
    diagonal = torch.tensor(
        [1.3, 2.1, 4.7],
        dtype=torch.float64,
        requires_grad=True,
    )
    right_hand_side = torch.tensor(
        [0.7, -1.2, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    initial = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    directions = (
        torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64),
        torch.tensor([-0.4, 0.15, 0.25], dtype=torch.float64),
        torch.tensor([0.1, 0.35, -0.2], dtype=torch.float64),
    )
    upstream = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64)

    def solve(current_diagonal, current_rhs, current_initial):
        return CG(
            Diag(current_diagonal),
            max_iter=2,
            tol=0,
            backward_mode="unrolled",
        ).run(current_initial, current_rhs)

    result = solve(diagonal, right_hand_side, initial)
    gradients = torch.autograd.grad(
        _real_inner(upstream, result),
        (diagonal, right_hand_side, initial),
    )
    analytical = sum(
        _real_inner(gradient, direction)
        for gradient, direction in zip(gradients, directions, strict=True)
    )

    epsilon = 1e-6
    arguments = (diagonal, right_hand_side, initial)
    with torch.no_grad():
        plus = solve(
            *(
                argument + epsilon * direction
                for argument, direction in zip(arguments, directions, strict=True)
            )
        )
        minus = solve(
            *(
                argument - epsilon * direction
                for argument, direction in zip(arguments, directions, strict=True)
            )
        )
        numerical = (_real_inner(upstream, plus) - _real_inner(upstream, minus)) / (
            2 * epsilon
        )

    torch.testing.assert_close(analytical, numerical, rtol=2e-6, atol=2e-8)


def test_fista_returns_and_evaluates_the_feasible_proximal_iterate():
    objective = lambda value: 0.5 * (value - 1.1).square().sum()
    result, evaluations = FISTA(
        f_grad=lambda value: value - 1.1,
        f_L=2.0,
        g_prox=BoxConstraint(1.0, 0.0, 1.0),
        max_iter=3,
        eval_func=objective,
        compile=False,
    ).run(torch.zeros(1))

    assert 0 <= result.item() <= 1
    torch.testing.assert_close(evaluations[-1], objective(result))


def test_weighted_box_constraint_is_invariant_to_step_and_lambda():
    value = torch.tensor([0.5], dtype=torch.float64)
    weight = Diag(torch.tensor([2.0], dtype=torch.float64))

    first = BoxConstraint(
        0.1,
        0.25,
        0.75,
        P=weight,
    )(value, 0.01)
    second = BoxConstraint(
        7.0,
        0.25,
        0.75,
        P=weight,
    )(value, 3.0)

    expected = torch.tensor([0.375], dtype=torch.float64)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


def test_weighted_box_constraint_swaps_bounds_for_negative_diagonal():
    result = BoxConstraint(
        1.0,
        0.25,
        0.75,
        P=Diag(torch.tensor([-2.0], dtype=torch.float64)),
    )(torch.tensor([0.0], dtype=torch.float64), 1.0)

    torch.testing.assert_close(result, torch.tensor([-0.125], dtype=torch.float64))


def test_weighted_box_constraint_zero_diagonal_passes_through_when_feasible():
    value = torch.tensor([3.0], dtype=torch.float64)
    result = BoxConstraint(
        1.0,
        -0.2,
        0.5,
        P=Diag(torch.tensor([0.0], dtype=torch.float64)),
    )(value, 1.0)

    torch.testing.assert_close(result, value)


def test_weighted_box_constraint_rejects_infeasible_zero_diagonal():
    with pytest.raises(ValueError):
        BoxConstraint(
            1.0,
            0.25,
            0.75,
            P=Diag(torch.tensor([0.0], dtype=torch.float64)),
        )(torch.tensor([0.0], dtype=torch.float64), 1.0)


@pytest.mark.parametrize("solver", [FISTA, POGM])
def test_proximal_solver_produces_plausible_blur_reconstruction(solver):
    torch.manual_seed(303)
    size = 16
    row, column = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij",
    )
    truth = (
        0.8 * (((column - 5).square() + (row - 6).square()) < 12)
        + ((column >= 10) & (column < 14) & (row >= 10) & (row < 14))
    ).clamp_max(1)
    truth = truth.to(torch.float64)[None, None]
    kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel / kernel.sum()
    forward = Convolve2d(truth.shape, kernel[None, None], padding=1)
    data = forward(truth) + 0.008 * torch.randn_like(truth)
    regularization = 0.006
    normal = forward.H * forward + 2 * regularization * Identity(truth.shape)
    right_hand_side = forward.H(data)
    initial = torch.zeros_like(truth)
    reference = CG(normal, max_iter=100, tol=1e-20).run(
        initial,
        right_hand_side,
    )

    result = solver(
        f_grad=lambda value: forward.H(forward(value) - data),
        f_L=1.0,
        g_prox=SquaredL2Regularizer(regularization),
        max_iter=60,
        compile=False,
    ).run(initial)

    def psnr(value):
        return 10 * torch.log10(1 / (value - truth).square().mean())

    relative_reference_error = _l2_norm(result - reference) / _l2_norm(reference)
    assert relative_reference_error < 0.005
    assert psnr(result) > psnr(data) + 3
