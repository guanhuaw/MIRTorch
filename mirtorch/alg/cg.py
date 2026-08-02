import logging
import math
from typing import Any, cast

import torch
from torch import Tensor

from mirtorch.util import squared_l2_norm

from .solver import (
    SolverResult,
    stopping_is_enabled,
    validate_stopping_tolerances,
)

logger = logging.getLogger(__name__)


class CG_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b: Tensor, solver, x0, saved, report):
        ctx.solver = solver
        result = solver._iterate(x0, b, report=report)
        if solver.eval_func is None:
            return result
        solution, evaluations = result
        saved.extend(evaluations)
        return solution

    @staticmethod
    def backward(ctx, *grad_outputs):
        dx = grad_outputs[0]
        solver = ctx.solver
        return (
            cg_block(
                torch.zeros_like(dx),
                dx,
                solver.A,
                solver.tol,
                solver.max_iter,
                solver.alert,
                None,
                solver.P,
                rtol=solver.rtol,
                atol=solver.atol,
            ),
            None,
            None,
            None,
            None,
        )


def cg_block(
    x0,
    b,
    A,
    tol,
    max_iter,
    alert,
    eval_func,
    P,
    *,
    differentiable=False,
    rtol=None,
    atol=None,
    report=None,
):
    """Run standard (preconditioned) conjugate-gradient iterations.

    ``tol`` retains the historical squared-residual semantics.  When either
    ``rtol`` or ``atol`` is provided, convergence instead follows
    ``||r|| <= atol + rtol * ||b||``.  A zero stopping threshold runs the full
    iteration count without synchronizing the device on every iteration.
    """
    residual = b - A * x0
    preconditioned = residual if P is None else P * residual
    direction = (
        preconditioned.clone() if differentiable else preconditioned.detach().clone()
    )
    solution = x0.clone() if differentiable else x0.detach().clone()
    rho = (residual.conj() * preconditioned).sum().real
    residual_squared = squared_l2_norm(residual)
    saved = []
    recurrences_valid = None
    iterations = 0
    converged = False

    if rtol is None and atol is None:
        stopping_squared = None if tol == 0 else tol
    else:
        relative_tolerance = 0.0 if rtol is None else rtol
        absolute_tolerance = 0.0 if atol is None else atol
        if stopping_is_enabled(relative_tolerance, absolute_tolerance):
            threshold = (
                absolute_tolerance + relative_tolerance * squared_l2_norm(b).sqrt()
            )
            stopping_squared = threshold.square()
        else:
            stopping_squared = None

    for iteration in range(1, max_iter + 1):
        if (
            stopping_squared is not None
            and (residual_squared <= stopping_squared).item()
        ):
            converged = True
            break

        iterations = iteration
        applied_direction = A * direction
        denominator = (direction.conj() * applied_direction).sum().real
        active = residual_squared != 0
        rho_valid = torch.isfinite(rho) & (rho > 0)
        denominator_valid = torch.isfinite(denominator) & (denominator > 0)
        update_valid = active & rho_valid & denominator_valid
        recurrence_valid = ~active | update_valid
        recurrences_valid = (
            recurrence_valid
            if recurrences_valid is None
            else recurrences_valid & recurrence_valid
        )

        safe_denominator = torch.where(
            denominator_valid,
            denominator,
            torch.ones_like(denominator),
        )
        safe_rho = torch.where(
            rho_valid,
            rho,
            torch.ones_like(rho),
        )
        step = torch.where(
            update_valid,
            safe_rho / safe_denominator,
            torch.zeros_like(rho),
        )
        solution = solution + step * direction
        next_residual = residual - step * applied_direction
        next_preconditioned = next_residual if P is None else P * next_residual
        next_rho = (next_residual.conj() * next_preconditioned).sum().real
        next_residual_squared = squared_l2_norm(next_residual)
        next_active = next_residual_squared != 0
        next_rho_valid = torch.isfinite(next_rho) & (next_rho > 0)
        next_recurrence_valid = ~next_active | next_rho_valid
        recurrences_valid = recurrences_valid & next_recurrence_valid
        safe_next_rho = torch.where(
            next_rho_valid,
            next_rho,
            torch.zeros_like(next_rho),
        )
        momentum = torch.where(
            update_valid & next_rho_valid,
            safe_next_rho / safe_rho,
            torch.zeros_like(next_rho),
        )
        direction = next_preconditioned + momentum * direction
        residual = next_residual
        rho = next_rho
        residual_squared = next_residual_squared

        if eval_func is not None:
            saved.append(eval_func(residual))
        if alert:
            logger.info(
                "Squared residual at CG iteration %d: %10.3e.",
                iteration,
                residual_squared,
            )

    if recurrences_valid is not None and not recurrences_valid.item():
        raise RuntimeError(
            "CG encountered a non-positive or non-finite recurrence inner product; "
            "A and P must be Hermitian positive definite on the active subspace"
        )
    if report is not None:
        if stopping_squared is not None and not converged:
            converged = bool((residual_squared <= stopping_squared).item())
        report.iterations = iterations
        report.converged = converged
        report.residual_norm = residual_squared.sqrt().detach()
    if eval_func is not None:
        return solution, saved
    return solution


class CG:
    r"""
    Solve :math:`Ax=b` with conjugate gradients, where ``A`` is Hermitian
    positive definite on the active Krylov subspace.

    ``backward_mode="implicit"`` (the default) solves another CG system in
    backward and avoids storing the iteration history.  It treats the operator
    as fixed and differentiates only the right-hand side. ``"unrolled"``
    differentiates the truncated iterations, including ``x0`` and tensors
    captured by the operator; use it when operator parameters require gradients.

    Supplying either ``rtol`` or ``atol`` selects the conventional criterion
    ``||r|| <= atol + rtol * ||b||`` and takes precedence over legacy ``tol``.

    Attributes:
        A: square Hermitian positive-definite ``LinearMap``
        tol: squared-residual stopping tolerance; zero runs ``max_iter`` steps
        max_iter: int, max number of iterations
        alert: bool, print the norm of residuals at the end
        eval_func: optional function evaluated on the residual after each iteration.
        P: LinearMap of a Preconditioner
        backward_mode: ``"implicit"`` or ``"unrolled"``
        rtol: optional relative residual tolerance
        atol: optional absolute residual tolerance

    Methods:
        run: run the CG algorithm
    """

    def __init__(
        self,
        A,
        max_iter=20,
        tol=1e-2,
        P=None,
        alert=False,
        eval_func=None,
        *,
        backward_mode: str = "implicit",
        rtol: float | None = None,
        atol: float | None = None,
    ):
        if A.size_in != A.size_out:
            raise ValueError("CG requires a square LinearMap")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        tol = float(tol)
        if not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be non-negative")
        if rtol is not None or atol is not None:
            rtol = None if rtol is None else float(rtol)
            atol = None if atol is None else float(atol)
            validate_stopping_tolerances(
                0.0 if rtol is None else rtol,
                0.0 if atol is None else atol,
            )
        if backward_mode not in ("implicit", "unrolled"):
            raise ValueError("backward_mode must be 'implicit' or 'unrolled'")
        self.A = A
        self.max_iter = max_iter
        self.tol = tol
        self.alert = alert
        self.eval_func = eval_func
        self.P = P
        self.backward_mode = backward_mode
        self.rtol = rtol
        self.atol = atol

    def _iterate(self, x0, b, *, differentiable=False, report=None):
        return cg_block(
            x0,
            b,
            self.A,
            self.tol,
            self.max_iter,
            self.alert,
            self.eval_func,
            self.P,
            differentiable=differentiable,
            rtol=self.rtol,
            atol=self.atol,
            report=report,
        )

    def run(self, x0, b, *, return_info: bool = False):
        r"""Run the CG iterations.
        Args:
            x0: Initialization
            b: RHS

        Returns:
            By default, the historical tensor result, plus ``saved`` when an
            ``eval_func`` is configured.  With ``return_info=True``, a
            :class:`~mirtorch.alg.SolverResult` containing the solution and
            diagnostics.
        """
        if list(self.A.size_out) != list(b.shape):
            raise ValueError("The size of A and b do not match.")
        if list(self.A.size_in) != list(x0.shape):
            raise ValueError("The size of A and x0 do not match.")
        report = (
            SolverResult(solution=x0, iterations=0, converged=False)
            if return_info
            else None
        )
        if self.backward_mode == "unrolled":
            result = self._iterate(
                x0,
                b,
                differentiable=True,
                report=report,
            )
            if self.eval_func is None:
                solution = cast(Tensor, result)
                saved = []
            else:
                solution, saved = cast(tuple[Tensor, list[Any]], result)
            if return_info:
                assert report is not None
                report.solution = solution
                report.history = saved
                return report
            return result
        saved = []
        solution = cast(Tensor, CG_func.apply(b, self, x0, saved, report))
        if return_info:
            assert report is not None
            report.solution = solution
            report.history = saved
            return report
        if self.eval_func is not None:
            return solution, saved
        return solution
