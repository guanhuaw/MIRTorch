import logging

import torch
from torch import Tensor

from mirtorch.util import squared_l2_norm

logger = logging.getLogger(__name__)


class CG_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b: Tensor, solver, x0, saved):
        ctx.solver = solver
        result = solver._iterate(x0, b)
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
            ),
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
):
    """Run standard (preconditioned) conjugate-gradient iterations."""
    residual = b - A * x0
    preconditioned = residual if P is None else P * residual
    direction = (
        preconditioned.clone() if differentiable else preconditioned.detach().clone()
    )
    solution = x0.clone() if differentiable else x0.detach().clone()
    residual_preconditioned = (residual.conj() * preconditioned).sum().real
    residual_squared = squared_l2_norm(residual)
    saved = []
    denominators_valid = None

    for iteration in range(1, max_iter + 1):
        if residual_squared.item() <= tol:
            break

        applied_direction = A * direction
        denominator = (direction.conj() * applied_direction).sum().real
        denominator_valid = torch.isfinite(denominator) & (denominator > 0)
        denominators_valid = (
            denominator_valid
            if denominators_valid is None
            else denominators_valid & denominator_valid
        )

        step = residual_preconditioned / denominator
        solution = solution + step * direction
        next_residual = residual - step * applied_direction
        next_preconditioned = next_residual if P is None else P * next_residual
        next_residual_preconditioned = (
            (next_residual.conj() * next_preconditioned).sum().real
        )
        momentum = next_residual_preconditioned / residual_preconditioned
        direction = next_preconditioned + momentum * direction
        residual = next_residual
        residual_preconditioned = next_residual_preconditioned
        residual_squared = squared_l2_norm(residual)

        if eval_func is not None:
            saved.append(eval_func(residual))
        if alert:
            logger.info(
                "Squared residual at CG iteration %d: %10.3e.",
                iteration,
                residual_squared,
            )

    if denominators_valid is not None and not denominators_valid.item():
        raise RuntimeError(
            "CG encountered a non-positive or non-finite <p, A p>; "
            "A must be Hermitian positive definite on the active subspace"
        )
    if eval_func is not None:
        return solution, saved
    return solution


class CG:
    r"""
    Solve :math:`Ax=b` with conjugate gradients, where ``A`` is Hermitian
    positive definite on the active Krylov subspace.

    ``backward_mode="implicit"`` (the default) solves another CG system in
    backward and avoids storing the iteration history. ``"unrolled"``
    differentiates the truncated iterations, including ``x0`` and tensors
    captured by the operator.

    Attributes:
        A: square Hermitian positive-definite ``LinearMap``
        tol: squared-residual stopping tolerance; zero runs ``max_iter`` steps
        max_iter: int, max number of iterations
        alert: bool, print the norm of residuals at the end
        eval_func: user-defined function to calculate the loss at each iteration.
        P: LinearMap of a Preconditioner
        backward_mode: ``"implicit"`` or ``"unrolled"``

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
    ):
        if A.size_in != A.size_out:
            raise ValueError("CG requires a square LinearMap")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if backward_mode not in ("implicit", "unrolled"):
            raise ValueError("backward_mode must be 'implicit' or 'unrolled'")
        self.A = A
        self.max_iter = max_iter
        self.tol = tol
        self.alert = alert
        self.eval_func = eval_func
        self.P = P
        self.backward_mode = backward_mode

    def _iterate(self, x0, b, *, differentiable=False):
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
        )

    def run(self, x0, b):
        r"""Run the CG iterations.
        Args:
            x0: Initialization
            b: RHS

        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        if list(self.A.size_out) != list(b.shape):
            raise ValueError("The size of A and b do not match.")
        if list(self.A.size_in) != list(x0.shape):
            raise ValueError("The size of A and x0 do not match.")
        if self.backward_mode == "unrolled":
            return self._iterate(x0, b, differentiable=True)
        saved = []
        solution = CG_func.apply(b, self, x0, saved)
        if self.eval_func is not None:
            return solution, saved
        return solution
