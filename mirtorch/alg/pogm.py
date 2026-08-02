import math
from collections.abc import Callable

import torch

from mirtorch.prox import Prox
from mirtorch.util import compile_callable, l2_norm, should_compile

from .solver import (
    SolverResult,
    stopping_is_enabled,
    validate_stopping_tolerances,
)


class POGM:
    r"""
    Optimized Proximal Gradient Method (POGM)
    Ref: D. Kim and J. A. Fessler. “Adaptive restart of the optimized gradient method for convex optimization”.
    In: J. Optim. Theory Appl. 178.1 (July 2018), 240–63 (cit. on pp. 5.26, 5.29).

    .. math::

        arg \min_x f(x) + g(x)

    where grad(f(x)) is L-Lipschitz continuous and g is proximal-friendly function.

    Attributes:
        max_iter (int): number of iterations to run
        f_grad (Callable): gradient of f
        f_L (float): L-Lipschitz value of f_grad
        g_prox (Prox): proximal operator g. For plain OGM, you could call Const() as a place-holder here
        restart (bool): use composite-gradient adaptive momentum restart
        eval_func: user-defined function to calculate the loss at each iteration.
        compile: use automatic compilation for real-valued CUDA runs.
        rtol: relative iterate-change stopping tolerance; zero disables it.
        atol: absolute iterate-change stopping tolerance; zero disables it.
    """

    def __init__(
        self,
        f_grad: Callable,
        f_L: float,
        g_prox: Prox,
        max_iter: int = 10,
        restart=False,
        eval_func: Callable | None = None,
        compile: bool = True,
        rtol: float = 0.0,
        atol: float = 0.0,
    ):
        f_L = float(f_L)
        if not math.isfinite(f_L) or f_L <= 0:
            raise ValueError("f_L must be positive")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        self.max_iter = max_iter
        self.f_grad = f_grad
        self.f_L = f_L
        self.prox = g_prox
        self._alpha = 1 / self.f_L  # value for 1/L
        if not isinstance(restart, bool):
            raise TypeError("restart must be a bool")
        rtol = float(rtol)
        atol = float(atol)
        validate_stopping_tolerances(rtol, atol)
        self.restart = restart
        self.rtol = rtol
        self.atol = atol
        self._stopping_enabled = stopping_is_enabled(rtol, atol)
        self.eval_func = eval_func
        self.compile = compile
        self._compiled_run = None

    def _run(self, x0: torch.Tensor, return_info: bool = False):
        momentum = 1.0
        previous_step = 1.0
        iterate = x0
        gradient_step = x0
        auxiliary = x0
        primary = x0
        saved = []
        iterations = 0
        converged = False
        residual_norm = None

        for i in range(1, self.max_iter + 1):
            gradient = self.f_grad(iterate)
            next_gradient_step = iterate - self._alpha * gradient
            if i == self.max_iter and not self.restart and not self._stopping_enabled:
                next_momentum = 0.5 * (1 + math.sqrt(1 + 8 * momentum**2))
            else:
                next_momentum = 0.5 * (1 + math.sqrt(1 + 4 * momentum**2))
            next_step = self._alpha * (2 * momentum + next_momentum - 1) / next_momentum
            next_auxiliary = (
                next_gradient_step
                + (momentum - 1) / next_momentum * (next_gradient_step - gradient_step)
                + momentum / next_momentum * (next_gradient_step - iterate)
                + self._alpha
                * (momentum - 1)
                / previous_step
                / next_momentum
                * (auxiliary - iterate)
            )
            next_iterate = self.prox(next_auxiliary, next_step)

            if self.restart:
                composite_gradient = (
                    gradient - (next_iterate - next_auxiliary) / next_step
                )
                next_primary = iterate - self._alpha * composite_gradient
                restart_measure = torch.sum(
                    composite_gradient.conj() * (next_primary - primary)
                ).real
                if restart_measure.item() > 0:
                    next_momentum = 1.0
                primary = next_primary

            if self._stopping_enabled or return_info:
                residual_norm = l2_norm(next_iterate - iterate)
                if self._stopping_enabled:
                    threshold = self.atol + self.rtol * l2_norm(next_iterate)
                    converged = bool((residual_norm <= threshold).item())

            iterate = next_iterate
            auxiliary = next_auxiliary
            gradient_step = next_gradient_step
            momentum = next_momentum
            previous_step = next_step
            iterations = i

            if self.eval_func is not None:
                saved.append(self.eval_func(iterate))

            if converged:
                break

        if return_info:
            return SolverResult(
                solution=iterate,
                iterations=iterations,
                converged=converged,
                residual_norm=residual_norm,
                history=saved,
            )

        if self.eval_func is not None:
            return iterate, saved
        return iterate

    def run(self, x0: torch.Tensor, *, return_info: bool = False):
        r"""
        Run the algorithm
        Args:
            x0: initialization
            return_info: return a :class:`~mirtorch.alg.SolverResult` with
                diagnostics.
        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        can_compile = (
            not return_info
            and self.eval_func is None
            and not self.restart
            and not self._stopping_enabled
        )
        if can_compile and should_compile(self.compile, x0):
            if self._compiled_run is None:
                self._compiled_run = compile_callable(self._run)
            return self._compiled_run(x0)
        return self._run(x0, return_info)
