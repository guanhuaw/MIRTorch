import logging
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

logger = logging.getLogger(__name__)


class FISTA:
    r"""
    Fast Iterative Soft Thresholding Algorithm (FISTA) / Fast Proximal Gradient Method (FPGM)

    .. math::

        arg \min_x f(x) + g(x)

    where grad(f(x)) is L-Lipschitz continuous and g is proximal-friendly function.

    Attributes:
        max_iter (int): number of iterations to run
        f_grad (Callable): gradient of f
        f_L (float): L-Lipschitz value of f_grad
        g_prox (Prox): proximal operator g
        restart (bool): use gradient-based adaptive momentum restart
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
        self.eval_func = eval_func
        self.compile = compile
        self._compiled_run = None
        if not isinstance(restart, bool):
            raise TypeError("restart must be a bool")
        rtol = float(rtol)
        atol = float(atol)
        validate_stopping_tolerances(rtol, atol)
        self.restart = restart
        self.rtol = rtol
        self.atol = atol
        self._stopping_enabled = stopping_is_enabled(rtol, atol)

    def _run(self, x0: torch.Tensor, return_info: bool = False):
        extrapolated = x0
        iterate = x0
        momentum = 1.0
        saved = []
        iterations = 0
        converged = False
        residual_norm = None
        for i in range(1, self.max_iter + 1):
            gradient = self.f_grad(extrapolated)
            next_iterate = self.prox(
                extrapolated - self._alpha * gradient,
                self._alpha,
            )
            next_momentum = 0.5 * (1 + math.sqrt(1 + 4 * momentum**2))

            should_restart = False
            if self.restart:
                restart_measure = torch.sum(
                    (extrapolated - next_iterate).conj() * (next_iterate - iterate)
                ).real
                should_restart = restart_measure.item() > 0

            if should_restart:
                next_momentum = 1.0
                extrapolated = next_iterate
            else:
                scale = (momentum - 1) / next_momentum
                extrapolated = next_iterate + scale * (next_iterate - iterate)

            if self._stopping_enabled or return_info:
                residual_norm = l2_norm(next_iterate - iterate)
                if self._stopping_enabled:
                    threshold = self.atol + self.rtol * l2_norm(next_iterate)
                    converged = bool((residual_norm <= threshold).item())

            iterate = next_iterate
            momentum = next_momentum
            iterations = i

            if self.eval_func is not None:
                cost = self.eval_func(iterate)
                saved.append(cost)
                logger.info("Cost function at iteration %d: %s", i, cost)

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
