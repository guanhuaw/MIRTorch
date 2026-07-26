import logging
import math
from collections.abc import Callable

import torch

from mirtorch._compile import compile_callable, should_compile
from mirtorch.prox import Prox

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
        restart (Union[...]): restart strategy, not yet implemented
        eval_func: user-defined function to calculate the loss at each iteration.
        compile: use automatic compilation for real-valued CUDA runs.
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
    ):
        if f_L <= 0:
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
        if restart:
            raise NotImplementedError
        self.restart = restart

    def _run(self, x0: torch.Tensor) -> torch.Tensor:
        xold = x0
        yold = x0
        told = 1.0
        for _ in range(self.max_iter):
            fgrad = self.f_grad(xold)
            ynew = self.prox(xold - self._alpha * fgrad, self._alpha)
            tnew = 0.5 * (1 + math.sqrt(1 + 4 * told**2))
            beta = (told - 1) / tnew
            told = tnew
            xold = ynew + beta * (ynew - yold)
            yold = ynew
        return xold

    def _run_with_evaluation(self, x0: torch.Tensor):
        assert self.eval_func is not None
        xold = x0
        yold = x0
        told = 1.0
        saved = []
        for i in range(1, self.max_iter + 1):
            fgrad = self.f_grad(xold)
            ynew = self.prox(xold - self._alpha * fgrad, self._alpha)
            tnew = 0.5 * (1 + math.sqrt(1 + 4 * told**2))
            beta = (told - 1) / tnew
            told = tnew
            xold = ynew + beta * (ynew - yold)
            yold = ynew
            cost = self.eval_func(xold)
            saved.append(cost)
            logger.info("Cost function at %dth iteration: %s", i, cost)
        return xold, saved

    def run(self, x0: torch.Tensor):
        r"""
        Run the algorithm

        Args:
            x0: initialization

        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        if self.eval_func is not None:
            return self._run_with_evaluation(x0)
        if should_compile(self.compile, x0):
            if self._compiled_run is None:
                self._compiled_run = compile_callable(self._run)
            return self._compiled_run(x0)
        return self._run(x0)
