import logging
import math
from collections.abc import Callable

import torch

from mirtorch.prox import Prox
from mirtorch.util import compile_callable, should_compile

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

    def _run(self, x0: torch.Tensor):
        extrapolated = x0
        iterate = x0
        momentum = 1.0
        saved = []
        for i in range(1, self.max_iter + 1):
            gradient = self.f_grad(extrapolated)
            next_iterate = self.prox(
                extrapolated - self._alpha * gradient,
                self._alpha,
            )
            next_momentum = 0.5 * (1 + math.sqrt(1 + 4 * momentum**2))
            scale = (momentum - 1) / next_momentum
            extrapolated = next_iterate + scale * (next_iterate - iterate)
            iterate = next_iterate
            momentum = next_momentum

            if self.eval_func is not None:
                cost = self.eval_func(iterate)
                saved.append(cost)
                logger.info("Cost function at iteration %d: %s", i, cost)

        if self.eval_func is not None:
            return iterate, saved
        return iterate

    def run(self, x0: torch.Tensor):
        r"""
        Run the algorithm

        Args:
            x0: initialization

        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        if self.eval_func is None and should_compile(self.compile, x0):
            if self._compiled_run is None:
                self._compiled_run = compile_callable(self._run)
            return self._compiled_run(x0)
        return self._run(x0)
