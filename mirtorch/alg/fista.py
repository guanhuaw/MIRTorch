from mirtorch.prox import Prox
import numpy as np
import torch
from typing import Callable
import logging

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
    """

    def __init__(
        self,
        f_grad: Callable,
        f_L: float,
        g_prox: Prox,
        max_iter: int = 10,
        restart=False,
        eval_func: Callable = None,
    ):
        self.max_iter = max_iter
        self.f_grad = f_grad
        self.f_L = f_L
        self.prox = g_prox
        self._alpha = 1 / self.f_L  # value for 1/L
        self.eval_func = eval_func
        if restart:
            raise NotImplementedError
        self.restart = restart

    def run(self, x0: torch.Tensor):
        r"""
        Run the algorithm

        Args:
            x0: initialization

        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calcuated by the eval_func.
        """

        def _update_momentum():
            nonlocal told, beta
            tnew = 0.5 * (1 + np.sqrt(1 + 4 * told**2))
            beta = (told - 1) / tnew
            told = tnew

        # initialize parameters
        xold = x0
        yold = x0
        told = 1.0
        beta = 0.0
        if self.eval_func is not None:
            saved = []
        for i in range(1, self.max_iter + 1):
            fgrad = self.f_grad(xold)
            ynew = self.prox(xold - self._alpha * fgrad, self._alpha)
            _update_momentum()
            xnew = ynew + beta * (ynew - yold)
            xold = xnew
            yold = ynew
            # log the cost function
            if self.eval_func is not None:
                saved.append(self.eval_func(xold))
                logger.info(f"Cost function at {i}th iteration: {self.eval_func(xold)}")
        if self.eval_func is not None:
            return xold, saved
        else:
            return xold
