import numpy as np
from typing import Callable

import torch
from mirtorch.prox import Prox


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
        restart (Union[...]): restart strategy, not yet implemented
        eval_func: user-defined function to calculate the loss at each iteration.

    TODO: add the restart
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
        if restart:
            raise NotImplementedError
        self.restart = restart
        self.eval_func = eval_func

    def run(self, x0: torch.Tensor):
        r"""
        Run the algorithm
        Args:
            x0: initialization
        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calcuated by the eval_func.
        """
        told = 1
        gamma_old = 1
        xold = x0
        omold = x0
        zold = x0
        if self.eval_func is not None:
            saved = []
        for i in range(1, self.max_iter + 1):
            fgrad = self.f_grad(xold)
            omnew = xold - self._alpha * fgrad
            if i == self.max_iter:
                tnew = 0.5 * (1 + np.sqrt(1 + 8 * told**2))
            else:
                tnew = 0.5 * (1 + np.sqrt(1 + 4 * told**2))
            gamma_new = self._alpha * (2 * told + tnew - 1) / tnew
            znew = (
                omnew
                + (told - 1) / tnew * (omnew - omold)
                + told / tnew * (omnew - xold)
                + self._alpha * (told - 1) / gamma_old / tnew * (zold - xold)
            )
            xnew = self.prox(znew, gamma_new)
            zold = znew
            told = tnew
            omold = omnew
            xold = xnew
            gamma_old = gamma_new

            if self.eval_func is not None:
                saved.append(self.eval_func(xold))

        if self.eval_func is not None:
            return xold, saved
        else:
            return xold
