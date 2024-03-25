from typing import Callable
import logging

import torch
from mirtorch.prox import Prox, Conj
from mirtorch.linear import LinearMap

logger = logging.getLogger(__name__)


class FBPD:
    r"""Forward-backward primal dual (FBPD) algorithm.

    Ref:
    L. Condat, A primal dual splitting method for convex optimization involving
    Lipschitzian, proximable and linear composite terms. Journal of Optimization Theory
    and Applications, 158(2):460-479, 2013.

    The cost function is:

    .. math::

        arg \min_x f(x) + g(x) + h(Gx)

    where f and h are proper convex functions, and g is a convex function with a L-Lipschitz continuous gradient.

    Attributes:
        g_grad: Callable to calculate the gradient of g
        f_prox: Prox: proximal operator of f
        h_prox: Prox: proximal operator of h
        g_L: float, Lipschitz value of g_grad
        G_norm: float of the norm of G'G, can be solved by power_iter()
        tau: float, step size
        max_iter: int, number of iterations to run
        eval_func: user-defined function to calculate the loss at each iteration.
    """

    def __init__(
        self,
        g_grad: Callable,
        f_prox: Prox,
        h_prox: Prox,
        g_L: float,
        G_norm: float,
        G: LinearMap | None = None,
        tau: float | None = None,
        max_iter: int = 10,
        eval_func: Callable | None = None,
        p: int = 1,
    ):
        self.max_iter = max_iter
        self.g_grad = g_grad
        self.f_prox = f_prox
        self.h_prox = h_prox
        self.h_conj_prox = Conj(self.h_prox)
        self.g_L = g_L
        self.G = G
        self.G_norm = G_norm
        self.p = p
        if tau is None:
            self.tau = 2.0 / (g_L + 2.0)
        else:
            self.tau = tau
        self.sigma = (1.0 / self.tau - self.g_L / 2.0) / self.G_norm
        self.eval_func = eval_func

    def run(self, x0: torch.Tensor):
        r"""
        Run the algorithm
        Args:
            x0: tensor, initialization
        Returns:
            xk: tensor, results
            saved: (optional) a list of intermediate results, calcuated by the eval_func.
        """
        uold = self.G * x0
        xold = x0
        if self.eval_func is not None:
            saved = []
        for i in range(1, self.max_iter + 1):
            xold_bar = self.g_grad(xold) + self.G.H * uold
            xnew = self.f_prox(xold - self.tau * xold_bar, self.tau)
            uold_bar = self.G * (2 * xnew - xold)
            unew = self.h_conj_prox(uold + self.sigma * uold_bar, self.sigma)
            xold = self.p * xnew + (1 - self.p) * xold
            uold = self.p * unew + (1 - self.p) * uold
            if self.eval_func is not None:
                saved.append(self.eval_func(xold))
                logger.info(
                    "The cost function at %dth iter in FBPD: %10.3e.", i, saved[-1]
                )
        if self.eval_func is not None:
            return xold, saved
        else:
            return xold
