import logging
import math
from collections.abc import Callable

import torch

from mirtorch.linear import LinearMap
from mirtorch.prox import Conj, Prox
from mirtorch.util import squared_l2_norm

from .solver import (
    SolverResult,
    stopping_is_enabled,
    validate_stopping_tolerances,
)

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
        G_norm: conservative upper bound on the squared operator norm
            :math:`\|G\|_2^2`.  If ``s`` is returned by
            ``power_iter(G, ...)``, pass a slightly inflated ``s**2`` because
            that estimate is not a certified upper bound.  The clearer keyword
            ``G_norm_squared`` is preferred for new code.
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
        G_norm: float | None = None,
        G: LinearMap | None = None,
        tau: float | None = None,
        max_iter: int = 10,
        eval_func: Callable | None = None,
        p: float = 1,
        *,
        G_norm_squared: float | None = None,
        sigma: float | None = None,
        rtol: float = 0.0,
        atol: float = 0.0,
    ):
        if G is None:
            raise ValueError("G must be provided")
        g_L = float(g_L)
        p = float(p)
        if not math.isfinite(g_L) or g_L < 0:
            raise ValueError("g_L must be non-negative")
        if G_norm is not None and G_norm_squared is not None:
            raise ValueError("provide only one of G_norm and G_norm_squared")
        norm_squared = G_norm_squared if G_norm_squared is not None else G_norm
        if norm_squared is not None:
            norm_squared = float(norm_squared)
        if norm_squared is None or not math.isfinite(norm_squared) or norm_squared <= 0:
            raise ValueError("G_norm_squared must be positive")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        if not math.isfinite(p) or not 0 < p <= 1:
            raise ValueError("p must be in the interval (0, 1]")
        rtol = float(rtol)
        atol = float(atol)
        validate_stopping_tolerances(rtol, atol)
        self.max_iter = max_iter
        self.g_grad = g_grad
        self.f_prox = f_prox
        self.h_prox = h_prox
        self.h_conj_prox = Conj(self.h_prox)
        self.g_L = g_L
        self.G = G
        # Keep the historical attribute while making its squared-norm meaning explicit.
        self.G_norm = norm_squared
        self.G_norm_squared = norm_squared
        self.p = p
        if tau is None:
            self.tau = 2.0 / (g_L + 2.0)
        else:
            self.tau = float(tau)
        if (
            not math.isfinite(self.tau)
            or self.tau <= 0
            or 1.0 / self.tau <= self.g_L / 2.0
        ):
            raise ValueError("tau must be positive and satisfy 1 / tau > g_L / 2")
        maximum_sigma = (1.0 / self.tau - self.g_L / 2.0) / norm_squared
        self.sigma = maximum_sigma if sigma is None else float(sigma)
        if (
            not math.isfinite(self.sigma)
            or self.sigma <= 0
            or self.sigma > maximum_sigma
        ):
            raise ValueError(
                "sigma must be positive and satisfy "
                "1 / tau - sigma * G_norm_squared >= g_L / 2"
            )
        self.eval_func = eval_func
        self.rtol = rtol
        self.atol = atol

    def run(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        *,
        return_info: bool = False,
    ):
        r"""
        Run the algorithm
        Args:
            x0: primal initialization
            u0: optional dual initialization.  By default the historical
                initialization ``G(x0)`` is used.
            return_info: return :class:`~mirtorch.alg.SolverResult` with
                diagnostics and warm-start state instead of the historical
                return value.
        Returns:
            xk: tensor, results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        if u0 is not None and list(u0.shape) != list(self.G.size_out):
            raise ValueError("The size of G and u0 do not match.")
        uold = self.G.apply(x0) if u0 is None else u0
        xold = x0
        saved = []
        iterations = 0
        converged = False
        residual_norm = None
        state_norm = None
        check_stopping = stopping_is_enabled(self.rtol, self.atol)
        for i in range(1, self.max_iter + 1):
            xold_bar = self.g_grad(xold) + self.G.adjoint(uold)
            xnew = self.f_prox(xold - self.tau * xold_bar, self.tau)
            uold_bar = self.G.apply(2 * xnew - xold)
            unew = self.h_conj_prox(uold + self.sigma * uold_bar, self.sigma)

            if check_stopping or return_info:
                residual_norm = torch.sqrt(
                    squared_l2_norm(xnew - xold) + squared_l2_norm(unew - uold)
                )
                state_norm = torch.sqrt(squared_l2_norm(xnew) + squared_l2_norm(unew))

            xold = self.p * xnew + (1 - self.p) * xold
            uold = self.p * unew + (1 - self.p) * uold
            iterations = i
            if self.eval_func is not None:
                saved.append(self.eval_func(xold))
                logger.info(
                    "The cost function at %dth iter in FBPD: %10.3e.", i, saved[-1]
                )

            if check_stopping:
                assert residual_norm is not None and state_norm is not None
                threshold = self.atol + self.rtol * state_norm
                if bool((residual_norm <= threshold).item()):
                    converged = True
                    break

        if return_info:
            return SolverResult(
                solution=xold,
                iterations=iterations,
                converged=converged,
                residual_norm=residual_norm,
                history=saved,
                state={"dual": uold},
            )
        if self.eval_func is not None:
            return xold, saved
        return xold
