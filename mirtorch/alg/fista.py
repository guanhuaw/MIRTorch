from mirtorch.prox import Prox
import numpy as np
import torch
from typing import Callable

class FISTA():
    r'''
    Fast Iterative Soft Thresholding Algorithm (FISTA) / Fast Proximal Gradient Method (FPGM)

    .. math::
        \arg \min_x f(x) + g(x)
     where grad(f(x)) is L-Lipschitz continuous and g is proximal operator
    Args:
        max_iter (int): number of iterations to run
        f_grad (Callable): gradient of f
        f_L (float): L-Lipschitz value of f_grad
        g_prox (Prox): proximal operator g
        restart (Union[...]): restart strategy, not yet implemented
    '''

    def __init__(self,
                 f_grad: Callable,
                 f_L: float,
                 g_prox: Prox,
                 max_iter: int = 10,
                 restart = False):
        self.max_iter = max_iter
        self.f_grad = f_grad
        self.f_L = f_L
        self.prox = g_prox
        self._alpha = 1/self.f_L # value for 1/L
        if restart:
            raise NotImplementedError
        self.restart = restart

    def run_alg(self,
                x0: torch.Tensor,
                eval_func: Callable = None):
        def _update_momentum():
            nonlocal told, beta
            tnew = .5 * (1 + np.sqrt(1 + 4 * told**2))
            beta = (told - 1) / tnew
            told = tnew
        # initalize parameters
        xold= x0
        yold = x0
        told = 1.0
        beta = 0.0
        saved = []
        for i in range(1, self.max_iter+1):
            fgrad = self.f_grad(xold)
            ynew = self.prox(xold - self._alpha * fgrad, self._alpha)
            _update_momentum()
            xnew = ynew + beta * (ynew - yold)
            xold = xnew
            yold = ynew
            if save_values:
                saved.append(xold)
        if save_values:
            return saved 
        return xold

FPGM = FISTA