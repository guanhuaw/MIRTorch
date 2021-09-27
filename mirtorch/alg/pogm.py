import numpy as np
from typing import Callable

import torch
from mirtorch.prox import Prox

class POGM():
    r'''
    Optimized Proximal Gradient Method (POGM) 

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
                save_values: bool = False,
                eval_func:Callable = None):
        told = 1
        sig = 1
        zetaold = 1
        xold = x0
        uold = x0
        zold = x0
        if eval_func is not None:
            saved = []
        for i in range(1, self.max_iter+1):
            fgrad = self.f_grad(xold)
            unew = xold - self._alpha * fgrad
            if i == self.max_iter:
                tnew = 0.5 * (1 + np.sqrt(1 + 8 * told**2))
            else:
                tnew = 0.5 * (1 + np.sqrt(1 + 4 * told**2))
            beta = (told - 1) / tnew
            gamma = sig * told / tnew

            znew = (unew + beta * (unew - uold) + gamma * (unew - xold) - beta * self._alpha / zetaold * (xold - zold))
            zetanew = self._alpha * (1 + beta + gamma)
            xnew = self.prox(znew, zetanew)

            uold = unew
            zold = znew
            zetaold = zetanew
            xold = xnew

            if eval_func is not None:
                saved.append(xold)
        
        if eval_func is not None:
            return xold, saved
        else:
            return xold




