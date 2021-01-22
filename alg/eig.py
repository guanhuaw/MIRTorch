import torch
import numpy as np
from ..linear import linearmaps

@torch.no_grad()
def power_iter(A, x0, max_iter = 100, tol = 1e-6):
    '''
        Input:
            A: Linear Maps
            max_iter: maximum number of iterations
            tol: stopping tolerance
            x0: initial guess of singular vector corresponding to max singular value
        Ouput:
            v1: principal right singular vector
            sig1: spectral norm of A
    '''
    x = x0
    ratio_old = float('inf')
    for iter in range(max_iter):
        Ax = A*x
        ratio = torch.norm(Ax)/torch.norm(x)
        if torch.abs(ratio - ratio_old)/ratio < tol:
            print('calculation of max singular value accomplished at %d iterations', iter)
            break
        ratio_old = ratio
        x = A.H*Ax
        x = x/torch.norm(x)
    return x, torch.norm(A*x)/torch.norm(x)