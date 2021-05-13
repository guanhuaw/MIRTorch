import numpy as np

class POGM():
    r'''
    TODO: docstring here
    '''

    def __init__(self, max_iter, fgrad, Lf, prox, restart = False):
        self.max_iter = max_iter
        self.grad = fgrad
        self.Lf = Lf
        self.prox = prox
        if restart:
            raise NotImplementedError
        self.restart = restart


    def update_params(tk_prev, iter):
        if iter >= 2 and iter < self.max_iter:
            mult = 4
        elif iter == self.max_iter:
            mult = 8
        tk = .5 * (1 + np.sqrt(mult * tk_prev * tk_prev + 1))
        gk = 1/self.Lf * (2*tk_prev + tk - 1)/tk
        return tk, gk


    def run_alg(self, x0):
        x_curr = x0
        w_curr = x0
        z_curr = x0
        tk = 1
        gk = 1
        for i in range(self.max_iter):
            tk_prev = tk
            gk_prev = gk
            w_prev = w_curr
            z_prev = z_curr
            #dont need to set x_prev because x is updated last

            #update params theta, gamma
            tk, gk = self.update_params(tk_prev, i+1)

            #update w
            w_curr = x_curr - 1/self.Lf*self.grad(x_curr)

            #update z
            z_curr = w_curr + (tk_prev-1)/tk * (w_curr - w_prev) 
            z_curr += tk_prev/tk * (w_curr - x_curr) 
            z_curr += (tk_prev-1)/(self.Lf*gk_prev*tk) * (z_prev - x_curr)

            #update x
            self.prox.Lambda = gk
            x_curr = self.prox(z_curr)
            
        return x_curr




