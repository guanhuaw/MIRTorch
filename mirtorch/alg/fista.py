class FISTA():
    def __init__(self, max_iter, step, fval, grad, prox, momentum = 1, restart = False):
        self.max_iter = max_iter
        self.step = step
        self.fval = fval
        self.grad = grad
        self.prox = prox
        self.momentum = momentum
        self.restart = restart

    def _update(self, iter):
        #looks like one possible implementation from jeff lecture slides
        self.momentum = (iter-1)/(iter+2)

    def run_alg(self, x0):
        x_curr = x0
        z_curr = x0 
        for i in range(max_iter):
            x_prev = x_curr
            z_prev = z_curr
            #compute new z_k and x_k
            z_curr = self.prox(x_prev - self.grad(x_prev)) 
            x_curr = z_curr + self.momentum * (z_curr - z_prev)
            #update momentum value for next iteration
            self._update(i)
        return x_curr