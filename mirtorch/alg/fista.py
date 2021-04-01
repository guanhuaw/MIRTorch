'''
See test_fista.ipynb to see more up to date implementation
'''
class FISTA():
    r'''
    Fast Iterative Soft Thresholding Algorithm (FISTA)

    .. math::
        \arg \min_x f(x) + g(x)
     where grad(f(x)) is L-Lipschitz continuous and g is proximal operator
    Args:
        max_iter (int): number of iterations to run
        fgrad (Callable): gradient of f
        Lf (float): L-Lipschitz value of fgrad
        prox (Prox): proximal operator g
        restart (Union[...]): restart strategy, not yet implemented
    '''
    def __init__(self, max_iter, fgrad, Lf, prox, restart = False):
        self.max_iter = max_iter
        self.step = 1
        self.grad = fgrad
        self.Lf = Lf
        self.prox = prox
        if restart:
            raise NotImplementedError
        self.restart = restart
    

    def _update(self):
        step_prev = self.step
        self.step = (1 + np.sqrt(1 + 4*step_prev*step_prev))/2
        self.momentum = (step_prev-1)/self.step


    def run_alg(self, x0):
        x_curr = x0
        z_curr = x0
        for i in range(max_iter):
            #update momentum parameters
            self._update()
            x_prev = x_curr
            z_prev = z_curr
            #compute new z_k and x_k
            z_curr = self.prox(x_prev - 1/self.Lf*self.grad(x_prev))
            x_curr = z_curr + self.momentum * (z_curr - z_prev)
        return x_curr
