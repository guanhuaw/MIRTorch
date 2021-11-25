import torch

class CG_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, A, max_iter, tol, alert, x0):
        ctx.save_for_backward(b)
        ctx.A = A
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.alert = alert
        return cg_block(x0, b, A, tol, max_iter, alert)

    @staticmethod
    def backward(ctx, dx):
        b = ctx.saved_tensors[0]
        # a better initialization?
        return cg_block(b, dx, ctx.A, ctx.tol, ctx.max_iter, ctx.alert), None, None, None, None, None


def cg_block(x0, b, A, tol, max_iter, alert):
    # solver for PSD Ax = b
    r0 = b - A * x0
    p0 = r0
    rk = r0
    pk = p0
    xk = x0
    rktrk = torch.sum(rk.conj() * rk).abs()
    num_loop = 0
    while torch.norm(rk.abs()).item() > tol and num_loop < max_iter:
        pktapk = torch.sum(pk.conj() * (A * pk)).abs()
        alpha = rktrk / pktapk
        xk1 = xk + alpha * pk
        rk1 = rk - alpha * A * pk
        rk1trk1 = torch.square(torch.norm(rk1))
        beta = rk1trk1 / rktrk
        pk1 = rk1 + beta * pk
        del xk
        del rk
        del pk
        xk = xk1
        rk = rk1
        pk = pk1
        rktrk = rk1trk1
        num_loop = num_loop + 1
    if alert:
        print(f'residual at {num_loop}th iter: {rktrk}')
    return xk


class CG():
    '''
    Solve the following equation: Ax = b, where A is PSD.
    Init:
        A: PSD matrix (Linear Map)
        b = RHS
        Tol: exiting tolerance
        max_iter: max number of iterations
    # TODO: check if A is PSD
    '''

    def __init__(self, A, max_iter=20, tol=1e-2, alert=False):
        self.solver = CG_func.apply
        self.A = A
        self.max_iter = max_iter
        self.tol = tol
        self.solver = CG_func.apply
        self.alert = alert

    def run(self, x0, b):
        assert list(self.A.size_out) == list(b.shape), "The size of A and b do not match."
        return self.solver(b, self.A, self.max_iter, self.tol, self.alert, x0)
