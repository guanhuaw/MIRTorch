import torch
import logging
from torch import Tensor

logger = logging.getLogger(__name__)


class CG_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b: Tensor, A, max_iter, tol, alert, x0, eval_func, P):
        ctx.save_for_backward(b)
        ctx.A = A
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.alert = alert
        ctx.eval_func = eval_func
        ctx.P = P
        return cg_block(x0, b, A, tol, max_iter, alert, eval_func, P)

    @staticmethod
    def backward(ctx, dx):
        b = ctx.saved_tensors[0]
        # a better initialization?
        return (
            cg_block(
                b, dx, ctx.A, ctx.tol, ctx.max_iter, ctx.alert, ctx.eval_func, ctx.P
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def cg_block(x0, b, A, tol, max_iter, alert, eval_func, P):
    # solver for PSD Ax = b
    if P is None:
        r0 = b - A * x0
        rk = r0
        p0 = r0.detach().clone()
        pk = p0
        xk = x0.detach().clone()
        rktrk = torch.square(torch.norm(rk))
        num_loop = 0
        if eval_func is not None:
            saved = []
        while rktrk.item() > tol and num_loop < max_iter:
            pktapk = torch.sum(pk.conj() * (A * pk)).abs()
            alpha = rktrk / pktapk
            xk1 = xk.add_(alpha * pk)
            rk1 = rk.sub_(alpha * A * pk)
            rk1trk1 = torch.square(torch.norm(rk1))
            beta = rk1trk1 / rktrk
            pk1 = (pk.mul_(beta)).add_(rk1)
            xk = xk1
            rk = rk1
            pk = pk1
            rktrk = rk1trk1
            num_loop = num_loop + 1
            if eval_func is not None:
                saved.append(eval_func(rk))
            if alert:
                logger.info(
                    "Residual at %dth iter in forward CG: %10.3e." % (num_loop, rktrk)
                )
    else:
        r0 = b - A * x0
        rk = r0
        zk = P * rk
        pk = zk.clone()
        xk = x0.detach().clone()
        rktzk = (rk.conj() * zk).sum().abs()
        num_loop = 0
        if eval_func is not None:
            saved = []
        while torch.square(torch.norm(rk)).item() > tol and num_loop < max_iter:
            pktapk = torch.sum(pk.conj() * (A * pk)).abs()
            alpha = rktzk / pktapk
            xk1 = xk.add_(alpha * pk)
            rk1 = rk.sub_(alpha * A * pk)
            zk1 = P * rk1
            rk1tzk1 = (rk1.conj() * zk1).sum().abs()
            beta = rk1tzk1 / rktzk
            pk1 = (pk.mul_(beta)).add_(zk1)
            xk = xk1
            rk = rk1
            pk = pk1
            rktzk = rk1tzk1
            num_loop = num_loop + 1
            if eval_func is not None:
                saved.append(eval_func(rk))
            if alert:
                logger.info(
                    "Residual at %dth iter in CG backpropagation: %10.3e."
                    % (num_loop, rktzk)
                )
                if torch.cuda.is_available():
                    logging.info(
                        "GPU memory usage at %dth iter in CG backpropagation: %10.3e."
                        % (num_loop, torch.cuda.memory_allocated() / 1024 / 1024 / 1024)
                    )

    if eval_func is not None:
        return xk, saved
    else:
        return xk


class CG:
    r"""
    Solve the equation :math:`Ax = b` with the conjugdate gradient (CG) method, where A is a positive semi-definite operator.
    The backpropagation still calls the CG to calculate the Jacobian to save the memory.

    Attributes:
        A: LinearMap of a PSD matrix
        tol: float, exiting tolerance
        max_iter: int, max number of iterations
        alert: bool, print the norm of residuals at the end
        eval_func: user-defined function to calculate the loss at each iteration.
        P: LinearMap of a Preconditioner

    Methods:
        run: run the CG algorithm
    """

    def __init__(self, A, max_iter=20, tol=1e-2, P=None, alert=False, eval_func=None):
        self.solver = CG_func.apply
        self.A = A
        self.max_iter = max_iter
        self.tol = tol
        self.solver = CG_func.apply
        self.alert = alert
        self.eval_func = eval_func
        self.P = P

    def run(self, x0, b):
        r"""Run the CG iterations.
        Args:
            x0: Initialization
            b: RHS

        Returns:
            xk: results
            saved: (optional) a list of intermediate results, calculated by the eval_func.
        """
        if list(self.A.size_out) != list(b.shape):
            raise ValueError("The size of A and b do not match.")
        return self.solver(
            b, self.A, self.max_iter, self.tol, self.alert, x0, self.eval_func, self.P
        )
