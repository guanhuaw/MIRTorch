import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def power_iter(A, x0, max_iter=100, tol=1e-6, alert=True):
    r"""
    Use power iteration to calculate the spectral norm of a LinearMap.

    Args:
        A: a LinearMap
        max_iter: maximum number of iterations
        tol: stopping tolerance
        x0: initial guess of singular vector corresponding to max singular value

    Returns:
        The spectral norm (x) and the principal right singular vector (sig1)
    """

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if torch.linalg.vector_norm(x0) == 0:
        raise ValueError("x0 must be nonzero")

    x = x0
    ratio_old = float("inf")
    for iteration in range(max_iter):
        Ax = A * x
        ratio = torch.norm(Ax) / torch.norm(x)
        if torch.abs(ratio - ratio_old) / ratio < tol:
            if alert:
                logger.info(
                    "The calculation of max singular value accomplished at %d iterations.",
                    iteration + 1,
                )
            break
        ratio_old = ratio
        x = A.adjoint(Ax)
        x = x / torch.norm(x)
    sig1 = torch.norm(A * x) / torch.norm(x)
    if alert:
        logger.info("The spectral norm is %s.", float(sig1))
    return x, sig1
