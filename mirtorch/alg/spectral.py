import logging
import math

import torch

from mirtorch.util import l2_norm

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
        A tuple containing the principal right singular vector and the
        estimated spectral norm.  The estimate is not a certified upper
        bound; step-size rules should use backtracking or a safety margin.
    """

    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    tol = float(tol)
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    initial_norm = l2_norm(x0)
    if initial_norm.item() == 0:
        raise ValueError("x0 must be nonzero")

    x = x0 / initial_norm
    for iteration in range(max_iter):
        Ax = A * x
        ratio = l2_norm(Ax)
        normal_x = A.adjoint(Ax)
        normal_norm = l2_norm(normal_x)
        scale = normal_norm.clamp_min(torch.finfo(normal_norm.dtype).tiny)
        next_x = torch.where(normal_norm > 0, normal_x / scale, x)

        # A stable eigen-residual is more informative than comparing two
        # successive singular-value estimates, especially for clustered spectra.
        if tol > 0:
            eigen_residual = l2_norm(normal_x - ratio.square() * x)
            relative_residual = eigen_residual / scale
        else:
            relative_residual = None
        if relative_residual is not None and relative_residual.item() < tol:
            if alert:
                logger.info(
                    "The calculation of max singular value accomplished at %d iterations.",
                    iteration + 1,
                )
            break
        x = next_x
    sig1 = l2_norm(A * x)
    if alert:
        logger.info("The spectral norm is %s.", float(sig1))
    return x, sig1
