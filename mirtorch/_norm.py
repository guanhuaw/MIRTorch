"""Norm helpers that work for complex tensors on every PyTorch backend."""

import torch
from torch import Tensor


def squared_l2_norm(value: Tensor) -> Tensor:
    """Return the squared Euclidean norm without a complex norm kernel."""
    return torch.sum(value.abs().square())


def l2_norm(value: Tensor) -> Tensor:
    """Return the Euclidean norm without a complex norm kernel."""
    return squared_l2_norm(value).sqrt()
