"""Shared utilities for backend-portable tensor operations."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

import torch
from torch import Tensor

__all__ = [
    "compile_callable",
    "is_compiling",
    "l2_norm",
    "should_compile",
    "squared_l2_norm",
]

P = ParamSpec("P")
R = TypeVar("R")


def squared_l2_norm(value: Tensor) -> Tensor:
    """Return the smooth squared Euclidean norm without a complex norm kernel."""
    result = value.real.square().sum()
    return result + value.imag.square().sum() if value.is_complex() else result


def l2_norm(value: Tensor) -> Tensor:
    """Return the Euclidean norm, choosing the zero subgradient at the origin.

    The norm itself is not differentiable at zero. Masking before the square
    root keeps that convention finite in inactive proximal-operator branches.
    """
    squared = squared_l2_norm(value)
    positive = squared > 0
    safe_squared = torch.where(positive, squared, torch.ones_like(squared))
    return torch.where(positive, safe_squared.sqrt(), torch.zeros_like(squared))


def is_compiling() -> bool:
    """Return whether execution is currently being traced by ``torch.compile``."""
    compiler = getattr(torch, "compiler", None)
    check = getattr(compiler, "is_compiling", None)
    return bool(check is not None and check())


def should_compile(enabled: bool, tensor: Tensor) -> bool:
    """Return whether an operation should use the automatic compiled path."""
    return (
        enabled
        and tensor.device.type == "cuda"
        and not tensor.is_complex()
        and hasattr(torch, "compile")
        and not is_compiling()
    )


def compile_callable(function: Callable[P, R]) -> Callable[P, R]:
    """Compile a callable when the installed PyTorch exposes the API."""
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return function
    return cast(Callable[P, R], compiler(function))
