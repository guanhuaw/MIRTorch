"""Small helpers for conservative automatic ``torch.compile`` use."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

import torch
from torch import Tensor

P = ParamSpec("P")
R = TypeVar("R")


def is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    check = getattr(compiler, "is_compiling", None)
    return bool(check is not None and check())


def should_compile(enabled: bool, tensor: Tensor) -> bool:
    return (
        enabled
        and tensor.device.type == "cuda"
        and not tensor.is_complex()
        and hasattr(torch, "compile")
        and not is_compiling()
    )


def compile_callable(function: Callable[P, R]) -> Callable[P, R]:
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return function
    return cast(Callable[P, R], compiler(function))
