"""Shared result type for iterative solvers."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SolverResult:
    """Result and diagnostics from an iterative solver.

    Solvers keep their historical tensor return by default.  Passing
    ``return_info=True`` to ``run`` returns this object instead.

    Attributes:
        solution: Final primal iterate.
        iterations: Number of completed iterations.
        converged: Whether an enabled stopping criterion was satisfied.
        residual_norm: Final norm used by the stopping criterion, if available.
        history: Values produced by the solver's evaluation callback.
        state: Additional tensors that can warm-start a later solve.
    """

    solution: torch.Tensor
    iterations: int
    converged: bool
    residual_norm: torch.Tensor | None = None
    history: list[Any] = field(default_factory=list)
    state: dict[str, torch.Tensor] = field(default_factory=dict)


def validate_stopping_tolerances(rtol: float, atol: float) -> None:
    """Validate non-negative finite relative and absolute tolerances."""
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def stopping_is_enabled(rtol: float, atol: float) -> bool:
    """Return whether a relative-step stopping criterion is enabled."""
    return rtol > 0 or atol > 0
