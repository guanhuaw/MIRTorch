"""Proximal operators, such as soft-thresholding, box-constraint and L2 norm.

Prox() class includes the common proximal operators used in iterative optimization.
2021-02. Neel Shah and Guanhua Wang, University of Michigan
"""

import math
from collections.abc import Sequence

import torch

from mirtorch.linear import LinearMap
from mirtorch.util import l2_norm

FloatLike = float | torch.Tensor
EPS = 1e-15


def _validate_regularization_parameter(value) -> float:
    """Convert a non-negative regularization parameter to ``float``."""
    parameter = float(value)
    if not math.isfinite(parameter):
        raise ValueError(f"Lambda must be finite, got {value}.")
    if parameter < 0:
        raise ValueError(f"Lambda should be non-negative, the Lambda here is {value}.")
    return parameter


def _solve_weighted_l2_scale(
    value: torch.Tensor,
    weights_squared: torch.Tensor,
    strength: torch.Tensor,
) -> torch.Tensor:
    r"""Solve ``||w*v / (s + w**2)|| = strength`` for ``s``.

    The proximal point is ``s*v / (s + w**2)``. Bisection uses the analytic
    upper bound ``||w*v|| / strength``; an implicit Newton correction gives
    the converged root its exact autograd derivative.
    """
    weight2 = weights_squared.detach()
    value2 = value.detach().abs().square()
    target = strength.detach()

    def weighted_norm(scale):
        return torch.sqrt(torch.sum(weight2 * value2 / (scale + weight2).square()))

    low = torch.zeros_like(target)
    high = torch.sqrt(torch.sum(weight2 * value2)) / target
    for _ in range(torch.finfo(value.dtype).bits):
        midpoint = (low + high) / 2
        root_is_higher = weighted_norm(midpoint) > target
        low = torch.where(root_is_higher, midpoint, low)
        high = torch.where(root_is_higher, high, midpoint)

    scale = ((low + high) / 2).detach()
    denominator = scale + weights_squared
    norm = torch.sqrt(
        torch.sum(weights_squared * value.abs().square() / denominator.square())
    )
    derivative = torch.sum(
        weights_squared * value.abs().square() / denominator.pow(3)
    ) / norm.clamp_min(torch.finfo(norm.dtype).tiny)
    derivative = derivative.detach().clamp_min(torch.finfo(derivative.dtype).tiny)
    return scale + (norm - strength) / derivative


class Prox:
    r"""
    Proximal operator base class
    Prox is currently supported to be called on a torch.Tensor
    The math definition is:

    .. math::

       Prox_f(v) = arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda  f(PTx)

    Attributes:
        T: LinearMap, optional, unitary LinearMap
        P: LinearMap, optional, diagonal matrix
        TODO: manually check if it is unitary or diagonal (maybe not so easy ...)
    """

    def __init__(self, T: LinearMap | None = None, P: LinearMap | None = None):
        self.T = T
        self.P = P

    def _apply(self, v: torch.Tensor, alpha: FloatLike):
        raise NotImplementedError

    def __call__(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.T is not None:
            v = self.T(v)

        if v.is_complex():
            out = self._complex(v) * self._apply(v.abs(), alpha)
        else:
            out = self._apply(v, alpha)

        if self.T is not None:
            out = self.T.H(out)
        return out

    def __repr__(self):
        return f"<{self.__class__.__name__} Prox>"

    def to(self, device: torch.device | str):
        """Move tensors and nested operators to a device in place."""

        def move(value):
            if isinstance(value, (torch.Tensor, LinearMap)):
                return value.to(device)
            if isinstance(value, Prox):
                return value.to(device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value

        for name, value in vars(self).items():
            setattr(self, name, move(value))
        return self

    def _diagonal_entries(self, v: torch.Tensor) -> torch.Tensor:
        """Return ``diag(P)`` for a documented diagonal weighting operator."""
        if self.P is None:
            return torch.ones_like(v)
        if list(v.shape) != list(self.P.size_in):
            raise ValueError(
                f"P expects shape {self.P.size_in}, but received {list(v.shape)}"
            )
        diagonal = self.P(
            torch.ones(
                self.P.size_in,
                dtype=v.dtype,
                device=v.device,
            )
        )
        if list(diagonal.shape) != list(v.shape):
            raise ValueError("P must be a square diagonal LinearMap")
        return diagonal

    def _diagonal_weights(self, v: torch.Tensor) -> torch.Tensor:
        """Return ``|diag(P)|`` for a documented diagonal weighting operator."""
        return self._diagonal_entries(v).abs()

    def _strength(self, v: torch.Tensor, parameter: float, alpha: FloatLike):
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != 1:
                raise ValueError("alpha must be a scalar")
            if alpha.is_complex():
                raise TypeError("alpha must be real")
            alpha_value = alpha.item()
        else:
            alpha_value = alpha
        if not math.isfinite(alpha_value):
            raise ValueError(f"alpha must be finite, got {alpha_value}.")
        if alpha_value < 0:
            raise ValueError(f"alpha should be non-negative, got {alpha}.")

        strength = torch.as_tensor(alpha, dtype=v.dtype, device=v.device) * parameter
        if strength.numel() != 1:
            raise ValueError("alpha must be a scalar")
        return strength

    def _complex(self, v) -> torch.Tensor:
        """
        Args:
            v: input tensor

        Returns:
            x: output proximal results
        """
        # To avoid the influence of noise
        # Without thresholding, numerical issues may happen for some unitary transform (wavelets)
        # TODO:"This is a temporary fix, we need to find a better solution."
        return v / v.abs().clamp_min(EPS)


class L1Regularizer(Prox):
    r"""
    Proximal operator for L1 regularizer, using soft threshold.

    .. math::

        arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda \| PTx \|_1


    Attributes:
        Lambda: floatm regularization parameter.
        P: LinearMap, optional, diagonal LinearMap
        T: LinearMap, optional, unitary LinearMap
    """

    def __init__(
        self,
        Lambda,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.Lambda = _validate_regularization_parameter(Lambda)

    def _apply(self, v, alpha) -> torch.Tensor:
        strength = self._strength(v, self.Lambda, alpha)
        threshold = strength * self._diagonal_weights(v)
        return torch.sign(v) * torch.clamp(v.abs() - threshold, min=0)


class L0Regularizer(Prox):
    r"""
    Proximal operator for L0 regularizer, using hard thresholding

    .. math::

        arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda \| PTx \|_0


    Attributes:
        Lambda: float, regularization parameter.
        P: LinearMap, optional, diagonal LinearMap
        T: LinearMap, optional, unitary LinearMap
    """

    def __init__(
        self,
        Lambda,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.Lambda = _validate_regularization_parameter(Lambda)

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        strength = self._strength(v, self.Lambda, alpha)
        threshold = torch.sqrt(2 * strength)
        if self.P is not None:
            threshold = threshold * (self._diagonal_weights(v) != 0)
        return torch.where(v.abs() > threshold, v, torch.zeros_like(v))


class L2Regularizer(Prox):
    r"""
    Proximal operator for L2 regularizer

    .. math::

        arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda \| PTx \|_2

    Attributes:
        Lambda: float, regularization parameter.
        P: LinearMap, optional, diagonal LinearMap
        T: LinearMap, optional, unitary LinearMap
    """

    def __init__(
        self,
        Lambda,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.Lambda = _validate_regularization_parameter(Lambda)

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        if self.Lambda == 0 or (not isinstance(alpha, torch.Tensor) and alpha == 0):
            return v

        strength = self._strength(v, self.Lambda, alpha)
        if self.P is None:
            norm = l2_norm(v)
            safe_norm = norm.clamp_min(torch.finfo(norm.dtype).tiny)
            scale = (1.0 - strength / safe_norm).clamp_min(0)
            return scale * v

        weights = self._diagonal_weights(v)
        weights_squared = weights.square()
        positive = weights > 0
        safe_weights_squared = torch.where(
            positive, weights_squared, torch.ones_like(weights_squared)
        )
        inverse_weighted_norm = torch.sqrt(
            torch.sum(
                torch.where(
                    positive,
                    v.abs().square() / safe_weights_squared,
                    torch.zeros_like(v),
                )
            )
        )
        zero_solution = inverse_weighted_norm <= strength
        has_root = (strength > 0) & ~zero_solution

        # Keep the unselected solver branch finite at alpha=0 and in the
        # zero-solution region. The synthetic values never affect the result.
        solver_value = torch.where(has_root, v, torch.ones_like(v))
        solver_weights_squared = torch.where(
            has_root, weights_squared, torch.ones_like(weights_squared)
        )
        solver_strength = torch.where(
            has_root, strength, torch.full_like(strength, 0.5)
        )
        scale = _solve_weighted_l2_scale(
            solver_value,
            solver_weights_squared,
            solver_strength,
        )
        result = v * scale / (scale + weights_squared)
        weighted_zero = torch.where(positive, torch.zeros_like(v), v)
        result = torch.where(zero_solution, weighted_zero, result)
        return torch.where(strength == 0, v, result)


class SquaredL2Regularizer(Prox):
    r"""
    Proximal operator for Squared L2 regularizer

    .. math::

        arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda \| PTx \|_2^2

    Attributes:
        Lambda: float, regularization parameter.
        P: LinearMap, optional, diagonal LinearMap
        T: LinearMap, optional, unitary LinearMap
    """

    def __init__(
        self,
        Lambda,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.Lambda = _validate_regularization_parameter(Lambda)

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        strength = self._strength(v, self.Lambda, alpha)
        weights_squared = self._diagonal_weights(v).square()
        return torch.div(v, 1 + 2 * strength * weights_squared)


class BoxConstraint(Prox):
    r"""
    Projection onto a box constraint.

    .. math::

        arg \min_{x:\ lower \leq PTx \leq upper}
        \frac{1}{2} \| x - v \|_2^2

    Scaling an indicator function by a positive step size or regularization
    parameter does not change its feasible set. Therefore ``alpha`` and
    ``Lambda`` do not change this projection.

    Attributes:
        Lambda: legacy regularization parameter retained for API compatibility
        lower: float, minimum value
        upper: float, maximum value
        T: LinearMap, optional, unitary LinearMap
        P: LinearMap, optional, real diagonal LinearMap
    """

    def __init__(
        self,
        Lambda,
        lower,
        upper,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.l = lower
        self.u = upper
        self.Lambda = _validate_regularization_parameter(Lambda)
        if self.l > self.u:
            raise ValueError("lower must not be greater than upper")

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.P is None:
            x = torch.clamp(v, self.l, self.u)
        else:
            diagonal = self._diagonal_entries(v)
            if diagonal.is_complex():
                if bool((diagonal.imag != 0).any().item()):
                    raise TypeError(
                        "P must have real diagonal entries for a box constraint"
                    )
                diagonal = diagonal.real

            lower = torch.as_tensor(self.l, dtype=v.dtype, device=v.device)
            upper = torch.as_tensor(self.u, dtype=v.dtype, device=v.device)
            zero = diagonal == 0
            zero_is_in_box = (lower <= 0) & (upper >= 0)
            if bool((zero & ~zero_is_in_box).any().item()):
                raise ValueError(
                    "BoxConstraint is infeasible where P has a zero diagonal "
                    "entry and the interval does not contain zero"
                )

            safe_diagonal = torch.where(zero, torch.ones_like(diagonal), diagonal)
            first_bound = lower / safe_diagonal
            second_bound = upper / safe_diagonal
            low = torch.minimum(first_bound, second_bound)
            high = torch.maximum(first_bound, second_bound)
            projected = torch.maximum(low, torch.minimum(v, high))
            x = torch.where(zero, v, projected)
        return x


class Stack(Prox):
    r"""
    Stack proximal operators.

    Attributes:
        proxs: list of proximal operators, required to have equal input and output shapes
    """

    def __init__(self, proxs):
        if not proxs:
            raise ValueError("At least one proximal operator is required")
        self.proxs = proxs
        super().__init__()

    def __call__(self, v, alphas, sizes=None) -> torch.Tensor:
        return self._apply(v, alphas, sizes)

    def _apply(self, v, alpha, sizes=None) -> torch.Tensor:
        alphas = alpha
        if sizes is None:
            if v.shape[0] % len(self.proxs):
                raise ValueError(
                    "The leading dimension must be divisible by the number of "
                    "proximal operators when sizes is omitted"
                )
            section_size = v.shape[0] // len(self.proxs)
            sizes = [section_size] * len(self.proxs)
        splits = torch.split(v, sizes, dim=0)
        if len(splits) != len(self.proxs):
            raise ValueError("sizes must define one section per proximal operator")
        if isinstance(alphas, torch.Tensor):
            if alphas.ndim == 0:
                alphas = alphas.expand(len(self.proxs))
        elif not isinstance(alphas, Sequence):
            alphas = [alphas] * len(self.proxs)
        if len(alphas) != len(self.proxs):
            raise ValueError("alphas must contain one value per proximal operator")
        seq = [self.proxs[i](splits[i], alphas[i]) for i in range(len(self.proxs))]
        return torch.cat(seq)


class Const(Prox):
    r"""
    Proximal operator a constant function, identical to an identity mapping

    .. math::

       arg \min_{x}  \frac{1}{2} \| x - v \|_2^2 + C

    Attributes:
        Lambda (float): regularization parameter.
        T (LinearMap): optional, unitary LinearMap
    """

    def __init__(
        self,
        Lambda=0,
        T: LinearMap | None = None,
        P: LinearMap | None = None,
    ):
        super().__init__(T, P)
        self.Lambda = float(Lambda)

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        return v


class Conj(Prox):
    r"""
    Proximal operator of the convex conjugate (Moreau's identity).

    .. math::

        Prox_{\alpha f^*}(v) = v - \alpha Prox_{frac{1}{\alpha} f}(\frac{1}{\alpha} v)

    Attributes:
        prox (Prox): Proximal operator function
    """

    def __init__(self, prox: Prox):
        self.prox = prox
        super().__init__()

    def _apply(self, v, alpha) -> torch.Tensor:
        if alpha <= 0:
            raise ValueError(f"alpha should be positive, the alpha here is {alpha}.")
        return v - alpha * self.prox(v / alpha, 1 / alpha)
