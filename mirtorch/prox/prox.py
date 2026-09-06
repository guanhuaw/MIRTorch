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


def _validate_regularization_parameter(value) -> float:
    """Convert a non-negative regularization parameter to ``float``."""
    parameter = float(value)
    if not math.isfinite(parameter):
        raise ValueError(f"Lambda must be finite, got {value}.")
    if parameter < 0:
        raise ValueError(f"Lambda should be non-negative, the Lambda here is {value}.")
    return parameter


def _solve_weighted_l2_radius(
    value_squared: torch.Tensor,
    weights_squared: torch.Tensor,
    strength: torch.Tensor,
) -> torch.Tensor:
    r"""Solve ``||w*v / (r + strength*w**2)|| = 1`` for ``r = ||w*x||``.

    Bisection finds the positive root in ``[0, ||w*v||]``, including at zero
    strength. Two differentiable Newton refinements recover first and second
    derivatives of the implicit root without retaining the bisection graph.
    """
    weight2 = weights_squared.detach()
    value2 = value_squared.detach()
    target = strength.detach()
    low = torch.zeros_like(target)
    high = torch.sqrt(torch.sum(weight2 * value2))
    for _ in range(torch.finfo(value2.dtype).bits):
        midpoint = (low + high) / 2
        norm_squared = torch.sum(
            weight2 * value2 / (midpoint + target * weight2).square()
        )
        root_is_higher = norm_squared > 1
        low = torch.where(root_is_higher, midpoint, low)
        high = torch.where(root_is_higher, high, midpoint)

    radius = ((low + high) / 2).detach()
    for _ in range(2):
        denominator = radius + strength * weights_squared
        numerator = weights_squared * value_squared
        residual = torch.sum(numerator / denominator.square()) - 1
        slope = 2 * torch.sum(numerator / denominator.pow(3))
        radius = radius + residual / slope
    return radius


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

        strength = (
            torch.as_tensor(alpha, dtype=v.real.dtype, device=v.device) * parameter
        )
        if strength.numel() != 1:
            raise ValueError("alpha must be a scalar")
        return strength


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
        threshold = strength if self.P is None else strength * self._diagonal_weights(v)
        magnitude = v.abs()
        active = magnitude > threshold
        denominator = torch.where(active, magnitude, torch.ones_like(magnitude))
        scale = torch.where(
            active, 1 - threshold / denominator, (threshold == 0).to(magnitude.dtype)
        )
        return scale * v


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
        return torch.where(
            (v.abs() > threshold) | (threshold == 0), v, torch.zeros_like(v)
        )


class L2Regularizer(Prox):
    r"""
    Proximal operator for L2 regularizer

    .. math::

        arg \min_x \frac{1}{2} \| x - v \|_2^2 + \alpha \lambda \| PTx \|_2

    Attributes:
        Lambda: float, regularization parameter.
        P: LinearMap, optional, diagonal LinearMap
        T: LinearMap, optional, unitary LinearMap

    First and second derivatives are supported away from shrinkage boundaries.
    At zero step size, step derivatives are right derivatives when the weighted
    input norm is nonzero; no joint derivative exists at a zero-norm boundary.
    Higher-order derivatives of the weighted root solve are not guaranteed.
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
            active = norm > strength
            denominator = torch.where(active, norm, torch.ones_like(norm))
            scale = torch.where(
                active, 1 - strength / denominator, (strength == 0).to(norm.dtype)
            )
            return scale * v

        diagonal = self._diagonal_entries(v)
        weights_squared = diagonal.real.square()
        if diagonal.is_complex():
            weights_squared = weights_squared + diagonal.imag.square()
        value_squared = v.real.square()
        if v.is_complex():
            value_squared = value_squared + v.imag.square()
        positive = weights_squared > 0
        safe_weights_squared = torch.where(
            positive, weights_squared, torch.ones_like(weights_squared)
        )
        inverse_weighted_norm = torch.sqrt(
            torch.sum(
                torch.where(
                    positive,
                    value_squared / safe_weights_squared,
                    torch.zeros_like(v.real),
                )
            )
        )
        zero_solution = inverse_weighted_norm <= strength
        has_root = ~zero_solution

        # Keep the unselected solver branch finite in the zero-solution
        # region. The synthetic values never affect the result.
        solver_value_squared = torch.where(
            has_root, value_squared, torch.ones_like(value_squared)
        )
        solver_weights_squared = torch.where(
            has_root, weights_squared, torch.ones_like(weights_squared)
        )
        solver_strength = torch.where(
            has_root, strength, torch.full_like(strength, 0.5)
        )
        radius = _solve_weighted_l2_radius(
            solver_value_squared,
            solver_weights_squared,
            solver_strength,
        )
        result = v * (radius / (radius + strength * weights_squared))
        weighted_zero = torch.where(positive, torch.zeros_like(v), v)
        result = torch.where(zero_solution, weighted_zero, result)
        # At the joint zero-norm/zero-strength boundary, keep the identity
        # derivative with respect to v for a fixed zero step.
        return torch.where((strength == 0) & zero_solution, v, result)


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
        if self.P is None:
            return torch.div(v, 1 + 2 * strength)
        diagonal = self._diagonal_entries(v)
        weights_squared = diagonal.real.square()
        if diagonal.is_complex():
            weights_squared = weights_squared + diagonal.imag.square()
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

    For complex inputs, the bounds apply to magnitudes and the input phase is
    preserved. At zero, a nonzero projected magnitude uses the positive real
    direction, where the phase is otherwise undefined.

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
        value = v.abs() if v.is_complex() else v
        low = torch.as_tensor(self.l, dtype=value.dtype, device=value.device)
        high = torch.as_tensor(self.u, dtype=value.dtype, device=value.device)
        if self.P is not None:
            diagonal = self._diagonal_entries(value)
            if diagonal.is_complex():
                if bool((diagonal.imag != 0).any().item()):
                    raise TypeError(
                        "P must have real diagonal entries for a box constraint"
                    )
                diagonal = diagonal.real

            zero = diagonal == 0
            zero_is_in_box = (low <= 0) & (high >= 0)
            if bool((zero & ~zero_is_in_box).any().item()):
                raise ValueError(
                    "BoxConstraint is infeasible where P has a zero diagonal "
                    "entry and the interval does not contain zero"
                )

            safe_diagonal = torch.where(zero, torch.ones_like(diagonal), diagonal)
            first_bound = low / safe_diagonal
            second_bound = high / safe_diagonal
            low = torch.minimum(first_bound, second_bound)
            high = torch.maximum(first_bound, second_bound)
            low = torch.where(zero, -torch.inf, low)
            high = torch.where(zero, torch.inf, high)

        projected = torch.clamp(value, min=low, max=high)
        if not v.is_complex():
            return projected

        nonzero = value != 0
        denominator = torch.where(nonzero, value, torch.ones_like(value))
        # A disk projection is locally the identity at its origin. Restoring
        # the phase as v / |v| would lose this derivative even with a safe divide.
        at_origin = projected.to(v.dtype) + v * ((low <= 0) & (high > 0))
        return torch.where(nonzero, v * (projected / denominator), at_origin)


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
