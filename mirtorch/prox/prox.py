"""Proximal operators, such as soft-thresholding, box-constraint and L2 norm.

Prox() class includes the common proximal operators used in iterative optimization.
2021-02. Neel Shah and Guanhua Wang, University of Michigan
"""

from collections.abc import Sequence

import torch

from mirtorch.linear import LinearMap

FloatLike = float | torch.Tensor
EPS = 1e-15


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

    def _scaled_parameter(self, v, parameter, alpha):
        if self.P is None:
            weighted = torch.as_tensor(parameter, dtype=v.dtype, device=v.device)
        else:
            weighted = self.P(
                torch.full(
                    self.P.size_in,
                    parameter,
                    dtype=v.dtype,
                    device=v.device,
                )
            )
        return weighted * torch.as_tensor(alpha, dtype=v.dtype, device=v.device)

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
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )

    def _apply(self, v, alpha) -> torch.Tensor:
        if alpha < 0:
            raise ValueError(
                f"alpha should be non-negative, the alpha here is {alpha}."
            )
        threshold = self._scaled_parameter(v, self.Lambda, alpha)
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
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if alpha < 0:
            raise ValueError(
                f"alpha should be non-negative, the alpha here is {alpha}."
            )
        threshold = self._scaled_parameter(v, self.Lambda, alpha)
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
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        threshold = self._scaled_parameter(v, self.Lambda, alpha)
        norm = torch.linalg.vector_norm(v)
        safe_norm = norm.clamp_min(torch.finfo(v.dtype).tiny)
        scale = (1.0 - threshold / safe_norm).clamp_min(0)
        return scale * v


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
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        threshold = self._scaled_parameter(v, self.Lambda, alpha)
        return torch.div(v, 1 + 2 * threshold)


class BoxConstraint(Prox):
    r"""
    Proximal operator for Box Constraint.

    .. math::

        arg \min_{x \in [lower, upper]} \frac{1}{2} \| x - v \|_2^2

    Attributes:
        Lambda: float, regularization parameter.
        lower: float, minimum value
        upper: float, maximum value
        T: LinearMap, optional, unitary LinearMap
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
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )
        if self.l > self.u:
            raise ValueError("lower must not be greater than upper")

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.P is None:
            x = torch.clamp(v, self.l, self.u)
        else:
            Lambda = self._scaled_parameter(v, self.Lambda, alpha)
            low = self.l / Lambda
            up = self.u / Lambda
            x = torch.maximum(low, torch.minimum(v, up))
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
