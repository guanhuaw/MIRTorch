"""Proximal operators, such as soft-thresholding, box-constraint and L2 norm.

Prox() class includes the common proximal operators used in iterative optimization.
2021-02. Neel Shah and Guanhua Wang, University of Michigan
"""

from mirtorch.linear import LinearMap
import torch
from typing import Union

FloatLike = Union[float, torch.FloatTensor]
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

    def __init__(self, T: LinearMap = None, P: LinearMap = None):
        self.T = T
        self.P = P

    def _apply(self, v: torch.Tensor, alpha: FloatLike):
        raise NotImplementedError

    def __call__(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.T is not None:
            v = self.T(v)

        if v.dtype == torch.cfloat or v.dtype == torch.cdouble:
            out = self._complex(v) * self._apply(v.abs(), alpha)
        else:
            out = self._apply(v, alpha)

        if self.T is not None:
            out = self.T.H(out)
        return out

    def __repr__(self):
        return f"<{self.__class__.__name__} Prox"

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
        angle = torch.zeros_like(v)
        msk = torch.abs(v) > EPS
        angle[msk] = v[msk] / torch.abs(v)[msk]
        angle[~msk] = v[~msk] / EPS
        return angle


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

    def __init__(self, Lambda, T: LinearMap = None, P: LinearMap = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if self.Lambda < 0:
            raise ValueError(
                f"Lambda should be non-negative, the Lambda here is {Lambda}."
            )
        if P is not None:
            self.Lambda = P(Lambda * torch.ones(P.size_in))

    def _softshrink(self, x, lambd) -> torch.Tensor:
        mask1 = x > lambd
        mask2 = x < -lambd
        return (
            mask1.float() * (-lambd)
            + mask1.float() * x
            + mask2.float() * lambd
            + mask2.float() * x
        )

    def _apply(self, v, alpha) -> torch.Tensor:
        if alpha < 0:
            raise ValueError(
                f"alpha should be non-negative, the alpha here is {alpha}."
            )
        if type(self.Lambda) is not torch.Tensor and type(alpha) is not torch.Tensor:
            # The softshrink function do not support tensor as Lambda.
            thresh = torch.nn.Softshrink(self.Lambda * alpha)
            x = thresh(v)
        else:
            x = self._softshrink(v, (self.Lambda * alpha).to(v.device))
        return x


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

    def __init__(self, Lambda, T: LinearMap = None, P: LinearMap = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            self.Lambda = P(Lambda * torch.ones(P.size_in))

    def _hardshrink(self, x: torch.Tensor, lambd: FloatLike) -> torch.Tensor:
        mask1 = x > lambd
        mask2 = x < -lambd
        return mask1 * x + mask2 * x

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        assert alpha >= 0, f"alpha should be greater than 0, the alpha here is {alpha}."
        if type(self.Lambda) is not torch.Tensor and type(alpha) is not torch.Tensor:
            # The hardthreshold function do not support tensor as Lambda.
            thresh = torch.nn.Hardshrink(self.Lambda * alpha)
            x = thresh(v)
        else:
            x = self._hardshrink(v, (self.Lambda * alpha).to(v.device))
        return x


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

    def __init__(self, Lambda, T: LinearMap = None, P: LinearMap = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            self.Lambda = P(Lambda * torch.ones(P.size_in))

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        if self.P is None:
            scale = 1.0 - self.Lambda * alpha / torch.max(
                torch.Tensor([self.Lambda * alpha]), torch.linalg.norm(v)
            )
        else:
            scale = torch.ones_like(v) - self.Lambda * alpha / torch.max(
                torch.Tensor(self.Lambda * alpha), torch.linalg.norm(v)
            )
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

    def __init__(self, Lambda, T: LinearMap = None, P: LinearMap = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            self.Lambda = P(Lambda * torch.ones(P.size_in))

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.P is None:
            x = torch.div(v, 1 + 2 * self.Lambda * alpha)
        else:
            x = torch.div(v, torch.ones(v.shape) + 2 * self.Lambda * alpha)
        return x


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

    def __init__(self, Lambda, lower, upper, T: LinearMap = None, P: LinearMap = None):
        super().__init__(T, P)
        self.l = lower
        self.u = upper
        self.Lambda = float(Lambda)

    def _apply(self, v: torch.Tensor, alpha: FloatLike) -> torch.Tensor:
        if self.P is None:
            x = torch.clamp(v, self.l, self.u)
        else:
            Lambda = self.P(self.Lambda * alpha * torch.ones(self.P.size_in))
            low = self.l / Lambda
            up = self.u / Lambda
            x = torch.minimum(up, torch.minimum(v, low))
        return x


class Stack(Prox):
    r"""
    Stack proximal operators.

    Attributes:
        proxs: list of proximal operators, required to have equal input and output shapes
    """

    def __init__(self, proxs):
        self.proxs = proxs
        super().__init__()

    def __call__(self, v, alphas, sizes=None) -> torch.Tensor:
        return self._apply(v, alphas, sizes)

    def _apply(self, v, alphas, sizes=None) -> torch.Tensor:
        if sizes is None:
            sizes = len(self.proxs)
        splits = torch.split(v, sizes)
        if not isinstance(alphas, torch.Tensor):
            alphas = [alphas] * sizes
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

    def __init__(self, Lambda=0, T: LinearMap = None, P: LinearMap = None):
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
        if alpha < 0:
            raise ValueError(
                f"alpha should be non-negative, the alpha here is {alpha}."
            )
        return v - alpha * self.prox(v / alpha, 1 / alpha)
