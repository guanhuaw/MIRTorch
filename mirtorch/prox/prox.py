"""
Proximal operators, including soft-thresholding, box-constraint and L2 norm.
2021-02. Neel Shah and Guanhua Wang, University of Michigan
"""

import torch


class Prox:
    r"""
    Proximal operator base class

    Prox is currently supported to be called on a torch.Tensor

    .. math::
       Prox_f(v) = \arg \min_x \frac{1}{2} \| x - v \|_2^2 + f(PTx)

    Args:
        T (LinearMap): optional, unitary LinearMap
        P (LinearMap): optional, diagonal matrix
        TODO: manually check if it is unitary or diagonal
    """

    def __init__(self, T = None, P = None):
        # if T is not None:
        #     assert('unitary' in T.property)
        self.T = T

        if P is not None:
            assert('diagonal' in P.property)
        self.P = P


    def _apply(self, v, alpha):
        raise NotImplementedError

    def __call__(self, v, alpha):
        #sigpy also has alpha value, maybe add that here after implementing basic functionality

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
        return f'<{self.__class__.__name__} Prox'

    def _complex(self, v):
        angle = torch.atan2(v.imag, v.real)
        exp = torch.complex(torch.cos(angle), torch.sin(angle))
        return exp

class L1Regularizer(Prox):
    r"""
    Proximal operator for L1 regularizer, using soft thresholding

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| PTx \|_1

    Args:
        Lambda (float): regularization parameter.
        P (LinearMap): optional, diagonal LinearMap
        T (LinearMap): optional, unitary LinearMap
    """


    def __init__(self, Lambda, T = None, P = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        assert self.Lambda > 0, "alpha should be greater than 0"
        if P is not None:
            # Should this be v.shape instead of P.size_in? TODO: Verify this through test
            self.Lambda = P(Lambda*torch.ones(P.size_in))

    def _softshrink(self, x, lambd):
        mask1 = x > lambd
        mask2 = x < -lambd
        out = torch.zeros_like(x)
        out += mask1.float() * -lambd + mask1.float() * x
        out += mask2.float() * lambd + mask2.float() * x
        return out

    def _apply(self, v, alpha):
        assert alpha > 0, "alpha should be greater than 0"
        if type(self.Lambda) is not torch.Tensor and type(alpha) is not torch.Tensor:
            # The softshrink function do not support tensor as Lambda.
            thresh = torch.nn.Softshrink(self.Lambda*alpha)
            x = thresh(v)
        else:
            #print(type(self.Lambda))
            x = self._softshrink(v, (self.Lambda*alpha).to(v.device))
        return x

class L0Regularizer(Prox):
    r"""
    Proximal operator for L0 regularizer, using hard thresholding

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| PTx \|_0

    Args:
        Lambda (float): regularization parameter.
        P (LinearMap): optional, diagonal LinearMap
        T (LinearMap): optional, unitary LinearMap
    """


    def __init__(self, Lambda, T = None, P = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            # Should this be v.shape instead of P.size_in? TODO: Verify this through test
            self.Lambda = P(Lambda*torch.ones(P.size_in))

    def _hardshrink(x, lambd):
        mask1 = x > lambd
        mask2 = x < -lambd
        out = torch.zeros_like(x)
        out += mask1.float() * x
        out += mask2.float() * x
        return out

    def _apply(self, v, alpha):
        assert alpha > 0, "alpha should be greater than 0"
        if type(self.Lambda) is not torch.Tensor and type(alpha) is not torch.Tensor:
            # The hardthreshold function do not support tensor as Lambda.
            thresh = torch.nn.Hardshrink(self.Lambda*alpha)
            x = thresh(v)
        else:
            x = self._hardshrink(v, (self.Lambda*alpha).to(v.device))
        return x


class L2Regularizer(Prox):
    r"""
    Proximal operator for L2 regularizer

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| Tx \|_2

    Args:
        Lambda (float): regularization parameter.
        T (LinearMap): optional, unitary LinearMap
    """
    def __init__(self, Lambda, T = None, P = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            self.Lambda = P(Lambda*torch.ones(P.size_in))

    
    def _apply(self, v, alpha):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        if self.P is None:
            scale = 1.0 - self.Lambda*alpha / torch.max(torch.Tensor([self.Lambda*alpha]), torch.linalg.norm(v))
            # x = torch.mul(scale, v)
        else:
            scale = torch.ones_like(v) - self.Lambda*alpha / torch.max(torch.Tensor(self.Lambda*alpha), torch.linalg.norm(v))
        return scale * v

class SquaredL2Regularizer(Prox):
    r"""
    Proximal operator for Squared L2 regularizer

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| Tx \|_2^2

    Args:
        Lambda (float): regularization parameter.
        T (LinearMap): optional, unitary LinearMap
    """
    def __init__(self, Lambda, T = None, P = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if P is not None:
            self.Lambda = P(Lambda*torch.ones(P.size_in))


    def _apply(self, v, alpha):
        if self.P is None:
            x = torch.div(v, 1 + 2*self.Lambda*alpha)
        else:
            x = torch.div(v, torch.ones(v.shape) + 2*self.Lambda*alpha)
        return x

class BoxConstraint(Prox):
    r"""
    Proximal operator for Box Constraint

    .. math::
        \arg \min_{x} \in [lower, upper]} \frac{1}{2} \| x - v \|_2^2

    Args:
        Lambda (float): regularization parameter.
        lower (scalar): minimum value
        upper (scalar): maximum value
        T (LinearMap): optional, unitary LinearMap
    """

    def __init__(self, Lambda, lower, upper, T = None, P = None):
        super().__init__(T)
        self.l = lower
        self.u = upper
        self.Lambda = float(Lambda)


    def _apply(self, v, alpha):
        if self.P is None:
            x = torch.clamp(v, self.l, self.u)
        else:
            Lambda = self.P(self.Lambda*alpha * torch.ones(self.P.size_in))
            l = self.l / Lambda
            u = self.u / Lambda
            x = torch.minimum(u, torch.minimum(v, l))
        return x

