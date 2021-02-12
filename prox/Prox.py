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
    """

    def __init__(self, T = None, P = None):
        if T is not None:
            assert('unitary' in T.property)
        self.T = T

        if P is not None:
            assert('diagonal' in P.property)
        self.P = P


    def _apply(self, v):
        raise NotImplementedError

    def __call__(self, v):
        #sigpy also has alpha value, maybe add that here after implementing basic functionality
        if v.dtype == torch.cfloat or v.dtype == torch.cdouble:
            return self._complex(v) * self._apply(v.abs())
        if self.T is not None:
            v = self.T(v)
        out = self._apply(v)
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
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| WTx \|_1

    Args:
        Lambda (float): regularization parameter.
        W (LinearMap): optional, diagonal LinearMap
        T (LinearMap): optional, unitary LinearMap
    """


    def __init__(self, Lambda, T = None, P = None):
        super().__init__(T, P)
        self.Lambda = float(Lambda)
        if self.P is not None:
            self.Lambda = P(Lambda*torch.ones(P.size_in)) # Pay attention here that Lambda is tensor, not value here
        self.T = T

    def _softshrink(x, lambd):
        mask1 = x > lambd
        mask2 = x < -lambd
        out = torch.zeros_like(x)
        out += mask1.float() * -lambd + mask1.float() * x
        out += mask2.float() * lambd + mask2.float() * x
        return out

    def _apply(self, v):
        if type(self.Lambda) is not torch.Tensor:
            thresh = torch.nn.Softshrink(self.Lambda) # Again, the softshrink function do not support tensor as Lambda
            x = thresh(v)
        else:
            x = self._softshrink(v, self.Lambda.to(v.device))
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
    def __init__(self, Lambda, T = None):
        super().__init__(T)
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        scale = 1.0 - self.Lambda / torch.max(torch.Tensor([self.Lambda]), torch.linalg.norm(v))
        x = torch.mul(scale, v)
        return x

class SquaredL2Regularizer(Prox):
    r"""
    Proximal operator for Squared L2 regularizer

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| Tx \|_2^2

    Args:
        Lambda (float): regularization parameter.
        T (LinearMap): optional, unitary LinearMap
    """

    def __init__(self, Lambda, T = None):
        super().__init__(T)
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # T here?
        x = torch.div(v, 1 + 2*self.Lambda)
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

    def __init__(self, Lambda, lower, upper, T = None):
        super().__init__(T)
        self.l = lower
        self.u = upper
        self.Lambda = float(Lambda)

    def _apply(self, v):
        x = torch.clamp(v, self.l, self.u)
        return x

