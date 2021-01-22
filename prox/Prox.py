import torch


class Prox:
    r"""
    Prox(v)imal operator base class

    Prox(v) is currently supported to be called on a torch.Tensor

    .. math::
       Prox(v)_f(v) \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + f(x(v))

    Args:

    """

    def __init__(self):
        #sigpy has size/shape input parameter, but I don't see why we would need it?: Maybe for now we do not need it: we do not.
        pass

    def __call__(self, v): 
        #sigpy also has alpha value, maybe add that here after implementing basic functionality
        return self._apply(v)

    def __repr__(self):
        return NotImplementedError

class L1Regularizer(Prox):
    r"""
    Prox(v)imal operator for L1 regularizer, using soft thresholding

    .. math::
        \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + \lambda \| x(v) \|_1

    Args:
        Lambda (float): regularization parameter.
    """
    
    def __init__(self, Lambda):
        super().__init__()
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        dtype = v.dtype
        thresh = torch.nn.Softshrink(self.Lambda)
        if dtype == torch.cfloat or dtype == torch.cdouble: v = torch.view_as_real(v)
        x = thresh(v)
        if dtype == torch.cfloat or dtype == torch.cdouble: x = torch.view_as_complex(x)
        return x
        #pseudo code here:
        # if dtype == torch.cfloat or dtype == torch.cdouble:
            # v_abs = torch.abs(v)
            # v_phs = torch.angle(v)
            # return torch.nn.functional.gumbel_softmax(v_abs, lambd = self.Lambda)*(torch.exp(1j*v_phs))
    # to do: add the repr
    # Consider this case:
    # \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + \lambda \|P x(v) \|_1
    # where P is orthogonal or diagonal, labeled in the property of LinearMap class


class L2Regularizer(Prox):
    r"""
    Prox(v)imal operator for L2 regularizer

    .. math::
        \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + \lambda \| x(v) \|_2

    Args:
        Lambda (float): regularization parameter.
    """
    def __init__(self, Lambda):
        super().__init__()
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf 
        dtype = v.dtype
        if dtype == torch.cfloat or dtype == torch.cdouble: v = torch.view_as_real(v)
        scale = 1.0 - self.Lambda / torch.max(torch.Tensor([self.Lambda]), torch.linalg.norm(v))
        if dtype == torch.cfloat or dtype == torch.cdouble: v = torch.view_as_complex(v)
        x = torch.mul(scale, v)
        return x

class SquaredL2Regularizer(Prox):

    r"""
    Prox(v)imal operator for Squared L2 regularizer

    .. math::
        \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + \lambda \| x(v) \|_2^2

    Args:
        Lambda (float): regularization parameter.
    """

    def __init__(self, Lambda):
        super().__init__()
        self.Lambda = float(Lambda)
    
    def _apply(self, v):
        x =  torch.div(v, 1 + 2*self.Lambda)
        return x

class BoxConstraint(Prox):

    r"""
    Prox(v)imal operator for Box(v) Constraint

    .. math::
        \arg \min_{x(v) \in [lower, upper]} \frac{1}{2} \| x(v) - v \|_2^2

    Args:
        Lambda (float): regularization parameter.
        lower (scalar): minimum value
        upper (scalar): max(v)imum value
    """

    def __init__(self, Lambda, lower, upper):
        super().__init__()
        self.l = lower
        self.u = upper
        self.Lambda = float(Lambda)
    
    def _apply(self, v):
        dtype = v.dtype
        if dtype == torch.cfloat or dtype == torch.cdouble: v = torch.view_as_real(v)
        x = torch.clamp(v, self.l, self.u)
        if dtype == torch.cfloat or dtype == torch.cdouble: x = torch.view_as_complex(x)
        return x
        
