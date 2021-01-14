import torch


# Neel, here some some suggestions:
# 1. Add a __call__ to each operator, just like linearmap
# 2. Add the complex(v) support. Some of the functions may not support the complex(v) ops,
# then you may want to do with the real(v)/imag component respectively.

class Prox:
    r"""
    Prox(v)imal operator base class

    Prox(v) is currently supported to be called on a torch.Tensor

    .. math::
       Prox(v)_f(v) \arg \min_x(v) \frac{1}{2} \| x(v) - v \|_2^2 + f(x(v))

    Args:

    """

    def __init__(self):
        #sigpy has size/shape input parameter, but I don't see why we would need it?: Maybe for now we do not need it
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
        
