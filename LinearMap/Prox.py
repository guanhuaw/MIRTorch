import torch

class Prox():
    r"""
    Proximal operator base class

    Prox is currently supported to be called on a torch.Tensor

    .. math::
       Prox_f(v) \arg \min_x \frac{1}{2} \| x - v \|_2^2 + f(x)

    Args:
        device (None or torch.device): device of output tensor, default on same device as input
    """

    def __init__(self, device = None):
        self.device = device
        #sigpy has size/shape input parameter, but I don't see why we would need it?: Maybe for now we do not need it
        

    def __call__(self, v): 
        #sigpy also has alpha value, maybe add that here after implementing basic functionality
        return self._apply(v)

class L1Regularizer(Prox):
    r"""
    Proximal operator for L1 regularizer, using soft thresholding

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| x \|_1

    Args:
        Lambda (float): regularization parameter.
        device (None or torch.device): device of output tensor, default on same device as input
    """
    
    def __init__(self, Lambda, device = None):
        super().__init__(device)
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf

        #utilize torch.nn.Softshrink
        thresh = torch.nn.Softshrink(self.Lambda)

        x = thresh(v)

        if self.device is not None:
            x = x.to(self.device)

        return x




class L2Regularizer(Prox):
    r"""
    Proximal operator for L2 regularizer

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| x \|_2

    Args:
        Lambda (float): regularization parameter.
        device (None or torch.device): device of output tensor, default on same device as input
    """
    def __init__(self, Lambda, device=None):
        super().__init__(device)
        self.Lambda = float(Lambda)

    def _apply(self, v):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        # Again, pls add the math expression
        
        scale = 1 - self.Lambda / torch.max(v, torch.linalg.norm(v))

        x = torch.mul(scale, v)

        if self.device is not None:
            x = x.to(self.device)

        return x

class SqauredL2Regularizer(Prox):

    r"""
    Proximal operator for Squared L2 regularizer

    .. math::
        \arg \min_x \frac{1}{2} \| x - v \|_2^2 + \lambda \| x \|_2^2

    Args:
        Lambda (float): regularization parameter.
        device (None or torch.device): device of output tensor, default on same device as input
    """

    def __init__(self, Lambda, device=None):

        super().__init__(device)
        self.Lambda = float(Lambda)
    
    def _apply(self, v):

        x = torch.div(v, 1 + 2*self.Lambda)

        if self.device is not None:
            x = x.to(self.device)
        
        return x

class BoxConstraint(Prox):

    r"""
    Proximal operator for Box Constraint

    .. math::
        \arg \min_{x \in [lower, upper]} \frac{1}{2} \| x - v \|_2^2

    Args:
        Lambda (float): regularization parameter.
        device (None or torch.device): device of output tensor, default on same device as input
    """

    def __init__(self, lower, upper, device=None):
        
        super().__init__(device)
        self.l = lower
        self.u = upper
        self.Lambda = float(Lambda)
    
    def _apply(self, v):

        x = torch.clamp(v, self.l, self.u)

        if self.device is not None:
            x = x.to(self.device)
        
        return x
        
