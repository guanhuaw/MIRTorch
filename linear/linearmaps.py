"""
Linear Operator implementations.
"""
import torch
import abc
import os
import numpy as np

# To Do: frame operator

'''
 Recommendation for linear operation:
 class forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_in):
        return forward_func(data_in)
    @staticmethod
    def backward(ctx, grad_data_in):
        return adjoint_func(grad_data_in)
forward_op = forward.apply # This may look wired to you. But the torch.autograd. Function requires .apply

class adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_in):
        return forward_func(data_in)
    @staticmethod
    def backward(ctx, grad_data_in):
        return adjoint_func(grad_data_in)
adjoint_op = adjoint.apply
'''


def check_device(x, y):
    assert x.device == y.device, "tensors should be on the same device"


class LinearMap():
    '''
        We followed the idea of Sigpy rather than ModOpt:
        Each son class (like FFT, Wavelet ...) defines it own _apply and _apply_adjoint
        This approach lacks the versatility of define new linear operator on the run, but is easier to implemented
    '''

    def __init__(self, size_in, size_out, device='cuda:0'):
        '''
        For initilization, this can be provided:
        size_in: the size of the input of the linear map (a list)
        size_out: the size of the output of the linear map (a list)
        '''
        self.size_in = size_in  # size_in: input data dimention
        self.size_out = size_out  # size_out: output data dimention
        self.device = device

    def __repr__(self):
        # Name of the linear operator
        pass

    def __call__(self, x):
        # For a instance of LinearOP class, apply it by A(x). Equal to A*x
        return self._apply(x)

    def _apply(self, x):
        # Worth noting that the function here should be differentiable, for example, composed of native torch functions,
        # or torch.autograd.Function, or nn.module
        raise NotImplementedError

    def _apply_adjoint(self, x):
        raise NotImplementedError

    def apply(self, x):
        assert x.shape == self.size_in, "input size and linear op size does not match"
        return self._apply(x)
    def adjoint(self, x):
        assert x.shape == self.size_out, "input size and adjoint op size does not match"
        return self._apply_adjoint(x)
    @property
    def H(self):
        return ConjTranspose(self)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, LinearMap):
            return Matmul(self, other)
        elif isinstance(other, torch.Tensor):
            if not other.shape:
                raise ValueError(
                    "Input tensor has empty shape. If want to scale the linear map, please use the standard scalar")
            return self.apply(other)
        else:
            raise NotImplementedError("Only scalers, Linearmaps or Tensors are allowed as arguments for this function.")

    def __rmul__(self, other):
        if np.isscalar(other):
            return Multiply(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return -1 * self

    def __matmul__(self,other):
        pass

class Add(LinearMap):
    '''
    Addition of linear operators.
    (A+B)*x = A(x) + B(x)
    '''

    def __init__(self, A, B):
        assert list(A.size_in) == list(B.size_in), "The input dimentions of two combined ops are not the same."
        assert list(A.size_out) == list(B.size_out), "The output dimentions of two combined ops are not the same."
        self.A = A
        self.B = B
        super().__init__(self.A.size_in, self.B.size_out)

    def _apply(self, x):
        return self.A(x) + self.B(x)

    def _apply_adjoint(self, x):
        return self.A.H(x) + self.B.H(x)


class Multiply(LinearMap):
    '''
    Scaling linear operators
    a*A*x = A(ax)
    '''

    def __init__(self, A, a):
        self.a = a
        self.A = A
        super().__init__(self.A.size_in, self.A.size_out)

    def _apply(self, x):
        ax = x * self.a
        return self.A(ax)

    def _apply_adjoint(self, x):
        ax = x * self.a
        return self.A.H(ax)


class Matmul(LinearMap):
    '''
    Matrix multiplication of linear operators.
    A*B*x = A(B(x))
    '''

    def __init__(self, A, B):
        self.A = A
        self.B = B
        assert list(self.B.size_out) == list(self.A.size_in), "Shapes do not match"
        super().__init__(self.B.size_in, self.A.size_out)

    def _apply(self, x):
        return self.A(self.B(x))

    def _apply_adjoint(self, x):
        return self.B.H(self.A.H(x))


class ConjTranspose(LinearMap):
    def __init__(self, A):
        self.A = A
        super().__init__(A.size_out, A.size_in)

    def _apply(self, x):
        return self.A.adjoint(x)

    def _apply_adjoint(self, x):
        return self.A.apply(x)

