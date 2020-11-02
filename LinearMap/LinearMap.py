"""
Linear Operator implementations.
"""
import torch
import abc
import os
import numpy as np
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

class Linearmap():

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
        self.size_in = size_in              # size_in: input data dimention
        self.size_out = size_out            # size_out: output data dimention
        self.device = device
    def check_device(self, x, y):
        # TODO: check if the output and input are on the same device (Guanhua)
        pass
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

    def _apply_adjoint(self,x):
        raise NotImplementedError

    def apply(self,x):
        assert(x.size == self.size_in)
        self._apply(x)

    def adjoint(self,x):
        assert (x.size == self.size_out)
        self._apply_adjoint(self,x)

    def H(self,x):
        return Transpose(self)

    # TODO: Reload the operator
    def __add__(self, other):
        return Add(self,other)

    def __mul__(self, other):
        if np.isscalar(other):
        elif isinstance(other, Linearmap):
        elif isinstance(other, torch.tensor):
        else:


    # def __sub__(self, other):
    #     return self.__add__(-other)

    # def __neg__(self):
    #     return -1 * self
        
    # @property
    # def H(self):
    #     pass

    # def _combine_compose_linops(self, linops):
    #     pass

class Add(Linearmap):
    '''
    Addition of linear operators.
    '''
    def __init__(self, other):
        # check shape/device: TODO: change to try catch

        assert(self.size_in == other.size_in)
        assert(self.size_out == other.size_out)
        self.other = other

        # ? How to define forward and adjoint op here
        super().__init__(self.size_in, self.size_out)

    def _apply(self, input_):
        output = 0
        with input_.device:
            output = self.other.forward_op(self.forward_op(input_))
        return output

class Matmul(Linearmap):
    '''
    Matrix multiplication of linear operators.
    '''
    def __init__(self, other):
        # check shape
        assert(self.size_out == other.size_in)
        super().__init__(self.size_in, other.size_out, forward_op, adjoint_op)

    def _apply(self, other):
        with other.device:
            output = Linearmap(self.size_in, # !
                               self.size_out, # !
                               torch.matmul(self.forward_op, other.forward_op),
                               self.adjoint_op + other.adjoint_op) # ???
        return output