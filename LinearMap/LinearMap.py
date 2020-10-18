"""
Linear Operator implementations.
"""
import torch


class Linearmap():
    '''
        Our major difference with Sigpy is that:
         In sigpy, for each linear op, they IMPLEMENTED it case by case. So each instance inherit Linop()
         In out, we directly call the forward and backward ops. Which is provided by 3rd package.

         Alternative: you can try using the nn.module as the base class. It also support manual forward() and backward()
    '''
    def __init__(self, size_in, size_out, forward_op, adjoint_op):
        '''
        For initilization, this can be provided:
        size_in: the size of the input of the linear map
        size_out: ...
        '''
        self.size_in = size_in          # size_in: input data dimention
        self.size_out = size_out        # size_out: output data dimention
        self.forward_op = forward_op    # forward function i.e. fft
        self.adjoint_op = adjoint_op    # adjoint function i.e. ifft

        class forward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return self.forward_op(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return self.adjoint_op(grad_data_in)
        self._apply = forward.apply # This may look wired to you. But the torch.autograd. Function requires .apply

        class adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data_in):
                return self.adjoint_op(data_in)
            @staticmethod
            def backward(ctx, grad_data_in):
                return self.forward_op(grad_data_in)
        self._adjoint_apply = adjoint.apply
        
    # Make sure that the input and output and forwardop, adjop are on the same divice (CPU/CPU)
    # ? input & output device
    def check_device(self):
        return self.forward_op.device == self.adjoint_op.device

    # # ?
    # def __call__(self):
    #     pass


    # # TODO add class
    # # you can try the sigpy approach or your own one
    # def __add__(self, other):
    #     with other.device:
    #         self.forward_op += other.forward_op
    #         self.adjoint_op += adjoint_op

    # def __mul__(self, other):
    #     pass

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
        # check shape
        assert(self.size_in == other.size_in)
        assert(self.size_out == other.size_out)
        self.other = other

        # ? How to define forward and adjoint op here
        super().__init__(self.size_in, self.size_out, forward_op, adjoint_op)

    def apply(self, input_):
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

    def apply(self, other):
        with other.device:
            output = Linearmap(self.size_in, # !
                               self.size_out, # !
                               torch.matmul(self.forward_op, other.forward_op),
                               self.adjoint_op + other.adjoint_op) # ???
        return output