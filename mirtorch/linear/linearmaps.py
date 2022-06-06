r"""Abstraction of linear operators as matrices.

    The implementation follow the SigPy(https://github.com/mikgroup/sigpy)
    and LinearmapAA (https://github.com/JeffFessler/LinearMapsAA.jl).
    Recommendation for linear operation：
    (but you do not have to do this if the BP is efficient enough).
    class forward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, data_in):
            return forward_func(data_in)
        @staticmethod
        def backward(ctx, grad_data_in):
            return adjoint_func(grad_data_in)
    forward_op = forward.apply

    class adjoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, data_in):
            return forward_func(data_in)
        @staticmethod
        def backward(ctx, grad_data_in):
            return adjoint_func(grad_data_in)
    adjoint_op = adjoint.apply
    2021-02. Guanhua Wang and Keyue Zhu, University of Michigan
    TODO: extended assignments, unary operators
"""
import torch
from torch import Tensor
import numpy as np
from typing import Union, Sequence, TypeVar

FloatLike = Union[float, torch.FloatTensor]


def check_device(x, y):
    r"""
    check if two tensors are on the same device
    Returns:

    """
    assert x.device == y.device, "Tensors should be on the same device"


T = TypeVar('T', bound='LinearMap')


class LinearMap:
    r"""
    Linear Operator: :math: `y = A*x`

    Attributes:
        size_in: the size of the input of the linear map (a list)
        size_out: the size of the output of the linear map (a list)

    TODO: size_in and size_out could also be a tuple of list (consider wavelets)

    """

    def __init__(self,
                 size_in: Sequence[int],
                 size_out: Sequence[int],
                 device='cpu'):
        r"""
        Initiate the linear operator.
        """
        self.size_in = list(size_in)  # size_in: input data dimension
        self.size_out = list(size_out)  # size_out: output data dimension
        self.device = device  # some linear operators do not depend on devices, like FFT.
        # self.property = property  # properties like 'unitary', 'Toeplitz', 'frame' ...

    def __repr__(self):
        return '<{oshape}x{ishape} {repr_str} Linop>'.format(
            oshape=self.size_out, ishape=self.size_in, repr_str=self.__class__.__name__)

    def __call__(self, x) -> Tensor:
        # for a instance A, we can apply it by calling A(x). Equal to A*x
        return self.apply(x)

    def _apply(self, x) -> Tensor:
        # worth noting that the function here should be differentiable,
        # for example, composed of native torch functions,
        # or torch.autograd.Function, or nn.module
        raise NotImplementedError

    def _apply_adjoint(self, x) -> Tensor:
        raise NotImplementedError

    def apply(self, x) -> Tensor:
        r"""
        Apply the forward operator
        """
        assert list(x.shape) == list(
            self.size_in), f"Shape of input data {x.shape} and forward linear op {self.size_in} do not match!"
        return self._apply(x)

    def adjoint(self, x) -> Tensor:
        r"""
        Apply the adjoint operator
        """
        assert list(x.shape) == list(
            self.size_out), f"Shape of input data {x.shape} and adjoint linear op {self.size_in} do not match!"
        return self._apply_adjoint(x)

    @property
    def H(self) -> T:
        r"""
        Apply the (Hermitian) transpose
        """
        return ConjTranspose(self)

    def __add__(self: T, other: T) -> T:
        r"""
        Reload the + symbol.
        """
        return Add(self, other)

    def __mul__(self: T, other) -> T:
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, LinearMap):
            return Matmul(self, other)
        elif isinstance(other, torch.Tensor):
            if not other.shape:
                # raise ValueError(
                #     "Input tensor has empty shape. If want to scale the linear map, please use the standard scalar")
                return Multiply(self, other)
            return self.apply(other)
        else:
            raise NotImplementedError(
                f"Only scalers, Linearmaps or Tensors, rather than '{type(other)}' are allowed as arguments for this function.")

    def __rmul__(self: T, other) -> T:
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, torch.Tensor) and not other.shape:
            return Multiply(self, other)
        else:
            return NotImplemented

    def __sub__(self: T, other: T) -> T:
        r"""
        Reload the - symbol.
        """
        return self.__add__(-other)

    def __neg__(self: T) -> T:
        r"""
        Reload the - symbol.
        """
        return -1 * self

    def to(self: T, *args, **kwargs):
        r"""
        Copy to different devices
        """
        for prop in self.__dict__.keys():
            if (isinstance(self.__dict__[prop], torch.Tensor) or isinstance(self.__dict__[prop], torch.nn.Module)):
                self.__dict__[prop] = self.__dict__[prop].to(*args, **kwargs)


class Add(LinearMap):
    r"""
    Addition of linear operators.

    .. math::
         (A+B)*x = A(x) + B(x)

    Attributes:
        A: the LHS LinearMap
        B: the RHS LinearMap
    """

    def __init__(self, A: LinearMap, B: LinearMap):
        assert list(A.size_in) == list(B.size_in), "The input dimentions of two combined ops are not the same."
        assert list(A.size_out) == list(B.size_out), "The output dimentions of two combined ops are not the same."
        self.A = A
        self.B = B
        super().__init__(self.A.size_in, self.B.size_out)

    def _apply(self: T, x: Tensor) -> Tensor:
        return self.A(x) + self.B(x)

    def _apply_adjoint(self: T, x: Tensor) -> Tensor:
        return self.A.H(x) + self.B.H(x)


class Multiply(LinearMap):
    r"""
    Scaling linear operators.

    .. math::
        a*A*x = A(ax)

    Attributes:
        a: scaling factor
        A: LinearMap
    """

    def __init__(self, A: LinearMap, a: FloatLike):
        self.a = a
        self.A = A
        super().__init__(self.A.size_in, self.A.size_out)

    def _apply(self: T, x: Tensor) -> Tensor:
        ax = x * self.a
        return self.A(ax)

    def _apply_adjoint(self: T, x: Tensor) -> Tensor:
        ax = x * self.a
        return self.A.H(ax)


class Matmul(LinearMap):
    r"""
    Matrix multiplication of linear operators.

    .. math::
        A*B*x = A(B(x))

    """

    def __init__(self, A: LinearMap, B: LinearMap):
        self.A = A
        self.B = B
        assert list(self.B.size_out) == list(self.A.size_in), "Shapes do not match"
        super().__init__(self.B.size_in, self.A.size_out)

    def _apply(self: T, x: Tensor) -> Tensor:
        # TODO: add frame operator
        return self.A(self.B(x))

    def _apply_adjoint(self: T, x: Tensor) -> Tensor:
        return self.B.H(self.A.H(x))


class ConjTranspose(LinearMap):
    def __init__(self, A: LinearMap):
        self.A = A
        super().__init__(A.size_out, A.size_in)

    def _apply(self: T, x: Tensor) -> Tensor:
        return self.A.adjoint(x)

    def _apply_adjoint(self: T, x: Tensor) -> Tensor:
        return self.A.apply(x)


class Vstack(LinearMap):
    # TODO
    pass


class Hstack(LinearMap):
    # TODO
    pass
