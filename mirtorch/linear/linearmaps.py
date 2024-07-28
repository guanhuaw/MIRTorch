from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from torch import Tensor

FloatLike = Union[float, torch.FloatTensor]


def check_device(x, y):
    r"""
    check if two tensors are on the same device
    """
    assert x.device == y.device, "Tensors should be on the same device"


class LinearMap:
    r"""
    Abstraction of linear operators as matrices :math:`y = A*x`.
    The implementation follow the `SigPy <https://github.com/mikgroup/sigpy>`_ and `LinearmapAA <https://github.com/JeffFessler/LinearMapsAA.jl>`_.

    Common operators, including +, -, *, are overloaded. One may freely compose operators as long as the size matches.

    New linear operators require to implement `_apply` (forward, :math:`A`) and `_adjoint` (conjugate adjoint, :math:`A'`) functions, as well as size.
    Recommendation for efficient backpropagation (but you do not have to do this if the AD is efficient enough):

    .. code-block:: python

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

    Attributes:
        size_in: the size of the input of the linear map (a list)
        size_out: the size of the output of the linear map (a list)
    """

    def __init__(self, size_in: List[int], size_out: List[int]):
        r"""
        Initiate the linear operator.
        """
        self.size_in = list(size_in)
        self.size_out = list(size_out)

    def __repr__(self):
        return (
            f"<LinearMap {self.__class__.__name__} of {self.size_out}x{self.size_in}>"
        )

    def __call__(self, x: Tensor) -> Tensor:
        # for a instance A, we can apply it by calling A(x). Equal to A*x
        return self.apply(x)

    def _apply(self, x: Tensor) -> Tensor:
        # worth noting that the function here should be differentiable,
        # for example, composed of native torch functions,
        # or torch.autograd.Function, or nn.module
        raise NotImplementedError

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def apply(self, x: Tensor) -> Tensor:
        r"""
        Apply the forward operator
        """
        assert list(x.shape) == list(
            self.size_in
        ), f"Shape of input data {x.shape} and forward linear op {self.size_in} do not match!"
        return self._apply(x)

    def adjoint(self, x: Tensor) -> Tensor:
        r"""
        Apply the adjoint operator
        """
        assert list(x.shape) == list(
            self.size_out
        ), f"Shape of input data {x.shape} and adjoint linear op {self.size_in} do not match!"
        return self._apply_adjoint(x)

    @property
    def H(self) -> LinearMap:
        r"""
        Apply the (Hermitian) transpose
        """
        return ConjTranspose(self)

    def __add__(self: LinearMap, other: LinearMap) -> LinearMap:
        r"""
        Reload the + symbol.
        """
        return Add(self, other)

    def __mul__(
        self: LinearMap, other: Union[str, int, LinearMap, Tensor]
    ) -> Union[LinearMap, Tensor]:
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, LinearMap):
            return Matmul(self, other)
        elif isinstance(other, Tensor):
            if not other.shape:
                return Multiply(self, other)
            return self.apply(other)
        else:
            raise NotImplementedError(
                (
                    f"Only scalers, Linearmaps or Tensors, rather than '{type(other)}' "
                    "fare allowed as arguments for this function."
                )
            )

    def __rmul__(
        self: LinearMap, other: Union[str, int, LinearMap, Tensor]
    ) -> LinearMap:
        r"""
        Reload the * symbol.
        """
        if np.isscalar(other):
            return Multiply(self, other)
        elif isinstance(other, Tensor) and not other.shape:
            return Multiply(self, other)
        else:
            return NotImplemented

    def __sub__(self: LinearMap, other: LinearMap) -> LinearMap:
        r"""
        Reload the - symbol.
        """
        return self.__add__(-other)

    def __neg__(self: LinearMap) -> LinearMap:
        r"""
        Reload the - symbol.
        """
        return -1 * self

    def to(self: LinearMap, device: Union[torch.device, str]) -> LinearMap:
        r"""
        Copy to different devices
        """
        for prop in self.__dict__.keys():
            if isinstance(self.__dict__[prop], Tensor) or isinstance(
                self.__dict__[prop], torch.nn.Module
            ):
                self.__dict__[prop] = self.__dict__[prop].to(device)


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
        assert list(A.size_in) == list(
            B.size_in
        ), "The input dimensions of two combined ops are not the same."
        assert list(A.size_out) == list(
            B.size_out
        ), "The output dimensions of two combined ops are not the same."
        self.A = A
        self.B = B
        super().__init__(self.A.size_in, self.B.size_out)

    def _apply(self: LinearMap, x: Tensor) -> Tensor:
        return self.A(x) + self.B(x)

    def _apply_adjoint(self: LinearMap, x: Tensor) -> Tensor:
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

    def _apply(self: LinearMap, x: Tensor) -> Tensor:
        ax = x * self.a
        return self.A(ax)

    def _apply_adjoint(self: LinearMap, x: Tensor) -> Tensor:
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

    def _apply(self: LinearMap, x: Tensor) -> Tensor:
        # TODO: add gram operator
        return self.A(self.B(x))

    def _apply_adjoint(self: LinearMap, x: Tensor) -> Tensor:
        return self.B.H(self.A.H(x))


class ConjTranspose(LinearMap):
    r"""
    Hermitian transpose of linear operators.
    """

    def __init__(self, A: LinearMap):
        self.A = A
        super().__init__(A.size_out, A.size_in)

    def _apply(self: LinearMap, x: Tensor) -> Tensor:
        return self.A.adjoint(x)

    def _apply_adjoint(self: LinearMap, x: Tensor) -> Tensor:
        return self.A.apply(x)


class BlockDiagonal(LinearMap):
    r"""
    Create a block-diagonal linear map from a list of linear maps. This assumes that each of the linear maps
    is a 2D linearmap, with identical input and output shapes.

    Attributes:
    A : List of 2D linear maps
    """

    def __init__(self, A: List[LinearMap]):
        self.A = A

        # dimension checks
        nz = len(A)
        assert all(
            [list(A[i].size_in) == list(A[i + 1].size_in) for i in range(nz - 1)]
        ), "Input dimensions of each linear map must be compatible to create a block diagonal linear map."
        assert all(
            [list(A[i].size_out) == list(A[i + 1].size_out) for i in range(nz - 1)]
        ), "Output dimensions of each linear map must be compatible to create a block diagonal linear map."
        size_in = list(A[0].size_in) + [nz]
        size_out = list(A[0].size_out) + [nz]
        super().__init__(tuple(size_in), tuple(size_out))

    def _apply(self: LinearMap, x: Tensor) -> Tensor:
        out = torch.zeros(
            self.size_out, dtype=x.dtype, device=x.device, layout=x.layout
        )
        nz = self.size_out[-1]

        # TODO: exploit parallelism
        for k in range(nz):
            out[..., k] = self.A[k].apply(x[..., k])
        return out

    def _apply_adjoint(self: LinearMap, x: Tensor):
        out = torch.zeros(self.size_in, dtype=x.dtype, device=x.device, layout=x.layout)
        nz = self.size_in[-1]

        # TODO: exploit parallelism
        for k in range(nz):
            out[..., k] = self.A[k].adjoint(x[..., k])
        return out


class Kron(LinearMap):
    r"""
    Create a LinearMap corresponding to the Kronecker product of a linear map with the identity matrix, i.e.,
    kron(I_n, A), where A is a LinearMap.

    Attributes:
    A: linear map
    n: dimension of identity matrix for Kronecker product

    Example: This could be used for 2D stack of spirals reconstruction where we have identical spiral trajectories
    in each slice, and we neglect the effects of off-resonance + no parallel imaging.
    """

    def __init__(self, A: LinearMap, n):
        self.A = A
        self.n = n
        size_in = list(A.size_in) + [n]
        size_out = list(A.size_out) + [n]
        super().__init__(tuple(size_in), tuple(size_out))

    def apply(self, x: Tensor):
        out = torch.zeros(
            self.size_out, dtype=x.dtype, device=x.device, layout=x.layout
        )

        # TODO: exploit parallelism.
        for k in range(self.n):
            out[..., k] = self.A.apply(x[..., k])

        return out

    def _apply_adjoint(self, x: Tensor):
        out = torch.zeros(self.size_in, dtype=x.dtype, device=x.device, layout=x.layout)

        # TODO: exploit parallelism.
        for k in range(self.n):
            out[..., k] = self.A.adjoint(x[..., k])

        return out


class Vstack(LinearMap):
    r"""
    Vertical stacking of linear operators.

    .. math::
        [A1; A2; ...; An] * x = [A1(x); A2(x); ...; An(x)]

    Attributes:
        A: List of LinearMaps to be stacked vertically
        dim: the dimension along which to stack the LinearMaps
    """

    def __init__(self, A: List[LinearMap], dim: int = 0):
        self.A = A

        # Check that all input sizes are the same
        assert all(
            [A[i].size_in == A[0].size_in for i in range(len(A))]
        ), "All input sizes must be the same"

        # Calculate the total output size
        size_out = [sum(A[i].size_out[0] for i in range(len(A)))] + list(
            A[0].size_out[1:]
        )

        self.dim = dim

        super().__init__(A[0].size_in, size_out)

    def _apply(self, x: Tensor) -> Tensor:
        return torch.cat([A_i(x) for A_i in self.A], dim=self.dim)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        outputs = []
        start = 0
        for A_i in self.A:
            end = start + A_i.size_out[0]
            outputs.append(A_i.H(x[start:end]))
            start = end
        return sum(outputs)


class Hstack(LinearMap):
    r"""
    Horizontal stacking of linear operators.

    .. math::
        [A1, A2, ..., An] * [x1; x2; ...; xn] = A1(x1) + A2(x2) + ... + An(xn)

    Attributes:
        A: List of LinearMaps to be stacked horizontally
    """

    def __init__(self, A: List[LinearMap], dim: int = 0):
        self.A = A

        # Check that all output sizes are the same
        assert all(
            [A[i].size_out == A[0].size_out for i in range(len(A))]
        ), "All output sizes must be the same"

        # Calculate the total input size
        size_in = [sum(A[i].size_in[0] for i in range(len(A)))] + list(A[0].size_in[1:])
        self.dim = dim

        super().__init__(size_in, A[0].size_out)

    def _apply(self, x: Tensor) -> Tensor:
        outputs = []
        start = 0
        for A_i in self.A:
            end = start + A_i.size_in[0]
            outputs.append(A_i(x[start:end]))
            start = end
        return sum(outputs)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return torch.cat([A_i.H(x) for A_i in self.A], dim=self.dim)
