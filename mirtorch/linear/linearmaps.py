from __future__ import annotations

from collections.abc import Sequence
from numbers import Number

import torch
from torch import Tensor

ScalarLike = complex | Tensor


def _is_scalar(value) -> bool:
    return isinstance(value, Number) or isinstance(value, Tensor) and value.ndim == 0


def _normalize_dim(dim: int, rank: int) -> int:
    if not -rank <= dim < rank:
        raise ValueError(f"dim={dim} is out of range for a {rank}-dimensional tensor")
    return dim % rank


def check_device(x, y):
    r"""
    check if two tensors are on the same device
    """
    if x.device != y.device:
        raise ValueError("Tensors should be on the same device")


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

    def __init__(self, size_in: Sequence[int], size_out: Sequence[int]):
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
        if list(x.shape) != self.size_in:
            raise ValueError(
                f"Shape of input data {list(x.shape)} and forward linear op "
                f"{self.size_in} do not match"
            )
        return self._apply(x)

    def adjoint(self, x: Tensor) -> Tensor:
        r"""
        Apply the adjoint operator
        """
        if list(x.shape) != self.size_out:
            raise ValueError(
                f"Shape of input data {list(x.shape)} and adjoint linear op "
                f"{self.size_out} do not match"
            )
        return self._apply_adjoint(x)

    @property
    def H(self) -> LinearMap:
        r"""
        Apply the (Hermitian) transpose
        """
        return ConjTranspose(self)

    def __add__(self, other: LinearMap) -> LinearMap:
        r"""
        Reload the + symbol.
        """
        return Add(self, other)

    def __mul__(self, other: ScalarLike | LinearMap) -> LinearMap | Tensor:
        r"""
        Reload the * symbol.
        """
        if isinstance(other, Tensor):
            if not other.shape:
                return Multiply(self, other)
            return self.apply(other)
        if isinstance(other, LinearMap):
            return Matmul(self, other)
        if isinstance(other, Number):
            return Multiply(self, other)
        raise NotImplementedError(
            f"Only scalars, LinearMaps, or Tensors—not {type(other)!r}—are supported."
        )

    def __rmul__(self, other: ScalarLike) -> LinearMap:
        r"""
        Reload the * symbol.
        """
        if _is_scalar(other):
            return Multiply(self, other)
        return NotImplemented

    def __sub__(self, other: LinearMap) -> LinearMap:
        r"""
        Reload the - symbol.
        """
        return self.__add__(-other)

    def __neg__(self) -> LinearMap:
        r"""
        Reload the - symbol.
        """
        return -1 * self

    def to(self, device: torch.device | str) -> LinearMap:
        r"""
        Copy to different devices
        """

        def move(value):
            if isinstance(value, (Tensor, torch.nn.Module)):
                return value.to(device)
            if isinstance(value, LinearMap):
                return value.to(device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value

        for prop, value in vars(self).items():
            setattr(self, prop, move(value))
        return self


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
        if A.size_in != B.size_in:
            raise ValueError("The input dimensions of the operators do not match")
        if A.size_out != B.size_out:
            raise ValueError("The output dimensions of the operators do not match")
        self.A = A
        self.B = B
        super().__init__(self.A.size_in, self.B.size_out)

    def _apply(self, x: Tensor) -> Tensor:
        return self.A(x) + self.B(x)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return self.A.adjoint(x) + self.B.adjoint(x)


class Multiply(LinearMap):
    r"""
    Scaling linear operators.

    .. math::
        a*A*x = A(ax)

    Attributes:
        a: scaling factor
        A: LinearMap
    """

    def __init__(self, A: LinearMap, a: ScalarLike):
        if not _is_scalar(a):
            raise TypeError("A LinearMap can only be multiplied by a scalar")
        self.a = a
        self.A = A
        super().__init__(self.A.size_in, self.A.size_out)

    def _apply(self, x: Tensor) -> Tensor:
        ax = x * self.a
        return self.A(ax)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        ax = x * self.a.conj() if isinstance(self.a, Tensor) else x * self.a.conjugate()
        return self.A.adjoint(ax)


class Matmul(LinearMap):
    r"""
    Matrix multiplication of linear operators.

    .. math::
        A*B*x = A(B(x))

    """

    def __init__(self, A: LinearMap, B: LinearMap):
        self.A = A
        self.B = B
        if self.B.size_out != self.A.size_in:
            raise ValueError("The inner dimensions of the operators do not match")
        super().__init__(self.B.size_in, self.A.size_out)

    def _apply(self, x: Tensor) -> Tensor:
        # TODO: add gram operator
        return self.A(self.B(x))

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return self.B.adjoint(self.A.adjoint(x))


class ConjTranspose(LinearMap):
    r"""
    Hermitian transpose of linear operators.
    """

    def __init__(self, A: LinearMap):
        self.A = A
        super().__init__(A.size_out, A.size_in)

    def _apply(self, x: Tensor) -> Tensor:
        return self.A.adjoint(x)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return self.A.apply(x)


class BlockDiagonal(LinearMap):
    r"""
    Create a block-diagonal linear map from a list of linear maps. This assumes that each of the linear maps
    is a 2D linearmap, with identical input and output shapes.

    Attributes:
    A : List of 2D linear maps
    """

    def __init__(self, A: list[LinearMap]):
        if not A:
            raise ValueError("At least one LinearMap is required")
        self.A = A

        # dimension checks
        nz = len(A)
        if not all(list(A[i].size_in) == list(A[i + 1].size_in) for i in range(nz - 1)):
            raise ValueError(
                "Input dimensions must match to create a block-diagonal LinearMap"
            )
        if not all(
            list(A[i].size_out) == list(A[i + 1].size_out) for i in range(nz - 1)
        ):
            raise ValueError(
                "Output dimensions must match to create a block-diagonal LinearMap"
            )
        size_in = list(A[0].size_in) + [nz]
        size_out = list(A[0].size_out) + [nz]
        super().__init__(tuple(size_in), tuple(size_out))

    def _apply(self, x: Tensor) -> Tensor:
        out = torch.empty(
            self.size_out, dtype=x.dtype, device=x.device, layout=x.layout
        )
        for k, operator in enumerate(self.A):
            out[..., k] = operator.apply(x[..., k])
        return out

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        out = torch.empty(self.size_in, dtype=x.dtype, device=x.device, layout=x.layout)
        for k, operator in enumerate(self.A):
            out[..., k] = operator.adjoint(x[..., k])
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

    def __init__(self, A: LinearMap, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.A = A
        self.n = n
        size_in = list(A.size_in) + [n]
        size_out = list(A.size_out) + [n]
        super().__init__(tuple(size_in), tuple(size_out))

    def _apply(self, x: Tensor) -> Tensor:
        out = torch.empty(
            self.size_out, dtype=x.dtype, device=x.device, layout=x.layout
        )
        for k in range(self.n):
            out[..., k] = self.A.apply(x[..., k])
        return out

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        out = torch.empty(self.size_in, dtype=x.dtype, device=x.device, layout=x.layout)
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

    def __init__(self, A: list[LinearMap], dim: int = 0):
        if not A:
            raise ValueError("At least one LinearMap is required")
        self.A = A
        self.dim = _normalize_dim(dim, len(A[0].size_out))

        # Check that all input sizes are the same
        if not all(operator.size_in == A[0].size_in for operator in A):
            raise ValueError("All input sizes must be the same")

        reference = A[0].size_out
        if not all(
            len(operator.size_out) == len(reference)
            and all(
                size == reference[index]
                for index, size in enumerate(operator.size_out)
                if index != self.dim
            )
            for operator in A
        ):
            raise ValueError("Output sizes must match outside the stacking dimension")

        size_out = list(reference)
        size_out[self.dim] = sum(operator.size_out[self.dim] for operator in A)
        super().__init__(A[0].size_in, size_out)

    def _apply(self, x: Tensor) -> Tensor:
        return torch.cat([A_i(x) for A_i in self.A], dim=self.dim)

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        sections = [operator.size_out[self.dim] for operator in self.A]
        inputs = torch.split(x, sections, dim=self.dim)
        outputs = [
            operator.adjoint(input_)
            for operator, input_ in zip(self.A, inputs, strict=True)
        ]
        return outputs[0] + sum(outputs[1:])


class Hstack(LinearMap):
    r"""
    Horizontal stacking of linear operators.

    .. math::
        [A1, A2, ..., An] * [x1; x2; ...; xn] = A1(x1) + A2(x2) + ... + An(xn)

    Attributes:
        A: List of LinearMaps to be stacked horizontally
    """

    def __init__(self, A: list[LinearMap], dim: int = 0):
        if not A:
            raise ValueError("At least one LinearMap is required")
        self.A = A
        self.dim = _normalize_dim(dim, len(A[0].size_in))

        # Check that all output sizes are the same
        if not all(operator.size_out == A[0].size_out for operator in A):
            raise ValueError("All output sizes must be the same")

        reference = A[0].size_in
        if not all(
            len(operator.size_in) == len(reference)
            and all(
                size == reference[index]
                for index, size in enumerate(operator.size_in)
                if index != self.dim
            )
            for operator in A
        ):
            raise ValueError("Input sizes must match outside the stacking dimension")

        size_in = list(reference)
        size_in[self.dim] = sum(operator.size_in[self.dim] for operator in A)
        super().__init__(size_in, A[0].size_out)

    def _apply(self, x: Tensor) -> Tensor:
        sections = [operator.size_in[self.dim] for operator in self.A]
        inputs = torch.split(x, sections, dim=self.dim)
        outputs = [
            operator(input_) for operator, input_ in zip(self.A, inputs, strict=True)
        ]
        return outputs[0] + sum(outputs[1:])

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return torch.cat([A_i.adjoint(x) for A_i in self.A], dim=self.dim)
