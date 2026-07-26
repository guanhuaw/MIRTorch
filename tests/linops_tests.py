import mirtorch
import mirtorch.linear.basics as basics
import pytest
import torch
from torch import Tensor

from mirtorch.linear import (
    Add,
    BlockDiagonal,
    ConjTranspose,
    Hstack,
    Identity,
    Kron,
    LinearMap,
    Matmul,
    Multiply,
    Vstack,
    Diff2dgram,
)


# Define a mock linear operator for testing purposes
class MockLinearOperator(LinearMap):
    def _apply(self, x: Tensor) -> Tensor:
        return 2 * x

    def _apply_adjoint(self, x: Tensor) -> Tensor:
        return 0.5 * x


@pytest.fixture
def tensor1():
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def tensor2():
    return torch.tensor([4.0, 5.0, 6.0])


@pytest.fixture
def linear_operator():
    return MockLinearOperator([3], [3])


def test_linear_map_initialization():
    lm = LinearMap([3], [3])
    assert lm.size_in == [3]
    assert lm.size_out == [3]


def test_package_exposes_version():
    assert isinstance(mirtorch.__version__, str)
    assert mirtorch.__version__


def test_linear_map_rejects_incorrect_shapes():
    operator = Identity([3])
    with pytest.raises(ValueError, match="forward linear op"):
        operator.apply(torch.ones(2))
    with pytest.raises(ValueError, match="adjoint linear op"):
        operator.adjoint(torch.ones(4))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires PyTorch 2 or newer")
def test_linear_map_supports_torch_compile(tensor1, linear_operator):
    compiled = torch.compile(linear_operator.apply, backend="eager", fullgraph=True)
    assert torch.equal(compiled(tensor1), linear_operator.apply(tensor1))


def test_diff2dgram_compiles_once_by_default(monkeypatch):
    compiled = []
    monkeypatch.setattr(
        basics,
        "should_compile",
        lambda enabled, _tensor: enabled,
    )
    monkeypatch.setattr(
        basics,
        "compile_callable",
        lambda function: compiled.append(function) or function,
    )
    image = torch.randn(1, 1, 8, 8)
    operator = Diff2dgram(image.shape)

    first = operator(image)
    second = operator(image)

    assert len(compiled) == 1
    assert torch.equal(first, second)
    assert torch.equal(Diff2dgram(image.shape, compile=False)(image), first)


def test_add_operator(tensor1, linear_operator):
    op = Add(linear_operator, linear_operator)
    result = op.apply(tensor1)
    expected = 4 * tensor1
    assert torch.allclose(result, expected)


def test_multiply_operator(tensor1, linear_operator):
    op = Multiply(linear_operator, 3)
    result = op.apply(tensor1)
    expected = linear_operator.apply(3 * tensor1)
    assert torch.allclose(result, expected)


def test_multiply_operator_complex_adjoint():
    operator = Multiply(Identity([3]), 1 + 2j)
    x = torch.randn(3, dtype=torch.complex64)
    y = torch.randn(3, dtype=torch.complex64)
    lhs = torch.vdot(operator(x), y)
    rhs = torch.vdot(x, operator.adjoint(y))
    assert torch.allclose(lhs, rhs)


def test_matmul_operator(tensor1, linear_operator):
    op = Matmul(linear_operator, linear_operator)
    result = op.apply(tensor1)
    expected = linear_operator.apply(linear_operator.apply(tensor1))
    assert torch.allclose(result, expected)


def test_conj_transpose_operator(tensor1, linear_operator):
    op = ConjTranspose(linear_operator)
    result = op.apply(tensor1)
    expected = linear_operator.adjoint(tensor1)
    assert torch.allclose(result, expected)


def test_block_diagonal_operator(tensor1, linear_operator):
    op = BlockDiagonal([linear_operator, linear_operator])
    x = torch.stack([tensor1, tensor1], dim=-1)
    result = op.apply(x)
    expected = torch.stack(
        [linear_operator.apply(tensor1), linear_operator.apply(tensor1)], dim=-1
    )
    assert torch.allclose(result, expected)


def test_kron_operator(tensor1, linear_operator):
    op = Kron(linear_operator, 2)
    x = torch.stack([tensor1, tensor1], dim=-1)
    result = op.apply(x)
    expected = torch.stack(
        [linear_operator.apply(tensor1), linear_operator.apply(tensor1)], dim=-1
    )
    assert torch.allclose(result, expected)


def test_vstack_operator(tensor1, linear_operator):
    op = Vstack([linear_operator, linear_operator])
    result = op.apply(tensor1)
    expected = torch.cat(
        [linear_operator.apply(tensor1), linear_operator.apply(tensor1)]
    )
    assert torch.allclose(result, expected)
    assert result.shape == (6,)  # 3 + 3

    # Test adjoint
    adjoint_input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    adjoint_result = op.adjoint(adjoint_input)
    expected_adjoint = linear_operator.adjoint(
        adjoint_input[:3]
    ) + linear_operator.adjoint(adjoint_input[3:])
    assert torch.allclose(adjoint_result, expected_adjoint)
    assert adjoint_result.shape == (3,)


def test_vstack_operator_nonzero_dimension():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    operator = Vstack([Identity(x.shape), Identity(x.shape)], dim=1)
    stacked = torch.cat([x, x], dim=1)
    assert operator.size_out == [2, 6]
    assert torch.equal(operator(x), stacked)
    assert torch.equal(operator.adjoint(stacked), 2 * x)


def test_hstack_operator(tensor1, tensor2, linear_operator):
    op = Hstack([linear_operator, linear_operator])
    input_tensor = torch.cat([tensor1, tensor2])
    result = op.apply(input_tensor)
    expected = linear_operator.apply(tensor1) + linear_operator.apply(tensor2)
    assert torch.allclose(result, expected)
    assert result.shape == (3,)

    # Test adjoint
    adjoint_input = torch.tensor([1.0, 2.0, 3.0])
    adjoint_result = op.adjoint(adjoint_input)
    expected_adjoint = torch.cat(
        [linear_operator.adjoint(adjoint_input), linear_operator.adjoint(adjoint_input)]
    )
    assert torch.allclose(adjoint_result, expected_adjoint)
    assert adjoint_result.shape == (6,)  # 3


def test_hstack_operator_nonzero_dimension():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    operator = Hstack([Identity(x.shape), Identity(x.shape)], dim=1)
    stacked = torch.cat([x, x], dim=1)
    assert operator.size_in == [2, 6]
    assert torch.equal(operator(stacked), 2 * x)
    assert torch.equal(operator.adjoint(x), torch.cat([x, x], dim=1))


@pytest.mark.parametrize("operator", [BlockDiagonal, Vstack, Hstack])
def test_stacked_operators_reject_empty_input(operator):
    with pytest.raises(ValueError, match="At least one"):
        operator([])
