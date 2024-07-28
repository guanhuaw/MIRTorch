import pytest
import torch
from typing import List
from torch import Tensor
from mirtorch.linear import LinearMap, Add, Multiply, Matmul, ConjTranspose, BlockDiagonal, Kron, Vstack, Hstack


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
    expected = torch.stack([linear_operator.apply(tensor1), linear_operator.apply(tensor1)], dim=-1)
    assert torch.allclose(result, expected)


def test_kron_operator(tensor1, linear_operator):
    op = Kron(linear_operator, 2)
    x = torch.stack([tensor1, tensor1], dim=-1)
    result = op.apply(x)
    expected = torch.stack([linear_operator.apply(tensor1), linear_operator.apply(tensor1)], dim=-1)
    assert torch.allclose(result, expected)


def test_vstack_operator(tensor1, linear_operator):
    op = Vstack([linear_operator, linear_operator])
    result = op.apply(tensor1)
    expected = torch.cat([linear_operator.apply(tensor1), linear_operator.apply(tensor1)])
    assert torch.allclose(result, expected)
    assert result.shape == (6,)  # 3 + 3

    # Test adjoint
    adjoint_input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    adjoint_result = op.adjoint(adjoint_input)
    expected_adjoint = linear_operator.adjoint(adjoint_input[:3]) + linear_operator.adjoint(adjoint_input[3:])
    assert torch.allclose(adjoint_result, expected_adjoint)
    assert adjoint_result.shape == (3,)

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
    expected_adjoint = torch.cat([linear_operator.adjoint(adjoint_input), linear_operator.adjoint(adjoint_input)])
    assert torch.allclose(adjoint_result, expected_adjoint)
    assert adjoint_result.shape == (6,)  # 3
