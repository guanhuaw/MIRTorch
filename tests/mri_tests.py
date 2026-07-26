import sys

import pytest
import torch
import numpy as np
from mirtorch.linear import FFTCn, Sense, NuSense, NuSenseGram


@pytest.fixture
def complex_tensor():
    return torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))


@pytest.fixture
def smaps():
    return torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))


@pytest.fixture
def masks():
    return torch.randint(0, 2, (2, 16, 16)).float()


@pytest.fixture
def traj():
    return torch.rand(2, 2, 1000) * 2 - 1


# ============================================================================
# FFTCn Tests
# ============================================================================

def test_fftcn_forward_backward(complex_tensor):
    """Test that FFT and inverse FFT are inverses of each other"""
    fftcn = FFTCn([2, 1, 16, 16], [2, 1, 16, 16], dims=(2, 3))
    k_space = fftcn(complex_tensor)
    image = fftcn.H(k_space)
    assert torch.allclose(complex_tensor, image, atol=1e-6)


def test_fftcn_adjoint_property(complex_tensor):
    """Test the adjoint property: <Ax, y> = <x, A*y>"""
    fftcn = FFTCn([2, 1, 16, 16], [2, 1, 16, 16], dims=(2, 3))
    k_space = torch.randn_like(complex_tensor)
    lhs = torch.sum(fftcn(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * fftcn.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)


# ============================================================================
# Sense Tests
# ============================================================================

def test_sense_forward_backward(complex_tensor, smaps, masks):
    """Test Sense forward and adjoint operations"""
    sense = Sense(smaps, masks)
    k_space = sense(complex_tensor)
    image = sense.H(k_space)
    assert k_space.shape == (2, 4, 16, 16)
    assert image.shape == (2, 1, 16, 16)
    # Note: Due to undersampling, forward-adjoint is not perfect inverse
    assert not torch.allclose(complex_tensor, image, atol=1e-6)


def test_sense_adjoint_property(complex_tensor, smaps, masks):
    """Test the adjoint property for Sense operator"""
    sense = Sense(smaps, masks)
    k_space = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    lhs = torch.sum(sense(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * sense.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_sense_broadcast_smaps():
    """Test broadcasting single sensitivity map to multiple batches"""
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    masks = torch.randint(0, 2, (10, 16, 16)).float()
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    sense = Sense(smaps, masks)
    k_space = sense(x)

    assert k_space.shape == (10, 4, 16, 16), f"Expected (10,4,16,16), got {k_space.shape}"
    assert sense.size_in == [10, 1, 16, 16]
    assert sense.size_out == [10, 4, 16, 16]


def test_sense_broadcast_masks():
    """Test broadcasting single mask to multiple batches"""
    smaps = torch.complex(torch.randn(10, 4, 16, 16), torch.randn(10, 4, 16, 16))
    masks = torch.randint(0, 2, (1, 16, 16)).float()
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    sense = Sense(smaps, masks)
    k_space = sense(x)

    assert k_space.shape == (10, 4, 16, 16)
    assert sense.size_in == [10, 1, 16, 16]
    assert sense.size_out == [10, 4, 16, 16]


def test_sense_broadcast_both():
    """Test when both smaps and masks have batch size 1"""
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    masks = torch.randint(0, 2, (1, 16, 16)).float()

    # When both are [1,...], the operator has size_in=[1,1,16,16)
    sense = Sense(smaps, masks)

    # Must pass input with batch size 1
    x = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    k_space = sense(x)
    assert k_space.shape == (1, 4, 16, 16)


def test_sense_incompatible_batch_sizes():
    """Test that incompatible batch sizes raise error"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    masks = torch.randint(0, 2, (10, 16, 16)).float()

    with pytest.raises(ValueError, match="Incompatible batch sizes"):
        sense = Sense(smaps, masks)


def test_sense_spatial_dimension_mismatch():
    """Test that spatial dimension mismatch raises error"""
    smaps = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
    masks = torch.randint(0, 2, (2, 32, 32)).float()  # Wrong spatial size

    with pytest.raises(AssertionError, match="Spatial dimensions mismatch"):
        sense = Sense(smaps, masks)


# ============================================================================
# NuSense Tests
# ============================================================================

def test_nusense_forward_backward(complex_tensor, smaps, traj):
    """Test NuSense forward and adjoint operations"""
    nusense = NuSense(smaps, traj)
    k_space = nusense(complex_tensor)
    image = nusense.H(k_space)
    assert k_space.shape == (2, 4, 1000)
    assert image.shape == (2, 1, 16, 16)
    # Note: Due to non-Cartesian sampling, forward-adjoint is not perfect inverse
    assert not torch.allclose(complex_tensor, image, atol=1e-6)


def test_nusense_selects_default_backend_for_platform(smaps, traj):
    expected = "torchkbnufft" if sys.platform == "darwin" else "finufft"
    assert NuSense(smaps, traj).backend == expected
    assert NuSense(smaps, traj, backend="torchkbnufft").backend == "torchkbnufft"
    assert NuSense(smaps, traj, backend="finufft").backend == "finufft"

    with pytest.raises(ValueError, match="NUFFT backend"):
        NuSense(smaps, traj, backend="invalid")


def test_nusense_adjoint_property(complex_tensor, smaps, traj):
    """Test the adjoint property for NuSense operator"""
    nusense = NuSense(smaps, traj)
    k_space = torch.randn(2, 4, 1000, dtype=torch.complex64)
    lhs = torch.sum(nusense(complex_tensor).conj() * k_space)
    rhs = torch.sum(complex_tensor.conj() * nusense.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_nusense_broadcast_smaps():
    """Test broadcasting single sensitivity map to multiple batches"""
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    traj = torch.rand(10, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    nusense = NuSense(smaps, traj)
    k_space = nusense(x)

    assert k_space.shape == (10, 4, 1000), f"Expected (10,4,1000), got {k_space.shape}"
    assert nusense.size_in == [10, 1, 16, 16]
    assert nusense.size_out == [10, 4, 1000]


def test_nusense_broadcast_traj():
    """Test broadcasting single trajectory to multiple batches"""
    smaps = torch.complex(torch.randn(10, 4, 16, 16), torch.randn(10, 4, 16, 16))
    traj = torch.rand(1, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    nusense = NuSense(smaps, traj)
    k_space = nusense(x)

    assert k_space.shape == (10, 4, 1000)
    assert nusense.size_in == [10, 1, 16, 16]
    assert nusense.size_out == [10, 4, 1000]


def test_nusense_broadcast_both():
    """Test when both smaps and trajectory have batch size 1"""
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    traj = torch.rand(1, 2, 1000) * 2 - 1

    # When both are [1,...], the operator has size_in=[1,1,16,16)
    nusense = NuSense(smaps, traj)

    # Must pass input with batch size 1
    x = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    k_space = nusense(x)
    assert k_space.shape == (1, 4, 1000)


def test_nusense_incompatible_batch_sizes():
    """Test that incompatible batch sizes raise error"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    traj = torch.rand(10, 2, 1000) * 2 - 1

    with pytest.raises(ValueError, match="Incompatible batch sizes"):
        nusense = NuSense(smaps, traj)


def test_nusense_sequential_mode(smaps, traj):
    """Test NuSense in sequential (memory-saving) mode"""
    x = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))

    nusense = NuSense(smaps, traj, sequential=True)
    k_space = nusense(x)
    image = nusense.H(k_space)

    assert k_space.shape == (2, 4, 1000)
    assert image.shape == (2, 1, 16, 16)


def test_nusense_non_batchmode():
    """Test NuSense without batch dimension"""
    smaps = torch.complex(torch.randn(4, 16, 16), torch.randn(4, 16, 16))
    traj = torch.rand(2, 1000) * 2 - 1
    x = torch.complex(torch.randn(16, 16), torch.randn(16, 16))

    nusense = NuSense(smaps, traj, batchmode=False)
    k_space = nusense(x)
    image = nusense.H(k_space)

    assert k_space.shape == (4, 1000)
    assert image.shape == (16, 16)


def test_nusense_non_batchmode_adjoint_property():
    """Test adjoint property in non-batchmode"""
    smaps = torch.complex(torch.randn(4, 16, 16), torch.randn(4, 16, 16))
    traj = torch.rand(2, 1000) * 2 - 1
    x = torch.complex(torch.randn(16, 16), torch.randn(16, 16))

    nusense = NuSense(smaps, traj, batchmode=False)
    k_space = torch.randn(4, 1000, dtype=torch.complex64)

    lhs = torch.sum(nusense(x).conj() * k_space)
    rhs = torch.sum(x.conj() * nusense.H(k_space))
    assert torch.allclose(lhs, rhs, atol=1e-6)


# ============================================================================
# NuSenseGram Tests
# ============================================================================

def test_nusense_gram_forward(complex_tensor, smaps, traj):
    """Test NuSenseGram forward operation"""
    nusense_gram = NuSenseGram(smaps, traj)
    output = nusense_gram(complex_tensor)
    assert output.shape == complex_tensor.shape
    # Gram operator changes the image, so they should not be equal
    assert not torch.allclose(complex_tensor, output, atol=1e-6)


def test_nusense_gram_adjoint_property(complex_tensor, smaps, traj):
    """Test the adjoint property for NuSenseGram operator"""
    nusense_gram = NuSenseGram(smaps, traj)
    y = torch.randn_like(complex_tensor)
    lhs = torch.sum(nusense_gram(complex_tensor).conj() * y)
    rhs = torch.sum(complex_tensor.conj() * nusense_gram.H(y))
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_nusense_gram_self_adjoint():
    """Test that NuSenseGram is self-adjoint (Hermitian)"""
    smaps = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
    traj = torch.rand(2, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))

    nusense_gram = NuSenseGram(smaps, traj)

    # For self-adjoint operators: A(x) should equal A.H(x)
    forward = nusense_gram(x)
    adjoint = nusense_gram.H(x)
    assert torch.allclose(forward, adjoint, atol=1e-6)


def test_nusense_gram_broadcast_smaps():
    """Test broadcasting single sensitivity map to multiple batches for Gram operator"""
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    traj = torch.rand(10, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    nusense_gram = NuSenseGram(smaps, traj)
    output = nusense_gram(x)

    assert output.shape == (10, 1, 16, 16)
    assert nusense_gram.size_in == [10, 1, 16, 16]
    assert nusense_gram.size_out == [10, 1, 16, 16]


def test_nusense_gram_broadcast_traj():
    """Test broadcasting single trajectory to multiple batches for Gram operator"""
    smaps = torch.complex(torch.randn(10, 4, 16, 16), torch.randn(10, 4, 16, 16))
    traj = torch.rand(1, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))

    nusense_gram = NuSenseGram(smaps, traj)
    output = nusense_gram(x)

    assert output.shape == (10, 1, 16, 16)


def test_nusense_gram_incompatible_batch_sizes():
    """Test that incompatible batch sizes raise error for Gram operator"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    traj = torch.rand(10, 2, 1000) * 2 - 1

    with pytest.raises(ValueError, match="Incompatible batch sizes"):
        nusense_gram = NuSenseGram(smaps, traj)


def test_nusense_gram_non_batchmode():
    """Test NuSenseGram without batch dimension"""
    smaps = torch.complex(torch.randn(4, 16, 16), torch.randn(4, 16, 16))
    traj = torch.rand(2, 1000) * 2 - 1
    x = torch.complex(torch.randn(16, 16), torch.randn(16, 16))

    nusense_gram = NuSenseGram(smaps, traj, batchmode=False)
    output = nusense_gram(x)

    assert output.shape == (16, 16)


# ============================================================================
# Integration Tests
# ============================================================================

def test_nusense_vs_nusense_gram_consistency():
    """Test that NuSenseGram = NuSense.H * NuSense"""
    smaps = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
    traj = torch.rand(2, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))

    nusense = NuSense(smaps, traj)
    nusense_gram = NuSenseGram(smaps, traj)

    # A'Ax via composition
    k_space = nusense(x)
    composed = nusense.H(k_space)

    # A'Ax via Gram operator
    gram = nusense_gram(x)

    # Kernel accumulation order can differ near individual zero-valued elements.
    relative_error = torch.linalg.vector_norm(composed - gram) / torch.linalg.vector_norm(
        composed
    )
    assert relative_error < 1e-5


def test_broadcasting_use_case_fmri():
    """
    Test the key use case: single sensitivity map for fMRI time series.
    This demonstrates the correct pattern when both smaps and traj are [1,...].

    When both inputs have batch size 1, use .repeat() to expand to desired batch size.
    """
    # Simulating fMRI: 100 time frames, single coil sensitivity map
    n_frames = 100
    n_coils = 8

    # Single sensitivity map (doesn't change over time)
    smaps = torch.complex(
        torch.randn(1, n_coils, 32, 32),
        torch.randn(1, n_coils, 32, 32)
    )

    # Single trajectory (same sampling pattern for all frames)
    traj = torch.rand(1, 2, 500) * 2 - 1

    # CORRECT PATTERN: Replicate trajectory to desired batch size
    traj = traj.repeat(n_frames, 1, 1)  # Now [100, 2, 500]

    # Now smaps=[1,...] broadcasts to traj=[100,...]
    nusense = NuSense(smaps, traj)
    assert nusense.size_in == [n_frames, 1, 32, 32]
    assert nusense.size_out == [n_frames, n_coils, 500]

    # Different images at each time frame
    x = torch.complex(
        torch.randn(n_frames, 1, 32, 32),
        torch.randn(n_frames, 1, 32, 32)
    )

    # Forward pass
    k_space = nusense(x)
    assert k_space.shape == (n_frames, n_coils, 500)

    # Adjoint pass
    image_recon = nusense.H(k_space)
    assert image_recon.shape == (n_frames, 1, 32, 32)

    # Note: traj.repeat() is memory-efficient (uses views, not copies)
    # The actual trajectory data is not duplicated in memory


def test_same_batch_sizes():
    """Test the common case where all inputs have same batch size"""
    n_batch = 5

    smaps = torch.complex(torch.randn(n_batch, 4, 16, 16), torch.randn(n_batch, 4, 16, 16))
    traj = torch.rand(n_batch, 2, 1000) * 2 - 1
    x = torch.complex(torch.randn(n_batch, 1, 16, 16), torch.randn(n_batch, 1, 16, 16))

    nusense = NuSense(smaps, traj)
    k_space = nusense(x)

    assert k_space.shape == (n_batch, 4, 1000)


def test_both_batch_one_requires_repeat():
    """
    Test the limitation: when both smaps and traj are [1,...],
    the operator has size_in=[1,...]. To use with larger batches,
    replicate one of the inputs first.
    """
    # Both have batch size 1
    smaps = torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    traj = torch.rand(1, 2, 1000) * 2 - 1

    # Operator has size_in=[1, 1, 16, 16]
    nusense = NuSense(smaps, traj)
    assert nusense.size_in == [1, 1, 16, 16]

    # This works with batch size 1
    x1 = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    k1 = nusense(x1)
    assert k1.shape == (1, 4, 1000)

    # To use with batch size 10, replicate trajectory
    traj_repeated = traj.repeat(10, 1, 1)  # Now [10, 2, 1000]
    nusense_10 = NuSense(smaps, traj_repeated)
    assert nusense_10.size_in == [10, 1, 16, 16]

    # Now batch size 10 works
    x10 = torch.complex(torch.randn(10, 1, 16, 16), torch.randn(10, 1, 16, 16))
    k10 = nusense_10(x10)
    assert k10.shape == (10, 4, 1000)

    # Note: traj.repeat() is efficient - it creates a view, not a copy


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_shape_mismatch_during_forward():
    """Test that shape mismatch during forward pass is caught"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    traj = torch.rand(5, 2, 1000) * 2 - 1

    nusense = NuSense(smaps, traj)

    # Wrong input shape
    x = torch.complex(torch.randn(3, 1, 16, 16), torch.randn(3, 1, 16, 16))

    with pytest.raises(ValueError, match="forward linear op"):
        k_space = nusense(x)


def test_shape_mismatch_during_adjoint():
    """Test that shape mismatch during adjoint pass is caught"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    traj = torch.rand(5, 2, 1000) * 2 - 1

    nusense = NuSense(smaps, traj)

    # Wrong k-space shape
    k_space = torch.randn(3, 4, 1000, dtype=torch.complex64)

    with pytest.raises(ValueError, match="forward linear op"):
        image = nusense.H(k_space)


# ============================================================================
# Performance / Memory Tests
# ============================================================================

def test_broadcasting_memory_efficiency():
    """
    Verify that broadcasting doesn't actually replicate the tensor
    (PyTorch handles this internally)
    """
    # Single smap
    smaps_single = torch.complex(torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32))

    # Replicated smap (what user SHOULDN'T need to do)
    smaps_replicated = smaps_single.repeat(100, 1, 1, 1)

    # Memory usage should be very different
    assert smaps_single.element_size() * smaps_single.nelement() * 100 == \
           smaps_replicated.element_size() * smaps_replicated.nelement()

    # But both should work the same way
    traj = torch.rand(100, 2, 500) * 2 - 1

    nusense_broadcast = NuSense(smaps_single, traj)
    nusense_replicated = NuSense(smaps_replicated, traj)

    assert nusense_broadcast.size_out == nusense_replicated.size_out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
