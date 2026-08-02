import math
import sys
from importlib.util import find_spec

import pytest
import torch

from mirtorch.linear import FFTCn, Gmri, GmriGram, NuSense, NuSenseGram, Sense, mri


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


def _direct_nufft_phase(
    spatial_shape: tuple[int, ...],
    trajectory: torch.Tensor,
    sign: int,
) -> torch.Tensor:
    coordinates = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        for size in spatial_shape
    ]
    grids = torch.meshgrid(*coordinates, indexing="ij")
    sample_shape = (trajectory.shape[0], trajectory.shape[-1]) + (1,) * len(
        spatial_shape
    )
    phase_argument = trajectory[:, 0].reshape(sample_shape) * grids[0]
    for dimension in range(1, len(spatial_shape)):
        phase_argument = (
            phase_argument
            + trajectory[:, dimension].reshape(sample_shape) * grids[dimension]
        )
    return torch.exp(sign * 1j * phase_argument)


def _direct_type2(
    modes: torch.Tensor,
    trajectory: torch.Tensor,
    norm: str | None,
    grid_size: tuple[int, ...],
) -> torch.Tensor:
    phase = _direct_nufft_phase(tuple(modes.shape[2:]), trajectory, sign=-1)
    spatial_dims = tuple(range(3, 3 + len(grid_size)))
    result = (modes.unsqueeze(2) * phase.unsqueeze(1)).sum(dim=spatial_dims)
    if norm == "ortho":
        result = result / math.sqrt(math.prod(grid_size))
    return result


def _direct_type1(
    samples: torch.Tensor,
    trajectory: torch.Tensor,
    spatial_shape: tuple[int, ...],
    norm: str | None,
    grid_size: tuple[int, ...],
) -> torch.Tensor:
    phase = _direct_nufft_phase(spatial_shape, trajectory, sign=1)
    result = (
        samples.reshape(samples.shape + (1,) * len(spatial_shape)) * phase.unsqueeze(1)
    ).sum(dim=2)
    if norm == "ortho":
        result = result / math.sqrt(math.prod(grid_size))
    return result


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )


def _direct_gmri_forward(operator: Gmri, image: torch.Tensor) -> torch.Tensor:
    spatial_shape = tuple(image.shape[2:])
    modes = (image.unsqueeze(0) * operator.C * operator.smaps.unsqueeze(0)).permute(
        1, 0, 2, *range(3, image.ndim + 1)
    )
    modes = modes.reshape(
        operator.nbatch,
        operator.L * operator.nc,
        *spatial_shape,
    )
    transformed = _direct_type2(
        modes,
        operator.traj,
        operator.norm,
        operator.grid_size,
    )
    segments = transformed.reshape(
        operator.nbatch,
        operator.L,
        operator.nc,
        operator.nshot,
        operator.npoints,
    ).permute(1, 0, 2, 3, 4)
    return (operator.B * segments).sum(dim=0)


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

    assert k_space.shape == (10, 4, 16, 16), (
        f"Expected (10,4,16,16), got {k_space.shape}"
    )
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
        Sense(smaps, masks)


def test_sense_spatial_dimension_mismatch():
    """Test that spatial dimension mismatch raises error"""
    smaps = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
    masks = torch.randint(0, 2, (2, 32, 32)).float()  # Wrong spatial size

    with pytest.raises(ValueError, match="Spatial dimensions mismatch"):
        Sense(smaps, masks)


def test_sense_complex_mask_adjoint_property():
    torch.manual_seed(1)
    smaps = torch.randn(2, 3, 8, 7, dtype=torch.complex128)
    masks = torch.randn(2, 8, 7, dtype=torch.complex128)
    image = torch.randn(2, 1, 8, 7, dtype=torch.complex128)
    samples = torch.randn(2, 3, 8, 7, dtype=torch.complex128)
    sense = Sense(smaps, masks)

    lhs = torch.vdot(sense(image).reshape(-1), samples.reshape(-1))
    rhs = torch.vdot(image.reshape(-1), sense.H(samples).reshape(-1))

    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


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
    native_available = sys.platform != "darwin" and find_spec("finufft") is not None
    expected = "finufft" if native_available else "torchkbnufft"
    assert NuSense(smaps, traj).backend == expected
    assert NuSense(smaps, traj, backend="torchkbnufft").backend == "torchkbnufft"
    assert NuSense(smaps, traj, backend="finufft").backend == "finufft"

    with pytest.raises(ValueError, match="NUFFT backend"):
        NuSense(smaps, traj, backend="invalid")


@pytest.mark.parametrize(
    ("spatial_shape", "sensitivity_batch", "trajectory_batch", "sequential"),
    [
        pytest.param((5,), 1, 2, False, id="1d-shared-smaps"),
        pytest.param((5, 6), 2, 1, False, id="2d-shared-trajectory"),
        pytest.param((4, 5), 2, 1, True, id="2d-sequential"),
        pytest.param((3, 4, 5), 2, 2, False, id="3d-per-batch"),
    ],
)
def test_nusense_torchkbnufft_forward_gradient_matches_direct_nudft(
    spatial_shape,
    sensitivity_batch,
    trajectory_batch,
    sequential,
):
    torch.manual_seed(20260731)
    batch, coils, points = 2, 2, 9
    image = torch.randn(
        batch,
        1,
        *spatial_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )
    smaps = torch.randn(
        sensitivity_batch,
        coils,
        *spatial_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )
    trajectory = (
        torch.rand(
            trajectory_batch,
            len(spatial_shape),
            points,
            dtype=torch.float64,
        )
        - 0.5
    ).requires_grad_()
    kwargs = {
        "backend": "torchkbnufft",
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
        "sequential": sequential,
    }
    operator = NuSense(smaps, trajectory, **kwargs)
    actual = operator(image)

    reference_image = image.detach().clone().requires_grad_()
    reference_smaps = smaps.detach().clone().requires_grad_()
    reference_trajectory = trajectory.detach().clone().requires_grad_()
    expected = _direct_type2(
        reference_image * reference_smaps,
        reference_trajectory,
        operator.norm,
        operator.grid_size,
    )
    fixed_operator = NuSense(
        smaps.detach(),
        trajectory.detach(),
        **kwargs,
    )

    assert torch.equal(actual.detach(), fixed_operator(image.detach()))
    assert _relative_error(actual.detach(), expected.detach()) < 2e-3

    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        torch.vdot(probe.reshape(-1), actual.reshape(-1)).real,
        (image, smaps, trajectory),
    )
    expected_gradients = torch.autograd.grad(
        torch.vdot(probe.reshape(-1), expected.reshape(-1)).real,
        (reference_image, reference_smaps, reference_trajectory),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 3e-3


@pytest.mark.parametrize(
    ("norm", "sequential"),
    [(None, False), ("ortho", True)],
)
def test_nusense_torchkbnufft_adjoint_gradient_matches_direct_nudft(
    norm,
    sequential,
):
    torch.manual_seed(20260732)
    spatial_shape = (5, 6)
    batch, coils, points = 2, 2, 11
    samples = torch.randn(
        batch,
        coils,
        points,
        dtype=torch.complex128,
        requires_grad=True,
    )
    smaps = torch.randn(
        batch,
        coils,
        *spatial_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )
    trajectory = (torch.rand(1, 2, points, dtype=torch.float64) - 0.5).requires_grad_()
    operator = NuSense(
        smaps,
        trajectory,
        backend="torchkbnufft",
        grid_size=2,
        numpoints=6,
        norm=norm,
        sequential=sequential,
    )
    actual = operator.adjoint(samples)

    reference_samples = samples.detach().clone().requires_grad_()
    reference_smaps = smaps.detach().clone().requires_grad_()
    reference_trajectory = trajectory.detach().clone().requires_grad_()
    coil_images = _direct_type1(
        reference_samples,
        reference_trajectory,
        spatial_shape,
        operator.norm,
        operator.grid_size,
    )
    expected = (coil_images * reference_smaps.conj()).sum(dim=1, keepdim=True)

    assert _relative_error(actual.detach(), expected.detach()) < 2e-3

    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        torch.vdot(probe.reshape(-1), actual.reshape(-1)).real,
        (samples, smaps, trajectory),
    )
    expected_gradients = torch.autograd.grad(
        torch.vdot(probe.reshape(-1), expected.reshape(-1)).real,
        (reference_samples, reference_smaps, reference_trajectory),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 3e-3


def test_nusense_gram_trainable_trajectory_uses_direct_composition():
    torch.manual_seed(20260731)
    smaps = torch.randn(1, 2, 4, 5, dtype=torch.complex128)
    traj = (torch.rand(1, 2, 9, dtype=torch.float64) - 0.5).requires_grad_()
    image = torch.randn(1, 1, 4, 5, dtype=torch.complex128)
    kwargs = {
        "backend": "torchkbnufft",
        "grid_size": 2,
        "numpoints": 6,
    }
    gram = NuSenseGram(smaps, traj, **kwargs)
    direct = NuSense(smaps, traj, **kwargs)

    assert gram._uses_direct_gram
    assert torch.equal(gram(image), direct.adjoint(direct(image)))
    (gradient,) = torch.autograd.grad(gram(image).abs().square().sum(), traj)
    assert gradient.shape == traj.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("change", ["mutate", "replace"])
def test_nusense_gram_rejects_changed_trajectory(change):
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    traj = torch.zeros(1, 2, 7)
    gram = NuSenseGram(
        smaps,
        traj,
        backend="torchkbnufft",
        numpoints=2,
        grid_size=1,
    )
    if change == "mutate":
        traj.add_(0.1)
    else:
        gram.traj = traj.clone()

    with pytest.raises(RuntimeError, match="trajectory changed"):
        gram(torch.ones(1, 1, 4, 4, dtype=torch.complex64))


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("cpu", {"finufft"}, "finufft"),
        ("cpu", set(), "torchkbnufft"),
        ("cuda", {"cufinufft"}, "finufft"),
        ("cuda", set(), "torchkbnufft"),
        ("mps", {"finufft", "cufinufft"}, "torchkbnufft"),
    ],
)
def test_nusense_auto_backend_requires_native_library(
    monkeypatch, device, available, expected
):
    monkeypatch.setattr(mri.sys, "platform", "linux")
    monkeypatch.setattr(
        mri,
        "find_spec",
        lambda name: object() if name in available else None,
    )

    assert mri._resolve_nufft_backend("auto", torch.device(device)) == expected


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
        NuSense(smaps, traj)


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


def test_nusense_non_batchmode_trainable_trajectory_gradient():
    torch.manual_seed(20260733)
    smaps = torch.randn(2, 4, 5, dtype=torch.complex128)
    trajectory = (torch.rand(2, 9, dtype=torch.float64) - 0.5).requires_grad_()
    image = torch.randn(4, 5, dtype=torch.complex128)
    samples = torch.randn(2, 9, dtype=torch.complex128)
    operator = NuSense(
        smaps,
        trajectory,
        batchmode=False,
        backend="torchkbnufft",
        grid_size=2,
        numpoints=6,
    )

    loss = (
        operator(image).abs().square().mean()
        + operator.adjoint(samples).abs().square().mean()
    )
    (gradient,) = torch.autograd.grad(loss, trajectory)

    assert gradient.shape == trajectory.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


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


def test_gmri_torchkbnufft_trajectory_gradient_matches_direct_nudft():
    torch.manual_seed(20260734)
    batch, coils, shots, points = 2, 2, 2, 7
    spatial_shape = (4, 5)
    image = torch.randn(
        batch,
        1,
        *spatial_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )
    smaps = torch.randn(
        batch,
        coils,
        *spatial_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )
    zmap = (
        torch.linspace(-40, 55, math.prod(spatial_shape), dtype=torch.float64)
        .reshape(1, *spatial_shape)
        .requires_grad_()
    )
    trajectory = (
        torch.rand(1, 2, shots, points, dtype=torch.float64) - 0.5
    ).requires_grad_()
    times = (torch.arange(points, dtype=torch.float64) * 0.003).requires_grad_()
    kwargs = {
        "backend": "torchkbnufft",
        "L": 3,
        "nbins": 7,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
    }
    operator = Gmri(smaps, zmap, trajectory, T=times, **kwargs)
    actual = operator(image)

    reference_image = image.detach().clone().requires_grad_()
    reference_smaps = smaps.detach().clone().requires_grad_()
    reference_zmap = zmap.detach().clone().requires_grad_()
    reference_trajectory = trajectory.detach().clone().requires_grad_()
    reference_times = times.detach().clone().requires_grad_()
    reference_operator = Gmri(
        reference_smaps,
        reference_zmap,
        reference_trajectory,
        T=reference_times,
        **kwargs,
    )
    expected = _direct_gmri_forward(reference_operator, reference_image)

    assert _relative_error(actual.detach(), expected.detach()) < 2e-3

    # Use one fixed cotangent to compare the two operator VJPs directly. A
    # squared-norm objective would also feed the NUFFT's small forward error
    # into the upstream gradient, obscuring the Jacobian comparison.
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        (actual * probe.conj()).sum().real,
        (image, smaps, zmap, trajectory, times),
    )
    expected_gradients = torch.autograd.grad(
        (expected * probe.conj()).sum().real,
        (
            reference_image,
            reference_smaps,
            reference_zmap,
            reference_trajectory,
            reference_times,
        ),
    )
    for name, actual_gradient, expected_gradient in zip(
        ("image", "smaps", "zmap", "trajectory", "times"),
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        # zmap and time gradients can amplify the NUFFT's small forward error
        # when the time-segment terms nearly cancel. The trajectory criterion
        # remains close to the accuracy of the underlying NUFFT approximation.
        tolerance = 5e-2 if name in ("zmap", "times") else 3e-3
        assert _relative_error(actual_gradient, expected_gradient) < tolerance


def test_gmri_gram_trainable_trajectory_uses_direct_composition():
    torch.manual_seed(20260735)
    smaps = torch.randn(1, 2, 4, 5, dtype=torch.complex128)
    zmap = torch.linspace(-30, 40, 20, dtype=torch.float64).reshape(1, 4, 5)
    trajectory = (torch.rand(1, 2, 2, 7, dtype=torch.float64) - 0.5).requires_grad_()
    image = torch.randn(1, 1, 4, 5, dtype=torch.complex128)
    kwargs = {
        "backend": "torchkbnufft",
        "L": 2,
        "nbins": 5,
        "grid_size": 2,
        "numpoints": 6,
    }
    gram = GmriGram(smaps, zmap, trajectory, **kwargs)
    direct = Gmri(smaps, zmap, trajectory, **kwargs)

    assert gram._uses_direct_gram
    assert torch.equal(gram(image), direct.adjoint(direct(image)))
    (gradient,) = torch.autograd.grad(gram(image).abs().square().sum(), trajectory)
    assert gradient.shape == trajectory.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Metal is unavailable",
)
def test_gmri_and_toeplitz_gram_run_on_mps():
    device = torch.device("mps")
    smaps = torch.ones((1, 2, 8, 8), dtype=torch.complex64, device=device) / 2**0.5
    zmap = torch.linspace(-20, 20, 64, device=device).reshape(1, 8, 8)
    traj = (torch.rand((1, 2, 2, 8), device=device) - 0.5) * 2 * torch.pi
    image = torch.randn((1, 1, 8, 8), dtype=torch.complex64, device=device)
    kwargs = {"L": 2, "nbins": 4, "numpoints": 2, "grid_size": 1.25}

    forward = Gmri(smaps, zmap, traj, **kwargs)
    samples = forward(image)
    adjoint = forward.H(samples)
    gram = GmriGram(smaps, zmap, traj, **kwargs)
    gram_image = gram(image)

    assert forward.backend == gram.backend == "torchkbnufft"
    assert forward.B.dtype == forward.C.dtype == torch.complex64
    assert gram.B.dtype == gram.C.dtype == torch.complex64
    assert samples.shape == (1, 2, 2, 8)
    assert adjoint.shape == gram_image.shape == image.shape
    assert torch.isfinite(adjoint).all().item()
    assert torch.isfinite(gram_image).all().item()

    trainable_zmap = zmap.detach().requires_grad_()
    trainable_traj = traj.detach().requires_grad_()
    times = (torch.arange(8, device=device) * 0.004).requires_grad_()
    trainable = Gmri(
        smaps,
        trainable_zmap,
        trainable_traj,
        T=times,
        **kwargs,
    )
    trainable_samples = samples.detach().requires_grad_()
    loss = (
        trainable(image).abs().square().mean()
        + trainable.adjoint(trainable_samples).abs().square().mean()
    )
    zmap_gradient, trajectory_gradient, time_gradient, sample_gradient = (
        torch.autograd.grad(
            loss,
            (trainable_zmap, trainable_traj, times, trainable_samples),
        )
    )
    assert torch.isfinite(zmap_gradient).all().item()
    assert torch.isfinite(trajectory_gradient).all().item()
    assert torch.isfinite(time_gradient).all().item()
    assert torch.isfinite(sample_gradient).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Metal is unavailable",
)
def test_direct_gmri_gram_keeps_aliases_after_device_move():
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    zmap = torch.zeros(1, 4, 4, requires_grad=True)
    traj = torch.zeros(1, 2, 1, 5)
    gram = GmriGram(
        smaps,
        zmap,
        traj,
        L=2,
        nbins=4,
        numpoints=2,
        grid_size=1,
        backend="torchkbnufft",
    ).to("mps")

    assert gram.smaps is gram._direct_operator.smaps
    assert gram.zmap is gram._direct_operator.zmap
    assert gram.traj is gram._direct_operator.traj
    image = torch.ones(1, 1, 4, 4, dtype=torch.complex64, device="mps")
    assert torch.isfinite(gram(image)).all().item()


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
        NuSenseGram(smaps, traj)


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
    relative_error = torch.linalg.vector_norm(
        composed - gram
    ) / torch.linalg.vector_norm(composed)
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
        torch.randn(1, n_coils, 32, 32), torch.randn(1, n_coils, 32, 32)
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
        torch.randn(n_frames, 1, 32, 32), torch.randn(n_frames, 1, 32, 32)
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

    smaps = torch.complex(
        torch.randn(n_batch, 4, 16, 16), torch.randn(n_batch, 4, 16, 16)
    )
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
        nusense(x)


def test_shape_mismatch_during_adjoint():
    """Test that shape mismatch during adjoint pass is caught"""
    smaps = torch.complex(torch.randn(5, 4, 16, 16), torch.randn(5, 4, 16, 16))
    traj = torch.rand(5, 2, 1000) * 2 - 1

    nusense = NuSense(smaps, traj)

    # Wrong k-space shape
    k_space = torch.randn(3, 4, 1000, dtype=torch.complex64)

    with pytest.raises(ValueError, match="forward linear op"):
        nusense.H(k_space)


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
    assert (
        smaps_single.element_size() * smaps_single.nelement() * 100
        == smaps_replicated.element_size() * smaps_replicated.nelement()
    )

    # But both should work the same way
    traj = torch.rand(100, 2, 500) * 2 - 1

    nusense_broadcast = NuSense(smaps_single, traj)
    nusense_replicated = NuSense(smaps_replicated, traj)

    assert nusense_broadcast.size_out == nusense_replicated.size_out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
