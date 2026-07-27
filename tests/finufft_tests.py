import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from mirtorch.alg import CG
from mirtorch.linear import Gmri, GmriGram, Identity, NuSense, NuSenseGram
from mirtorch.linear._finufft import FinufftSenseBackend


def _finufft_device() -> torch.device:
    if torch.cuda.is_available() and importlib.util.find_spec("cufinufft"):
        return torch.device("cuda")
    if sys.platform == "darwin":
        pytest.skip(
            "The PyPI FINUFFT and PyTorch wheels load duplicate OpenMP "
            "runtimes on macOS; CPU execution is disabled until a supported "
            "packaging solution is available."
        )
    if importlib.util.find_spec("finufft"):
        return torch.device("cpu")
    pytest.skip("FINUFFT/cuFINUFFT is not installed")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )


def _exact_type1(
    backend: FinufftSenseBackend,
    samples: torch.Tensor,
    traj: torch.Tensor,
    mode_size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    mode_size = backend.im_size if mode_size is None else mode_size
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=traj.dtype,
            device=traj.device,
        )
        for size in mode_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    phase_argument = sum(
        traj[:, dimension].reshape(
            traj.shape[0],
            traj.shape[-1],
            *([1] * len(mode_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(mode_size))
    )
    phase = torch.exp(1j * phase_argument)
    sample_shape = (samples.shape[0], samples.shape[1], samples.shape[2]) + (1,) * len(
        mode_size
    )
    return (samples.reshape(sample_shape) * phase.unsqueeze(1)).sum(dim=2)


def _exact_type2(
    backend: FinufftSenseBackend,
    modes: torch.Tensor,
    traj: torch.Tensor,
) -> torch.Tensor:
    image = torch.ones(
        modes.shape[0],
        1,
        *modes.shape[2:],
        dtype=modes.dtype,
        device=modes.device,
    )
    return _direct_sense(image, modes, traj, scale=1)


def _direct_gmri_phases(
    zmap: torch.Tensor,
    traj: torch.Tensor,
    dt: float,
    im_size: tuple[int, ...],
) -> torch.Tensor:
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=traj.dtype,
            device=traj.device,
        )
        for size in im_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    spatial_phase = sum(
        traj[:, dimension].reshape(
            traj.shape[0],
            *traj.shape[2:],
            *([1] * len(im_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(im_size))
    )
    times = (
        torch.arange(
            traj.shape[-1],
            dtype=traj.dtype,
            device=traj.device,
        )
        * dt
        / 1000
    )
    field_shape = (zmap.shape[0], 1, 1, 1) + im_size
    time_shape = (1, 1, 1, traj.shape[-1]) + (1,) * len(im_size)
    field_phase = 2 * math.pi * zmap.reshape(field_shape) * times.reshape(time_shape)
    return spatial_phase.unsqueeze(1) + field_phase


def _direct_gmri(
    image: torch.Tensor,
    smaps: torch.Tensor,
    zmap: torch.Tensor,
    traj: torch.Tensor,
    dt: float,
    scale: float,
) -> torch.Tensor:
    batch = image.shape[0]
    if smaps.shape[0] == 1:
        smaps = smaps.expand(batch, *smaps.shape[1:])
    if zmap.shape[0] == 1:
        zmap = zmap.expand(batch, *zmap.shape[1:])
    if traj.shape[0] == 1:
        traj = traj.expand(batch, *traj.shape[1:])

    im_size = tuple(image.shape[2:])
    phase = torch.exp(-1j * _direct_gmri_phases(zmap, traj, dt, im_size))
    spatial_dims = tuple(range(4, 4 + len(im_size)))
    return ((image * smaps).unsqueeze(2).unsqueeze(2) * phase).sum(
        dim=spatial_dims
    ) * scale


def _direct_gmri_adjoint(
    samples: torch.Tensor,
    smaps: torch.Tensor,
    zmap: torch.Tensor,
    traj: torch.Tensor,
    dt: float,
    im_size: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    batch = samples.shape[0]
    if smaps.shape[0] == 1:
        smaps = smaps.expand(batch, *smaps.shape[1:])
    if zmap.shape[0] == 1:
        zmap = zmap.expand(batch, *zmap.shape[1:])
    if traj.shape[0] == 1:
        traj = traj.expand(batch, *traj.shape[1:])

    phase = torch.exp(1j * _direct_gmri_phases(zmap, traj, dt, im_size))
    sample_shape = samples.shape + (1,) * len(im_size)
    coil_images = (samples.reshape(sample_shape) * phase).sum(dim=(2, 3)) * scale
    return (smaps.conj() * coil_images).sum(dim=1, keepdim=True)


def test_finufft_plan_reuses_fixed_coordinates_and_invalidates_mutations(monkeypatch):
    plans = []

    class FakePlan:
        def __init__(self, *_args, **_kwargs):
            self.setpts_calls = 0
            plans.append(self)

        def setpts(self, *_coordinates):
            self.setpts_calls += 1

        def execute(self, data):
            return data

    class FakeLibrary:
        Plan = FakePlan

    backend = FinufftSenseBackend(
        im_size=(5,),
        grid_size=(10,),
        norm=None,
        batchmode=True,
        sequential=False,
        eps=1e-6,
    )
    monkeypatch.setattr(backend, "_library", lambda _device: FakeLibrary())
    data = torch.ones(1, 5, dtype=torch.complex64)
    coordinates = torch.zeros(1, 5)

    backend._execute(1, data, coordinates, mode_size=(5,))
    backend._execute(1, data, coordinates, mode_size=(5,))
    assert len(plans) == 1
    assert plans[0].setpts_calls == 1

    coordinates.add_(0.1)
    backend._execute(1, data, coordinates, mode_size=(5,))
    assert plans[0].setpts_calls == 2

    backend.clear_plans()
    assert not backend._plans
    assert not backend._coordinate_cache


def test_finufft_gmri_batches_all_segments_into_one_transform(monkeypatch):
    calls = {"type1": 0, "type2": 0}

    def counted_type1(backend, samples, traj, mode_size=None):
        calls["type1"] += 1
        return _exact_type1(backend, samples, traj, mode_size)

    def counted_type2(backend, modes, traj):
        calls["type2"] += 1
        return _exact_type2(backend, modes, traj)

    monkeypatch.setattr(FinufftSenseBackend, "type1", counted_type1)
    monkeypatch.setattr(FinufftSenseBackend, "type2", counted_type2)
    smaps = torch.randn(1, 2, 5, 4, dtype=torch.complex128)
    zmap = torch.linspace(-30, 40, 20, dtype=torch.float64).reshape(1, 5, 4)
    traj = torch.randn(1, 2, 2, 9, dtype=torch.float64) * 0.1
    operator = Gmri(
        smaps,
        zmap,
        traj,
        L=5,
        nbins=10,
        backend="finufft",
        eps=1e-12,
    )

    image = torch.randn(1, 1, 5, 4, dtype=torch.complex128)
    samples = torch.randn(1, 2, 2, 9, dtype=torch.complex128)
    operator(image)
    operator.H(samples)

    assert calls == {"type1": 1, "type2": 1}


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS wheel-specific check")
def test_finufft_macos_duplicate_openmp_fails_safely():
    finufft_spec = importlib.util.find_spec("finufft")
    if finufft_spec is None or finufft_spec.origin is None:
        pytest.skip("FINUFFT is not installed")

    torch_omp = Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"
    finufft_omp = (
        Path(finufft_spec.origin).resolve().parent / ".dylibs" / "libomp.dylib"
    )
    if not torch_omp.exists() or not finufft_omp.exists():
        pytest.skip("The installed wheels do not bundle duplicate OpenMP runtimes")

    image = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    smaps = torch.ones_like(image)
    traj = torch.zeros(1, 2, 3)
    operator = NuSense(smaps, traj, backend="finufft")

    with pytest.raises(RuntimeError, match="separate OpenMP runtimes"):
        operator(image)


@pytest.mark.parametrize("norm", [None, "ortho"])
@pytest.mark.parametrize("shared_traj", [False, True])
def test_finufft_toeplitz_embedding_matches_exact_normal(
    monkeypatch,
    norm,
    shared_traj,
):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    torch.manual_seed(26)
    dtype = torch.complex128
    batch, coils, height, width, count = 2, 3, 5, 4, 23
    image = torch.randn(batch, 1, height, width, dtype=dtype)
    smaps_batch = batch if shared_traj else 1
    traj_batch = 1 if shared_traj else batch
    smaps = torch.randn(smaps_batch, coils, height, width, dtype=dtype)
    traj = (torch.rand(traj_batch, 2, count, dtype=torch.float64) * 2 - 1) * torch.pi

    gram = NuSenseGram(
        smaps,
        traj,
        backend="finufft",
        norm=norm,
        grid_size=2,
        eps=1e-12,
    )
    scale = 1 if norm is None else 1 / math.sqrt((2 * height) * (2 * width))
    samples = _direct_sense(image, smaps, traj, scale)
    expected = _direct_sense_adjoint(
        samples,
        smaps,
        traj,
        (height, width),
        scale,
    )

    assert _relative_error(gram(image), expected) < 2e-14
    assert _relative_error(gram.H(image), expected) < 2e-14


def test_finufft_toeplitz_rejects_trainable_or_changed_trajectory(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    trainable_traj = torch.zeros(1, 2, 7, requires_grad=True)

    with pytest.raises(ValueError, match="fixed trajectories"):
        NuSenseGram(smaps, trainable_traj, backend="finufft")

    traj = torch.zeros(1, 2, 7)
    gram = NuSenseGram(smaps, traj, backend="finufft")
    traj.add_(0.1)
    with pytest.raises(RuntimeError, match="trajectory changed"):
        gram(torch.ones(1, 1, 4, 4, dtype=torch.complex64))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal is not available",
)
def test_cached_finufft_toeplitz_kernel_runs_on_metal(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    torch.manual_seed(27)
    image = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    smaps = torch.randn(1, 2, 4, 4, dtype=torch.complex64)
    traj = (torch.rand(1, 2, 17) * 2 - 1) * torch.pi
    traj.add_(0.01)
    gram = NuSenseGram(smaps, traj, backend="finufft")
    expected = gram(image)

    gram.to("mps")
    actual = gram(image.to("mps")).cpu()

    assert _relative_error(actual, expected) < 2e-6


def _direct_sense(
    image: torch.Tensor,
    smaps: torch.Tensor,
    traj: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch = image.shape[0]
    if smaps.shape[0] == 1:
        smaps = smaps.expand(batch, *smaps.shape[1:])
    if traj.shape[0] == 1:
        traj = traj.expand(batch, *traj.shape[1:])

    im_size = image.shape[2:]
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=traj.dtype,
            device=traj.device,
        )
        for size in im_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    phase_argument = sum(
        traj[:, dimension].reshape(
            batch,
            traj.shape[-1],
            *([1] * len(im_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(im_size))
    )
    phase = torch.exp(-1j * phase_argument)
    coil_images = image * smaps
    spatial_dims = tuple(range(3, 3 + len(im_size)))
    return (coil_images.unsqueeze(2) * phase.unsqueeze(1)).sum(dim=spatial_dims) * scale


def _direct_sense_adjoint(
    samples: torch.Tensor,
    smaps: torch.Tensor,
    traj: torch.Tensor,
    im_size: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    batch = samples.shape[0]
    if smaps.shape[0] == 1:
        smaps = smaps.expand(batch, *smaps.shape[1:])
    if traj.shape[0] == 1:
        traj = traj.expand(batch, *traj.shape[1:])

    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=traj.dtype,
            device=traj.device,
        )
        for size in im_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    phase_argument = sum(
        traj[:, dimension].reshape(
            batch,
            traj.shape[-1],
            *([1] * len(im_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(im_size))
    )
    phase = torch.exp(1j * phase_argument)
    sample_shape = (batch, samples.shape[1], samples.shape[2]) + (1,) * len(im_size)
    coil_images = (samples.reshape(sample_shape) * phase.unsqueeze(1)).sum(
        dim=2
    ) * scale
    return (smaps.conj() * coil_images).sum(dim=1, keepdim=True)


def test_finufft_matches_legacy_forward_adjoint_and_tensor_gradients():
    device = _finufft_device()
    torch.manual_seed(21)
    dtype = torch.complex64
    real_dtype = torch.float32
    batch, coils, height, width, samples = 2, 3, 16, 12, 173

    image_data = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps_data = torch.randn(
        1,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    traj = (
        torch.rand(
            batch,
            2,
            samples,
            dtype=real_dtype,
            device=device,
        )
        * 2
        - 1
    ) * torch.pi
    upstream = torch.randn(
        batch,
        coils,
        samples,
        dtype=dtype,
        device=device,
    )

    legacy_image = image_data.clone().requires_grad_()
    legacy_smaps = smaps_data.clone().requires_grad_()
    legacy = NuSense(
        legacy_smaps,
        traj,
        backend="torchkbnufft",
        grid_size=2,
        numpoints=6,
        norm="ortho",
    )
    legacy_samples = legacy(legacy_image)
    legacy_loss = (upstream.conj() * legacy_samples).sum().real
    legacy_grad_image, legacy_grad_smaps = torch.autograd.grad(
        legacy_loss,
        (legacy_image, legacy_smaps),
    )

    finufft_image = image_data.clone().requires_grad_()
    finufft_smaps = smaps_data.clone().requires_grad_()
    finufft = NuSense(
        finufft_smaps,
        traj,
        backend="finufft",
        grid_size=2,
        norm="ortho",
        eps=1e-6,
    )
    finufft_samples = finufft(finufft_image)
    finufft_loss = (upstream.conj() * finufft_samples).sum().real
    finufft_grad_image, finufft_grad_smaps = torch.autograd.grad(
        finufft_loss,
        (finufft_image, finufft_smaps),
    )

    assert _relative_error(finufft_samples, legacy_samples) < 2e-3
    assert _relative_error(finufft_grad_image, legacy_grad_image) < 2e-3
    assert _relative_error(finufft_grad_smaps, legacy_grad_smaps) < 2e-3

    adjoint_input = torch.randn_like(upstream)
    finufft_adjoint = finufft.H(adjoint_input)
    legacy_adjoint = legacy.H(adjoint_input)
    assert _relative_error(finufft_adjoint, legacy_adjoint) < 2e-3

    lhs = (finufft_samples.conj() * adjoint_input).sum()
    rhs = (finufft_image.conj() * finufft_adjoint).sum()
    assert torch.allclose(lhs, rhs, rtol=2e-5, atol=2e-5)


def test_finufft_forward_and_snopy_vjp_match_direct_nudft():
    device = _finufft_device()
    torch.manual_seed(22)
    dtype = torch.complex128
    batch, coils, height, width, samples = 2, 2, 6, 5, 19

    image_data = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps_data = torch.randn(
        1,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    shared_traj = (
        torch.rand(1, 2, samples, dtype=torch.float64, device=device) * 2 - 1
    ) * torch.pi
    traj_data = shared_traj.expand(batch, -1, -1).clone()
    upstream = torch.randn(
        batch,
        coils,
        samples,
        dtype=dtype,
        device=device,
    )

    image = image_data.clone().requires_grad_()
    smaps = smaps_data.clone().requires_grad_()
    traj = traj_data.clone().requires_grad_()
    operator = NuSense(
        smaps,
        traj,
        backend="finufft",
        norm=None,
        eps=1e-12,
    )
    actual = operator(image)
    actual_loss = (upstream.conj() * actual).sum().real
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (image, smaps, traj),
    )

    direct_image = image_data.clone().requires_grad_()
    direct_smaps = smaps_data.clone().requires_grad_()
    direct_traj = traj_data.clone().requires_grad_()
    expected = _direct_sense(
        direct_image,
        direct_smaps,
        direct_traj,
        scale=1,
    )
    expected_loss = (upstream.conj() * expected).sum().real
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (direct_image, direct_smaps, direct_traj),
    )

    assert _relative_error(actual, expected) < 2e-11
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 2e-11


def test_finufft_adjoint_vjp_matches_direct_nudft():
    device = _finufft_device()
    torch.manual_seed(23)
    dtype = torch.complex128
    batch, coils, height, width, count = 2, 2, 6, 5, 17

    samples_data = torch.randn(
        batch,
        coils,
        count,
        dtype=dtype,
        device=device,
    )
    shared_smaps = torch.randn(
        1,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps_data = shared_smaps.expand(batch, -1, -1, -1).clone()
    traj_data = (
        torch.rand(1, 2, count, dtype=torch.float64, device=device) * 2 - 1
    ) * torch.pi
    upstream = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )

    samples = samples_data.clone().requires_grad_()
    smaps = smaps_data.clone().requires_grad_()
    traj = traj_data.clone().requires_grad_()
    operator = NuSense(
        smaps,
        traj,
        backend="finufft",
        norm=None,
        eps=1e-12,
    )
    actual = operator.H(samples)
    actual_loss = (upstream.conj() * actual).sum().real
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (samples, smaps, traj),
    )

    direct_samples = samples_data.clone().requires_grad_()
    direct_smaps = smaps_data.clone().requires_grad_()
    direct_traj = traj_data.clone().requires_grad_()
    expected = _direct_sense_adjoint(
        direct_samples,
        direct_smaps,
        direct_traj,
        (height, width),
        scale=1,
    )
    expected_loss = (upstream.conj() * expected).sum().real
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (direct_samples, direct_smaps, direct_traj),
    )

    assert _relative_error(actual, expected) < 2e-11
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 2e-11


@pytest.mark.parametrize("sequential", [False, True])
def test_finufft_nonbatch_mode_matches_legacy(sequential):
    device = _finufft_device()
    torch.manual_seed(24)
    dtype = torch.complex64
    coils, height, width, count = 2, 10, 8, 79
    image = torch.randn(height, width, dtype=dtype, device=device)
    smaps = torch.randn(coils, height, width, dtype=dtype, device=device)
    traj = (torch.rand(2, count, dtype=torch.float32, device=device) * 2 - 1) * torch.pi

    legacy = NuSense(
        smaps,
        traj,
        batchmode=False,
        sequential=sequential,
        backend="torchkbnufft",
    )
    finufft = NuSense(
        smaps,
        traj,
        batchmode=False,
        sequential=sequential,
        backend="finufft",
        eps=1e-6,
    )

    assert _relative_error(finufft(image), legacy(image)) < 2e-3
    samples = torch.randn(coils, count, dtype=dtype, device=device)
    assert _relative_error(finufft.H(samples), legacy.H(samples)) < 2e-3


def test_finufft_legacy_ortho_scale_uses_oversampled_grid():
    device = _finufft_device()
    torch.manual_seed(25)
    dtype = torch.complex128
    height, width, count = 6, 5, 13
    image = torch.randn(1, 1, height, width, dtype=dtype, device=device)
    smaps = torch.ones(1, 1, height, width, dtype=dtype, device=device)
    traj = (
        torch.rand(1, 2, count, dtype=torch.float64, device=device) * 2 - 1
    ) * torch.pi

    operator = NuSense(
        smaps,
        traj,
        backend="finufft",
        grid_size=2,
        norm="ortho",
        eps=1e-12,
    )
    actual = operator(image)
    grid_size = (math.floor(height * 2), math.floor(width * 2))
    expected = _direct_sense(
        image,
        smaps,
        traj,
        scale=1 / math.sqrt(math.prod(grid_size)),
    )
    assert _relative_error(actual, expected) < 2e-11


def test_finufft_and_legacy_cg_reconstructions_match():
    device = _finufft_device()
    torch.manual_seed(41)
    dtype = torch.complex64
    batch, coils, height, width, count = 1, 4, 16, 16, 800

    truth = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps = torch.randn(
        batch,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps = smaps / torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True))
    traj = (
        torch.rand(batch, 2, count, dtype=torch.float32, device=device) * 2 - 1
    ) * torch.pi

    legacy = NuSense(
        smaps,
        traj,
        backend="torchkbnufft",
        grid_size=2,
        numpoints=6,
    )
    finufft = NuSense(
        smaps,
        traj,
        backend="finufft",
        grid_size=2,
        eps=1e-6,
    )
    measurements = legacy(truth)
    regularization = 0.05
    identity = Identity(legacy.size_in)
    legacy_normal = legacy.H * legacy + regularization * identity
    finufft_normal = finufft.H * finufft + regularization * identity
    finufft_toeplitz_normal = (
        NuSenseGram(
            smaps,
            traj,
            backend="finufft",
            grid_size=2,
            eps=1e-6,
        )
        + regularization * identity
    )

    legacy_reconstruction = CG(
        legacy_normal,
        max_iter=12,
        tol=0,
    ).run(
        torch.zeros_like(truth),
        legacy.H(measurements),
    )
    finufft_reconstruction = CG(
        finufft_normal,
        max_iter=12,
        tol=0,
    ).run(
        torch.zeros_like(truth),
        finufft.H(measurements),
    )
    finufft_toeplitz_reconstruction = CG(
        finufft_toeplitz_normal,
        max_iter=12,
        tol=0,
    ).run(
        torch.zeros_like(truth),
        finufft.H(measurements),
    )

    assert _relative_error(finufft_reconstruction, legacy_reconstruction) < 1e-3
    assert (
        _relative_error(
            finufft_toeplitz_reconstruction,
            finufft_reconstruction,
        )
        < 2e-4
    )


def test_finufft_toeplitz_matches_direct_gram_legacy_and_gradients():
    device = _finufft_device()
    torch.manual_seed(42)
    dtype = torch.complex64
    batch, coils, height, width, count = 2, 3, 12, 10, 211

    image_data = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps_data = torch.randn(
        1,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    traj = (
        torch.rand(batch, 2, count, dtype=torch.float32, device=device) * 2 - 1
    ) * torch.pi
    upstream = torch.randn_like(image_data)

    direct_image = image_data.clone().requires_grad_()
    direct_smaps = smaps_data.clone().requires_grad_()
    direct = NuSense(
        direct_smaps,
        traj,
        backend="finufft",
        norm="ortho",
        grid_size=2,
        eps=1e-6,
    )
    direct_output = direct.H(direct(direct_image))
    direct_loss = (upstream.conj() * direct_output).sum().real
    direct_gradients = torch.autograd.grad(
        direct_loss,
        (direct_image, direct_smaps),
    )

    gram_image = image_data.clone().requires_grad_()
    gram_smaps = smaps_data.clone().requires_grad_()
    gram = NuSenseGram(
        gram_smaps,
        traj,
        backend="finufft",
        norm="ortho",
        grid_size=2,
        eps=1e-6,
    )
    gram_output = gram(gram_image)
    gram_loss = (upstream.conj() * gram_output).sum().real
    gram_gradients = torch.autograd.grad(
        gram_loss,
        (gram_image, gram_smaps),
    )

    legacy = NuSenseGram(
        smaps_data,
        traj,
        backend="torchkbnufft",
        norm="ortho",
        grid_size=2,
        numpoints=6,
    )
    legacy_output = legacy(image_data)

    assert _relative_error(gram_output, direct_output) < 2e-5
    assert _relative_error(gram_output, legacy_output) < 3e-3
    for gram_gradient, direct_gradient in zip(
        gram_gradients,
        direct_gradients,
        strict=True,
    ):
        assert _relative_error(gram_gradient, direct_gradient) < 3e-5

    probe = torch.randn_like(image_data)
    lhs = (probe.conj() * gram(image_data)).sum()
    rhs = (gram(probe).conj() * image_data).sum()
    assert torch.allclose(lhs, rhs, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("im_size", [(7,), (5, 4), (4, 3, 3)])
def test_finufft_toeplitz_matches_direct_gram_in_each_dimension(im_size):
    device = _finufft_device()
    torch.manual_seed(43)
    batch, coils, count = 1, 2, 19
    image = torch.randn(
        batch,
        1,
        *im_size,
        dtype=torch.complex64,
        device=device,
    )
    smaps = torch.randn(
        batch,
        coils,
        *im_size,
        dtype=torch.complex64,
        device=device,
    )
    traj = (
        torch.rand(
            batch,
            len(im_size),
            count,
            dtype=torch.float32,
            device=device,
        )
        * 2
        - 1
    ) * torch.pi

    direct = NuSense(
        smaps,
        traj,
        backend="finufft",
        norm=None,
        eps=1e-6,
    )
    gram = NuSenseGram(
        smaps,
        traj,
        backend="finufft",
        norm=None,
        eps=1e-6,
    )

    assert _relative_error(gram(image), direct.H(direct(image))) < 3e-5


def test_finufft_toeplitz_nonbatch_mode_matches_direct_gram():
    device = _finufft_device()
    torch.manual_seed(44)
    coils, height, width, count = 2, 7, 6, 31
    image = torch.randn(
        height,
        width,
        dtype=torch.complex64,
        device=device,
    )
    smaps = torch.randn(
        coils,
        height,
        width,
        dtype=torch.complex64,
        device=device,
    )
    traj = (torch.rand(2, count, dtype=torch.float32, device=device) * 2 - 1) * torch.pi

    direct = NuSense(
        smaps,
        traj,
        batchmode=False,
        backend="finufft",
        norm="ortho",
        eps=1e-6,
    )
    gram = NuSenseGram(
        smaps,
        traj,
        batchmode=False,
        backend="finufft",
        norm="ortho",
        eps=1e-6,
    )

    assert _relative_error(gram(image), direct.H(direct(image))) < 3e-5


def test_finufft_gmri_matches_legacy_and_exact_segment_gradients(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    monkeypatch.setattr(FinufftSenseBackend, "type2", _exact_type2)
    torch.manual_seed(45)
    dtype = torch.complex128
    batch, coils, height, width, shots, points = 2, 3, 12, 10, 3, 31
    image_data = torch.randn(batch, 1, height, width, dtype=dtype)
    smaps_data = torch.randn(1, coils, height, width, dtype=dtype)
    zmap = torch.linspace(-180, 150, height * width, dtype=torch.float64).reshape(
        1,
        height,
        width,
    )
    traj_data = (
        torch.rand(batch, 2, shots, points, dtype=torch.float64) * 2 - 1
    ) * torch.pi
    upstream = torch.randn(batch, coils, shots, points, dtype=dtype)
    kwargs = {
        "L": 5,
        "nbins": 24,
        "dt": 0.02,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
    }

    legacy = Gmri(
        smaps_data,
        zmap,
        traj_data,
        backend="torchkbnufft",
        **kwargs,
    )
    legacy_output = legacy(image_data)

    image = image_data.clone().requires_grad_()
    smaps = smaps_data.clone().requires_grad_()
    traj = traj_data.clone().requires_grad_()
    operator = Gmri(
        smaps,
        zmap,
        traj,
        backend="finufft",
        eps=1e-12,
        **kwargs,
    )
    actual = operator(image)
    actual_gradients = torch.autograd.grad(
        (upstream.conj() * actual).sum().real,
        (image, smaps, traj),
    )

    direct_image = image_data.clone().requires_grad_()
    direct_smaps = smaps_data.clone().requires_grad_()
    direct_traj = traj_data.clone().requires_grad_()
    flat_traj = direct_traj.flatten(2)
    scale = 1 / math.sqrt((2 * height) * (2 * width))
    expected = torch.zeros_like(actual)
    for segment in range(operator.L):
        segment_output = _direct_sense(
            direct_image * operator.C[segment],
            direct_smaps,
            flat_traj,
            scale,
        ).reshape(actual.shape)
        expected = expected + operator.B[segment] * segment_output
    expected_gradients = torch.autograd.grad(
        (upstream.conj() * expected).sum().real,
        (direct_image, direct_smaps, direct_traj),
    )

    assert _relative_error(actual, legacy_output) < 2e-3
    assert _relative_error(actual, expected) < 2e-12
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert _relative_error(actual_gradient, expected_gradient) < 2e-12

    adjoint_input = torch.randn_like(upstream)
    assert (
        _relative_error(
            operator.H(adjoint_input),
            legacy.H(adjoint_input),
        )
        < 2e-3
    )


def test_finufft_gmri_runs_on_supported_cpu_or_cuda():
    device = _finufft_device()
    torch.manual_seed(47)
    dtype = torch.complex64
    batch, coils, height, width, shots, points = 1, 2, 12, 10, 3, 41
    image = torch.randn(
        batch,
        1,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    smaps = torch.randn(
        batch,
        coils,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    zmap = torch.linspace(
        -180,
        150,
        height * width,
        dtype=torch.float32,
        device=device,
    ).reshape(batch, height, width)
    traj = (
        torch.rand(
            batch,
            2,
            shots,
            points,
            dtype=torch.float32,
            device=device,
        )
        * 2
        - 1
    ) * torch.pi
    kwargs = {
        "L": 5,
        "nbins": 24,
        "dt": 0.02,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
    }
    finufft = Gmri(
        smaps,
        zmap,
        traj,
        backend="finufft",
        eps=1e-6,
        **kwargs,
    )
    legacy = Gmri(
        smaps,
        zmap,
        traj,
        backend="torchkbnufft",
        **kwargs,
    )

    assert _relative_error(finufft(image), legacy(image)) < 3e-3
    samples = torch.randn(
        batch,
        coils,
        shots,
        points,
        dtype=dtype,
        device=device,
    )
    assert _relative_error(finufft.H(samples), legacy.H(samples)) < 3e-3

    finufft_gram = GmriGram(
        smaps,
        zmap,
        traj,
        backend="finufft",
        eps=1e-6,
        **kwargs,
    )
    legacy_gram = GmriGram(
        smaps,
        zmap,
        traj,
        backend="torchkbnufft",
        **kwargs,
    )
    assert _relative_error(finufft_gram(image), legacy_gram(image)) < 3e-3


def test_finufft_gmri_gram_autocorrelation_matches_b0_normal(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    torch.manual_seed(46)
    dtype = torch.complex128
    batch, coils, height, width, shots, points = 1, 2, 9, 8, 4, 39
    image = torch.randn(batch, 1, height, width, dtype=dtype)
    smaps = torch.randn(batch, coils, height, width, dtype=dtype)
    smaps = smaps / torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True))
    zmap = (torch.rand(batch, height, width, dtype=torch.float64) * 2 - 1) * 350
    traj = (torch.rand(batch, 2, shots, points, dtype=torch.float64) * 2 - 1) * torch.pi
    dt = 0.08
    scale = 1 / math.sqrt((2 * height) * (2 * width))
    samples = _direct_gmri(image, smaps, zmap, traj, dt, scale)
    expected = _direct_gmri_adjoint(
        samples,
        smaps,
        zmap,
        traj,
        dt,
        (height, width),
        scale,
    )
    kwargs = {
        "L": 8,
        "nbins": 40,
        "dt": dt,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
        "backend": "finufft",
        "eps": 1e-12,
    }
    gram = GmriGram(smaps, zmap, traj, **kwargs)
    actual = gram(image)

    assert gram.B.imag.abs().max() == 0
    assert _relative_error(actual, expected) < 2e-3

    torchkbnufft_gram = GmriGram(
        smaps,
        zmap,
        traj,
        backend="torchkbnufft",
        **{key: value for key, value in kwargs.items() if key != "backend"},
    )
    assert _relative_error(actual, torchkbnufft_gram(image)) < 2e-3

    probe = torch.randn_like(image)
    lhs = (probe.conj() * gram(image)).sum()
    rhs = (gram(probe).conj() * image).sum()
    assert torch.allclose(lhs, rhs, rtol=2e-12, atol=2e-12)


def test_finufft_gmri_gram_rejects_changed_trajectory(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    zmap = torch.zeros(1, 4, 4)
    traj = torch.zeros(1, 2, 2, 7)
    gram = GmriGram(
        smaps,
        zmap,
        traj,
        L=2,
        nbins=4,
        backend="finufft",
    )

    traj.add_(0.1)
    with pytest.raises(RuntimeError, match="trajectory changed"):
        gram(torch.ones(1, 1, 4, 4, dtype=torch.complex64))


def test_finufft_gmri_gram_uses_direct_path_for_trajectory_gradient(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    monkeypatch.setattr(FinufftSenseBackend, "type2", _exact_type2)
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex128)
    zmap = torch.zeros(1, 4, 4, dtype=torch.float64)
    traj = (torch.rand(1, 2, 2, 7, dtype=torch.float64) - 0.5).requires_grad_()
    image = torch.randn(1, 1, 4, 4, dtype=torch.complex128)
    gram = GmriGram(
        smaps,
        zmap,
        traj,
        L=2,
        nbins=4,
        backend="finufft",
        eps=1e-12,
    )

    assert gram._uses_direct_gram
    (gradient,) = torch.autograd.grad(gram(image).abs().square().sum(), traj)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
