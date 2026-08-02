import math

import torch

from mirtorch.alg import CG
from mirtorch.linear import Gmri, GmriGram, Identity
from mirtorch.linear._finufft import FinufftSenseBackend


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )


def _synthetic_cartesian_case(
    size: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    coordinates = torch.linspace(-1, 1, size, dtype=real_dtype)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    phantom = 0.6 * ((x / 0.72).square() + (y / 0.88).square() <= 1)
    phantom = phantom + 0.35 * (
        ((x + 0.26) / 0.18).square() + ((y - 0.12) / 0.24).square() <= 1
    )
    phantom = phantom - 0.18 * (
        ((x - 0.23) / 0.14).square() + ((y + 0.25) / 0.18).square() <= 1
    )
    image = (
        (phantom * torch.exp(1j * (0.25 * x - 0.16 * y)))
        .reshape(1, 1, size, size)
        .to(complex_dtype)
    )

    coil_count = 3
    angles = torch.arange(coil_count, dtype=real_dtype) * (2 * math.pi / coil_count)
    smaps = (
        torch.stack(
            [
                (1 + 0.22 * (x * torch.cos(angle) + y * torch.sin(angle)))
                * torch.exp(
                    1j
                    * (
                        0.35 * (x * torch.sin(angle) - y * torch.cos(angle))
                        + 0.13 * angle
                    )
                )
                for angle in angles
            ]
        )
        .reshape(1, coil_count, size, size)
        .to(complex_dtype)
    )
    smaps = smaps / torch.sqrt(smaps.abs().square().sum(dim=1, keepdim=True))

    zmap = (
        110 * x - 75 * y + 35 * torch.sin(math.pi * x) * torch.cos(math.pi * y)
    ).reshape(1, size, size)

    frequencies = torch.arange(-(size // 2), (size - 1) // 2 + 1, dtype=real_dtype) * (
        2 * math.pi / size
    )
    ky, kx = torch.meshgrid(frequencies, frequencies, indexing="ij")
    trajectory = torch.stack((ky, kx)).reshape(1, 2, size, size)
    return image, smaps, zmap, trajectory


def _direct_b0_encoding(
    image: torch.Tensor,
    smaps: torch.Tensor,
    zmap: torch.Tensor,
    trajectory: torch.Tensor,
    *,
    dt: float | None = None,
    times: torch.Tensor | None = None,
    grid_size: tuple[int, ...],
) -> torch.Tensor:
    spatial_shape = tuple(image.shape[2:])
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        for size in spatial_shape
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    spatial_phase = sum(
        trajectory[:, dimension].reshape(
            trajectory.shape[0],
            trajectory.shape[2],
            trajectory.shape[3],
            *([1] * len(spatial_shape)),
        )
        * mode_grids[dimension]
        for dimension in range(len(spatial_shape))
    )
    if times is None:
        if dt is None:
            raise ValueError("dt is required when explicit times are omitted")
        times = (
            torch.arange(
                trajectory.shape[-1],
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            * dt
        )
    elif dt is not None:
        raise ValueError("provide either dt or explicit times, not both")
    times = times / 1000
    field_phase = (
        2
        * math.pi
        * zmap[:, None, None]
        * times.reshape(1, 1, -1, *([1] * len(spatial_shape)))
    )
    phase = torch.exp(-1j * (spatial_phase + field_phase)).unsqueeze(1)
    spatial_dims = tuple(range(4, 4 + len(spatial_shape)))
    return ((image * smaps).unsqueeze(2).unsqueeze(2) * phase).sum(
        dim=spatial_dims
    ) / math.sqrt(math.prod(grid_size))


def _exact_type1(
    backend: FinufftSenseBackend,
    samples: torch.Tensor,
    trajectory: torch.Tensor,
    mode_size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    mode_size = backend.im_size if mode_size is None else mode_size
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        for size in mode_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    phase_argument = sum(
        trajectory[:, dimension].reshape(
            trajectory.shape[0],
            trajectory.shape[-1],
            *([1] * len(mode_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(mode_size))
    )
    phase = torch.exp(1j * phase_argument)
    sample_shape = samples.shape + (1,) * len(mode_size)
    return (samples.reshape(sample_shape) * phase.unsqueeze(1)).sum(dim=2)


def _exact_type2(
    backend: FinufftSenseBackend,
    modes: torch.Tensor,
    trajectory: torch.Tensor,
) -> torch.Tensor:
    mode_vectors = [
        torch.arange(
            -(size // 2),
            (size - 1) // 2 + 1,
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        for size in backend.im_size
    ]
    mode_grids = torch.meshgrid(*mode_vectors, indexing="ij")
    phase_argument = sum(
        trajectory[:, dimension].reshape(
            trajectory.shape[0],
            trajectory.shape[-1],
            *([1] * len(backend.im_size)),
        )
        * mode_grids[dimension]
        for dimension in range(len(backend.im_size))
    )
    phase = torch.exp(-1j * phase_argument)
    spatial_dims = tuple(range(3, 3 + len(backend.im_size)))
    return (modes.unsqueeze(2) * phase.unsqueeze(1)).sum(dim=spatial_dims)


def test_b0_encoding_matches_direct_physics_and_has_exact_adjoint():
    torch.manual_seed(20260726)
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    height, width = 9, 8
    coordinates_y = torch.linspace(-1, 1, height, dtype=real_dtype)
    coordinates_x = torch.linspace(-1, 1, width, dtype=real_dtype)
    y, x = torch.meshgrid(coordinates_y, coordinates_x, indexing="ij")
    image = (
        (
            (
                ((x / 0.75).square() + (y / 0.85).square() <= 1)
                + 0.35
                * (((x + 0.25) / 0.2).square() + ((y - 0.2) / 0.25).square() <= 1)
            )
            * torch.exp(1j * (0.2 * x - 0.1 * y))
        )
        .reshape(1, 1, height, width)
        .to(complex_dtype)
    )

    coil_count = 3
    angles = torch.arange(coil_count, dtype=real_dtype) * (2 * math.pi / coil_count)
    smaps = (
        torch.stack(
            [
                (1 + 0.15 * (x * torch.cos(angle) + y * torch.sin(angle)))
                * torch.exp(1j * 0.2 * (x * torch.sin(angle) - y * torch.cos(angle)))
                for angle in angles
            ]
        )
        .reshape(1, coil_count, height, width)
        .to(complex_dtype)
    )
    smaps = smaps / torch.sqrt(smaps.abs().square().sum(dim=1, keepdim=True))
    zmap = (
        180 * x - 130 * y + 50 * torch.sin(math.pi * x) * torch.cos(math.pi * y)
    ).reshape(1, height, width)

    shot_count, point_count = 7, 13
    shot_angles = torch.arange(shot_count, dtype=real_dtype) * (math.pi / shot_count)
    radius = torch.linspace(-math.pi, math.pi, point_count, dtype=real_dtype)
    trajectory = torch.stack(
        (
            torch.cos(shot_angles)[:, None] * radius,
            torch.sin(shot_angles)[:, None] * radius,
        )
    ).unsqueeze(0)
    dt = 0.12
    common = {
        "nbins": 40,
        "dt": dt,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
        "backend": "torchkbnufft",
    }
    coarse = Gmri(smaps, zmap, trajectory, L=1, **common)
    accurate = Gmri(smaps, zmap, trajectory, L=4, **common)
    reference = _direct_b0_encoding(
        image,
        smaps,
        zmap,
        trajectory,
        dt=dt,
        grid_size=accurate.grid_size,
    )

    coarse_error = _relative_error(coarse(image), reference)
    accurate_error = _relative_error(accurate(image), reference)
    assert accurate_error < 1e-3
    assert accurate_error < coarse_error / 100

    probe_image = torch.randn_like(image)
    probe_samples = torch.randn_like(reference)
    lhs = torch.vdot(accurate(probe_image).reshape(-1), probe_samples.reshape(-1))
    rhs = torch.vdot(
        probe_image.reshape(-1),
        accurate.H(probe_samples).reshape(-1),
    )
    adjoint_error = (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs())
    assert adjoint_error < 1e-12


def test_b0_cg_reconstruction_recovers_a_synthetic_phantom():
    image, smaps, zmap, trajectory = _synthetic_cartesian_case()
    common = {
        "L": 5,
        "nbins": 24,
        "dt": 0.08,
        "grid_size": 2,
        "numpoints": 6,
        "norm": "ortho",
        "backend": "torchkbnufft",
    }
    operator = Gmri(smaps, zmap, trajectory, **common)
    measurements = operator(image)
    adjoint_image = operator.H(measurements)
    normal = operator.H * operator + 1e-6 * Identity(operator.size_in)
    reconstruction = CG(normal, max_iter=8, tol=0).run(
        torch.zeros_like(image),
        adjoint_image,
    )

    reconstruction_error = _relative_error(reconstruction, image)
    data_residual = _relative_error(operator(reconstruction), measurements)
    adjoint_error = _relative_error(adjoint_image, image)
    correlation = (
        torch.vdot(reconstruction.reshape(-1), image.reshape(-1)).abs()
        / torch.linalg.vector_norm(reconstruction)
        / torch.linalg.vector_norm(image)
    )
    assert reconstruction_error < 1e-3
    assert data_residual < 1e-3
    assert reconstruction_error < adjoint_error / 100
    assert correlation > 0.999

    toeplitz_gram = GmriGram(smaps, zmap, trajectory, **common)
    toeplitz_normal_error = _relative_error(
        toeplitz_gram(image),
        adjoint_image,
    )
    toeplitz_reconstruction = CG(
        toeplitz_gram + 1e-6 * Identity(operator.size_in),
        max_iter=8,
        tol=0,
    ).run(torch.zeros_like(image), adjoint_image)
    assert toeplitz_normal_error < 2e-3
    assert _relative_error(toeplitz_reconstruction, image) < 2e-3
    assert _relative_error(toeplitz_reconstruction, reconstruction) < 2e-3


def test_gmri_all_parameter_gradients_match_references(monkeypatch):
    monkeypatch.setattr(FinufftSenseBackend, "type1", _exact_type1)
    monkeypatch.setattr(FinufftSenseBackend, "type2", _exact_type2)
    torch.manual_seed(602)
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    height, width, coil_count, shot_count, point_count = 4, 5, 2, 2, 7

    parameters = [
        torch.randn(1, 1, height, width, dtype=complex_dtype, requires_grad=True),
        torch.randn(
            1,
            coil_count,
            height,
            width,
            dtype=complex_dtype,
            requires_grad=True,
        ),
        (torch.randn(1, 2, shot_count, point_count, dtype=real_dtype) * 0.35)
        .clone()
        .requires_grad_(),
        torch.linspace(-57, 73, height * width, dtype=real_dtype)
        .reshape(1, height, width)
        .clone()
        .requires_grad_(),
        (
            torch.arange(point_count, dtype=real_dtype) * 0.031
            + torch.linspace(0, 0.003, point_count, dtype=real_dtype)
        ).requires_grad_(),
    ]
    image, smaps, trajectory, zmap, times = parameters
    probe = torch.randn(
        1,
        coil_count,
        shot_count,
        point_count,
        dtype=complex_dtype,
    )
    common = {
        "L": 3,
        "nbins": 9,
        "numpoints": 3,
        "grid_size": 1.5,
        "norm": "ortho",
        "backend": "finufft",
        "eps": 1e-12,
    }

    operator = Gmri(smaps, zmap, trajectory, T=times, **common)
    objective = torch.vdot(probe.reshape(-1), operator(image).reshape(-1)).real
    gradients = torch.autograd.grad(objective, parameters)
    directions = [
        direction / torch.linalg.vector_norm(direction)
        for direction in (torch.randn_like(parameter) for parameter in parameters)
    ]
    analytic_derivatives = [
        (
            (gradient.conj() * direction).sum().real
            if parameter.is_complex()
            else (gradient * direction).sum()
        )
        for parameter, gradient, direction in zip(
            parameters,
            gradients,
            directions,
            strict=True,
        )
    ]

    # MIRT percentile locations are model-selection nodes. Their gradients are
    # intentionally stopped because differentiating the near-singular basis
    # fit gives a poor derivative of the physical signal. Validate the time
    # gradient against the exact B0 encoding instead of against a perturbation
    # that also redesigns those nodes.
    reference_times = times.detach().clone().requires_grad_()
    reference_samples = _direct_b0_encoding(
        image.detach(),
        smaps.detach(),
        zmap.detach(),
        trajectory.detach(),
        times=reference_times,
        grid_size=operator.grid_size,
    )
    reference_objective = torch.vdot(
        probe.reshape(-1),
        reference_samples.reshape(-1),
    ).real
    reference_time_gradient = torch.autograd.grad(
        reference_objective,
        reference_times,
    )[0]
    reference_time_derivative = (reference_time_gradient * directions[-1]).sum()

    detached_parameters = [parameter.detach() for parameter in parameters]

    def evaluate(values):
        eval_image, eval_smaps, eval_trajectory, eval_zmap, eval_times = values
        eval_operator = Gmri(
            eval_smaps,
            eval_zmap,
            eval_trajectory,
            T=eval_times,
            **common,
        )
        return torch.vdot(
            probe.reshape(-1),
            eval_operator(eval_image).reshape(-1),
        ).real

    step = 1e-3
    for index, (analytic, direction) in enumerate(
        zip(analytic_derivatives[:-1], directions[:-1], strict=True)
    ):
        plus = [parameter.clone() for parameter in detached_parameters]
        minus = [parameter.clone() for parameter in detached_parameters]
        plus[index] = plus[index] + step * direction
        minus[index] = minus[index] - step * direction
        finite_difference = (evaluate(plus) - evaluate(minus)) / (2 * step)
        scale = torch.maximum(
            torch.maximum(analytic.abs(), finite_difference.abs()),
            torch.tensor(1e-10, dtype=real_dtype),
        )
        relative_error = (analytic - finite_difference).abs() / scale
        assert relative_error < 5e-5

    time_scale = torch.maximum(
        torch.maximum(
            analytic_derivatives[-1].abs(),
            reference_time_derivative.abs(),
        ),
        torch.tensor(1e-10, dtype=real_dtype),
    )
    time_relative_error = (
        analytic_derivatives[-1] - reference_time_derivative
    ).abs() / time_scale
    assert time_relative_error < 2e-4
