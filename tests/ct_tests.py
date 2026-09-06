"""CT physics checks use independent ray/rectangle intersections, not A.H."""

import math

import numpy as np
import pytest
import torch

from mirtorch.linear import CT

_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
elif torch.backends.mps.is_available():
    _DEVICES.append("mps")


def _intersection_length(start, end, bounds):
    """Independent slab clipping of a finite ray against one pixel."""
    lower, upper = 0.0, 1.0
    direction = [b - a for a, b in zip(start, end)]
    for origin, delta, (minimum, maximum) in zip(start, direction, bounds):
        if abs(delta) < 1e-14:
            # Half-open pixels assign a ray on a shared edge to one pixel.
            if not minimum <= origin < maximum:
                return 0.0
            continue
        first, last = sorted(((minimum - origin) / delta, (maximum - origin) / delta))
        lower, upper = max(lower, first), min(upper, last)
        if upper <= lower:
            return 0.0
    return (upper - lower) * math.hypot(*direction)


def _reference_matrix(shape, angles, detector_count, spacing, voxel_size, geometry):
    """Brute-force each ray against every pixel; deliberately no grid traversal."""
    ny, nx = shape
    dy, dx = voxel_size
    reach = math.hypot(ny * dy, nx * dx) / 2 + max(dy, dx)
    rows = []
    for angle in angles:
        tangent = (math.cos(angle), math.sin(angle))
        direction = (-math.sin(angle), math.cos(angle))
        for detector in range(detector_count):
            offset = (detector - (detector_count - 1) / 2) * spacing
            if geometry == "fan":
                start = tuple(-10.0 * value for value in direction)
                end = tuple(8.0 * d + offset * u for d, u in zip(direction, tangent))
            else:
                start = tuple(
                    -reach * d + offset * u for d, u in zip(direction, tangent)
                )
                end = tuple(reach * d + offset * u for d, u in zip(direction, tangent))
            rows.append(
                [
                    _intersection_length(
                        start,
                        end,
                        (
                            ((column - nx / 2) * dx, (column + 1 - nx / 2) * dx),
                            ((row - ny / 2) * dy, (row + 1 - ny / 2) * dy),
                        ),
                    )
                    for row in range(ny)
                    for column in range(nx)
                ]
            )
    return torch.tensor(rows, dtype=torch.float64)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (1, 1)])
def test_ct_matches_independent_ray_pixel_intersections(geometry, shape):
    angles = torch.tensor([0.0, 0.31, math.pi / 2, 1.91], dtype=torch.float64)
    kwargs = (
        {"source_distance": 10.0, "detector_distance": 8.0} if geometry == "fan" else {}
    )
    projector = CT(
        shape,
        angles,
        6,
        voxel_size=(0.7, 1.2),
        detector_spacing=0.9,
        geometry=geometry,
        **kwargs,
    )
    matrix = _reference_matrix(shape, angles.tolist(), 6, 0.9, (0.7, 1.2), geometry)
    image = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) + 0.3
    views = torch.linspace(-0.2, 1.1, 24, dtype=torch.float64).reshape(4, 6)

    torch.testing.assert_close(
        projector(image), (matrix @ image.flatten()).reshape(4, 6)
    )
    torch.testing.assert_close(
        projector.H(views), (matrix.T @ views.flatten()).reshape(shape)
    )


def test_axis_aligned_projections_have_physical_length_and_orientation():
    image = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    vertical = CT(image.shape, [0.0], 3, voxel_size=(2.0, 1.0))
    horizontal = CT(
        image.shape, [math.pi / 2], 2, voxel_size=(2.0, 1.0), detector_spacing=2.0
    )

    torch.testing.assert_close(vertical(image)[0], image.sum(0) * 2)
    torch.testing.assert_close(horizontal(image)[0], image.sum(1), rtol=1e-6, atol=1e-6)


def test_fan_projection_has_geometric_magnification():
    # A small square centred at x=1 casts a detector shadow centred near M*x.
    shape = (81, 81)
    image = torch.zeros(shape, dtype=torch.float64)
    image[39:42, 59:62] = 1.0
    angles = torch.zeros(1, dtype=torch.float64)
    fan = CT(
        shape,
        angles,
        161,
        voxel_size=0.05,
        detector_spacing=0.05,
        geometry="fan",
        source_distance=10.0,
        detector_distance=10.0,
    )
    parallel = CT(shape, angles, 161, voxel_size=0.05, detector_spacing=0.05)
    positions = (torch.arange(161, dtype=torch.float64) - 80) * 0.05
    fan_centroid = (fan(image)[0] * positions).sum() / fan(image).sum()
    parallel_centroid = (parallel(image)[0] * positions).sum() / parallel(image).sum()

    torch.testing.assert_close(
        parallel_centroid, torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(fan_centroid, 2 * parallel_centroid, rtol=0.0, atol=0.03)


def test_discrete_disk_converges_toward_continuous_chord_lengths():
    angles = torch.tensor([0.0, 0.37, 1.1], dtype=torch.float64)
    detector = torch.arange(-6, 7, dtype=torch.float64) * 0.1
    analytic = 2 * (0.7**2 - detector.square()).sqrt()
    errors = []
    for size in (24, 96):
        spacing = 4.0 / size
        axis = (torch.arange(size, dtype=torch.float64) - (size - 1) / 2) * spacing
        rows, columns = torch.meshgrid(axis, axis, indexing="ij")
        disk = (rows.square() + columns.square() < 0.7**2).to(torch.float64)
        projector = CT(disk.shape, angles, 13, voxel_size=spacing, detector_spacing=0.1)
        errors.append((projector(disk) - analytic).square().mean().sqrt())

    # The binary voxel phantom approximates a circle; equality is not exact.
    assert errors[1] < errors[0] / 2
    assert errors[1] < 0.04


@pytest.mark.parametrize("device", _DEVICES)
def test_missed_rays_and_zero_image_are_finite(device):
    projector = CT(
        (3, 4), torch.tensor([0.0, 0.3], device=device), 9, detector_spacing=3.0
    )
    image = torch.ones((3, 4), device=device)
    views = projector(image)

    assert torch.isfinite(views).all()
    assert torch.count_nonzero(views[:, [0, 1, 7, 8]]) == 0
    torch.testing.assert_close(
        projector(torch.zeros_like(image)), torch.zeros_like(views)
    )
    torch.testing.assert_close(
        projector.H(torch.zeros_like(views)), torch.zeros_like(image)
    )


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_ct_adjoint_and_higher_order_image_gradients(geometry, dtype):
    kwargs = (
        {"source_distance": 10.0, "detector_distance": 8.0} if geometry == "fan" else {}
    )
    projector = CT(
        (2, 3),
        torch.tensor([0.2, 1.1], dtype=torch.float64),
        4,
        geometry=geometry,
        **kwargs,
    )
    torch.manual_seed(43)
    image = torch.randn((2, 3), dtype=dtype, requires_grad=True)
    views = torch.randn((2, 4), dtype=dtype, requires_grad=True)
    forward = projector(image)
    adjoint = projector.H(views)

    torch.testing.assert_close(
        torch.vdot(forward.flatten(), views.flatten()),
        torch.vdot(image.flatten(), adjoint.flatten()),
    )
    torch.testing.assert_close(
        torch.autograd.grad(forward, image, views, retain_graph=True)[0], adjoint
    )
    torch.testing.assert_close(torch.autograd.grad(adjoint, views, image)[0], forward)
    for operation, value in ((projector, image), (projector.H, views)):
        assert torch.autograd.gradcheck(operation, (value,))
        assert torch.autograd.gradgradcheck(operation, (value,))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_ct_devices_chunks_and_cache_match(device, dtype):
    angles = torch.tensor([0.0, 0.27, 1.34], device=device)
    image = torch.randn((4, 5), dtype=dtype).to(device).requires_grad_()
    views = torch.randn((3, 8), dtype=dtype).to(device).requires_grad_()
    reference = CT(image.shape, angles.cpu(), 8, ray_chunk_size=24)
    expected = reference(image.cpu())
    expected_adjoint = reference.H(views.cpu())
    for budget, chunk in ((0, 1), (0, 7), (128 * 1024**2, 5)):
        projector = CT(
            image.shape, angles, 8, ray_chunk_size=chunk, max_cache_bytes=budget
        )
        actual = projector(image)
        adjoint = projector.H(views)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            adjoint.cpu(), expected_adjoint, rtol=2e-5, atol=2e-5
        )
        image_vjp = torch.autograd.grad(actual, image, views, create_graph=True)[0]
        torch.testing.assert_close(image_vjp, adjoint, rtol=2e-5, atol=2e-5)
        mixed_gradient = torch.autograd.grad(image_vjp.real.sum(), views)[0]
        torch.testing.assert_close(
            mixed_gradient, projector(torch.ones_like(image)), rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            torch.autograd.grad(adjoint, views, image)[0], actual, rtol=2e-5, atol=2e-5
        )


def test_intersection_cache_stays_within_its_memory_budget():
    arguments = ((7, 9), torch.tensor([0.17, 0.81]), 13)
    cached = CT(*arguments, max_cache_bytes=32_000, ray_chunk_size=5)
    uncached = CT(*arguments, max_cache_bytes=1, ray_chunk_size=5)
    assert cached._plan is not None
    storage = sum(
        tensor.numel() * tensor.element_size()
        for chunk in cached._plan
        for tensor in chunk
    )
    assert storage <= 32_000
    assert uncached._plan is None


@pytest.mark.parametrize("cache_bytes", [0, 128 * 1024**2])
def test_input_angle_mutation_does_not_change_fixed_geometry(cache_bytes):
    angles = torch.tensor([0.23, 0.87], dtype=torch.float64)
    projector = CT((3, 4), angles, 5, max_cache_bytes=cache_bytes)
    image = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    expected = projector(image)
    angles.add_(0.7)

    torch.testing.assert_close(projector(image), expected)


def test_expected_counts_matches_beer_lambert_and_parameter_gradients():
    projector = CT(
        (3, 4), torch.tensor([0.0, 0.3], dtype=torch.float64), 5, voxel_size=0.2
    )
    image = torch.full((3, 4), 0.4, dtype=torch.float64, requires_grad=True)
    incident = torch.linspace(100.0, 120.0, 5, dtype=torch.float64, requires_grad=True)
    background = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    probe = torch.linspace(0.1, 1.0, 10, dtype=torch.float64).reshape(2, 5)
    transmission = torch.exp(-projector(image))
    counts = projector.expected_counts(image, incident, background)
    gradients = torch.autograd.grad(counts, (image, incident, background), probe)

    torch.testing.assert_close(counts, incident * transmission + background)
    torch.testing.assert_close(
        gradients[0], -projector.H(probe * incident * transmission)
    )
    torch.testing.assert_close(gradients[1], (probe * transmission).sum(0))
    torch.testing.assert_close(gradients[2], probe.sum())
    torch.testing.assert_close(
        projector.expected_counts(torch.zeros_like(image), incident, background),
        (incident + background).expand(2, 5),
    )
    assert (projector.expected_counts(2 * image, incident, background) <= counts).all()
    assert torch.autograd.gradcheck(
        projector.expected_counts, (image, incident, background)
    )
    assert torch.autograd.gradgradcheck(
        projector.expected_counts, (image, incident, background)
    )


@pytest.mark.parametrize("field", ["image", "incident", "background"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), 1j])
def test_expected_counts_rejects_nonphysical_inputs(field, invalid):
    projector = CT((2, 2), [0.0], 3)
    arguments = {"image": torch.ones(2, 2), "incident": 100.0, "background": 1.0}
    arguments[field] = torch.full((2, 2), invalid) if field == "image" else invalid
    with pytest.raises((TypeError, ValueError)):
        projector.expected_counts(
            arguments["image"], arguments["incident"], arguments["background"]
        )


def test_count_calibration_python_floats_preserve_double_precision():
    projector = CT((1, 1), torch.tensor([0.0], dtype=torch.float64), 1)
    attenuation = torch.zeros((1, 1), dtype=torch.float64)
    incident, background = 20_000.123456789, 0.123456789012345
    actual = projector.expected_counts(attenuation, incident, background)
    expected = projector.expected_counts(
        attenuation,
        torch.tensor(incident, dtype=torch.float64),
        torch.tensor(background, dtype=torch.float64),
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        projector.expected_counts(attenuation, 0.0, background),
        torch.full_like(attenuation, background),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("field", ["incident", "background"])
def test_count_calibration_rejects_values_not_representable_in_signal_dtype(field):
    projector = CT((2, 3), [0.0], 4)
    arguments = {"incident": 100.0, "background": 1.0}
    arguments[field] = torch.tensor(1e40, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        projector.expected_counts(torch.zeros(2, 3), **arguments)


@pytest.mark.parametrize("device", _DEVICES)
def test_cpu_double_calibration_casts_directly_to_device_signal_dtype(device):
    projector = CT((2, 3), torch.tensor([0.0, 0.3], device=device), 4)
    attenuation = torch.zeros((2, 3), device=device)
    incident = torch.tensor(
        [1.2345678901, 2.3456789012, 3.4567890123, 4.5678901234],
        dtype=torch.float64,
        requires_grad=True,
    )
    background = torch.tensor(0.123456789, dtype=torch.float64, requires_grad=True)
    actual = projector.expected_counts(attenuation, incident, background)
    expected = (incident.float() + background.float()).expand(2, 4)
    torch.testing.assert_close(actual.cpu(), expected)
    gradients = torch.autograd.grad(actual.sum(), (incident, background))
    torch.testing.assert_close(gradients[0], torch.full_like(incident, 2.0))
    torch.testing.assert_close(gradients[1], torch.full_like(background, 8.0))


@pytest.mark.parametrize("field", ["incident", "background"])
@pytest.mark.parametrize(
    "invalid",
    [np.complex64(1 + 2j), np.array(1 + 2j), np.bool_(True), np.array(True)],
)
def test_count_calibration_rejects_numpy_complex_and_bool(field, invalid):
    projector = CT((2, 3), [0.0], 4)
    arguments = {"incident": 100.0, "background": 1.0}
    arguments[field] = invalid
    with pytest.raises(TypeError, match="real"):
        projector.expected_counts(torch.zeros(2, 3), **arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_shape": (0, 3)},
        {"image_shape": (2, 3, 4)},
        {"image_shape": (2.0, 3)},
        {"angles": []},
        {"angles": [float("nan")]},
        {"angles": [[0.0]]},
        {"angles": torch.tensor([0.2], requires_grad=True)},
        {"detector_count": 0},
        {"detector_count": 2.5},
        {"voxel_size": -1.0},
        {"voxel_size": (1.0, 0.0)},
        {"detector_spacing": 0.0},
        {"geometry": "cone"},
        {"geometry": "fan"},
        {"geometry": "fan", "source_distance": 0.5, "detector_distance": 10.0},
        {"geometry": "fan", "source_distance": 10.0, "detector_distance": 0.5},
        {"ray_chunk_size": 0},
        {"max_cache_bytes": -1},
    ],
)
def test_ct_rejects_invalid_geometry(kwargs):
    arguments = {"image_shape": (3, 4), "angles": [0.0, 0.2], "detector_count": 5}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        CT(**arguments)


def test_expected_counts_rejects_invalid_broadcast_shape():
    projector = CT((2, 3), [0.0, 0.2], 4)
    with pytest.raises((ValueError, RuntimeError)):
        projector.expected_counts(torch.ones(2, 3), torch.ones(3, 4))


@pytest.mark.parametrize("device", [value for value in _DEVICES if value != "cpu"])
def test_ct_device_transfer_and_mismatch(device):
    projector = CT((3, 4), [0.0, 0.23], 5)
    image = torch.randn(3, 4)
    expected = projector(image)
    with pytest.raises(ValueError):
        projector(image.to(device))
    projector.to(device)
    torch.testing.assert_close(
        projector(image.to(device)).cpu(), expected, rtol=2e-5, atol=2e-5
    )
