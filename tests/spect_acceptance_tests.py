import math

import pytest
import torch

from mirtorch.linear.spect import SPECT, project, required_rotation_shape


def _phantom_model(dtype=torch.float64, device="cpu"):
    nx, ny, nz, nview = 16, 16, 2, 16
    x_grid = torch.linspace(-1, 1, nx, dtype=dtype, device=device)[:, None, None]
    y_grid = torch.linspace(-1, 1, ny, dtype=dtype, device=device)[None, :, None]
    z_grid = torch.arange(nz, device=device)[None, None, :]
    body = (x_grid.square() / 0.92**2 + y_grid.square() / 0.82**2 <= 1).expand(
        nx, ny, nz
    )
    mumap = 0.12 * body.to(dtype)

    image = torch.zeros((nx, ny, nz), dtype=dtype, device=device)
    sources = (
        (-0.28, -0.20, 0.28, 0.20, 0, 1.00),
        (0.30, 0.28, 0.20, 0.25, 0, 0.65),
        (0.00, 0.00, 0.45, 0.20, 1, 0.80),
        (-0.35, 0.30, 0.16, 0.16, 1, 0.55),
    )
    for center_x, center_y, radius_x, radius_y, plane, activity in sources:
        ellipse = (x_grid - center_x).square() / radius_x**2 + (
            y_grid - center_y
        ).square() / radius_y**2 <= 1
        image += activity * ellipse * (z_grid == plane)
    image *= body

    kernel_size = 5
    kernel_axis = (
        torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    )
    kernel_x, kernel_z = torch.meshgrid(kernel_axis, kernel_axis, indexing="ij")
    psfs = torch.empty(
        kernel_size,
        kernel_size,
        ny,
        nview,
        dtype=dtype,
        device=device,
    )
    for depth in range(ny):
        for view in range(nview):
            sigma_x = 0.55 + 0.70 * depth / (ny - 1)
            sigma_z = (
                0.50
                + 0.45 * depth / (ny - 1)
                + 0.03 * math.sin(2 * math.pi * view / nview)
            )
            psf = torch.exp(
                -0.5 * ((kernel_x / sigma_x).square() + (kernel_z / sigma_z).square())
            )
            psfs[:, :, depth, view] = psf / psf.sum()
    return image, body, mumap, psfs


def _nrmse(actual, expected):
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    )


def _correlation(actual, expected):
    actual = actual - actual.mean()
    expected = expected - expected.mean()
    return torch.sum(actual * expected) / (
        torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(expected)
    )


def test_spect_phantom_projection_backprojection_and_mlem_are_plausible():
    truth, support, mumap, psfs = _phantom_model()
    nview = psfs.shape[-1]
    model = SPECT(
        truth.shape,
        (truth.shape[0], truth.shape[2], nview),
        mumap,
        psfs,
        dy=0.1,
        view_chunk_size=4,
    )
    projections = model(truth)

    assert torch.isfinite(projections).all()
    assert torch.all(projections >= 0)
    assert projections.max() > 1

    torch.manual_seed(2026)
    probe_image = torch.randn_like(truth)
    probe_views = torch.randn_like(projections)
    lhs = torch.sum(model(probe_image) * probe_views)
    rhs = torch.sum(probe_image * model.H(probe_views))
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)

    epsilon = torch.finfo(truth.dtype).eps
    sensitivity = model.H(torch.ones_like(projections))
    backprojection = model.H(projections) / sensitivity.clamp_min(epsilon)
    backprojection *= torch.sum(backprojection * truth) / torch.sum(
        backprojection.square()
    )

    reconstruction = support.to(truth.dtype)
    reconstruction *= projections.sum() / model(reconstruction).sum()
    objectives = []
    for _ in range(15):
        expected_counts = model(reconstruction).clamp_min(epsilon)
        reconstruction *= model.H(projections / expected_counts)
        reconstruction /= sensitivity.clamp_min(epsilon)
        reconstruction *= support
        expected_counts = model(reconstruction).clamp_min(epsilon)
        objectives.append(
            torch.sum(expected_counts - projections * torch.log(expected_counts))
        )

    objective_changes = torch.diff(torch.stack(objectives))
    assert torch.all(objective_changes <= 1e-10)
    assert torch.isfinite(reconstruction).all()
    assert torch.all(reconstruction >= 0)
    assert _nrmse(reconstruction, truth) < 0.60
    assert _nrmse(reconstruction, truth) < 0.75 * _nrmse(backprojection, truth)
    assert _correlation(reconstruction, truth) > 0.83
    assert (
        _correlation(reconstruction, truth) > _correlation(backprojection, truth) + 0.20
    )
    assert _nrmse(model(reconstruction), projections) < 0.12


def test_spect_gradcheck_covers_image_mumap_and_psfs():
    torch.manual_seed(123)
    size_in = (3, 4, 2)
    nview = 3
    image = (torch.rand(size_in, dtype=torch.float64) + 0.2).requires_grad_()
    mumap = (torch.rand(size_in, dtype=torch.float64) * 0.05 + 0.01).requires_grad_()
    psfs = (
        torch.rand(2, 3, size_in[1], nview, dtype=torch.float64) + 0.1
    ).requires_grad_()
    weights = torch.randn(size_in[0], size_in[2], nview, dtype=torch.float64)

    def objective(image_parameter, mumap_parameter, psf_parameter):
        projections = project(
            image_parameter,
            mumap_parameter,
            psf_parameter,
            dy=0.3,
            view_chunk_size=2,
        )
        return torch.sum(projections * weights)

    assert torch.autograd.gradcheck(
        objective,
        (image, mumap, psfs),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an Apple Metal device",
)
def test_spect_adjoint_and_parameter_gradients_on_mps():
    device = torch.device("mps")
    torch.manual_seed(123)
    size_in = (4, 5, 2)
    nview = 4
    image = (torch.rand(size_in) + 0.2).to(device).requires_grad_()
    mumap = (torch.rand(size_in) * 0.05 + 0.01).to(device).requires_grad_()
    psfs = (torch.rand(2, 3, size_in[1], nview) + 0.1).to(device).requires_grad_()
    weights = torch.randn(size_in[0], size_in[2], nview).to(device)

    model = SPECT(
        size_in,
        weights.shape,
        mumap.detach(),
        psfs.detach(),
        dy=0.3,
        view_chunk_size=2,
    )
    probe_image = torch.randn(size_in).to(device)
    probe_views = torch.randn(weights.shape).to(device)
    lhs = torch.sum(model(probe_image) * probe_views)
    rhs = torch.sum(probe_image * model.H(probe_views))
    torch.testing.assert_close(lhs.cpu(), rhs.cpu(), rtol=2e-5, atol=2e-5)

    def objective(image_parameter, mumap_parameter, psf_parameter):
        projections = project(
            image_parameter,
            mumap_parameter,
            psf_parameter,
            dy=0.3,
            view_chunk_size=2,
        )
        return torch.sum(projections * weights)

    gradients = torch.autograd.grad(objective(image, mumap, psfs), (image, mumap, psfs))
    parameters = (image, mumap, psfs)
    for seed, parameter_index in enumerate(range(3), start=1):
        torch.manual_seed(seed)
        direction = torch.randn(parameters[parameter_index].shape).to(device)
        direction /= torch.linalg.vector_norm(direction)
        step = 2e-3
        plus = [parameter.detach() for parameter in parameters]
        minus = [parameter.detach() for parameter in parameters]
        plus[parameter_index] = plus[parameter_index] + step * direction
        minus[parameter_index] = minus[parameter_index] - step * direction
        finite_difference = (objective(*plus) - objective(*minus)) / (2 * step)
        autodiff = torch.sum(gradients[parameter_index] * direction)
        torch.testing.assert_close(
            finite_difference.cpu(),
            autodiff.cpu(),
            rtol=1.2e-2,
            atol=3e-4,
        )


def test_full_grid_joint_gradients_and_second_derivatives():
    torch.manual_seed(234)
    shape = (3, 4, 2)
    width, depth = required_rotation_shape(shape, 0.7, dx=1.1)
    image = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    mumap = (0.05 + 0.01 * torch.rand_like(image)).requires_grad_()
    psfs = (0.1 + torch.rand(2, 3, depth, 2, dtype=image.dtype)).requires_grad_()
    angles = torch.tensor([23.4, 119.2], dtype=image.dtype, requires_grad=True)
    probe = torch.randn(width, shape[2], 2, dtype=image.dtype)

    def objective(x, mu, psf, theta):
        model = SPECT(
            shape,
            probe.shape,
            mu,
            psf,
            0.7,
            angles=theta,
            dx=1.1,
            attenuation="voxel",
            view_chunk_size=1,
        )
        return (model(x) * probe).sum()

    parameters = (image, mumap, psfs, angles)
    assert torch.autograd.gradcheck(objective, parameters, fast_mode=True)
    assert torch.autograd.gradgradcheck(objective, parameters, fast_mode=True)


_DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    _DEVICES.append("mps")
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("complex_signal", [False, True])
def test_full_grid_values_adjoints_and_gradients_across_devices(device, complex_signal):
    torch.manual_seed(235)
    shape = (4, 5, 2)
    width, depth = required_rotation_shape(shape, 0.7, dx=1.1)
    signal_dtype = torch.complex64 if complex_signal else torch.float32
    image = torch.randn(shape, dtype=signal_dtype)
    mumap = 0.05 + 0.01 * torch.rand(shape)
    psfs = 0.1 + torch.rand(2, 3, depth, 3)
    angles = torch.tensor([23.4, 119.2, 249.1])
    probe = torch.randn(width, shape[2], 3, dtype=signal_dtype)

    def evaluate(target):
        x, mu, psf, theta = (
            value.to(target).detach().requires_grad_()
            for value in (image, mumap, psfs, angles)
        )
        weights = probe.to(target)
        model = SPECT(
            shape,
            probe.shape,
            mu,
            psf,
            0.7,
            angles=theta,
            dx=1.1,
            attenuation="voxel",
            view_chunk_size=2,
        )
        forward = model(x)
        adjoint = model.H(weights)
        loss = (forward.conj() * weights).real.sum()
        gradients = torch.autograd.grad(loss, (x, mu, psf, theta))
        torch.testing.assert_close(gradients[0], adjoint, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            (forward.conj() * weights).sum(),
            (x.conj() * adjoint).sum(),
            rtol=2e-5,
            atol=3e-5,
        )
        return tuple(value.detach().cpu() for value in (forward, adjoint, *gradients))

    for actual, expected in zip(evaluate(device), evaluate("cpu")):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
