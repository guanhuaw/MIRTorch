import math

import numpy as np
import pytest
import torch

from mirtorch.linear.spect import (
    SPECT,
    backproject,
    backproject_angle,
    parallel_hole_psfs,
    project,
    project_angle,
    required_rotation_shape,
)


def _model_data(dtype=torch.float64):
    torch.manual_seed(42)
    nx, ny, nz, nview = 5, 6, 4, 5
    mumap = torch.rand(nx, ny, nz, dtype=dtype) * 0.02
    psfs = torch.rand(2, 3, ny, nview, dtype=dtype)
    psfs /= psfs.sum(dim=(0, 1), keepdim=True)
    return mumap, psfs


@pytest.mark.parametrize("signal_dtype", [torch.float64, torch.complex128])
def test_spect_is_an_exact_discrete_adjoint(signal_dtype):
    mumap, psfs = _model_data()
    size_in = mumap.shape
    size_out = (size_in[0], size_in[2], psfs.shape[-1])
    spect = SPECT(size_in, size_out, mumap, psfs, dy=0.4, view_chunk_size=2)
    torch.manual_seed(1)
    if signal_dtype.is_complex:
        image = torch.randn(size_in, dtype=torch.float64) + 1j * torch.randn(
            size_in, dtype=torch.float64
        )
        views = torch.randn(size_out, dtype=torch.float64) + 1j * torch.randn(
            size_out, dtype=torch.float64
        )
    else:
        image = torch.randn(size_in, dtype=signal_dtype)
        views = torch.randn(size_out, dtype=signal_dtype)

    lhs = torch.vdot(spect(image).flatten(), views.flatten())
    rhs = torch.vdot(image.flatten(), spect.H(views).flatten())

    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("signal_dtype", [torch.float64, torch.complex128])
def test_single_angle_projector_is_an_exact_adjoint(signal_dtype):
    mumap, psfs = _model_data()
    psf = psfs[..., 0]
    torch.manual_seed(2)
    if signal_dtype.is_complex:
        image = torch.randn(mumap.shape, dtype=torch.float64) + 1j * torch.randn(
            mumap.shape, dtype=torch.float64
        )
        view = torch.randn(
            (mumap.shape[0], mumap.shape[2]), dtype=torch.float64
        ) + 1j * torch.randn((mumap.shape[0], mumap.shape[2]), dtype=torch.float64)
    else:
        image = torch.randn(mumap.shape, dtype=signal_dtype)
        view = torch.randn((mumap.shape[0], mumap.shape[2]), dtype=signal_dtype)

    forward = project_angle(image, mumap, psf, 0.4, 37.0)
    adjoint = backproject_angle(view, mumap, psf, 0.4, 37.0)

    torch.testing.assert_close(
        torch.vdot(forward.flatten(), view.flatten()),
        torch.vdot(image.flatten(), adjoint.flatten()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_attenuation_uses_trapezoidal_depth_integral():
    nx, ny, nz = 3, 4, 2
    dy = 0.5
    attenuation = 0.2
    mumap = torch.full((nx, ny, nz), attenuation, dtype=torch.float64)
    psf = torch.ones((1, 1, ny), dtype=torch.float64)
    image = torch.arange(nx * ny * nz, dtype=torch.float64).reshape(nx, ny, nz)

    actual = project_angle(image, mumap, psf, dy, viewangle=0.0)
    depth = torch.arange(ny, dtype=torch.float64) + 0.5
    expected = (image * torch.exp(-dy * attenuation * depth)[None, :, None]).sum(dim=1)

    torch.testing.assert_close(actual, expected)


def test_chunked_and_unchunked_models_match():
    mumap, psfs = _model_data()
    size_out = (mumap.shape[0], mumap.shape[2], psfs.shape[-1])
    image = torch.randn(mumap.shape, dtype=torch.float64)
    views = torch.randn(size_out, dtype=torch.float64)
    chunked = SPECT(mumap.shape, size_out, mumap, psfs, 0.4, view_chunk_size=2)
    unchunked = SPECT(mumap.shape, size_out, mumap, psfs, 0.4, view_chunk_size=None)

    torch.testing.assert_close(chunked(image), unchunked(image))
    torch.testing.assert_close(chunked.H(views), unchunked.H(views))


def test_free_functions_and_linear_map_match():
    mumap, psfs = _model_data(torch.float32)
    image = torch.randn(mumap.shape)
    size_out = (mumap.shape[0], mumap.shape[2], psfs.shape[-1])
    views = torch.randn(size_out)
    spect = SPECT(mumap.shape, size_out, mumap, psfs, dy=0.4, view_chunk_size=2)

    torch.testing.assert_close(
        spect(image),
        project(image, mumap, psfs, dy=0.4, view_chunk_size=2),
    )
    torch.testing.assert_close(
        spect.H(views),
        backproject(views, mumap, psfs, dy=0.4, view_chunk_size=2),
    )


def test_model_parameters_remain_differentiable():
    mumap, psfs = _model_data()
    mumap = mumap.requires_grad_()
    psfs = psfs.requires_grad_()
    image = torch.randn(mumap.shape, dtype=torch.float64, requires_grad=True)

    loss = project(image, mumap, psfs, dy=0.4, view_chunk_size=2).square().sum()
    loss.backward()

    for gradient in (image.grad, mumap.grad, psfs.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_trainable_angles_rebuild_each_chunk_for_repeated_gradients():
    mumap, psfs = _model_data()
    angles = torch.linspace(13.0, 301.0, psfs.shape[-1], requires_grad=True)
    size_out = (mumap.shape[0], mumap.shape[2], psfs.shape[-1])
    model = SPECT(
        mumap.shape,
        size_out,
        mumap,
        psfs,
        dy=0.4,
        view_chunk_size=2,
        angles=angles,
    )
    image = torch.randn(mumap.shape, dtype=mumap.dtype)

    for _ in range(2):
        (gradient,) = torch.autograd.grad(model(image).square().sum(), angles)
        assert torch.isfinite(gradient).all()

    assert model.angles.numel() == psfs.shape[-1]
    assert not hasattr(model, "_rotation_indices")


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"size_in": (5, 6)}, ValueError, "size_in must have three"),
        ({"size_out": (5, 4, 4)}, ValueError, "size_out must be"),
        ({"mumap_shape": (5, 5, 4)}, ValueError, "mumap shape"),
        ({"psf_depth": 0}, ValueError, "psfs must have nonempty"),
        ({"psf_dtype": torch.float32}, TypeError, "same dtype"),
        ({"dy": 0.0}, ValueError, "finite and positive"),
        ({"dy": "1"}, TypeError, "real scalar"),
        ({"view_chunk_size": 0}, ValueError, "must be positive"),
    ],
)
def test_constructor_validation(change, error, message):
    mumap, psfs = _model_data()
    mumap_shape = change.get("mumap_shape", mumap.shape)
    if mumap_shape != mumap.shape:
        mumap = torch.zeros(mumap_shape, dtype=mumap.dtype)
    psf_depth = change.get("psf_depth", psfs.shape[2])
    if psf_depth != psfs.shape[2]:
        psfs = torch.zeros(
            psfs.shape[0],
            psfs.shape[1],
            psf_depth,
            psfs.shape[3],
            dtype=psfs.dtype,
        )
    psfs = psfs.to(change.get("psf_dtype", psfs.dtype))
    size_in = change.get("size_in", (5, 6, 4))
    size_out = change.get("size_out", (5, 4, 5))

    with pytest.raises(error, match=message):
        SPECT(
            size_in,
            size_out,
            mumap,
            psfs,
            change.get("dy", 0.4),
            view_chunk_size=change.get("view_chunk_size", 2),
        )


@pytest.mark.parametrize("kernel_shape", [(3, 3), (2, 4), (1, 2)])
def test_psfs_are_impulse_responses_not_mirrored_correlations(kernel_shape):
    shape = (9, 1, 9)
    image = torch.zeros(shape, dtype=torch.float64)
    image[4, 0, 4] = 1
    psf = torch.arange(1, 1 + kernel_shape[0] * kernel_shape[1], dtype=image.dtype)
    psf = psf.reshape(*kernel_shape, 1)
    actual = project_angle(image, torch.zeros_like(image), psf, 1.0, 0.0)
    expected = torch.zeros(9, 9, dtype=image.dtype)
    x0, z0 = 4 - kernel_shape[0] // 2, 4 - kernel_shape[1] // 2
    expected[x0 : x0 + kernel_shape[0], z0 : z0 + kernel_shape[1]] = psf[..., 0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kernel_shape", [(3, 2), (2, 4)])
def test_depth_blur_matches_independent_spatial_convolution(kernel_shape):
    torch.manual_seed(65)
    shape = (5, 3, 4)
    image = torch.randn(shape, dtype=torch.float64)
    psf = torch.rand(*kernel_shape, shape[1], dtype=image.dtype)
    expected = torch.zeros(shape[0], shape[2], dtype=image.dtype)
    for x in range(shape[0]):
        for z in range(shape[2]):
            for kx in range(kernel_shape[0]):
                for kz in range(kernel_shape[1]):
                    source_x = x - kx + kernel_shape[0] // 2
                    source_z = z - kz + kernel_shape[1] // 2
                    if 0 <= source_x < shape[0] and 0 <= source_z < shape[2]:
                        expected[x, z] += (
                            image[source_x, :, source_z] * psf[kx, kz]
                        ).sum()
    actual = project_angle(image, torch.zeros_like(image), psf, 1.0, 0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_full_depth_grid_removes_artificial_corner_loss():
    shape = (32, 32, 1)
    width, depth = required_rotation_shape(shape, 1.0)
    mumap = torch.zeros(shape, dtype=torch.float64)
    image = torch.zeros_like(mumap)
    image[2, 2, 0] = 1
    full = SPECT(
        shape,
        (width, 1, 3),
        mumap,
        torch.ones(1, 1, depth, 3, dtype=image.dtype),
        1.0,
        angles=[0.0, 45.0, 90.0],
    )
    counts = full(image).sum(dim=(0, 1))
    torch.testing.assert_close(counts[[0, 2]], torch.ones(2, dtype=image.dtype))
    # A bilinear rotation is not exactly flux preserving, but the corner source
    # must not disappear at 45 degrees as it did in the image-sized depth grid.
    assert 0.8 < counts[1] < 1.2
    narrower = SPECT(
        shape,
        (shape[0], 1, 3),
        mumap,
        full.psfs,
        1.0,
        angles=full.angles,
    )
    torch.testing.assert_close(narrower(image).sum(dim=(0, 1)), counts)
    image.zero_()
    image[2, 29, 0] = 1
    assert full(image)[..., 1].sum() > 0.8
    # This source is genuinely outside a detector only 32 bins wide at 45 deg.
    assert narrower(image)[..., 1].sum() == 0


@pytest.mark.parametrize("shape", [(5, 6, 2), (6, 5, 2)])
def test_full_grid_zero_angle_is_an_exact_centered_embedding(shape):
    width, depth = required_rotation_shape(shape, 0.7, dx=1.3)
    image = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape)
    psf = torch.ones(1, 1, depth, dtype=image.dtype)
    projected = project_angle(
        image,
        torch.zeros_like(image),
        psf,
        0.7,
        0.0,
        detector_width=width,
        dx=1.3,
    )
    expected = torch.zeros(width, shape[2], dtype=image.dtype)
    start = (width - shape[0]) // 2
    expected[start : start + shape[0]] = image.sum(dim=1)
    torch.testing.assert_close(projected, expected, rtol=1e-14, atol=1e-13)


def test_rotation_uses_physical_coordinates_for_anisotropic_voxels():
    shape = (9, 9, 1)
    dx, dy = 2.0, 1.0
    width, depth = required_rotation_shape(shape, dy, dx=dx)
    image = torch.zeros(shape, dtype=torch.float64)
    image[4, 6, 0] = 1  # 2 length units off center in depth.
    projected = project_angle(
        image,
        torch.zeros_like(image),
        torch.ones(1, 1, depth, dtype=image.dtype),
        dy,
        90.0,
        detector_width=width,
        dx=dx,
    )[:, 0]
    coordinates = (torch.arange(width, dtype=image.dtype) - (width - 1) / 2) * dx
    centroid = (projected * coordinates).sum() / projected.sum()
    torch.testing.assert_close(centroid, torch.tensor(-2.0, dtype=image.dtype))


def test_voxel_attenuation_matches_analytic_uniform_slab():
    shape = (3, 7, 2)
    mu, dy = 1.2, 0.6
    image = torch.ones(shape, dtype=torch.float64)
    mumap = torch.full_like(image, mu)
    psf = torch.ones(1, 1, shape[1], dtype=image.dtype)
    actual = project_angle(image, mumap, psf, dy, 0.0, attenuation="voxel")
    # Activity is per voxel, hence the slab integral is divided by dy.
    expected = -torch.expm1(torch.tensor(-mu * dy * shape[1], dtype=image.dtype)) / (
        mu * dy
    )
    torch.testing.assert_close(
        actual, expected.expand_as(actual), rtol=1e-13, atol=1e-13
    )
    midpoint = project_angle(image, mumap, psf, dy, 0.0)
    assert (midpoint - actual).abs().max() > 0.01


def test_voxel_attenuation_has_correct_first_and_second_derivatives_at_zero():
    dy = 0.7
    mumap = torch.zeros(1, 1, 1, dtype=torch.float64, requires_grad=True)
    model = SPECT(
        mumap.shape,
        (1, 1, 1),
        mumap,
        torch.ones(1, 1, 1, 1, dtype=mumap.dtype),
        dy,
        attenuation="voxel",
    )
    value = model(torch.ones_like(mumap)).sum()
    gradient = torch.autograd.grad(value, mumap, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(), mumap)[0]
    torch.testing.assert_close(value, torch.tensor(1.0, dtype=mumap.dtype))
    torch.testing.assert_close(gradient, torch.full_like(mumap, -dy / 2))
    torch.testing.assert_close(curvature, torch.full_like(mumap, dy**2 / 3))


def test_parallel_hole_response_broadens_with_distance_and_keeps_calibration_gradients():
    distances = torch.tensor(
        [[20.0], [100.0], [200.0]], dtype=torch.float64, requires_grad=True
    )
    intrinsic = torch.tensor(3.0, dtype=distances.dtype, requires_grad=True)
    psfs = parallel_hole_psfs(
        distances,
        (31, 31),
        (1.0, 1.0),
        hole_diameter=1.5,
        hole_length=25.0,
        intrinsic_fwhm=intrinsic,
    )
    torch.testing.assert_close(psfs.sum(dim=(0, 1)), torch.ones_like(distances))
    axis = torch.arange(31, dtype=distances.dtype) - 15
    variance = (psfs * axis[:, None, None, None].square()).sum(dim=(0, 1))
    assert torch.all(torch.diff(variance[:, 0]) > 0)
    fitted_variance = 1 / (2 * torch.log(psfs[15, 15] / psfs[16, 15]))
    expected_variance = ((1.5 / 25.0 * distances).square() + intrinsic.square()) / (
        8 * math.log(2)
    )
    torch.testing.assert_close(fitted_variance, expected_variance)
    assert torch.autograd.gradcheck(
        lambda d, intrinsic: parallel_hole_psfs(
            d,
            (9, 7),
            (2.0, 3.0),
            hole_diameter=1.5,
            hole_length=25.0,
            intrinsic_fwhm=intrinsic,
        ),
        (distances, intrinsic),
    )
    assert torch.autograd.gradgradcheck(
        lambda d, intrinsic: parallel_hole_psfs(
            d,
            (9, 7),
            (2.0, 3.0),
            hole_diameter=1.5,
            hole_length=25.0,
            intrinsic_fwhm=intrinsic,
        ),
        (distances, intrinsic),
        fast_mode=True,
    )


@pytest.mark.parametrize("name", ["mumap", "psfs"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_spect_rejects_nonphysical_attenuation_and_response(name, value):
    mumap, psfs = _model_data()
    values = mumap if name == "mumap" else psfs
    values.flatten()[0] = value
    with pytest.raises(ValueError, match=f"{name} must be finite and nonnegative"):
        SPECT(mumap.shape, (5, 4, 5), mumap, psfs, 0.4)


@pytest.mark.parametrize("mode", ["bad", None])
def test_spect_rejects_unknown_attenuation_mode(mode):
    mumap, psfs = _model_data()
    with pytest.raises(ValueError, match="attenuation must be"):
        SPECT(mumap.shape, (5, 4, 5), mumap, psfs, 0.4, attenuation=mode)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"distances": torch.ones(2)}, "distances must have"),
        ({"distances": -torch.ones(2, 1)}, "nonnegative"),
        ({"kernel_shape": (0, 3)}, "kernel_shape"),
        ({"voxel_size": (1.0,)}, "voxel_size"),
        ({"hole_length": 0.0}, "hole_length"),
        ({"hole_diameter": torch.ones(3)}, "broadcast"),
    ],
)
def test_parallel_hole_response_validation(kwargs, match):
    options = {
        "distances": torch.ones(2, 1),
        "kernel_shape": (3, 3),
        "voxel_size": (1.0, 1.0),
        "hole_diameter": 1.0,
        "hole_length": 20.0,
        "intrinsic_fwhm": 3.0,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        parallel_hole_psfs(**options)


@pytest.mark.parametrize("name", ["hole_diameter", "hole_length", "intrinsic_fwhm"])
@pytest.mark.parametrize(
    "value",
    [
        np.complex64(1 + 2j),
        np.array(1 + 2j),
        np.array([1 + 2j]),
        np.bool_(True),
        np.array(True),
        np.array([True]),
    ],
)
def test_parallel_hole_rejects_numpy_complex_and_boolean_calibration(name, value):
    options = {"hole_diameter": 1.0, "hole_length": 20.0, "intrinsic_fwhm": 3.0}
    options[name] = value
    with pytest.raises(TypeError, match=f"{name} must be real"):
        parallel_hole_psfs(torch.ones(2, 1), (3, 3), (1.0, 1.0), **options)


@pytest.mark.parametrize(
    "angles",
    [
        np.complex64(1 + 2j),
        np.array(1 + 2j),
        np.array([1 + 2j]),
        np.bool_(True),
        np.array(True),
        np.array([True]),
    ],
)
def test_spect_rejects_numpy_complex_and_boolean_angles(angles):
    mumap = torch.zeros(3, 3, 1)
    psfs = torch.ones(1, 1, 3, 1)
    with pytest.raises(TypeError, match="angles must be real"):
        SPECT(mumap.shape, (3, 1, 1), mumap, psfs, 1.0, angles=angles)


def test_parallel_hole_preserves_python_scalar_double_precision():
    distances = torch.full((2, 1), 100.0, dtype=torch.float64)
    calibration = {
        "hole_diameter": 0.1234567890123,
        "hole_length": 2.2345678901234,
        "intrinsic_fwhm": 0.3456789012345,
    }
    actual = parallel_hole_psfs(distances, (5, 5), (1.0, 1.0), **calibration)
    expected = parallel_hole_psfs(
        distances,
        (5, 5),
        (1.0, 1.0),
        **{
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in calibration.items()
        },
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("name", ["hole_diameter", "hole_length", "intrinsic_fwhm"])
def test_parallel_hole_backpropagates_mps_results_to_cpu_double_calibration(name):
    calibration = {
        "hole_diameter": 1.234,
        "hole_length": 20.456,
        "intrinsic_fwhm": 3.567,
    }
    distances = torch.tensor([[50.0], [100.0]], dtype=torch.float32)
    weights = torch.arange(25, dtype=torch.float32).reshape(5, 5, 1, 1).square()

    def evaluate(device):
        parameter = torch.tensor(
            calibration[name], dtype=torch.float64, requires_grad=True
        )
        options = {**calibration, name: parameter}
        psfs = parallel_hole_psfs(
            distances.to(device),
            (5, 5),
            (1.0, 1.0),
            **options,
        )
        gradient = torch.autograd.grad((psfs * weights.to(device)).sum(), parameter)[0]
        assert gradient.device.type == "cpu"
        assert gradient.dtype == torch.float64
        return psfs.detach().cpu(), gradient

    for actual, expected in zip(evaluate("mps"), evaluate("cpu")):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_spect_backpropagates_mps_results_to_cpu_double_angles():
    torch.manual_seed(245)
    shape = (3, 4, 2)
    width, depth = required_rotation_shape(shape, 0.7)
    image = torch.rand(shape)
    mumap = 0.1 * torch.ones(shape)
    psfs = torch.rand(3, 3, depth, 2)
    probe = torch.randn(width, shape[2], 2)

    def evaluate(device):
        angles = torch.tensor([23.45, 123.45], dtype=torch.float64, requires_grad=True)
        model = SPECT(
            shape, probe.shape, mumap.to(device), psfs.to(device), 0.7, angles=angles
        )
        forward = model(image.to(device))
        gradient = torch.autograd.grad((forward * probe.to(device)).sum(), angles)[0]
        assert gradient.device.type == "cpu"
        assert gradient.dtype == torch.float64
        return forward.detach().cpu(), gradient

    for actual, expected in zip(evaluate("mps"), evaluate("cpu")):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
