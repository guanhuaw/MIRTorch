import pytest
import torch

from mirtorch.linear.spect import (
    SPECT,
    backproject,
    backproject_angle,
    project,
    project_angle,
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
        ({"psf_depth": 5}, ValueError, "psfs depth dimension"),
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
