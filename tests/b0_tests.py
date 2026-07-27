import numpy as np
import pytest
import torch

from mirtorch.linear import Gmri, GmriGram
from mirtorch.linear.mri import mri_exp_approx, readout_times


def _numpy_mri_exp_approx(b0, bins, segments, times, autocorrelation):
    histogram, bin_edges = np.histogram(2 * np.pi * b0.reshape(-1), bins)
    bin_width = bin_edges[1] - bin_edges[0]
    if autocorrelation:
        histogram = np.correlate(histogram, histogram, mode="full")
        bin_centers = np.arange(1 - bins, bins) * bin_width
    else:
        bin_centers = bin_edges[1:] - bin_width / 2

    frequencies = 1j * bin_centers
    segment_times = np.linspace(times[0], times[-1], segments) / 1000
    basis = np.exp(-segment_times[:, None] * frequencies[None, :])
    weights = np.diag(np.sqrt(histogram))
    interpolator = np.linalg.pinv(weights @ basis.T) @ weights
    temporal = (interpolator @ np.exp(-frequencies[:, None] * times[None, :] / 1000)).T
    if autocorrelation:
        temporal = temporal.real
    spatial = np.exp(-segment_times[:, None] * (2j * np.pi * b0.reshape(-1))[None, :]).T
    return temporal, spatial, segment_times


def test_default_readout_times_use_exact_dwell_spacing():
    template = torch.zeros(1, dtype=torch.float64)
    times = readout_times(5, 0.004, template=template)
    assert torch.equal(
        times,
        torch.tensor([0.0, 0.004, 0.008, 0.012, 0.016], dtype=torch.float64),
    )


def test_constant_field_map_uses_a_finite_histogram_range():
    zmap = torch.full((4, 4), 25.0, dtype=torch.float64)
    times = torch.arange(8, dtype=torch.float64) * 0.04
    temporal, spatial, segment_times = mri_exp_approx(
        zmap,
        bins=8,
        lseg=3,
        t=times,
    )
    assert torch.isfinite(temporal).all()
    assert torch.isfinite(spatial).all()
    assert torch.isfinite(segment_times).all()


@pytest.mark.parametrize("autocorrelation", [False, True])
def test_torch_time_segmentation_matches_legacy_numpy_for_explicit_times(
    autocorrelation,
):
    zmap = torch.linspace(-200, 180, 63, dtype=torch.float64).reshape(7, 9)
    times = torch.arange(39, dtype=torch.float64) * 0.08

    actual = mri_exp_approx(
        zmap,
        bins=20,
        lseg=6,
        t=times,
        autocorrelation=autocorrelation,
    )
    expected = _numpy_mri_exp_approx(
        zmap.numpy(),
        bins=20,
        segments=6,
        times=times.numpy(),
        autocorrelation=autocorrelation,
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert np.allclose(
            actual_value.detach().numpy(),
            expected_value,
            rtol=5e-11,
            atol=5e-12,
        )


def test_time_segmentation_retains_zmap_and_time_gradients():
    zmap = torch.linspace(-50, 70, 20, dtype=torch.float64).reshape(4, 5)
    zmap.requires_grad_()
    times = (torch.arange(9, dtype=torch.float64) * 0.03).requires_grad_()

    temporal, spatial, segment_times = mri_exp_approx(
        zmap,
        bins=8,
        lseg=4,
        t=times,
    )
    loss = temporal.abs().square().sum() + spatial.real.sum() + segment_times.sum()
    zmap_gradient, time_gradient = torch.autograd.grad(loss, (zmap, times))

    assert torch.isfinite(zmap_gradient).all()
    assert torch.isfinite(time_gradient).all()
    assert zmap_gradient.abs().sum() > 0
    assert time_gradient.abs().sum() > 0


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_zmap_gradient_matches_global_shift_finite_difference(segments):
    zmap = torch.linspace(-50, 70, 20, dtype=torch.float64).reshape(4, 5)
    zmap.requires_grad_()
    times = torch.arange(9, dtype=torch.float64) * 0.03

    def objective(field_map):
        temporal, spatial, segment_times = mri_exp_approx(
            field_map,
            bins=8,
            lseg=segments,
            t=times,
        )
        return (
            temporal.real.sum()
            + 0.17 * temporal.imag.sum()
            + 0.03 * spatial.real.sum()
            - 0.02 * spatial.imag.sum()
            + segment_times.sum()
        )

    analytic = torch.autograd.grad(objective(zmap), zmap)[0].sum()
    step = 1e-4
    finite_difference = (
        objective(zmap.detach() + step) - objective(zmap.detach() - step)
    ) / (2 * step)
    torch.testing.assert_close(
        analytic,
        finite_difference,
        rtol=2e-4,
        atol=1e-9,
    )


@pytest.mark.parametrize("operator_type", [Gmri, GmriGram])
def test_b0_operators_backpropagate_to_zmap_and_explicit_times(operator_type):
    torch.manual_seed(4)
    zmap = torch.linspace(-30, 40, 16, dtype=torch.float64).reshape(1, 4, 4)
    zmap.requires_grad_()
    times = (torch.arange(7, dtype=torch.float64) * 0.02).requires_grad_()
    smaps = torch.randn(1, 1, 4, 4, dtype=torch.complex128)
    traj = torch.randn(1, 2, 1, 7, dtype=torch.float64) * 0.2
    image = torch.randn(1, 1, 4, 4, dtype=torch.complex128)
    operator = operator_type(
        smaps,
        zmap,
        traj,
        L=3,
        nbins=8,
        T=times,
        backend="torchkbnufft",
    )

    loss = operator(image).abs().square().sum()
    zmap_gradient, time_gradient = torch.autograd.grad(loss, (zmap, times))

    assert torch.isfinite(zmap_gradient).all()
    assert torch.isfinite(time_gradient).all()
    assert zmap_gradient.abs().sum() > 0
    assert time_gradient.abs().sum() > 0


def _b0_cache_inputs():
    torch.manual_seed(77)
    smaps = torch.randn(1, 1, 4, 4, dtype=torch.complex128)
    zmap = torch.linspace(-30, 40, 16, dtype=torch.float64).reshape(1, 4, 4)
    traj = torch.randn(1, 2, 1, 7, dtype=torch.float64) * 0.2
    image = torch.randn(1, 1, 4, 4, dtype=torch.complex128)
    kwargs = {
        "L": 3,
        "nbins": 8,
        "numpoints": 2,
        "grid_size": 1.5,
        "backend": "torchkbnufft",
    }
    return smaps, zmap, traj, image, kwargs


@pytest.mark.parametrize("parameter", ["zmap", "T", "dt"])
def test_gmri_refreshes_coefficients_after_parameter_replacement(parameter):
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    times = torch.arange(7, dtype=torch.float64) * 0.02
    operator = Gmri(
        smaps,
        zmap,
        traj,
        T=times if parameter == "T" else None,
        **kwargs,
    )
    original = operator(image)

    if parameter == "zmap":
        replacement = zmap + 23
        operator.zmap = replacement
        fresh = Gmri(smaps, replacement, traj, **kwargs)
    elif parameter == "T":
        replacement = times + 0.01
        operator.T = replacement
        fresh = Gmri(smaps, zmap, traj, T=replacement, **kwargs)
    else:
        replacement = operator.dt * 3
        operator.dt = replacement
        fresh = Gmri(smaps, zmap, traj, dt=replacement, **kwargs)

    updated = operator(image)
    assert not torch.allclose(updated, original, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(updated, fresh(image), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("change", ["replace", "enable_gradient"])
def test_gmri_gram_rejects_changed_static_coefficients(change):
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    operator = GmriGram(smaps, zmap, traj, **kwargs)
    if change == "replace":
        operator.zmap = zmap.clone()
    else:
        operator.zmap.requires_grad_()
    with pytest.raises(RuntimeError, match="field map or readout times changed"):
        operator(image)


@pytest.mark.parametrize("change", ["mutate", "replace"])
def test_gmri_gram_rejects_changed_fixed_trajectory(change):
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    operator = GmriGram(smaps, zmap, traj, **kwargs)
    if change == "mutate":
        traj.add_(0.01)
    else:
        operator.traj = operator.traj.clone()

    with pytest.raises(RuntimeError, match="trajectory changed"):
        operator(image)


def test_direct_gmri_gram_rejects_replaced_public_tensor():
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    operator = GmriGram(
        smaps,
        zmap.requires_grad_(),
        traj,
        **kwargs,
    )
    operator.zmap = operator.zmap.clone()

    with pytest.raises(RuntimeError, match="zmap was replaced"):
        operator(image)


def test_gmri_gram_uses_direct_normal_when_kernel_budget_is_exceeded(monkeypatch):
    smaps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    zmap = torch.linspace(-20, 20, 16).reshape(1, 4, 4)
    traj = torch.randn(1, 2, 1, 7) * 0.1
    image = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    monkeypatch.setattr("mirtorch.linear.mri._MAX_B0_KERNEL_BYTES", 1)
    with pytest.warns(RuntimeWarning, match="direct Gmri"):
        gram = GmriGram(
            smaps,
            zmap,
            traj,
            L=2,
            nbins=6,
            numpoints=2,
            grid_size=1.5,
            backend="torchkbnufft",
        )
    assert gram._uses_direct_gram
    expected = gram._direct_operator.H(gram._direct_operator(image))
    assert torch.equal(gram(image), expected)


def test_gmri_gram_limits_internal_toeplitz_workspace(monkeypatch):
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    monkeypatch.setattr("mirtorch.linear.mri._MAX_B0_WORKSPACE_BYTES", 1)
    gram = GmriGram(smaps, zmap, traj, **kwargs)

    assert gram._segment_chunk_size == 1
    assert gram._coil_chunk_size == 1
    assert torch.isfinite(gram(image)).all()


def test_gmri_limits_internal_workspace(monkeypatch):
    smaps, zmap, traj, image, kwargs = _b0_cache_inputs()
    monkeypatch.setattr("mirtorch.linear.mri._MAX_B0_WORKSPACE_BYTES", 1)
    operator = Gmri(smaps, zmap, traj, **kwargs)

    assert operator._segment_chunk_size == 1
    assert torch.isfinite(operator(image)).all()
