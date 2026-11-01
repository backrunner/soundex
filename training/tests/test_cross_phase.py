"""P68 observation-only phase extension and equal-width control."""

import pytest
import torch
from test_band_phase import fixture

from models.band_phase import BandPhaseCorrection


@pytest.mark.parametrize("mode", ["temporal_repeat", "adjacent"])
def test_new_observations_preserve_frozen_magnitude_state_and_causality(mode):
    base, _, x, ctx, cond = fixture()
    model = BandPhaseCorrection(base, seed=68, phase_support="native", phase_observation=mode)
    expected, _ = base(x, ctx, cond, base.initial_state(2))
    zero, _ = model(x, ctx, cond, model.initial_state(2))
    assert torch.equal(zero, expected)
    assert model.state_size == 12354
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33281
    with torch.no_grad():
        model.phase_head[-1].weight.normal_(0, 0.05)
    actual, state = model(x, ctx, cond, model.initial_state(2))
    assert torch.equal(actual[:, 0], expected[:, 0])
    out, h = [], model.initial_state(2)
    for start, stop in ((0, 3), (3, 8), (8, 14)):
        value, h = model(x[:, :, start:stop], ctx[:, start:stop], cond[:, start:stop], h)
        out.append(value)
    torch.testing.assert_close(torch.cat(out, 2), actual, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(h, state, rtol=2e-5, atol=2e-5)
    changed = x.clone()
    changed[1, :, 8:] += 2
    other, _ = model(changed, ctx, cond, model.initial_state(2))
    # The frozen parent itself differs by float32 rounding on this full-batch
    # perturbation. Use the existing chunk-equivalence tolerance, and separately
    # require exactly zero derivatives through future/other-channel inputs.
    torch.testing.assert_close(other[:, :, :8], actual[:, :, :8], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(other[0], actual[0], rtol=0, atol=0)
    observed = x.clone().requires_grad_(True)
    output, _ = model(observed, ctx, cond, model.initial_state(2))
    gradient = torch.autograd.grad(output[1, :, :8].sum(), observed)[0]
    assert not gradient[1, :, 8:].any()
    assert not gradient[0].any()


@pytest.mark.parametrize("mode", ["temporal_repeat", "adjacent"])
def test_cross_frequency_boundary_and_common_phase_invariance(mode):
    base, _, x, ctx, cond = fixture()
    model = BandPhaseCorrection(
        base, seed=68, phase_support="native", phase_observation=mode
    ).double()
    x, ctx, cond = x.double(), ctx.double(), cond.double()
    with torch.no_grad():
        model.phase_head[-1].weight.normal_(0, 0.05)

    def correction(value):
        predicted, _ = model(value, ctx, cond, model.initial_state(2))
        parent, _ = base(value, ctx, cond, base.initial_state(2))
        return predicted[:, 1] - parent[:, 1]

    before = correction(x)
    global_shift = x.clone()
    global_shift[:, 1] += 0.3
    torch.testing.assert_close(correction(global_shift), before, atol=1e-12, rtol=1e-10)
    shifted = x.clone()
    shifted[:, 1] += torch.linspace(-0.7, 0.9, 129, dtype=x.dtype)
    difference = (correction(shifted) - before).abs().max()
    if mode == "adjacent":
        assert difference > 1e-8
    else:
        assert difference < 1e-12


@pytest.mark.parametrize("level", [-200.0, -190.0, -100.0])
def test_adjacent_features_have_finite_silent_and_weak_gradients(level):
    base, _, x, ctx, cond = fixture()
    model = BandPhaseCorrection(base, phase_support="native", phase_observation="adjacent")
    x[:, 0] = level
    x[:, 1] = 0
    actual, _ = model(x, ctx * 0, cond, model.initial_state(2))
    actual[:, 1].square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.phase_head.parameters()
    )
    if level == -200:
        expected, _ = base(x, ctx * 0, cond, base.initial_state(2))
        assert torch.equal(actual, expected)
