"""P65 phase-only causality, preservation and reference/gradient gates."""

import torch

from models.band_phase import BandPhaseCorrection
from models.band_tcn import make_prototype
from models.direct_spectrum_objective import direct_spectrum_loss


def fixture(frames=14):
    torch.manual_seed(65)
    base = make_prototype("band", seed=65, nonzero=True)
    model = BandPhaseCorrection(base).eval()
    x = torch.randn(2, 2, frames, 129)
    x[:, 0] = x[:, 0] * 10 - 40
    context = torch.rand(2, frames, 1538)
    cond = torch.tensor([1.0, 0.6, 1.0]).expand(2, frames, 3).clone()
    return base, model, x, context, cond


def test_zero_rotation_equals_parent_and_base_has_no_gradient():
    base, model, x, ctx, cond = fixture()
    expected, _ = base(x, ctx, cond, base.initial_state(2))
    actual, _ = model(x, ctx, cond, model.initial_state(2))
    assert torch.equal(expected, actual)
    actual[:, 1].square().mean().backward()
    assert all(p.grad is None for p in base.parameters())
    assert model.phase_head[-1].weight.grad.abs().sum() > 0


def test_nonzero_rotation_preserves_magnitude_endpoints_and_chunking():
    base, model, x, ctx, cond = fixture()
    with torch.no_grad():
        model.phase_head[-1].weight.normal_(0, 0.05)
    expected, _ = base(x, ctx, cond, base.initial_state(2))
    actual, state = model(x, ctx, cond, model.initial_state(2))
    assert torch.equal(actual[:, 0], expected[:, 0])
    assert torch.equal(actual[:, 1, :, [0, 128]], expected[:, 1, :, [0, 128]])
    chunks, h = [], model.initial_state(2)
    for start, end in [(0, 3), (3, 8), (8, 14)]:
        out, h = model(x[:, :, start:end], ctx[:, start:end], cond[:, start:end], h)
        chunks.append(out)
    torch.testing.assert_close(torch.cat(chunks, 2), actual, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(h, state, atol=2e-5, rtol=2e-5)
    changed = x.clone()
    changed[1, :, 7:] += 2
    other, _ = model(changed, ctx, cond, model.initial_state(2))
    torch.testing.assert_close(other[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(other[:, :, :7], actual[:, :, :7], rtol=0, atol=0)


def test_silence_and_weak_signal_gradients_are_finite():
    base, model, x, ctx, cond = fixture()
    for level in [-200.0, -190.0, -100.0]:
        quiet = torch.zeros_like(x)
        quiet[:, 0] = level
        model.zero_grad()
        actual, _ = model(quiet, ctx * 0, cond, model.initial_state(2))
        actual[:, 1].square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.phase_head.parameters()
        )
        if level == -200.0:
            expected, _ = base(quiet, ctx * 0, cond, base.initial_state(2))
            assert torch.equal(actual, expected)


def test_reference_correct_phase_gradient_and_finite_difference():
    _, model, x, ctx, cond = fixture()
    model = model.double()
    x, ctx, cond = x.double(), ctx.double(), cond.double()
    output, _ = model(x, ctx, cond, model.initial_state(2))
    # Direct objective remains the existing original-reference objective.
    batch = {
        "x": output.detach().clone(),
        "y": output.detach().clone(),
        "rate": torch.full((2, 1), 48000),
        "identity": torch.tensor([False, True]),
    }
    phase = output.detach().clone().requires_grad_(True)
    loss = direct_spectrum_loss(phase, batch, burn=0)["total"]
    gradient = torch.autograd.grad(loss, phase)[0]
    assert abs(float(loss.detach())) < 1e-10
    assert gradient.abs().max() < 1e-9
    shifted = output.detach().clone()
    shifted[:, 1, :, 60] += 0.1
    shifted.requires_grad_(True)
    loss = direct_spectrum_loss(shifted, batch, burn=0)["total"]
    derivative = torch.autograd.grad(loss, shifted)[0][0, 1, 5, 60]
    step = 1e-5
    plus, minus = shifted.detach().clone(), shifted.detach().clone()
    plus[0, 1, 5, 60] += step
    minus[0, 1, 5, 60] -= step
    numeric = (
        direct_spectrum_loss(plus, batch, burn=0)["total"]
        - direct_spectrum_loss(minus, batch, burn=0)["total"]
    ) / (2 * step)
    torch.testing.assert_close(derivative, numeric, atol=1e-8, rtol=1e-4)


def test_phase_carrier_ablation_preserves_other_features_and_source_aggregation():
    from scripts.assess_band_phase import compare, repeated_current

    features = torch.randn(2, 4, 648)
    features[..., :129].abs_()
    before = features.clone()
    changed = repeated_current(None, (features,))[0]
    assert torch.equal(features, before)
    assert torch.equal(changed[..., :387], before[..., :387])
    assert torch.equal(changed[..., 645:], before[..., 645:])
    torch.testing.assert_close(
        changed[..., 387:516], before[..., :129].square() / (1 + before[..., :129].square())
    )
    assert not changed[..., 516:645].any()
    original = [
        {
            "case": i,
            "role": 0,
            "source": name,
            "source_id": name,
            "codec": "test",
            "group": "music/paired",
            "scores": {"mr": score},
        }
        for i, (name, score) in enumerate((("a", 1.0), ("a", 3.0), ("b", 6.0)))
    ]
    updated = [
        {**row, "scores": {"mr": row["scores"]["mr"] + delta}}
        for row, delta in zip(original, (1, -1, -2), strict=True)
    ]
    result = compare(original, updated)["source_equal_summary"]["music/paired"]
    assert result["cases"] == 3 and result["sources"] == 2
    assert result["metrics"]["mr"]["before"] == 4.0
    assert result["metrics"]["mr"]["after"] == 3.0
    assert len(result["metrics"]["mr"]["worse_cases"]) == 1


def test_carrier_head_does_not_encode_static_cross_bin_phase_offsets():
    # A representational boundary, not an assertion about realizable music PCM.
    # Rotating each input bin by its own constant angle leaves the correction
    # unchanged: only relative-to-base phase and temporal advance are observed.
    base, model, x, ctx, cond = fixture()
    model.double()
    x, ctx, cond = x.double(), ctx.double(), cond.double()
    with torch.no_grad():
        model.phase_head[-1].weight.normal_(0, 0.05)
    shifted = x.clone()
    shifted[:, 1] += torch.linspace(-0.7, 0.9, 129, dtype=x.dtype)
    predicted, _ = model(x, ctx, cond, model.initial_state(2))
    predicted_shifted, _ = model(shifted, ctx, cond, model.initial_state(2))
    parent, _ = base(x, ctx, cond, base.initial_state(2))
    parent_shifted, _ = base(shifted, ctx, cond, base.initial_state(2))
    torch.testing.assert_close(
        predicted[:, 1] - parent[:, 1],
        predicted_shifted[:, 1] - parent_shifted[:, 1],
        atol=1e-12,
        rtol=1e-10,
    )
