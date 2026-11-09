"""P74 causal observation path, matched geometry and finite derivative checks."""

import pytest
import torch

from models.phase_temporal import phase_temporal_model


def fixture(kind="phase", frames=16):
    torch.manual_seed(74)
    model = phase_temporal_model(kind, 74, nonzero=True)
    original = torch.randn(2, 2, frames, 129)
    original[:, 0] = original[:, 0] * 10 - 40
    context = torch.rand(2, frames, 1538)
    condition = torch.tensor([1.0, 0.6, 1.0]).expand(2, frames, 3).clone()
    return model, original, context, condition


def test_matched_parameters_initial_function_and_history_geometry():
    a, x, ctx, cond = fixture("phase")
    b = phase_temporal_model("magnitude", 74, nonzero=True)
    assert a.state_size == b.state_size == 12612
    assert a.base.initial_state(2).shape == (2, 12354)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    for model in (a, b):
        for projection in model.base.relation_projections:
            torch.nn.init.zeros_(projection.weight)
    aa, _ = a(x, ctx, cond, a.initial_state(2))
    bb, _ = b(x, ctx, cond, b.initial_state(2))
    assert torch.equal(aa, bb)


@pytest.mark.parametrize("kind", ["magnitude", "phase"])
def test_chunking_future_and_physical_channel_isolation(kind):
    model, x, ctx, cond = fixture(kind)
    expected, final = model(x, ctx, cond, model.initial_state(2))
    pieces, state = [], model.initial_state(2)
    for start, end in ((0, 3), (3, 9), (9, 16)):
        out, state = model(x[:, :, start:end], ctx[:, start:end], cond[:, start:end], state)
        pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces, 2), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state, final, atol=2e-5, rtol=2e-5)
    changed = x.clone()
    changed[1, :, 8:] += 2
    actual, _ = model(changed, ctx, cond, model.initial_state(2))
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :, :8], expected[:, :, :8], atol=2e-5, rtol=2e-5)
    observed = x.clone().requires_grad_()
    out, _ = model(observed, ctx, cond, model.initial_state(2))
    grad = torch.autograd.grad(out[0, :, :8].sum(), observed)[0]
    assert not grad[1].any()
    assert not grad[0, :, 8:].any()


def test_phase_history_reaches_later_magnitude_only_in_candidate():
    for kind in ("magnitude", "phase"):
        model, x, ctx, cond = fixture(kind)
        x.requires_grad_()
        out, _ = model(x, ctx, cond, model.initial_state(2))
        grad = torch.autograd.grad(out[:, 0, -1, 80:120].sum(), x)[0]
        phase_history = grad[:, 1, :8].abs().sum()
        assert (phase_history > 0) if kind == "phase" else (phase_history == 0)


def test_relations_ignore_common_rotation_but_observe_relative_phase():
    model, x, _ctx, _cond = fixture()
    base = model.base
    magnitude = torch.pow(10.0, x[:, 0] / 20)
    scale = magnitude.square().mean(-1, keepdim=True).sqrt()
    history = base.initial_state(2)[:, base.temporal_state_size :]
    original, _ = base.relations(x, magnitude, scale, history)
    common = x.clone()
    common[:, 1] += 0.7
    rotated, _ = base.relations(common, magnitude, scale, history)
    for a, b in zip(original, rotated, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    common[:, 1, 0, 70:80] += 0.5
    changed, _ = base.relations(common, magnitude, scale, history)
    assert not torch.allclose(changed[1], original[1])
    assert not torch.allclose(changed[3], original[3])


@pytest.mark.parametrize("level", [-200.0, -190.0, -100.0])
def test_silence_weak_input_and_finite_difference(level):
    model, x, ctx, cond = fixture(frames=5)
    model = model.double()
    x, ctx, cond = x.double(), ctx.double() * 0, cond.double()
    x[:, 0] = level
    x.requires_grad_()
    out, _ = model(x, ctx, cond, model.initial_state(2))
    value = out[:, 0].sum()
    grad = torch.autograd.grad(value, x)[0]
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    if level == -200:
        assert not grad[:, 1].any()
    direction = torch.zeros_like(x)
    direction[0, 1, 1, 70] = 1
    step = 1e-3
    with torch.no_grad():
        high, _ = model(x + step * direction, ctx, cond, model.initial_state(2))
        low, _ = model(x - step * direction, ctx, cond, model.initial_state(2))
    finite = (high[:, 0].sum() - low[:, 0].sum()) / (2 * step)
    torch.testing.assert_close(finite, (grad * direction).sum(), atol=2e-7, rtol=2e-3)
