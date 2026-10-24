"""Adapter isolation, causal state, reference gradients and nonzero export tests."""

import copy
from types import SimpleNamespace

import onnx
import pytest
import torch

from evaluation.adapter_pilot import assert_frozen, step
from evaluation.adapter_runtime import export
from models.complex_synthesis import complex_features, synthesize
from models.restoration_head import RestorationHead
from models.spectral_adapter import SpectralAdapter, from_base, load_adapter
from scripts.audit_update_directions import metrics
from scripts.run_adapter_pilot import checkpoint, optimizer_for

CONFIG = {
    "arm": "complex",
    "width": 8,
    "generation_budget": True,
    "budget_scope": "residual",
    "residual_slope": 1 / 64,
}


def fixture(dtype=torch.float32, frames=100):
    torch.manual_seed(48)
    base = RestorationHead(**CONFIG).to(dtype).eval()
    with torch.no_grad():
        base.backbone.decoder[-1].weight.normal_(0, 0.1)
    x = torch.randn(2, 2, frames, 129, dtype=dtype)
    x[:, 0] = x[:, 0] * 10 - 40
    context = torch.rand(2, frames, 1538, dtype=dtype)
    condition = torch.tensor([[[44100 / 48000, 0.5, 1]], [[1, 0.5, 1]]], dtype=dtype).expand(
        -1, 100, -1
    )
    return base, x, context, condition


@pytest.mark.parametrize("support", ["full", "high"])
def test_zero_start_matches_base_exactly_and_only_adapter_is_trainable(support):
    base, x, context, condition = fixture()
    model = from_base({"model_config": CONFIG, "model": base.state_dict()}, support).eval()
    a, state, budget = base.forward_details(x, context, condition, base.initial_state(2))
    b, other, auxiliary = model.forward_details(x, context, condition, model.initial_state(2))
    assert torch.equal(a, b) and torch.equal(state, other) and torch.equal(budget, auxiliary)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "adapter.weight",
        "adapter.bias",
    }


def test_nonzero_high_adapter_preserves_low_bins_state_and_base_weights():
    base, x, context, condition = fixture()
    model = from_base({"model_config": CONFIG, "model": base.state_dict()}, "high").eval()
    with torch.no_grad():
        model.adapter.weight.normal_(0, 0.1)
    a, state = base(x, context, condition, base.initial_state(2))
    b, other = model(x, context, condition, model.initial_state(2))
    low = model.support_mask(condition) == 0
    assert torch.equal(a.permute(0, 2, 3, 1)[low], b.permute(0, 2, 3, 1)[low])
    assert torch.equal(state, other) and not torch.equal(a, b)
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=1e-4)
    complex_features(b).abs().mean().backward()
    assert model.adapter.weight.grad.norm() > 0
    optimizer.step()
    assert all(
        torch.equal(p, before[n]) and p.grad is None
        for n, p in model.named_parameters()
        if n in before
    )


def test_support_physical_rates_and_smooth_endpoints():
    model = SpectralAdapter(CONFIG, "high").double()
    # Evaluate physical frequencies at bin one by controlling the rate ratio.
    hz = torch.tensor(
        [0, 3999.99, 4000, 4000.01, 6000, 7999.99, 8000, 8000.01], dtype=torch.float64
    )
    condition = torch.zeros(len(hz), 1, 3, dtype=torch.float64)
    condition[:, 0, 0] = hz / (48000 / 256)
    mask = model.support_mask(condition)[:, 0, 1]
    assert torch.all(mask[:3] == 0) and torch.all(mask[-2:] == 1)
    assert mask[4] == 0.5 and mask[3] < 1e-9 and 1 - mask[5] < 1e-9


def test_causality_chunk_reset_channel_isolation_and_silence():
    _, x, context, condition = fixture()
    model = SpectralAdapter(CONFIG, "high").eval()
    with torch.no_grad():
        model.adapter.weight.normal_(0, 0.1)
        full, state = model(x, context, condition, model.initial_state(2))
        pieces, history = [], model.initial_state(2)
        for start, end in ((0, 13), (13, 80), (80, 100)):
            part, history = model(
                x[:, :, start:end], context[:, start:end], condition[:, start:end], history
            )
            pieces.append(part)
        torch.testing.assert_close(torch.cat(pieces, 2), full, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(history, state, atol=2e-5, rtol=2e-5)
        mono, _ = model(x[:1], context[:1], condition[:1], model.initial_state(1))
        torch.testing.assert_close(mono, full[:1], atol=2e-5, rtol=2e-5)
        altered = x.clone()
        altered[:, :, 60:] += 1
        future, _ = model(altered, context, condition, model.initial_state(2))
        assert torch.equal(full[:, :, :60], future[:, :, :60])
        silence = x.clone()
        silence[:, 0], silence[:, 1] = -200, 0
        quiet, _ = model(silence, torch.zeros_like(context), condition, model.initial_state(2))
        assert torch.isfinite(quiet).all()
        reset, _ = model(x, context, condition, model.initial_state(2))
        assert torch.equal(full, reset)


def test_nonzero_export_and_state_parity(tmp_path):
    model = SpectralAdapter(CONFIG, "high").eval()
    with torch.no_grad():
        model.adapter.weight.normal_(0, 0.1)
    report = export(model, tmp_path)
    assert report["parity"]["sequential_hops"] == 320
    metadata = {p.key: p.value for p in onnx.load(tmp_path / "diagnostic.onnx").metadata_props}
    assert metadata["soundex.experimental"] == "p20-context-head-diagnostic"
    assert metadata["soundex.deployable"] == "false"
    assert metadata["soundex.study"] == "p48-frozen-feature-adapter"


def test_existing_original_reference_objective_adapter_directional_derivative():
    base, x, context, condition = fixture(torch.float64, frames=120)
    model = from_base({"model_config": CONFIG, "model": base.state_dict()}, "high").double()
    target = x.clone()
    target[0, 0, 89:, 80:] += 2
    batch = {
        "x": x,
        "y": target,
        "clean_wave": synthesize(complex_features(target)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0, dtype=torch.float64),
        "active": torch.ones(2, 120),
    }
    objective = SimpleNamespace(batch=batch)
    loss = metrics(model, objective, context)[0]["total"]
    gradient = torch.autograd.grad(loss, model.adapter.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    direction = torch.randn_like(gradient)
    direction /= direction.norm()
    values = []
    with torch.no_grad():
        for sign in (-1, 1):
            model.adapter.weight.copy_(sign * 1e-4 * direction)
            values.append(metrics(model, objective, context)[0]["total"])
        model.adapter.weight.zero_()
    torch.testing.assert_close(
        (values[1] - values[0]) / 2e-4, (gradient * direction).sum(), atol=1e-7, rtol=2e-3
    )


def test_real_objective_optimizer_resume_and_frozen_base():
    base, x, context, condition = fixture(frames=120)
    initial = {"model_config": CONFIG, "model": copy.deepcopy(base.state_dict())}
    model = from_base(initial, "high").eval()
    target = x.clone()
    target[0, 0, 89:, 80:] += 2
    batch = {
        "x": x,
        "y": target,
        "clean_wave": synthesize(complex_features(target)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0),
        "active": torch.ones(2, 120),
    }
    items = [(SimpleNamespace(batch=batch), context)] * 5
    optimizer = optimizer_for(model)
    first = step(model, items, optimizer)
    assert first["parameter_delta_l2"] > 0
    assert first["fit_before"]["budget"] == first["guard_after"]["budget"]
    saved = copy.deepcopy(checkpoint(model, initial, optimizer, [first], "test"))
    second = step(model, items, optimizer)
    resumed = load_adapter(saved).eval()
    resumed_optimizer = optimizer_for(resumed)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    assert step(resumed, items, resumed_optimizer) == second
    assert all(torch.equal(p, resumed.state_dict()[k]) for k, p in model.state_dict().items())
    assert_frozen(model, initial)
    assert_frozen(resumed, initial)
