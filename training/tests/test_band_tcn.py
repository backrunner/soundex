"""Nonzero P62 causal/token state, original-reference gradients and export checks."""

from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from evaluation.head_export import export_graph, verify
from evaluation.spectral_training import objective_value
from models.band_tcn import band_slices, make_prototype
from models.complex_synthesis import complex_features, synthesize
from models.envelope_probe import probe_inputs
from models.fit_diagnostic import BURN
from scripts.bench_band_tcn import native_parity


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(frames=100, dtype=torch.float32):
    generator = torch.Generator().manual_seed(62)
    x = torch.randn(2, 2, frames, 129, generator=generator, dtype=dtype)
    x[:, 0] = x[:, 0] * 8 - 40
    context = torch.rand(2, frames, 1538, generator=generator, dtype=dtype)
    condition = torch.tensor([[[44100 / 48000, 0.5, 1]], [[1, 0.5, 1]]], dtype=dtype)
    return x, context, condition.expand(-1, frames, -1)


def test_all_fft_bins_are_assigned_exactly_once():
    for bins in (129, 513, 1025):
        slices = band_slices(bins)
        assert [i for lo, hi in slices for i in range(lo, hi)] == list(range(bins))
        # Boundaries align in normalized frequency; physical Hz follow rate.
        assert [lo / (bins - 1) for lo, _ in slices] == [lo / 128 for lo, _ in band_slices(129)]


@torch.no_grad()
def test_zero_start_identity_and_nonzero_history_effect():
    x, context, condition = inputs()
    model = make_prototype("band")
    output, state = model(x, context, condition, model.initial_state(2))
    assert torch.equal(output, x) and state.abs().max() > 0
    model = make_prototype("band", nonzero=True)
    full, _ = model(x, context, condition, model.initial_state(2))
    # Repeated-current-frame ablation changes history, not the last observation.
    repeated, _ = model(
        x[:, :, -1:].expand_as(x),
        context[:, -1:].expand_as(context),
        condition,
        model.initial_state(2),
    )
    assert not torch.equal(full, x)
    assert (full[:, :, -1] - repeated[:, :, -1]).abs().max() > 1e-5


@torch.no_grad()
def test_chunks_hops_future_reset_and_physical_channel_isolation():
    model = make_prototype("band", nonzero=True)
    x, context, condition = inputs()
    full, final = model(x, context, condition, model.initial_state(2))
    for boundaries in ((0, 7, 65, 100), tuple(range(101))):
        pieces, state = [], model.initial_state(2)
        for start, end in pairwise(boundaries):
            part, state = model(
                x[:, :, start:end], context[:, start:end], condition[:, start:end], state
            )
            pieces.append(part)
        torch.testing.assert_close(torch.cat(pieces, 2), full, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(state, final, atol=2e-5, rtol=2e-5)
    altered_x, altered_context = x.clone(), context.clone()
    altered_x[:, :, 60:] += 3
    altered_context[:, 60:] *= 2
    future, _ = model(altered_x, altered_context, condition, model.initial_state(2))
    torch.testing.assert_close(full[:, :, :60], future[:, :, :60], atol=2e-5, rtol=2e-5)
    # Float32 batched kernels can round differently with changed future rows.
    # Require exact prefix equality in double as a separate causality check.
    precise = make_prototype("band", nonzero=True).double()
    original64, _ = precise(
        x.double(), context.double(), condition.double(), precise.initial_state(2)
    )
    future64, _ = precise(
        altered_x.double(), altered_context.double(), condition.double(), precise.initial_state(2)
    )
    assert torch.equal(original64[:, :, :60], future64[:, :, :60])
    for channel in (0, 1):
        mono, state = model(
            x[channel : channel + 1],
            context[channel : channel + 1],
            condition[channel : channel + 1],
            model.initial_state(1),
        )
        torch.testing.assert_close(mono, full[channel : channel + 1], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(state, final[channel : channel + 1], atol=2e-5, rtol=2e-5)
    reset, reset_state = model(x, context, condition, model.initial_state(2))
    assert torch.equal(full, reset) and torch.equal(final, reset_state)


@pytest.mark.parametrize("db", [-200.0, -100.0, -40.0])
def test_silence_weak_and_normal_gradients_are_finite(db):
    model = make_prototype("band", nonzero=True).double()
    x, context, condition = inputs(frames=8, dtype=torch.float64)
    x[:, 0] = db
    if db == -200:
        context.zero_()
    output, state = model(x, context, condition, model.initial_state(2))
    complex_features(output).abs().square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(state).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert state.abs().max() > 0  # quiet frames also advance history


def test_original_reference_objective_reaches_local_encoder_and_mixer():
    model = make_prototype("band", nonzero=True).double()
    x, context, condition = inputs(frames=120, dtype=torch.float64)
    target = x.clone()
    target[0, 0, 89:, 80:] += 2
    batch = {
        "x": x,
        "y": target,
        "clean_wave": synthesize(complex_features(target)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0, dtype=torch.float64),
        "active": torch.ones(2, 120, dtype=torch.float64),
    }
    item = (SimpleNamespace(batch=batch), context)
    _, _, condition = probe_inputs(batch, context)
    # Training intentionally detaches warmup. A parameter finite difference must
    # hold that history fixed, rather than differentiate through recomputed history.
    with torch.no_grad():
        _, warm_state = model(
            x[:, :, :BURN], context[:, :BURN], condition[:, :BURN], model.initial_state(2)
        )

    def fixed_warmup(current, _batch, _context):
        tail, _, budget = current.forward_details(
            x[:, :, BURN:], context[:, BURN:], condition[:, BURN:], warm_state
        )
        return torch.cat((x[:, :, :BURN], tail), 2), budget

    def value():
        with patch("evaluation.spectral_training.predict_budget", fixed_warmup):
            return objective_value(model, item, "common")

    torch.testing.assert_close(value(), objective_value(model, item, "common"), rtol=0, atol=0)
    parameters = (model.backbone.encoders[-1][0].weight, model.backbone.mixers[-1][0].weight)
    gradients = torch.autograd.grad(value(), parameters)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        direction = gradient / gradient.norm()
        saved = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(saved - 1e-5 * direction)
            left = value()
            parameter.copy_(saved + 1e-5 * direction)
            right = value()
            parameter.copy_(saved)
        torch.testing.assert_close((right - left) / 2e-5, gradient.norm(), rtol=2e-3, atol=1e-7)


def test_nonzero_export_has_sequential_state_parity(tmp_path):
    model = make_prototype("band", nonzero=True)
    session = export_graph(model, tmp_path / "band.onnx")
    result = verify(model, session)
    assert result["sequential_hops"] == 320 and result["reset"] and result["silence"]


@pytest.mark.parametrize("rate", [44100, 48000])
def test_actual_rust_frontend_and_physical_channel_parity(tmp_path, rate):
    root = Path(__file__).resolve().parents[2]
    binary = root / "target/release/soundex-context-replay"
    if not binary.exists():
        binary = root / "target/debug/soundex-context-replay"
    if not binary.exists():
        pytest.skip("build soundex-context-replay for native integration")
    model = make_prototype("band", nonzero=True)
    graph = tmp_path / "band.onnx"
    export_graph(model, graph)
    report = native_parity(model, graph, binary, tmp_path / "native", rate)
    assert report["silent_channel_peak"] < 1e-9
