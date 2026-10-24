"""Known-answer checks for offline band attribution and counterfactual counting."""

from types import SimpleNamespace

import pytest
import torch

from evaluation.output_coupling import (
    budget_ablation,
    budget_coordinate_control,
    decompose,
    feature_paths,
    masks,
)
from models.complex_synthesis import complex_features
from models.restoration_head import RestorationHead, predict_budget


@pytest.mark.parametrize("column", [False, True])
def test_physical_band_partition_and_exact_noop(column):
    rates = torch.tensor([44100.0, 48000.0])
    if column:
        rates = rates[:, None]
    partition = masks(rates)
    assert torch.all(sum(v.int() for v in partition.values()) == 1)
    assert partition["low4k"][0].sum() == 24
    assert partition["low4k"][1].sum() == 22
    start = torch.randn(2, 2, 112, 129, generator=torch.Generator().manual_seed(47))
    for value in feature_paths(start, start, rates).values():
        assert torch.equal(value, start)


def test_linear_ola_decomposition_and_low_bin_cause():
    start = torch.zeros(2, 2, 112, 129)
    start[:, 0] = -40
    current = start.clone()
    current[:, 0, 80:, 3] += 3
    batch = {"rate": torch.tensor([[44100.0], [48000.0]]), "x": start, "y": start}
    batch["clean_wave"] = torch.zeros(2, 112 * 128)
    result = decompose(start, current, batch)
    assert result["additivity_max_error"] < 1e-15
    assert max(result["parts"]["high8k"]["full"]["rms"]) == 0
    assert min(result["parts"]["low4k"]["low4k_projection_share"]) == pytest.approx(1)
    assert torch.equal(feature_paths(start, current, batch["rate"])["without_low4k_update"], start)


def test_high_bin_can_leak_into_offline_low_band_after_ola():
    start = torch.zeros(2, 2, 112, 129)
    start[:, 0] = -80
    current = start.clone()
    current[:, 0, 95, 60] = 0
    batch = {"rate": torch.tensor([44100.0, 48000.0]), "x": start, "y": start}
    batch["clean_wave"] = torch.zeros(2, 112 * 128)
    result = decompose(start, current, batch)
    assert min(result["parts"]["high8k"]["low4k"]["rms"]) > 0
    assert max(result["parts"]["low4k"]["full"]["rms"]) == 0
    assert result["additivity_max_error"] < 1e-15


def test_budget_ablation_counts_attempts_once_and_keeps_other_failures():
    def point(*, budget=True, pcm=False, other=False):
        violations = [{"metric": "budget"}] if budget else []
        if other:
            violations.append({"metric": "paired_direct_complex"})
        return {
            "radius": 0.001,
            "metric_guard": {"spectral_progress": True, "violations": violations},
            "pcm_guard": {"violations": [1] if pcm else [], "measurements": []},
            "current": {},
        }

    points = [[point(), point()], [point(pcm=True)], [point(other=True)], [point(budget=False)]]
    history = [
        {"step": i + 1, "accepted": i == 3, "points": p, "base": {}} for i, p in enumerate(points)
    ]
    result = budget_ablation(history)
    assert result["counts"]["actual_accepted"] == 1
    assert result["counts"]["would_accept_recorded_proposal_without_budget"] == 2
    assert result["counts"]["additional_accepted_attempts_without_budget"] == 1
    assert len(result["budget_only_cases"]) == 2


def test_budget_coordinate_control_preserves_nonzero_output_and_restores_hooks(monkeypatch):
    torch.manual_seed(47)
    model = (
        RestorationHead(
            "complex", 8, residual_slope=1 / 64, generation_budget=True, budget_scope="residual"
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.1)
    features = torch.randn(2, 2, 100, 129, dtype=torch.float64)
    features[:, 0] -= 30
    batch = {
        "x": features,
        "rate": torch.tensor([[44100.0], [48000.0]], dtype=torch.float64),
        "cutoff": torch.full((2, 100), 8000.0, dtype=torch.float64),
        "active": torch.ones(2, 100),
    }
    context = torch.rand(2, 100, 1538, dtype=torch.float64)
    item = (SimpleNamespace(batch=batch), context)
    original, budget = predict_budget(model, batch, context)
    gauged, doubled = budget_coordinate_control(model, item)
    torch.testing.assert_close(
        complex_features(gauged), complex_features(original), atol=1e-14, rtol=1e-12
    )
    torch.testing.assert_close(doubled, 2 * budget, atol=1e-14, rtol=1e-12)
    restored, _ = predict_budget(model, batch, context)
    assert torch.equal(restored, original)
    assert not torch.equal(original, features)

    def failure(*_args):
        raise RuntimeError("known probe failure")

    monkeypatch.setattr("evaluation.output_coupling.predict_budget", failure)
    with pytest.raises(RuntimeError, match="known probe failure"):
        budget_coordinate_control(model, item)
    assert not model.budget_head._forward_hooks
    assert not model.backbone.decoder[-1]._forward_hooks
