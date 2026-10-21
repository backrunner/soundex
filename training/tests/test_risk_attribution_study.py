import copy
import json
from types import SimpleNamespace

import pytest
import torch

from models.risk_balance import training_terms as previous_terms
from scripts import analyze_risk_attribution_study as analyzer
from scripts.analyze_risk_attribution_study import compare, factorial, screen
from scripts.run_broad_risk_study import protocol as parent_protocol
from scripts.run_broad_risk_study import training_terms as parent_terms
from scripts.run_risk_attribution_study import ARMS, protocol, training_terms
from tests.test_multires_spectrum import feature_batch


def test_factorial_protocol_preserves_budget_and_architecture():
    parent = parent_protocol(
        {"direct_loss": {}, "multires_loss": {}, "weight": 1, "development": []}
    )
    child = protocol(parent)
    assert child["steps"] == parent["steps"] == 16192
    assert child["seeds"] == parent["seeds"] and child["rate"] == parent["rate"]
    assert set(child["arms"]) == set(ARMS) == {"compressed", "identity_mid"}
    assert all(c == parent["model_configs"]["control"] for c in child["model_configs"].values())
    assert child["loss_routes"] == {a: {a: 1} for a in ARMS}
    assert "no new loss or DSP" in child["contrast"]
    assert protocol(parent, preflight=True)["steps"] == 4
    with pytest.raises(ValueError, match="budget"):
        protocol({**parent, "steps": 1024})


def test_factorial_gradients_change_only_registered_factor():
    batch = feature_batch()
    batch["dry"] = batch["clean_wave"].clone()
    prediction = batch["x"].clone()
    prediction[:, 0, 89:, 48:] -= 3
    prediction.requires_grad_()
    budget = torch.full(
        (2, prediction.shape[2] - 80, 5), 0.2, dtype=prediction.dtype, requires_grad=True
    )
    objective = SimpleNamespace(
        batch=batch, region=slice(89 * 128, prediction.shape[2] * 128 - 128)
    )
    new = training_terms(prediction, budget, objective, 1)
    prior = previous_terms(prediction, budget, objective, 1)
    parent = parent_terms(prediction, budget, objective, 1)
    torch.testing.assert_close(new["compressed"], prior["compressed"], rtol=0, atol=0)

    def grads(loss):
        return torch.autograd.grad(loss, (prediction, budget), retain_graph=True)

    comp, old = grads(new["compressed"]), grads(prior["compressed"])
    for a, b in zip(comp, old, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    reduced, full = grads(new["identity_mid"]), grads(parent["control"])
    torch.testing.assert_close(reduced[0][0], full[0][0])
    torch.testing.assert_close(reduced[0][1], full[0][1] * 0.3)
    # Budget supervision retains equal paired/identity weighting in every cell.
    for a, b in ((reduced[1], full[1]), (comp[1], full[1])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def score(a, b):
    return {
        "recordings": {"a": a, "b": b},
        "recording_mean": (a + b) / 2,
        "components_recording_mean": {"dynamics": (a + b) / 4},
    }


def test_factorial_interaction_is_in_common_score_units_and_records_stay_paired():
    cells = {
        name: {"group": score(*values)}
        for name, values in {
            "raw_full": (2, 4),
            "raw_mid": (1, 3),
            "log_full": (1, 2),
            "log_mid": (0.5, 1),
        }.items()
    }
    result = factorial(cells)
    assert result["compression_at_identity1"]["group"]["difference"] == -1.5
    assert result["identity_reduction_with_compression"]["group"]["difference"] == -0.75
    assert result["interaction"]["group"] == 0.25
    assert compare(score(0, 0), score(0, 0))["delta_pct"] is None
    wrong = copy.deepcopy(cells)
    wrong["log_mid"]["group"]["recordings"] = {"a": 0.5, "c": 1}
    with pytest.raises(ValueError, match="populations"):
        factorial(wrong)


def test_spectral_improvement_does_not_hide_identity_or_reference_dynamics_regression():
    control = {
        f"{corpus}/{role}/native_output": score(2, 4)
        for corpus in ("music_library", "speech_library")
        for role in ("paired", "identity")
    }
    candidate = copy.deepcopy(control)
    candidate["music_library/paired/native_output"] = score(1, 3)
    assert screen(candidate, control)["all_pass"]
    candidate["music_library/paired/native_output"]["components_recording_mean"]["dynamics"] = 2
    result = screen(candidate, control)
    assert result["gates"]["music_mean_improves"] and not result["all_pass"]
    candidate = copy.deepcopy(control)
    candidate["music_library/paired/native_output"] = score(1, 3)
    candidate["speech_library/identity/native_output"] = score(3, 5)
    assert not screen(candidate, control)["all_pass"]


def test_full_assessment_combines_both_seeds_and_rejects_changed_input_baseline(
    tmp_path, monkeypatch
):
    root, reference = tmp_path / "new", tmp_path / "old"
    root.mkdir()
    reference.mkdir()
    config = {
        "smoke": False,
        "reference_study": str(reference),
        "review_scope": "reused",
        "steps": 16192,
        "seeds": [1, 2],
        "development": [],
        "direct_loss": {},
        "multires_loss": {},
        "rate": 1e-4,
        "schedule_hashes": {"1": "a", "2": "b"},
    }
    (root / "registration.json").write_text(json.dumps(config))
    (reference / "registration.json").write_text(json.dumps(config))
    groups = [
        f"{corpus}/{role}/native_output"
        for corpus in ("music_library", "speech_library")
        for role in ("paired", "identity")
    ]

    def cell(a, b):
        return {"spectral": {group: score(a, b) for group in groups}}

    old = {"initial": cell(4, 6)}
    new = {"initial": cell(4, 6)}
    for seed in config["seeds"]:
        old[f"control-16192-{seed}"] = cell(2, 4)
        old[f"compressed_identity_mid-16192-{seed}"] = cell(1, 3)
        new[f"compressed-16192-{seed}"] = cell(1.5, 3.5)
        new[f"identity_mid-16192-{seed}"] = cell(1.8, 3.8)
    (reference / "assessment.json").write_text(json.dumps({"cases": old}))
    monkeypatch.setattr(
        analyzer, "collect_cases", lambda *_, **__: (config, copy.deepcopy(new), {})
    )
    result = analyzer.run(root)
    assert set(result["factorial"]) == {"1", "2"}
    assert len(result["cases"]) == 9 and not result["promotion"]
    assert all(result["research_screen_pass"].values())
    new["initial"]["spectral"][groups[0]]["recordings"]["a"] = 5
    with pytest.raises(ValueError, match="metric drift"):
        analyzer.run(root)
