"""Finite-step descent, no-op, restoration and numerical-floor regressions."""

import pytest
import torch

from evaluation.adapter_scale import FACTORS, backtrack


def setup():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    return p, [p.detach().clone()], [torch.tensor([-10.0])]


def test_overshoot_backtracks_on_actual_objective_and_restores():
    p, original, delta = setup()
    calls = []

    def evaluate():
        calls.append(float(p.detach()))
        return {"total": float(p.detach().square().sum())}

    result = backtrack([p], original, delta, -20.0, 1.0, evaluate)
    assert result["selected_scale"] == FACTORS[2]
    assert [x["accepted"] for x in result["points"]] == [False, False, True]
    assert calls == [1 + s * -10 for s in FACTORS[:3]]
    assert torch.equal(p, original[0])


def test_non_descent_is_explicit_noop_without_metric_access():
    p, original, delta = setup()
    result = backtrack([p], original, delta, 1.0, 1.0, lambda: pytest.fail("not evaluated"))
    assert result["selected_scale"] == 0 and result["points"] == []
    assert torch.equal(p, original[0])


def test_tiny_or_zero_change_cannot_pass_and_all_attempts_remain():
    p, original, delta = setup()
    result = backtrack([p], original, delta, -20.0, 1.0, lambda: {"total": 1 - 1e-8})
    assert result["selected_scale"] == 0 and len(result["points"]) == len(FACTORS)
    assert not any(v["accepted"] for v in result["points"])
    assert torch.equal(p, original[0])
    result = backtrack([p], original, [torch.zeros_like(p)], -1.0, 1.0, lambda: {"total": 0.0})
    assert result["selected_scale"] == 0


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_errors_restore_original_parameters(failure):
    p, original, delta = setup()

    def evaluate():
        if failure == "exception":
            raise RuntimeError("metric failure")
        return {"total": float("nan")}

    with pytest.raises((RuntimeError, ValueError)):
        backtrack([p], original, delta, -20.0, 1.0, evaluate)
    assert torch.equal(p, original[0])


def test_invalid_local_derivative_rejected():
    p, original, delta = setup()
    with pytest.raises(ValueError, match="finite"):
        backtrack([p], original, delta, float("nan"), 1.0, lambda: {})
    assert torch.equal(p, original[0])


def test_actual_frozen_guard_review_writes_receipt_and_resumes(tmp_path):
    from types import SimpleNamespace

    from test_spectral_adapter import CONFIG, fixture

    from evaluation.reporting import sha256_file
    from models.complex_synthesis import complex_features, synthesize
    from models.spectral_adapter import from_base
    from scripts.audit_adapter_scale import recheck

    base, x, context, condition = fixture(frames=120)
    initial = {"model_config": CONFIG, "model": base.state_dict()}
    initial_path = tmp_path / "initial.pth"
    torch.save(initial, initial_path)
    model = from_base(initial, "high")
    batch = {
        "x": x,
        "y": x.clone(),
        "clean_wave": synthesize(complex_features(x)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0),
        "active": torch.ones(2, 120),
    }
    item = (SimpleNamespace(batch=batch), context)
    pool = SimpleNamespace(select=lambda _: item)
    proposal_path = tmp_path / "proposal.pth"
    torch.save(
        {
            "delta": {n: torch.full_like(p, 1e-6) for n, p in model.adapter.named_parameters()},
            "report": {"search": {"selected_scale": 0.001}},
        },
        proposal_path,
    )
    config = {
        "starts": {"seed": {"path": str(initial_path)}},
        "groups": [{"index": 1, "draws": [0] * 5}],
        "cases": [{"index": 1, "samples": {"guard": [{"row": {"corpus": "music_library"}}]}}],
    }
    entries = {
        "probe": {
            "path": str(proposal_path),
            "sha256": sha256_file(proposal_path),
            "seed": "seed",
            "support": "high",
            "index": 1,
        }
    }
    first = recheck(config, entries, pool, {"draws": [[0]]}, tmp_path)
    assert (tmp_path / "review/probe.receipt.json").exists()
    assert set(first["probe"]["guard"]) == {"start", "adam", "fixed", "backtrack"}
    pool.select = lambda _: pytest.fail("completed guard must not be rescored")
    assert recheck(config, entries, pool, {"draws": [[0]]}, tmp_path) == first
