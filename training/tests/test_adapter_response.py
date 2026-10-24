"""P51 counterfactual separation controls and real causal feature capture."""

import copy

import pytest
import torch

from evaluation.adapter_response import (
    coefficients,
    energy,
    features,
    grams,
    install,
    response_bands,
    separate,
)
from models.spectral_adapter import from_base


def test_identical_roles_reduce_to_scalar_not_false_selectivity():
    torch.manual_seed(51)
    x = torch.randn(60, 5, dtype=torch.float64)
    gram = x.T @ x / len(x)
    weights = torch.randn(4, 5).double()
    projected, scalar, report = separate(weights, gram, gram)
    torch.testing.assert_close(projected, scalar, atol=2e-6, rtol=2e-6)
    assert report["scalar"] == pytest.approx(0.5, abs=2e-6)
    assert report["relative_normal_equation_residual_after_float32"] < 1e-6


def test_separable_roles_preserve_pair_better_than_matched_scalar():
    weights = torch.ones(2, 2, dtype=torch.float64)
    gp = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    gi = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    projected, scalar, report = separate(weights, gp, gi)
    assert energy(projected - weights, gp) < energy(scalar - weights, gp) * 1e-8
    assert energy(projected, gi) == pytest.approx(energy(scalar, gi), rel=1e-6)
    assert report["scalar"] < 1e-5


def test_zero_penalty_singular_features_and_basis_rescaling():
    torch.manual_seed(52)
    p, i = torch.randn(20, 4, dtype=torch.float64), torch.randn(20, 4, dtype=torch.float64)
    p[:, 2], i[:, 2] = 0, 0
    gp, gi = p.T @ p / 20, i.T @ i / 20
    weights = torch.randn(3, 4).double()
    zero, _, _ = separate(weights, gp, gi, penalty=0)
    assert torch.equal(zero, weights)
    projected, _, _ = separate(weights, gp, gi)
    scale = torch.tensor([2.0, 0.1, 1.0, 4.0], dtype=torch.float64)
    alternate, _, _ = separate(
        weights / scale, gp * scale[:, None] * scale[None, :], gi * scale[:, None] * scale[None, :]
    )
    torch.testing.assert_close(alternate * scale, projected, atol=1e-7, rtol=1e-6)
    assert torch.equal(projected[:, 2], weights[:, 2])


def test_capture_excludes_warmup_restores_hooks_and_does_not_change_model():
    from test_adapter_continuation import real_item

    initial, item = real_item()
    model = from_base(initial, "high").eval()
    before = copy.deepcopy(model.state_dict())
    z = features(model, item)
    assert z.shape == (2, 31, 17) and torch.all(z[..., -1] == 1)
    assert not model.adapter._forward_pre_hooks
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    # Here input observations are exactly equal across the paired/identity views.
    same = (
        type(item[0])(batch={k: torch.cat((v[:1], v[:1])) for k, v in item[0].batch.items()}),
        torch.cat((item[1][:1], item[1][:1])),
    )
    same[0].batch["identity"] = torch.tensor([False, True])
    same_z = features(model, same)
    values = grams(same_z)
    assert torch.equal(values["paired"], values["identity"])
    weights = coefficients(model)
    weights[:, -1] = 1e-3
    install(model, weights)
    response = response_bands(model, z, item[0].batch["rate"])
    assert response["normalized_power_0_4000"] == [0, 0]
    assert min(response["normalized_power_8000_12000"]) > 0
    original = model.adapter.forward
    model.adapter.forward = lambda _: (_ for _ in ()).throw(RuntimeError("probe failure"))
    with pytest.raises(RuntimeError, match="probe failure"):
        features(model, item)
    assert not model.adapter._forward_pre_hooks
    model.adapter.forward = original


def test_actual_candidate_barrier_reviews_and_receipt_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_adapter_continuation import real_item

    from evaluation.reporting import sha256_file
    from scripts import audit_adapter_response as runner
    from scripts.run_adapter_continuation import raw_review

    initial, item = real_item()
    original = tmp_path / "initial.pth"
    torch.save(initial, original)
    base_entry = {"path": str(original), "sha256": sha256_file(original)}
    parents = {"seed": base_entry}
    model = from_base(initial, "high").eval()
    weights = coefficients(model)
    weights[:, -1] = 1e-5
    install(model, weights)
    for arm in ("persistent", "reset"):
        path = tmp_path / f"{arm}.pth"
        torch.save(
            {
                "format": "soundex-spectral-adapter-v1",
                "model": model.state_dict(),
                "base_config": model.base_config,
                "support": "high",
            },
            path,
        )
        parents["seed-" + arm] = {"path": str(path), "sha256": sha256_file(path)}
    sample = {"row": {"corpus": "music_library"}, "spec": {"row_id": "fixture"}}
    config = {
        "prior": str(tmp_path / "prior"),
        "starts": {"seed": base_entry},
        "parents": parents,
        "cases": {
            role: [{"index": 0, "samples": {"calibration": [sample]}}] for role in ("fit", "review")
        },
    }
    schedule = {"fit": [[0]], "review": [[0]]}
    pool = SimpleNamespace(select=lambda _: item)
    raw_review(config, parents, pool, schedule, tmp_path / "prior")
    output = tmp_path / "current"
    stats = {"seed": {"grams": grams(features(model, item))}}
    entries = runner.candidates(config, stats, output)
    assert len(entries) == 4 and (output / "candidate-registration.json").exists()
    runner.review(config, entries, pool, schedule, output)
    assert len(list((output / "review").rglob("*.receipt.json"))) == 8
    monkeypatch.setattr(
        runner, "review_case", lambda *_: pytest.fail("completed review must be reused")
    )
    runner.review(config, entries, pool, schedule, output)
    assert runner.candidates(config, stats, output) == entries
