"""P53 corpus indexing, fixed RMS budgets, finite gradients and exact resume."""

import copy
import shutil
from types import SimpleNamespace

import pytest
import torch

from evaluation.balanced_preservation import (
    ARMS,
    SHARES,
    CorpusPool,
    calibrate,
    coefficients,
    grouped_penalties,
    update,
)
from evaluation.physical_preservation import penalties
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_adapter_pilot import optimizer_for


def grouped_item():
    from test_physical_preservation import model_item

    initial, model, item = model_item()
    batch = {k: v.repeat_interleave(5, dim=0) for k, v in item[0].batch.items()}
    levels = torch.tensor([0.1, 0.3, 0.7, 1.0, 0.05] * 2)
    for key in ("x", "y", "clean_wave"):
        batch[key] = batch[key] * levels.reshape(-1, *([1] * (batch[key].ndim - 1)))
    corpora = ["music_library"] * 4 + ["speech_library"]
    return (
        initial,
        model,
        (SimpleNamespace(batch=batch, corpora=corpora), item[1].repeat_interleave(5, dim=0)),
    )


def test_corpus_metadata_and_original_term_equivalence():
    _, model, item = grouped_item()
    terms = grouped_penalties(model, item)
    original = penalties(model, item)[0]
    assert torch.equal(terms["physical"], original["physical"])
    for kind in ("logit", "physical"):
        torch.testing.assert_close(
            sum(SHARES[c] * terms[f"{kind}/{c}"] for c in SHARES), original[kind]
        )
    # Vary corpus order together with tensor rows; no first-four indexing.
    indices = torch.tensor([4, 1, 2, 3, 0, 9, 6, 7, 8, 5])
    shuffled = (
        SimpleNamespace(
            batch={k: v[indices] for k, v in item[0].batch.items()},
            corpora=[item[0].corpora[i] for i in indices[:5]],
        ),
        item[1][indices],
    )
    for k, v in grouped_penalties(model, shuffled).items():
        torch.testing.assert_close(v, terms[k])
    bad = copy.deepcopy(item)
    bad[0].corpora[-1] = "unknown"
    with pytest.raises(ValueError, match="known corpora"):
        grouped_penalties(model, bad)
    bad = copy.deepcopy(item)
    bad[0].batch["identity"][0] = True
    with pytest.raises(ValueError, match="metadata"):
        grouped_penalties(model, bad)
    pool = CorpusPool(
        SimpleNamespace(select=lambda _: item, rows=[{"corpus": c} for c in shuffled[0].corpora])
    )
    assert pool.select(list(range(5)))[0].corpora == shuffled[0].corpora
    assert not model.adapter._forward_hooks


def test_rms_not_pooled_norm_cross_terms_and_scale_invariance():
    # Exactly cancelling batch means must still receive a finite RMS budget.
    gs = {"common": torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)}
    for kind in ("physical", "logit"):
        gs[kind + "/music_library"] = torch.tensor([[2.0, 0.0], [-2.0, 0.0]]).double()
        gs[kind + "/speech_library"] = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]).double()
    gs["physical"] = 0.8 * gs["physical/music_library"] + 0.2 * gs["physical/speech_library"]
    result = coefficients(gs)
    assert gs["common"].mean(0).norm() == 0
    for arm in ARMS:
        report = result["diagnostics"][arm]
        assert report["combined_rms_to_common"] == pytest.approx(0.1)
        if arm.startswith("balanced"):
            values = list(report["component_rms_to_common"].values())
            assert values[0] / values[1] == pytest.approx(4)
    changed = {k: v * (1e-12 if k.endswith("speech_library") else 1) for k, v in gs.items()}
    rescaled = coefficients(changed)
    for arm in ARMS:
        a = sum(w * gs[k] for k, w in result["weights"][arm].items())
        b = sum(w * changed[k] for k, w in rescaled["weights"][arm].items())
        torch.testing.assert_close(a, b)
    changed["physical/speech_library"] *= 0
    with pytest.raises(ValueError, match="inactive"):
        coefficients(changed)
    gs["common"] *= float("nan")
    with pytest.raises(ValueError, match="finite"):
        coefficients(gs)


def test_fixed_weights_real_gradients_and_p52_global_update_equivalence():
    from evaluation.adapter_pilot import assert_frozen
    from evaluation.physical_preservation import update as p52_update

    initial, model, item = grouped_item()
    reference = copy.deepcopy(model)
    scales = calibrate(model, [item] * 2)
    for arm in ARMS:
        assert scales["diagnostics"][arm]["combined_rms_to_common"] == pytest.approx(0.1)
    weight = scales["weights"]["rms-physical"]["physical"]
    result = update(model, [item] * 4, optimizer_for(model), "rms-physical", scales)
    control = p52_update(
        reference,
        [item] * 4,
        optimizer_for(reference),
        "physical",
        {"weights": {"physical": weight}},
    )
    assert result == control
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items())
    assert_frozen(model, initial)
    for corpus in SHARES:
        # A target perturbation changes this corpus only, with finite derivatives.
        changed = copy.deepcopy(item)
        index = changed[0].corpora.index(corpus) + 5
        changed[0].batch["clean_wave"][index, 89 * 128 + 8] += 1e-5
        before, after = grouped_penalties(model, item), grouped_penalties(model, changed)
        for key in before:
            if key.startswith("logit") or ("/" in key and not key.endswith(corpus)):
                assert torch.equal(before[key], after[key])
        value = after[f"physical/{corpus}"]
        gradients = torch.autograd.grad(value, list(model.adapter.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients)


def test_real_training_resume_frozen_barrier_and_review_receipts(tmp_path, monkeypatch):
    from scripts import run_balanced_preservation as runner
    from scripts.run_adapter_continuation import raw_review

    initial, model, item = grouped_item()
    base = tmp_path / "base.pth"
    torch.save(initial, base)
    parent = tmp_path / "parent.pth"
    torch.save(
        {
            "format": "soundex-spectral-adapter-v1",
            "model": model.state_dict(),
            "base_config": model.base_config,
            "support": "high",
        },
        parent,
    )
    base_entry = {"path": str(base), "sha256": sha256_file(base)}
    parent_entry = {"path": str(parent), "sha256": sha256_file(parent)}
    sample = {"row": {"corpus": "music_library"}, "spec": {"row_id": "fixture"}}
    config = {
        "starts": {"seed": base_entry},
        "parents": {"seed": parent_entry},
        "steps": 2,
        "calibration_indices": [0],
        "prior": str(tmp_path / "p52"),
        "p51": str(tmp_path / "p51"),
        "p50": str(tmp_path / "p50"),
        "cases": {
            role: [{"index": i, "samples": {"calibration": [sample] * 5}} for i in range(8)]
            for role in ("fit", "review")
        },
    }
    schedule = {"fit": [[0]] * 8, "review": [[0]]}
    pool = SimpleNamespace(select=lambda _: item)
    output = tmp_path / "current"
    scales = runner.calibration(config, pool, schedule, output)
    assert runner.calibration(config, pool, schedule, output) == scales
    whole = runner.train("seed", "balanced-physical", config, scales, pool, schedule, output)
    actual, calls = runner.update, 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interruption")
        return actual(*args)

    monkeypatch.setattr(runner, "update", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.train(
            "seed", "balanced-physical", config, scales, pool, schedule, tmp_path / "resumed"
        )
    monkeypatch.setattr(runner, "update", actual)
    resumed = runner.train(
        "seed", "balanced-physical", config, scales, pool, schedule, tmp_path / "resumed"
    )
    a, b = (torch.load(p, weights_only=False) for p in (whole, resumed))
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    for k, state in a["optimizer"]["state"].items():
        assert all(torch.equal(v, b["optimizer"]["state"][k][f]) for f, v in state.items())
    entries = {}
    for arm in ARMS:
        path = runner.train("seed", arm, config, scales, pool, schedule, output)
        entries["seed-" + arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "seed": "seed",
            "arm": arm,
        }
    parents = {"seed": base_entry, "seed-persistent": parent_entry, "seed-reset": parent_entry}
    raw_review(config, parents, pool, schedule, tmp_path / "p50")
    for role in ("fit", "review"):
        dest = tmp_path / "p51/review" / role
        dest.mkdir(parents=True)
        for arm in ("separated", "scalar"):
            shutil.copy2(
                tmp_path / "p50/review" / role / "seed-persistent.json",
                dest / f"seed-persistent-{arm}.json",
            )
    for role in ("fit", "review"):
        dest = tmp_path / "p52/review" / role
        dest.mkdir(parents=True)
        for arm in ("control", "logit", "physical"):
            shutil.copy2(
                tmp_path / "p50/review" / role / "seed-persistent.json",
                dest / f"seed-{arm}.json",
            )
    write_json_report(entries, output / "review-registration.json")
    runner.review(config, entries, pool, schedule, output)
    assert len(list((output / "review").rglob("*.receipt.json"))) == 6
    monkeypatch.setattr(
        runner, "review_case", lambda *_: pytest.fail("completed receipt must be reused")
    )
    runner.review(config, entries, pool, schedule, output)
    write_json_report({}, output / "review-registration.json")
    with pytest.raises(ValueError, match="barrier"):
        runner.review(config, entries, pool, schedule, output)
