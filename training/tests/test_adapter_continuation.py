"""Global source isolation, exact calibration continuity and optimizer resume checks."""

from types import SimpleNamespace

import pytest
import torch

from evaluation.adapter_continuation import schedule_roles, update
from evaluation.adapter_pilot import assert_frozen
from evaluation.adapter_scale import calibrate
from evaluation.reporting import sha256_file
from models.complex_synthesis import complex_features, synthesize
from models.spectral_adapter import from_base
from scripts.run_adapter_pilot import optimizer_for


def bank():
    rows, specs = {}, []
    for corpus in ("music_library", "speech_library"):
        for index in range(120):
            for alias in range(2):
                key = f"{corpus}-{index}-{alias}"
                rows[key] = {
                    "track_id": key,
                    "source_checksum": f"{corpus}-{index}",
                    "corpus": corpus,
                    "split": "train",
                    "held_out": False,
                }
                specs.append({"row_id": key, "start": 0, "gain_db": 0.0})
    # A shared track joins two distinct checksums as well as their aliases.
    rows["music_library-1-0"]["track_id"] = rows["music_library-0-0"]["track_id"]
    return specs, rows


def test_global_role_partition_aliases_mix_and_determinism():
    specs, rows = bank()
    result = schedule_roles(specs, rows, steps=8, review_batches=16)
    assert result == schedule_roles(specs, rows, steps=8, review_batches=16)
    assert len(result["fit"]) == 32 and len(result["review"]) == 16
    for field in ("track_id", "source_checksum"):
        sets = {
            role: {rows[specs[i]["row_id"]][field] for d in draws for i in d}
            for role, draws in result.items()
        }
        assert not sets["fit"] & sets["review"]
    for draws in result.values():
        for draw in draws:
            assert [rows[specs[i]["row_id"]]["corpus"] for i in draw] == ["music_library"] * 4 + [
                "speech_library"
            ]
            assert len({rows[specs[i]["row_id"]]["source_checksum"] for i in draw}) == 5


def test_nontrain_schedule_rejected():
    specs, rows = bank()
    rows[specs[0]["row_id"]]["held_out"] = True
    with pytest.raises(ValueError, match="TRAIN"):
        schedule_roles(specs, rows)


def real_item():
    from test_spectral_adapter import CONFIG, fixture

    base, x, context, condition = fixture(frames=120)
    initial = {"arm": CONFIG["arm"], "model_config": CONFIG, "model": base.state_dict()}
    target = x.clone()
    target[0, 0, :, 65:] += 0.2
    batch = {
        "x": x,
        "y": target,
        "clean_wave": synthesize(complex_features(target)).detach(),
        "identity": torch.tensor([False, True]),
        "rate": condition[:, :1, 0] * 48000,
        "cutoff": torch.full((2, 120), 8000.0),
        "active": torch.ones(2, 120),
    }
    return initial, (SimpleNamespace(batch=batch), context)


def test_first_continuous_step_matches_p49_and_moments_accumulate():
    initial, item = real_item()
    model = from_base(initial, "high").eval()
    reference, delta = calibrate(model, [item] * 4)
    optimizer = optimizer_for(model)
    first = update(model, [item] * 4, optimizer)
    for name, parameter in model.adapter.named_parameters():
        assert torch.equal(parameter, delta[name] * 0.001)
    assert first["fit_before"] == reference["base"]
    assert first["fit_after"] == reference["controls"]["fixed"]["fit"]
    assert first["changed_parameters"] > 0
    update(model, [item] * 4, optimizer)
    assert all(int(state["step"]) == 2 for state in optimizer.state.values())
    assert_frozen(model, initial)


def test_checkpoint_resume_matches_uninterrupted_and_raw_review_receipts(tmp_path, monkeypatch):
    from scripts import run_adapter_continuation as runner

    initial, item = real_item()
    initial_path = tmp_path / "initial.pth"
    torch.save(initial, initial_path)
    entry = {"path": str(initial_path), "sha256": sha256_file(initial_path)}
    sample = {"row": {"corpus": "music_library"}, "spec": {"row_id": "fixture"}}
    config = {
        "steps": 2,
        "starts": {"seed": entry},
        "cases": {
            role: [{"index": 0, "samples": {"calibration": [sample]}}] for role in ("fit", "review")
        },
    }
    schedule = {"fit": [[0]] * 8, "review": [[0]]}
    pool = SimpleNamespace(select=lambda _: item)
    first = runner.train("seed", entry, "persistent", config, pool, schedule, tmp_path / "whole")
    calls = 0
    actual = runner.update

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return actual(*args)

    monkeypatch.setattr(runner, "update", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.train("seed", entry, "persistent", config, pool, schedule, tmp_path / "resume")
    monkeypatch.setattr(runner, "update", actual)
    resumed = runner.train("seed", entry, "persistent", config, pool, schedule, tmp_path / "resume")
    a, b = [torch.load(p, weights_only=False) for p in (first, resumed)]
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    for key, state in a["optimizer"]["state"].items():
        assert all(
            torch.equal(value, b["optimizer"]["state"][key][field])
            for field, value in state.items()
        )
    reset = runner.train("seed", entry, "reset", config, pool, schedule, tmp_path / "whole")
    c = torch.load(reset, weights_only=False)
    assert all(int(s["step"]) == 1 for s in c["optimizer"]["state"].values())
    endpoints = {
        "seed": entry,
        "seed-persistent": {"path": str(first), "sha256": sha256_file(first)},
        "seed-reset": {"path": str(reset), "sha256": sha256_file(reset)},
    }
    runner.raw_review(config, endpoints, pool, schedule, tmp_path / "review")
    assert len(list((tmp_path / "review/review").rglob("*.receipt.json"))) == 6
    monkeypatch.setattr(
        runner, "review_case", lambda *_: pytest.fail("must reuse completed review")
    )
    runner.raw_review(config, endpoints, pool, schedule, tmp_path / "review")
