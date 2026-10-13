import copy

import pytest
import torch

from evaluation.reporting import write_json_report
from scripts import run_consistency_followup as runner
from scripts.analyze_consistency_followup import grouped, support_checks
from scripts.run_consistency_followup import registered, verify_receipt


def test_recording_mean_does_not_overweight_multiple_clips():
    rows = [{"corpus": "music_library", "track_id": "a", "metrics": {"total": x}} for x in (1, 3)]
    rows += [
        {"corpus": "music_library", "track_id": "b", "metrics": {"total": 8}},
        {"corpus": "speech_library", "track_id": "c", "metrics": {"total": 4}},
    ]
    value = grouped(rows)
    assert value["clip"]["music_library"]["total"] == 4
    assert value["recording"]["music_library"]["total"] == 5
    for row in rows:
        row["metrics"]["rows"] = [{"spectral": 0.1}]
    assert grouped(rows) == value


def test_identity_only_gain_cannot_pass_restoration_screen():
    proxy = {"total": 1.0, "paired_restoration": 1.0, "identity_preservation": 1.0}
    native = {
        f"output.paired.{k}": 1.0
        for k in ("high_sc", "high_flux", "high_energy_error_db", "band_0_4000_sc")
    }
    start = {
        "proxy": {w: {"music_library": proxy.copy()} for w in ("clip", "recording")},
        "native": {w: {"music_library": native.copy()} for w in ("clip", "recording")},
    }
    end = copy.deepcopy(start)
    for weighting in ("clip", "recording"):
        end["proxy"][weighting]["music_library"].update(total=0.75, identity_preservation=0.5)
    checks = support_checks(start, end, start)
    assert not all(checks.values())
    assert not checks["clip_music_paired_restoration_below_start_and_initial"]
    assert not checks["recording_native_high_sc_below_start_and_initial"]


def test_registration_and_completed_artifacts_reject_drift(tmp_path):
    path = tmp_path / "registration.json"
    registered({"steps": [512, 2048]}, path)
    registered({"steps": [512, 2048]}, path)
    with pytest.raises(ValueError, match="registered artifact changed"):
        registered({"steps": [512, 4096]}, path)
    from evaluation.reporting import sha256_file

    registered({"files": {path.name: sha256_file(path)}}, tmp_path / "complete.json")
    verify_receipt(tmp_path, "files")
    path.write_text("{}")
    with pytest.raises(ValueError, match="completed artifact changed"):
        verify_receipt(tmp_path, "files")


def test_continuation_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    from types import SimpleNamespace

    torch.set_num_threads(1)

    def factory(_):
        return torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1)
        )

    class Objective:
        def __init__(self):
            self.batch = {}

        def __call__(self, prediction):
            return {"total": (prediction - 0.1).square().mean()}

    class Pool:
        def __init__(self):
            self.rows = [{"corpus": c} for c in ["music_library"] * 4 + ["speech_library"]]

        def select(self, _):
            return Objective(), torch.ones(5, 2)

    score = {"weighted": {"total": 0.1}, "auxiliary": 0.05, "combined": 0.105}
    monkeypatch.setattr(runner, "make_rendered_model", factory)
    monkeypatch.setattr(runner, "features", lambda m, o, c: m(c))
    monkeypatch.setattr(runner, "consistency_loss", lambda p, b: p.square().mean())
    monkeypatch.setattr(runner, "evaluate_bank", lambda *args: copy.deepcopy(score))
    monkeypatch.setattr(runner, "print", lambda *args, **kwargs: None, raising=False)
    args = SimpleNamespace(p16=tmp_path / "p16", p17=tmp_path / "p17")
    p16 = args.p16 / "size-40-5"
    p17 = args.p17 / "seed-5-lr-0.0001"
    p16.mkdir(parents=True)
    p17.mkdir(parents=True)
    write_json_report({"crop_draws": [[0, 1, 2, 3, 4]] * 2048}, p16 / "draws.json")
    torch.manual_seed(17)
    model = factory(None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model(torch.ones(5, 2)).square().mean().backward()
    optimizer.step()
    for value in optimizer.state.values():
        value["step"].fill_(512)
    torch.save(
        {
            "step": 512,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        p17 / "step-512.pth",
    )
    write_json_report(score, p17 / "evaluation-512.json")
    registration = {"rate": 0.0001, "training_checks": list(range(512, 2049, 256))}
    config = {"consistency_weight": 0.1}
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    runner.train(5, args, full, config, registration, Pool())
    save = runner.atomic_save

    def interrupt(value, path):
        save(value, path)
        if path.name == "latest.pth" and value["step"] == 832:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train(5, args, resumed, config, registration, Pool())
    monkeypatch.setattr(runner, "atomic_save", save)
    runner.train(5, args, resumed, config, registration, Pool())
    a, b = [
        torch.load(root / "seed-5/step-2048.pth", weights_only=False) for root in (full, resumed)
    ]
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for key, value in a["optimizer"]["state"].items():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                value[name], b["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    runner.train(5, args, resumed, config, registration, Pool())
