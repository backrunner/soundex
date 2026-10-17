"""Register and reuse actual schedule files; verify downstream analyzer contract."""

from pathlib import Path

import pytest

from data.causal_study import SAMPLES, digest
from evaluation.reporting import write_json_report
from scripts import run_broad_budget_study as study


def test_registration_binds_schedules_and_downstream_replay_counts(tmp_path, monkeypatch):
    rows = {}
    for i in range(6):
        name = str(i)
        rows[name] = {
            "row_id": name,
            "track_id": name,
            "source_checksum": name,
            "start_sample": 0,
            "sample_rate": 48000,
            "length_samples": SAMPLES * 6,
            "corpus": "speech_library" if i == 4 else "music_library",
            "split": "test" if i == 5 else "train",
            "held_out": i == 5,
            "sampling_weight": 1.0,
            "codec_id": "fixture",
            "channel_role": "mono",
        }
    bank = [{"row_id": str(i), "start": 0, "gain_db": 0.0} for i in range(5)]
    draws = [list(range(5))] * 4
    parent = tmp_path / "parent"
    root = tmp_path / "study"
    parent.mkdir()
    root.mkdir()
    native = tmp_path / "native"
    native.write_bytes(b"fixture")
    config = {
        "bank": bank,
        "common": bank,
        "seeds": [7, 8],
        "rate": 1e-4,
        "weight": 0.1365237430457469,
        "development": [{"row_id": "5", "start": 0, "gain_db": 0.0}],
        "shifted": [{"row_id": "0", "start": SAMPLES * 4, "gain_db": 0.0}],
        "tolerance": 1e-6,
        "input_hashes": {},
        "schedule_hashes": {str(seed): digest(draws) for seed in (7, 8)},
    }
    write_json_report(config, parent / "registration.json")
    write_json_report({"status": "complete"}, parent / "summary.json")
    for seed in config["seeds"]:
        write_json_report({"draws": draws}, parent / f"draws-{seed}.json")
    monkeypatch.setattr(study, "load_sources", lambda paths: (rows, {}))
    monkeypatch.setattr(study, "STEPS", 4)
    monkeypatch.setattr(study, "ENDPOINTS", [2, 4])
    registered = study.register(parent, root, native)
    assert registered["native_count"] == (1 + 4 * 2 * 2) * 1 * 2
    assert registered["joint_native_count"] == registered["native_count"] + 2
    assert registered["paced_arms"] == registered["arms"]
    assert not registered["score_entire_bank"]
    assert study.register(parent, root, native) == registered
    schedule = Path(registered["schedule_files"]["broad-7"])
    schedule.write_text("changed")
    with pytest.raises(ValueError, match="input drift"):
        study.register(parent, root, native)
