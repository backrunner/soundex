"""Matched draws, control architecture, and exact checkpoint recovery tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.causal_study import make_schedule
from export_study import make_model
from models.causal_restorer import CausalRestorer
from scripts import run_causal_study as runner
from tests.test_causal_objective import make_batch


def test_frozen_draws_preserve_recording_and_corpus_mass():
    rows = {}
    for i in range(255):
        corpus = "music_library" if i < 253 else "speech_library"
        rows[str(i)] = {
            "row_id": str(i),
            "corpus": corpus,
            "track_id": str(i),
            "split": "train",
            "held_out": False,
            "length_samples": 48000,
            "sample_rate": 48000,
            "source_metadata": {"genre": "pop" if i < 252 else "country"},
        }
    a, report = make_schedule(rows, 42, 8192)
    b, _ = make_schedule(rows, 42, 8192)
    c, _ = make_schedule(rows, 43, 8192)
    assert a == b and a != c
    assert report["ratios"]["speech_library"] == 1 / 254
    assert report["available_recordings"]["music_library"] == 253
    assert 10 < report["draws_by_corpus"]["speech_library"] < 65
    assert report["drawn_recordings"] == 255
    assert all(0 <= r["start"] <= 48000 - 32768 and -6 <= r["gain_db"] <= 3 for r in a)


def test_legacy_control_matches_existing_shape_small_budget():
    assert sum(p.numel() for p in make_model("legacy").parameters()) == 905734


def test_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    batch = make_batch(
        np.random.default_rng(4).normal(0, 0.1, 32768).astype(np.float32), 48000, tmp_path
    )

    class Clips:
        def __init__(self):
            self.rows = {"a": {"corpus": "music_library"}, "b": {"corpus": "speech_library"}}
            self.failed = False
            self.calls = 0

        def batch(self, specs, device):
            self.calls += 1
            # First two updates, then two validation clips, then interrupt next update.
            if self.failed and self.calls == 5:
                raise RuntimeError("simulated interruption")
            return {k: v.to(device) for k, v in batch.items()}

    monkeypatch.setattr(runner, "make_model", lambda _: CausalRestorer("tcn", width=8))

    def export(_, path):
        Path(path).write_bytes(b"fixture")

    monkeypatch.setattr(runner, "export_study", export)
    monkeypatch.setattr(
        runner,
        "actual_score",
        lambda *_, limiter_mode="legacy": {"diagnostic_mock": True, "limiter_mode": limiter_mode},
    )
    config = {
        "steps": 4,
        "batch_size": 1,
        "learning_rate": 3e-4,
        "validation_interval": 2,
        "stream_binary": "mock",
    }
    specs = [{"row_id": "a"}, {"row_id": "b"}]
    schedule = [{"row_id": "a"}] * 4
    uninterrupted = tmp_path / "full"
    runner.run_arm("tcn", 42, config, {}, schedule, Clips(), specs, uninterrupted, "cpu")
    interrupted = tmp_path / "resumed"
    clips = Clips()
    clips.failed = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run_arm("tcn", 42, config, {}, schedule, clips, specs, interrupted, "cpu")
    runner.run_arm("tcn", 42, config, {}, schedule, Clips(), specs, interrupted, "cpu")
    states = [
        torch.load(p / "last.pth", map_location="cpu", weights_only=True)
        for p in (uninterrupted, interrupted)
    ]
    for key in states[0]["model"]:
        torch.testing.assert_close(states[0]["model"][key], states[1]["model"][key], rtol=0, atol=0)
    assert states[0]["validation"] == states[1]["validation"]
    assert json.loads((interrupted / "complete.json").read_text())["steps"] == 4
