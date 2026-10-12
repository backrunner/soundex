from functools import partial

import pytest
import torch

from models.factorized_envelope import factor_objective, make_factor_model
from scripts import run_envelope_study as runner
from train import spectral_features


class SyntheticClips:
    def __init__(self):
        self.rows = {}
        self.values = {}
        for index, corpus in enumerate(("music_library", "speech_library")):
            g = torch.Generator().manual_seed(50 + index)
            pcm = torch.randn(1, 96 * 128, generator=g) * 0.1
            clean = spectral_features(torch.nn.functional.pad(pcm * 1.1, (128, 0)), 256, 128)[0]
            for suffix in ("", ":identity"):
                key = corpus + suffix
                self.rows[key] = {
                    "row_id": key,
                    "track_id": corpus,
                    "corpus": corpus,
                    "codec_id": "synthetic",
                    "sample_rate": 48000,
                    "channel_role": "mono",
                }
                wave = pcm * (1.1 if suffix else 1.0)
                self.values[key] = {
                    "pcm": wave[0],
                    "x": spectral_features(torch.nn.functional.pad(wave, (128, 0)), 256, 128)[0],
                    "y": clean,
                    "rate": torch.tensor([48000.0]),
                    "cutoff": torch.full((96,), 16000.0),
                    "active": torch.ones(96, dtype=torch.bool),
                }

    def batch(self, specs, device="cpu"):
        values = [self.values[s["row_id"]] for s in specs]
        return {key: torch.stack([v[key] for v in values]).to(device) for key in values[0]}


def test_resume_matches_uninterrupted_training_and_receipts(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    clips = SyntheticClips()
    specs = [
        {"row_id": corpus, "start": 0, "gain_db": 0.0}
        for corpus in ("music_library", "speech_library")
    ]
    config = {
        "device": "cpu",
        "learning_rate": 1e-4,
        "steps": 3,
        "validation_interval": 1,
        "validation_specs": specs,
        "training_probe_specs": specs,
        "history_ablation_arms": [],
        "checkpoint_format": "factor-envelope-v1",
    }
    arguments = dict(
        model_factory=partial(make_factor_model, "factorized", width=16),
        training_objective=factor_objective,
        selection_objective=factor_objective,
        context_arm="past",
        burn=80,
        include_initial=True,
    )
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    runner.run_arm(config, 5, "factorized", specs * 6, clips, full, **arguments)
    save = runner.atomic_save

    def interrupt(checkpoint, path):
        save(checkpoint, path)
        if checkpoint["step"] == 1 and path.name == "latest.pth":
            raise RuntimeError("test interruption")

    monkeypatch.setattr(runner, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        runner.run_arm(config, 5, "factorized", specs * 6, clips, resumed, **arguments)
    monkeypatch.setattr(runner, "atomic_save", save)
    runner.run_arm(config, 5, "factorized", specs * 6, clips, resumed, **arguments)
    a, b = [
        torch.load(root / "factorized-5/latest.pth", weights_only=False) for root in (full, resumed)
    ]
    assert a["step"] == b["step"] == 3
    assert a["validation"] == b["validation"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="completed arm changed"):
        runner.run_arm(
            {**config, "learning_rate": 2e-4},
            5,
            "factorized",
            specs * 6,
            clips,
            resumed,
            **arguments,
        )
    (resumed / "factorized-5/evaluation.json").write_text("{}")
    with pytest.raises(ValueError, match="completed arm changed"):
        runner.run_arm(config, 5, "factorized", specs * 6, clips, resumed, **arguments)
