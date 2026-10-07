"""Shared deterministic checkpoint fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from checkpoint import build_checkpoint, save_checkpoint
from checkpoint_state import canonical_sha256
from configuration import load_config
from models.generator import SoundExGenerator
from validation import BestCheckpointTracker, aggregate_validation_rows, build_validation_state


@pytest.fixture
def resolved_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    value = load_config(path)
    assert isinstance(value, dict)
    return value


@pytest.fixture
def checkpoint_factory(tmp_path: Path, resolved_config: dict[str, Any]):
    def create(
        name: str = "model.pth",
        *,
        config: dict[str, Any] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        checkpoint_config = resolved_config if config is None else config
        with torch.random.fork_rng():
            torch.manual_seed(7)
            generator_config = checkpoint_config["model"]["generator"]
            generator = SoundExGenerator(
                channels=generator_config["channels"],
                bottleneck_blocks=generator_config["bottleneck_blocks"],
                expand_ratio=generator_config["expand_ratio"],
            )
        recipe = checkpoint_config["data"]["recipe"]
        recipe_sha256 = canonical_sha256(recipe)
        manifest_sha256 = "a" * 64
        data_provenance = {
            "recipe": recipe,
            "recipe_sha256": recipe_sha256,
            "manifests": [
                {
                    "corpus": "musdb18_hq",
                    "path": "/datasets/musdb18_hq/manifest.jsonl",
                    "sha256": manifest_sha256,
                    "summary": {"rows": 12},
                }
            ],
            "manifest_set_sha256": canonical_sha256(
                [{"corpus": "musdb18_hq", "sha256": manifest_sha256}]
            ),
            "effective_source_counts": {
                "strategy": "weighted",
                "train_available": {"musdb18_hq": 8},
                "val_size": 4,
                "validation_row_ids": ["validation-fixture"],
            },
        }
        data_provenance["validation_manifest_sha256"] = canonical_sha256(
            {
                "manifest_set_sha256": data_provenance["manifest_set_sha256"],
                "row_ids": ["validation-fixture"],
            }
        )
        validation_report = aggregate_validation_rows(
            [
                {
                    "row_id": "validation-fixture",
                    "corpus": "musdb18_hq",
                    "codec_id": "mp3-cbr-64",
                    "quality": "cbr:64",
                    "sample_rate": 44_100,
                    "channel_role": "mid",
                    "metrics": {
                        "high_band_magnitude": 1.5,
                        "low_band_identity": 0.1,
                    },
                }
            ]
        )
        tracker = BestCheckpointTracker()
        assert tracker.consider(10, validation_report["overall"])
        validation_state = build_validation_state(
            report=validation_report,
            manifest_sha256=data_provenance["validation_manifest_sha256"],
            tracker=tracker,
            interval_epochs=1,
            epoch=10,
        )
        checkpoint = build_checkpoint(
            epoch=10,
            global_step=120,
            generator_state=generator.state_dict(),
            discriminator_state={"fixture": torch.tensor([1.0])},
            generator_optimizer_state={"state": {}, "param_groups": []},
            discriminator_optimizer_state={"state": {}, "param_groups": []},
            generator_scheduler_state={"last_epoch": 10},
            discriminator_scheduler_state={"last_epoch": 10},
            scaler_state=None,
            resolved_config=checkpoint_config,
            data_provenance=data_provenance,
            validation_state=validation_state,
            provenance={
                "python_version": "3.12.fixture",
                "pytorch_version": torch.__version__,
                "ffmpeg_version": None,
                "source_git_sha": None,
            },
        )
        path = tmp_path / name
        save_checkpoint(checkpoint, path)
        return path, checkpoint

    return create
