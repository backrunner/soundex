"""Deterministic validation and checkpoint-selection tests."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.losses import GeneratorLoss
from train import validate
from validation import (
    BestCheckpointTracker,
    aggregate_validation_rows,
    build_validation_state,
    should_run_validation,
)


class _FixedValidationDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        time = torch.arange(256, dtype=torch.float32) / 44_100.0
        clean = 0.2 * torch.sin(2.0 * math.pi * (440.0 + index * 110.0) * time)
        return {
            "clean": clean,
            "degraded": clean * (0.8 + index * 0.02),
            "metadata": {
                "row_id": f"row-{index}",
                "corpus": "musdb18_hq" if index < 2 else "slakh2100",
                "codec_id": "mp3-cbr-64",
                "codec_mode": "cbr",
                "codec_setting": 64.0,
                "sample_rate": 44_100,
                "channel_role": "mid",
                "cutoff_hz": 8000.0,
            },
        }


def _row(row_id: str, high: float, low: float, corpus: str) -> dict[str, object]:
    return {
        "row_id": row_id,
        "corpus": corpus,
        "codec_id": "mp3-cbr-64",
        "quality": "cbr:64",
        "sample_rate": 44_100,
        "channel_role": "mid",
        "metrics": {
            "high_band_magnitude": high,
            "low_band_identity": low,
            "phase": high / 10.0,
        },
    }


def test_aggregation_is_bitwise_stable_across_worker_order() -> None:
    rows = [
        _row("row-c", 3.0, 0.3, "slakh2100"),
        _row("row-a", 1.0, 0.1, "musdb18_hq"),
        _row("row-b", 2.0, 0.2, "musdb18_hq"),
    ]

    forward = aggregate_validation_rows(rows)
    shuffled = aggregate_validation_rows([rows[1], rows[2], rows[0]])

    assert forward == shuffled
    assert forward["row_ids"] == ["row-a", "row-b", "row-c"]
    assert forward["overall"]["high_band_magnitude"] == 2.0
    assert forward["strata"]["corpus"]["musdb18_hq"]["row_count"] == 2


def test_complete_validation_is_identical_across_workers_and_row_order() -> None:
    dataset = _FixedValidationDataset()
    single_worker = DataLoader(dataset, batch_size=2, num_workers=0)
    two_workers = DataLoader(dataset, batch_size=2, num_workers=2)
    reversed_rows = DataLoader(Subset(dataset, [2, 1, 0]), batch_size=2, num_workers=0)
    loss = GeneratorLoss(waveform_weight=0.0, fft_size=64, hop_size=16)
    audio = {"fft_size": 64, "hop_size": 16, "crossover_width_hz": 1000.0}
    model = torch.nn.Identity()

    first = validate(model, single_worker, loss, torch.device("cpu"), audio, 4)
    second = validate(model, two_workers, loss, torch.device("cpu"), audio, 4)
    third = validate(model, reversed_rows, loss, torch.device("cpu"), audio, 4)

    assert first == second == third


def test_best_checkpoint_uses_high_band_then_low_band_tie_breaker() -> None:
    tracker = BestCheckpointTracker()

    assert tracker.consider(1, {"high_band_magnitude": 2.0, "low_band_identity": 0.1})
    assert tracker.consider(2, {"high_band_magnitude": 1.5, "low_band_identity": 0.4})
    assert tracker.consider(3, {"high_band_magnitude": 1.5, "low_band_identity": 0.2})
    assert not tracker.consider(4, {"high_band_magnitude": 1.6, "low_band_identity": 0.0})
    assert tracker.best_epoch == 3

    restored = BestCheckpointTracker.from_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()


def test_resume_keeps_absolute_validation_schedule() -> None:
    assert should_run_validation(1, 5)
    assert should_run_validation(5, 5)
    assert not should_run_validation(6, 5)
    assert should_run_validation(10, 5)


def test_balanced_selection_rejects_magnitude_gain_with_worse_total() -> None:
    tracker = BestCheckpointTracker(
        primary_metric="total", tie_breaker_metric="high_band_magnitude"
    )
    assert tracker.consider(1, {"total": 2.0, "high_band_magnitude": 10.0})
    assert not tracker.consider(2, {"total": 2.1, "high_band_magnitude": 9.0})
    assert tracker.consider(3, {"total": 1.9, "high_band_magnitude": 10.1})
    assert tracker.consider(4, {"total": 1.9, "high_band_magnitude": 10.0})
    assert not tracker.consider(5, {"total": 1.9, "high_band_magnitude": 10.0})
    assert BestCheckpointTracker.from_state_dict(tracker.state_dict()).state_dict() == (
        tracker.state_dict()
    )


def test_validation_state_binds_report_to_manifest() -> None:
    report = aggregate_validation_rows([_row("row-a", 1.0, 0.1, "musdb18_hq")])
    tracker = BestCheckpointTracker()
    assert tracker.consider(5, report["overall"])

    state = build_validation_state(
        report=report,
        manifest_sha256="a" * 64,
        tracker=tracker,
        interval_epochs=5,
        epoch=5,
    )

    assert state["report"] == report
    assert state["manifest_sha256"] == "a" * 64
    assert state["best"]["best_epoch"] == 5
