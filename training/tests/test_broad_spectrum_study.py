import copy
from types import SimpleNamespace

import pytest

from scripts.run_broad_spectrum_study import (
    ScheduledBatches,
    protocol,
    review_inventory,
    validate_schedule,
)


def fixture():
    rows = {
        str(i): {
            "track_id": str(i),
            "source_checksum": str(i),
            "start_sample": 0,
            "sample_rate": 48000,
            "length_samples": 96000,
            "split": "train",
            "held_out": False,
            "corpus": "music_library" if i < 4 else "speech_library",
        }
        for i in range(5)
    }
    return rows, {
        "specs": [{"row_id": str(i), "start": 0, "gain_db": 0.0} for i in range(5)],
        "draws": [[0, 1, 2, 3, 4]],
    }


def test_fixed_broad_protocol_keeps_losses_but_not_fitted_training_specs():
    parent = {
        "seeds": [12, 13],
        "loss_routes": {
            "control": {"raw": 1, "budget": 1},
            "mr_add": {"raw": 1, "multires": 1, "budget": 1},
        },
        "direct_loss": {"version": 1},
        "multires_loss": {"version": "v1"},
        "weight": 0.1365,
        "rate": 1e-4,
        "specs": ["fitted"],
    }
    broad = {"development": ["validation"]}
    full = protocol(parent, broad)
    assert full["steps"] == full["primary_endpoint"] == 16192
    assert full["arms"] == ["control", "mr_add"]
    assert full["specs"] == full["development"] == ["validation"]
    assert full["loss_routes"] == parent["loss_routes"]
    assert full["model_configs"]["control"] == full["model_configs"]["mr_add"]
    assert full["seeds"] == [12, 13]
    preflight = protocol(parent, broad, preflight=True)
    assert preflight["steps"] == 4 and preflight["seeds"] == [12]


def test_scheduled_batch_access_preserves_noncyclic_resume_positions():
    selected = []
    pool = SimpleNamespace(select=lambda indices: selected.append(indices) or indices)
    draws = [[4, 1], [2, 9], [7, 6]]
    batches = ScheduledBatches(pool, draws)
    assert len(batches) == 3
    assert batches[2] == [7, 6]
    assert batches[0] == [4, 1]
    assert selected == [draws[2], draws[0]]


def test_validation_inventory_does_not_use_training_coverage_or_allow_training_rows():
    rows, schedule = fixture()
    with pytest.raises(ValueError, match="validation-only"):
        review_inventory(schedule["specs"], rows)
    for row in rows.values():
        row.update(split="validation", codec_id="mp3", channel_role="mono")
    report = review_inventory(schedule["specs"] * 2, rows)
    assert report["clips"] == 10 and report["recordings"] == 5
    assert report["views"] == {"mono": 10}


@pytest.mark.parametrize(
    "kind", ["count", "negative", "duplicate", "corpus", "recording", "validation", "protected"]
)
def test_schedule_rejects_sampling_and_isolation_drift(kind):
    rows, schedule = fixture()
    validate_schedule(schedule, rows, [], 1)
    schedule = copy.deepcopy(schedule)
    probes = []
    if kind == "count":
        schedule["draws"] = []
    elif kind == "negative":
        schedule["draws"][0][0] = -1
    elif kind == "duplicate":
        schedule["draws"][0][0] = 1
    elif kind == "corpus":
        rows["0"]["corpus"] = "speech_library"
    elif kind == "recording":
        rows["0"]["track_id"] = "1"
    elif kind == "validation":
        rows["0"]["split"] = "validation"
    else:
        probes = [schedule["specs"][0]]
    with pytest.raises(ValueError):
        validate_schedule(schedule, rows, probes, 1)
