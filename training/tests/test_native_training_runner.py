# SPDX-License-Identifier: Apache-2.0
"""A fresh native run cannot start with unresolved or duplicate music references."""

import json
from pathlib import Path

import pytest

from scripts.run_native_training import main


@pytest.mark.parametrize(
    ("flags", "minimum", "error"),
    [(["large_dc_offset"], 1, "unresolved"), ([], 2, "below")],
)
def test_quality_and_count_fail_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], minimum: int, error: str
) -> None:
    entry = {
        "id": "recording",
        "path": "original.wav",
        "audio_sha256": "a" * 64,
        "split_group": "composer:work",
        "source_quality": {
            "audio_sha256": "a" * 64,
            "decoded_pcm_sha256": "b" * 64,
            "review_flags": flags,
            "sample_rate": 48000,
            "duration_seconds": 60,
        },
    }
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(entry) + "\n")
    run = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_native_training",
            "--catalog",
            str(catalog),
            "--run-dir",
            str(run),
            "--processed-root",
            str(tmp_path / "processed"),
            "--config",
            str(tmp_path / "profile"),
            "--minimum-tracks",
            str(minimum),
        ],
    )
    with pytest.raises(ValueError, match=error):
        main()
    assert json.loads((run / "progress.json").read_text())["stage"] == "failed"
    assert not (run / "source").exists()
