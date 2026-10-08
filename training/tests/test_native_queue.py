# SPDX-License-Identifier: Apache-2.0
"""A queued run stops below target rather than training with a smaller collection."""

import json
from pathlib import Path

import pytest
import yaml

from data.protocol import DataProtocolError, load_recipe_from_profile
from scripts.wait_for_native_training import main, verify_supplements


def test_matching_recipe_does_not_allow_an_invalid_prepared_manifest(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs/diverse_lossless.yaml"
    recipe = load_recipe_from_profile(config)
    version = tmp_path / "slakh2100/processed" / f"slakh2100-{recipe.hash[:16]}"
    version.mkdir(parents=True)
    (version / "recipe.json").write_text(json.dumps({"recipe_hash": recipe.hash}))
    (version / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(DataProtocolError, match="required field"):
        verify_supplements(config, tmp_path)


def test_completed_collection_below_target_never_starts_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "base.jsonl"
    catalog.write_text("")
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "progress.json").write_text(
        json.dumps({"completed": 0, "total": 1, "errors": [{"id": "missing"}]})
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"data": {}}))
    job = tmp_path / "job"
    monkeypatch.setattr("scripts.wait_for_native_training.verify_supplements", lambda *_: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "wait_for_native_training",
            "--base-catalog",
            str(catalog),
            "--acquisition-dir",
            str(collection),
            "--job-dir",
            str(job),
            "--processed-root",
            str(tmp_path / "processed"),
            "--config",
            str(profile),
            "--project-root",
            str(tmp_path),
            "--source-commit",
            "HEAD",
            "--minimum-tracks",
            "1000",
        ],
    )
    with pytest.raises(ValueError, match="below target"):
        main()
    progress = json.loads((job / "progress.json").read_text())
    assert progress["stage"] == "failed"
    assert not (job / "music-catalog.jsonl").exists()
    assert not (job / "training-run").exists()
