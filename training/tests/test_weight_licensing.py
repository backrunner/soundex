# SPDX-License-Identifier: Apache-2.0
"""Check publication rights review bindings without gating training ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import weight_licensing
from configuration import load_config
from data.dataset import DatasetSourceSpec
from data.protocol import sha256_file
from weight_licensing import check_license_review

CONFIG = Path(__file__).parents[1] / "configs/default.yaml"


def approved_manifest(tmp_path: Path) -> tuple[dict, DatasetSourceSpec]:
    config = load_config(CONFIG)
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"corpus": "slakh2100"}) + "\n")
    return config, DatasetSourceSpec("slakh2100", "Slakh", path, "synthetic")


def test_model_license_review_is_bound_and_requires_application_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source = approved_manifest(tmp_path)
    checkpoint_path = tmp_path / "source.pth"
    checkpoint_path.write_bytes(b"checkpoint-loader-is-tested-separately")
    checkpoint_hash = sha256_file(checkpoint_path)
    manifests = [{"corpus": "slakh2100", "sha256": sha256_file(source.manifest_path)}]
    checkpoint = {
        "resolved_config": config,
        "data": {
            "manifest_set_sha256": "b" * 64,
            "manifests": [{**manifests[0], "path": str(source.manifest_path)}],
        },
        "hashes": {"resolved_config_sha256": "c" * 64},
    }
    monkeypatch.setattr(weight_licensing, "load_checkpoint", lambda _path: checkpoint)
    review = {
        "schema_version": 1,
        "license": "Apache-2.0",
        "rights_confirmed": True,
        "reviewer": "Test maintainer",
        "reviewed_at": "2026-10-08",
        "permitted_uses": [
            "training",
            "application_integration",
            "redistribution",
            "modification",
            "commercial_use",
        ],
        "artifact_sha256": "a" * 64,
        "manifest_set_sha256": "b" * 64,
        "source_checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_hash},
        "training_manifests": manifests,
        "initialization_and_resume_review": "Fresh random initialization, no parent or teacher.",
        "source_grants_and_credits": "Verified publisher grant and source attribution.",
    }
    review_path = tmp_path / "rights.json"
    fields = {"Weight license": "Apache-2.0", "License file": "LICENSE"}
    metadata = {
        "soundex.source_checkpoint_sha256": checkpoint_hash,
        "soundex.manifest_set_sha256": "b" * 64,
        "soundex.resolved_config_sha256": "c" * 64,
    }

    def check(value: dict) -> str:
        review_path.write_text(json.dumps(value))
        fields["Data rights review / SHA-256"] = f"{review_path.name} {sha256_file(review_path)}"
        return check_license_review(fields, metadata, "a" * 64, tmp_path / "card.md")

    assert len(check(review)) == 64
    for patch, message in (
        ({"rights_confirmed": False}, "does not approve"),
        ({"artifact_sha256": "0" * 64}, "binding mismatch"),
        ({"permitted_uses": ["training"]}, "application use"),
        ({"training_manifests": []}, "training manifests"),
        ({"initialization_and_resume_review": "TODO"}, "requires completed"),
    ):
        with pytest.raises(ValueError, match=message):
            check({**review, **patch})
    check(review)
    checkpoint_path.write_bytes(b"substituted-checkpoint")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        check_license_review(fields, metadata, "a" * 64, tmp_path / "card.md")
