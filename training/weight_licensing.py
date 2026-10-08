# SPDX-License-Identifier: Apache-2.0
"""Bind the official Apache weight route to actual sources and a maintainer review."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from checkpoint import load_checkpoint
from data.protocol import sha256_file

_REQUIRED_USES = frozenset(
    {"training", "application_integration", "redistribution", "modification", "commercial_use"}
)
_HASH = re.compile(r"[0-9a-f]{64}")


def check_license_review(
    fields: Mapping[str, str], metadata: Mapping[str, str], artifact_hash: str, card: Path
) -> str:
    """Require a hash-bound human rights review, not a tier label or free-text claim."""
    if fields.get("Weight license") != "Apache-2.0" or fields.get("License file") != "LICENSE":
        raise ValueError("official model card must specify Apache-2.0 and LICENSE")
    declaration = fields.get("Data rights review / SHA-256", "")
    # Paths can contain spaces; the final token is the review checksum.
    parts = declaration.rsplit(None, 1)
    if len(parts) != 2 or not _HASH.fullmatch(parts[1]):
        raise ValueError("data rights review path and SHA-256 are required")
    review_path = (card.parent / parts[0]).resolve()
    if sha256_file(review_path) != parts[1]:
        raise ValueError("data rights review SHA-256 mismatch")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        not isinstance(review, dict)
        or review.get("schema_version") != 1
        or isinstance(review.get("schema_version"), bool)
    ):
        raise ValueError("unsupported data rights review schema")
    if review.get("license") != "Apache-2.0" or review.get("rights_confirmed") is not True:
        raise ValueError("data rights review does not approve an Apache-2.0 release")
    if (
        not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or "TODO" in review["reviewer"]
    ):
        raise ValueError("data rights review requires a named reviewer")
    date.fromisoformat(review.get("reviewed_at", ""))
    for key in ("initialization_and_resume_review", "source_grants_and_credits"):
        if not isinstance(review.get(key), str) or not review[key].strip() or "TODO" in review[key]:
            raise ValueError(f"data rights review requires completed {key}")
    uses = review.get("permitted_uses")
    if (
        not isinstance(uses, list)
        or not all(isinstance(value, str) for value in uses)
        or not _REQUIRED_USES.issubset(uses)
    ):
        raise ValueError("data rights review does not cover application use and redistribution")
    if (
        review.get("artifact_sha256") != artifact_hash
        or review.get("manifest_set_sha256") != metadata["soundex.manifest_set_sha256"]
    ):
        raise ValueError("data rights review artifact or manifest binding mismatch")
    source = review.get("source_checkpoint")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != metadata["soundex.source_checkpoint_sha256"]
        or not isinstance(source.get("path"), str)
    ):
        raise ValueError("data rights review checkpoint binding mismatch")
    checkpoint_path = (review_path.parent / source["path"]).resolve()
    if sha256_file(checkpoint_path) != source["sha256"]:
        raise ValueError("data rights review source checkpoint checksum mismatch")
    checkpoint = load_checkpoint(checkpoint_path)
    for record in checkpoint["data"]["manifests"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("checkpoint training manifest hash mismatch")
    if (
        checkpoint["data"]["manifest_set_sha256"] != review["manifest_set_sha256"]
        or checkpoint["hashes"]["resolved_config_sha256"]
        != metadata["soundex.resolved_config_sha256"]
    ):
        raise ValueError("source checkpoint provenance does not match released artifact")
    expected = [
        {"corpus": row["corpus"], "sha256": row["sha256"]}
        for row in checkpoint["data"]["manifests"]
    ]
    if review.get("training_manifests") != expected:
        raise ValueError("data rights review does not list the checkpoint training manifests")
    return parts[1]
