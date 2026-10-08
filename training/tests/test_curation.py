# SPDX-License-Identifier: Apache-2.0
"""Source evidence and actual repertoire coverage cannot be replaced by track count."""

from pathlib import Path
from typing import Any

import pytest

from data.curation import REVIEW_SCOPE, audit_curation
from data.protocol import sha256_file


def fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = tmp_path / "publisher.html"
    evidence.write_text("Publisher's original recording, CC BY 4.0, Korean pop repertoire")
    receipt = {
        "url": "https://publisher.example/original",
        "path": evidence.name,
        "sha256": sha256_file(evidence),
    }
    entry = {"id": "song", "audio_sha256": "a" * 64, "genre": "pop"}
    review = {
        "status": "accepted",
        "scope": REVIEW_SCOPE,
        "audio_sha256": entry["audio_sha256"],
        "commercial_training": True,
        "apache_weight_distribution": True,
        "unresolved_issues": [],
        "evidence": [receipt],
        "genre_basis": "publisher says pop",
        "region": {
            "name": "korea",
            "basis": "publisher-repertoire",
            "explanation": "publisher repertoire",
            "evidence": receipt,
        },
        "vocal_music": True,
        "vocal_language": "ko",
    }
    for key in (
        "composition_basis",
        "recording_basis",
        "performers_and_samples_basis",
        "grant_basis",
        "credit_text",
        "reviewed_at",
        "artist_id",
    ):
        review[key] = "retained publisher statement"
    policy = {
        "minimum_music_recordings": 1,
        "minimum_distinct_artists": 1,
        "minimum_artists_per_region": 1,
        "minimum_genres_per_region": 1,
        "minimum_artists_per_genre": 1,
        "maximum_artist_fraction": 1,
        "maximum_genre_fraction": 1,
        "maximum_unknown_region_fraction": 0,
        "regions": {"korea": 1},
        "genres": {"pop": 1},
        "regional_vocal_music": {"korea": 1},
    }
    return entry, review, policy


def test_review_binds_actual_audio_and_retained_evidence(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    _, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert report["ready_for_this_regional_run"]
    for changed in (
        {"audio_sha256": "b" * 64},
        {"commercial_training": False},
        {"unresolved_issues": ["unidentified samples"]},
    ):
        approved, report = audit_curation(
            [entry], {"song": {**review, **changed}}, policy, tmp_path
        )
        assert not approved and not report["ready_for_this_regional_run"]
    (tmp_path / "publisher.html").write_text("different grant")
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert not approved and report["held_for_source_review"]


def test_title_or_nationality_does_not_establish_repertoire(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    review["region"]["basis"] = "artist-nationality"
    _, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert report["regions"] == {"unknown": 1}
    assert not report["ready_for_this_regional_run"]


@pytest.mark.parametrize("changes", [{"vocal_music": False}, {"vocal_language": "en"}])
def test_instrumentals_and_wrong_language_do_not_fill_regional_vocals(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    entry, review, policy = fixture(tmp_path)
    _, report = audit_curation([entry], {"song": {**review, **changes}}, policy, tmp_path)
    assert not report["regional_vocal_music"].get("korea")
    assert not report["ready_for_this_regional_run"]


def test_more_tracks_do_not_replace_artist_and_genre_breadth(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    policy.update(minimum_distinct_artists=3, minimum_genres_per_region=3)
    entries = [{**entry, "id": str(i)} for i in range(20)]
    _, report = audit_curation(entries, {e["id"]: review for e in entries}, policy, tmp_path)
    assert report["source_reviewed_music_recordings"] == 20
    assert not report["ready_for_this_regional_run"]


def test_invalid_policy_cannot_approve_incomplete_region(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    policy["regions"] = {"invented": 1}
    with pytest.raises(ValueError, match="unsupported"):
        audit_curation([entry], {"song": review}, policy, tmp_path)
