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


def test_equivalent_genre_spelling_does_not_create_a_false_coverage_gap(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    entry["genre"] = "Hip Hop"
    review["genre_basis"] = "publisher says Hip Hop"
    policy["genres"] = {"hiphop": 1}
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert report["ready_for_this_regional_run"]
    assert report["genres"] == {"hiphop": 1}
    assert approved[0]["publisher_original_genre"] == "Hip Hop"


def test_reviewed_alternate_arrangements_count_as_one_work(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    review["work_group"] = "composer:shared-melody"
    entries = [{**entry, "id": str(i), "split_group": str(i)} for i in range(5)]
    policy["minimum_music_recordings"] = 5
    approved, report = audit_curation(entries, {e["id"]: review for e in entries}, policy, tmp_path)
    assert len(approved) == 1
    assert approved[0]["split_group"] == "composer:shared-melody"
    assert not report["ready_for_this_regional_run"]
    assert len(report["held_for_source_review"]) == 4


def test_source_review_corrects_credits_and_genre_without_rewriting_receipts(
    tmp_path: Path,
) -> None:
    entry, review, policy = fixture(tmp_path)
    entry.update(author="Recommended artist", genre="unlabeled")
    review.update(credited_author="Actual composer", reviewed_genre="pop")
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert report["ready_for_this_regional_run"]
    assert approved[0]["author"] == "Actual composer"
    assert approved[0]["publisher_original_author"] == "Recommended artist"
    assert entry["author"] == "Recommended artist"
    repeated, _ = audit_curation(approved, {"song": review}, policy, tmp_path)
    assert repeated[0]["publisher_original_author"] == "Recommended artist"
    assert repeated[0]["publisher_original_genre"] == "unlabeled"


def test_licensed_sound_effect_cannot_fill_music_target(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    review["content_kind"] = "instrument-effect"
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert not approved
    assert report["held_for_source_review"][0]["reasons"] == ["not_a_music_recording"]


@pytest.mark.parametrize("genre", [None, "", 123])
def test_malformed_reviewed_genre_is_held(tmp_path: Path, genre: Any) -> None:
    entry, review, policy = fixture(tmp_path)
    review["reviewed_genre"] = genre
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert not approved
    assert report["held_for_source_review"][0]["reasons"] == ["invalid_reviewed_genre"]


def test_pilot_advisories_report_gaps_without_weakening_source_review(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    policy.update(genres={"pop": 3}, advisory_metrics=["genres"])
    approved, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert approved and report["ready_for_this_regional_run"]
    assert not report["coverage_targets_met"]
    assert report["coverage_gaps"][0]["metric"] == "genres"
    assert report["blocking_gaps"] == []
    approved, report = audit_curation(
        [entry], {"song": {**review, "commercial_training": False}}, policy, tmp_path
    )
    assert not approved and not report["ready_for_this_regional_run"]
    assert any(gap["metric"] == "music" for gap in report["blocking_gaps"])


def test_pilot_count_and_region_remain_required(tmp_path: Path) -> None:
    entry, review, policy = fixture(tmp_path)
    policy.update(minimum_music_recordings=2, regions={"korea": 2}, advisory_metrics=["genres"])
    _, report = audit_curation([entry], {"song": review}, policy, tmp_path)
    assert not report["ready_for_this_regional_run"]
    assert {gap["metric"] for gap in report["blocking_gaps"]} == {"music", "regions"}


@pytest.mark.parametrize(
    "metrics", [["music"], ["regions"], ["genres", "genres"], "genres", [None]]
)
def test_advisories_cannot_disable_required_selection_checks(tmp_path: Path, metrics: Any) -> None:
    entry, review, policy = fixture(tmp_path)
    policy["advisory_metrics"] = metrics
    with pytest.raises(ValueError, match="advisory_metrics"):
        audit_curation([entry], {"song": review}, policy, tmp_path)
