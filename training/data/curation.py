# SPDX-License-Identifier: Apache-2.0
"""Audit a requested source review and regional coverage without changing generic ingestion."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from data.protocol import sha256_file

REVIEW_SCOPE = "commercial-codec-restoration-and-apache-2.0-weight-distribution"
REGIONS = {"china", "japan", "korea", "europe", "north_america"}
VOCAL_LANGUAGES = {"china": {"zh", "yue"}, "japan": {"ja"}, "korea": {"ko"}}
ADVISORY_METRICS = {"genres", "genre-artists", "regional-vocal-music"}
GENRE_ALIASES = {
    "hip hop": "hiphop",
    "hip-hop": "hiphop",
    "r&b": "rnb",
    "rhythm and blues": "rnb",
    "classical music": "classical",
}


def validate_policy(policy: dict[str, Any]) -> None:
    """Reject malformed targets rather than silently approving incomplete coverage."""
    advisory = policy.get("advisory_metrics", [])
    if (
        not isinstance(advisory, list)
        or any(not isinstance(metric, str) or metric not in ADVISORY_METRICS for metric in advisory)
        or len(set(advisory)) != len(advisory)
    ):
        raise ValueError("advisory_metrics must contain unique supported coverage metrics")
    for key in (
        "minimum_music_recordings",
        "minimum_distinct_artists",
        "minimum_artists_per_region",
        "minimum_genres_per_region",
        "minimum_artists_per_genre",
    ):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    for key in (
        "maximum_artist_fraction",
        "maximum_genre_fraction",
        "maximum_unknown_region_fraction",
    ):
        value = policy.get(key)
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f"fraction in [0, 1] required: {key}")
    for key in ("regions", "genres", "regional_vocal_music"):
        values = policy.get(key, {})
        if not isinstance(values, dict) or (key != "regional_vocal_music" and not values):
            raise ValueError(f"nonempty target mapping required: {key}")
        for name, value in values.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"positive integer target required: {key}.{name}")
            if key != "genres" and name not in REGIONS:
                raise ValueError(f"unsupported repertoire region: {name}")


def evidence_errors(evidence: dict[str, Any], root: Path) -> list[str]:
    """Bind a primary publisher page/version to its retained receipt bytes."""
    if not isinstance(evidence, dict):
        return ["missing_primary_evidence"]
    if urlparse(str(evidence.get("url", ""))).scheme != "https":
        return ["missing_primary_evidence_url"]
    path = root / str(evidence.get("path", ""))
    if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
        return ["missing_or_changed_primary_evidence"]
    return []


def review_errors(entry: dict[str, Any], review: dict[str, Any], root: Path) -> list[str]:
    """A license badge or a review for different audio cannot stand in for this review."""
    if review.get("status") != "accepted" or review.get("scope") != REVIEW_SCOPE:
        return ["source_review_pending_or_incompatible"]
    errors = []
    if review.get("audio_sha256") != entry["audio_sha256"]:
        errors.append("source_review_audio_binding_mismatch")
    for permission in ("commercial_training", "apache_weight_distribution"):
        if review.get(permission) is not True:
            errors.append("unresolved_" + permission)
    if review.get("unresolved_issues") != []:
        errors.append("unresolved_source_rights")
    for field in (
        "composition_basis",
        "recording_basis",
        "performers_and_samples_basis",
        "grant_basis",
        "credit_text",
        "reviewed_at",
        "artist_id",
    ):
        if not isinstance(review.get(field), str) or not review[field].strip():
            errors.append("missing_" + field)
    if "credited_author" in review and (
        not isinstance(review["credited_author"], str) or not review["credited_author"].strip()
    ):
        errors.append("invalid_credited_author")
    evidence = review.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        errors.append("missing_primary_evidence")
    else:
        for receipt in evidence:
            errors.extend(evidence_errors(receipt, root))
    return sorted(set(errors))


def regional_information(review: dict[str, Any], root: Path) -> tuple[str, list[str]]:
    """Nationality, title and instrument names alone do not establish repertoire coverage."""
    region = review.get("region")
    if not isinstance(region, dict):
        return "unknown", []
    allowed_bases = {"publisher-repertoire", "artist-repertoire", "recorded-market"}
    if region.get("basis") not in allowed_bases:
        return "unknown", ["unverified_regional_basis"]
    errors = evidence_errors(region.get("evidence", {}), root)
    if not region.get("explanation"):
        errors.append("missing_regional_explanation")
    if region.get("name") not in REGIONS:
        errors.append("unverified_region_name")
    return str(region.get("name", "unknown")) if not errors else "unknown", errors


def audit_curation(
    entries: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    evidence_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Report actual approved distinct works, creators, regional repertoire and genre gaps."""
    validate_policy(policy)
    approved, held = [], []
    reviewed_works: set[str] = set()
    regions: Counter[str] = Counter()
    genres: Counter[str] = Counter()
    artists: Counter[str] = Counter()
    regional_artists: dict[str, set[str]] = defaultdict(set)
    genre_artists: dict[str, set[str]] = defaultdict(set)
    regional_genres: dict[str, set[str]] = defaultdict(set)
    regional_vocals: Counter[str] = Counter()
    for entry in entries:
        review = reviews.get(entry["id"], {})
        errors = review_errors(entry, review, evidence_root)
        region, regional_errors = regional_information(review, evidence_root)
        if errors:
            held.append({"id": entry["id"], "reasons": errors})
            continue
        if review.get("content_kind", "music") != "music":
            held.append({"id": entry["id"], "reasons": ["not_a_music_recording"]})
            continue
        work = review.get("work_group", entry.get("split_group", entry["id"]))
        if not isinstance(work, str) or not work.strip():
            held.append({"id": entry["id"], "reasons": ["missing_reviewed_work_group"]})
            continue
        if work in reviewed_works:
            held.append({"id": entry["id"], "reasons": ["duplicate_reviewed_work"]})
            continue
        original_genre = entry.get("genre", "unlabeled")
        reviewed_genre = review.get("reviewed_genre", original_genre)
        if not isinstance(reviewed_genre, str) or not reviewed_genre.strip():
            held.append({"id": entry["id"], "reasons": ["invalid_reviewed_genre"]})
            continue
        label = reviewed_genre.strip().casefold()
        genre, artist = GENRE_ALIASES.get(label, label), review["artist_id"]
        if genre not in {"unlabeled", "unknown"} and not review.get("genre_basis"):
            held.append({"id": entry["id"], "reasons": ["unreviewed_genre"]})
            continue
        approved.append(
            {
                **entry,
                "author": review.get("credited_author", entry.get("author", "unknown")),
                "publisher_original_author": entry.get(
                    "publisher_original_author", entry.get("author", "unknown")
                ),
                "genre": genre,
                "publisher_original_genre": entry.get("publisher_original_genre", original_genre),
                "split_group": work,
                "source_review": review,
                "music_region": region,
            }
        )
        reviewed_works.add(work)
        regions[region] += 1
        genres[genre] += 1
        artists[artist] += 1
        regional_artists[region].add(artist)
        genre_artists[genre].add(artist)
        if genre not in {"unlabeled", "unknown"}:
            regional_genres[region].add(genre)
        if (
            review.get("vocal_music") is True
            and review.get("vocal_language") in VOCAL_LANGUAGES.get(region, set())
            and not regional_errors
        ):
            regional_vocals[region] += 1
    gaps: list[dict[str, Any]] = []

    def minimum(metric: str, name: str, actual: int, required: int) -> None:
        if actual < required:
            gaps.append({"metric": metric, "name": name, "actual": actual, "required": required})

    count = len(approved)
    minimum("music", "distinct-recordings", count, policy["minimum_music_recordings"])
    minimum("artists", "distinct-creators", len(artists), policy["minimum_distinct_artists"])
    for region, required in policy["regions"].items():
        minimum("regions", region, regions[region], required)
        minimum(
            "regional-artists",
            region,
            len(regional_artists[region]),
            policy["minimum_artists_per_region"],
        )
        minimum(
            "regional-genres",
            region,
            len(regional_genres[region]),
            policy["minimum_genres_per_region"],
        )
    for genre, required in policy["genres"].items():
        minimum("genres", genre, genres[genre], required)
        minimum(
            "genre-artists", genre, len(genre_artists[genre]), policy["minimum_artists_per_genre"]
        )
    for region, required in policy.get("regional_vocal_music", {}).items():
        minimum("regional-vocal-music", region, regional_vocals[region], required)
    if count:
        for metric, counts, limit in (
            ("artist-dominance", artists, policy["maximum_artist_fraction"]),
            ("genre-dominance", genres, policy["maximum_genre_fraction"]),
            (
                "unknown-region",
                {"unknown": regions["unknown"]},
                policy["maximum_unknown_region_fraction"],
            ),
        ):
            for name, value in counts.items():
                if value / count > limit:
                    gaps.append(
                        {
                            "metric": metric,
                            "name": name,
                            "actual_fraction": value / count,
                            "maximum_fraction": limit,
                        }
                    )
    advisory = set(policy.get("advisory_metrics", []))
    blocking_gaps = [gap for gap in gaps if gap["metric"] not in advisory]
    report = {
        "signal_audited_candidates": len(entries),
        "source_reviewed_music_recordings": count,
        "regions": dict(sorted(regions.items())),
        "genres": dict(sorted(genres.items())),
        "artists": dict(sorted(artists.items())),
        "regional_vocal_music": dict(sorted(regional_vocals.items())),
        "held_for_source_review": held,
        "coverage_gaps": gaps,
        "blocking_gaps": blocking_gaps,
        "advisory_metrics": sorted(advisory),
        "coverage_targets_met": not gaps,
        "ready_for_this_regional_run": not blocking_gaps,
        "scope": REVIEW_SCOPE,
        "not_a_legal_opinion_or_comprehensive_market_coverage_claim": True,
    }
    return approved, report
