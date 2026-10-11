"""Identity controls and split-group audit without retagging original paired rows."""

from __future__ import annotations

from collections import Counter

from data.causal_study import PreparedClips

IDENTITY_SUFFIX = ":identity"


def audit_groups(rows):
    groups = {}
    coverage = {}
    for r in rows.values():
        meta = r.get("source_metadata", {})
        group = (
            meta.get("split_group")
            or meta.get("source_review", {}).get("work_group")
            or r["track_id"]
        )
        key = (r["corpus"], group)
        old = groups.setdefault(key, r["split"])
        if old != r["split"]:
            raise ValueError(f"work-group split leakage: {key}")
        coverage.setdefault(r["split"], {})[r["track_id"]] = {
            "corpus": r["corpus"],
            "genre": meta.get("genre", "unlabeled"),
            "region": meta.get("music_region", "unlabeled"),
        }
    return {
        split: {
            key: dict(Counter(v[key] for v in tracks.values()))
            for key in ("corpus", "genre", "region")
        }
        for split, tracks in coverage.items()
    }


class IdentityClips(PreparedClips):
    def __init__(self, rows, detector, cache):
        expanded = dict(rows)
        for key, row in rows.items():
            expanded[key + IDENTITY_SUFFIX] = {
                **row,
                "row_id": key + IDENTITY_SUFFIX,
                "degraded_path": row["clean_path"],
                "degraded_checksum": row["clean_checksum"],
                "codec_id": "identity",
                "quality": "native-lossless",
            }
        super().__init__(expanded, detector, cache)


def identity_specs(specs):
    return [{**s, "row_id": s["row_id"] + IDENTITY_SUFFIX} for s in specs]


def add_identity_controls(schedule, batch_size):
    # One slot every second update; no content-based/validation-based selection.
    return [
        {**s, "row_id": s["row_id"] + IDENTITY_SUFFIX} if i % (2 * batch_size) == 0 else dict(s)
        for i, s in enumerate(schedule)
    ]


def review_specs(rows):
    """One deterministic full paired file per test recording; never used for selection.

    These recordings may have been reviewed by older studies: this is not a fresh
    independent test. Selection is frozen before reading new candidate metrics.
    """
    chosen = {}
    for row in sorted(rows.values(), key=lambda r: r["row_id"]):
        if row["split"] == "test" and row["held_out"]:
            chosen.setdefault(row["track_id"], row["row_id"])
    if not chosen:
        raise ValueError("no test recordings for review")
    return [{"row_id": key, "start": 0, "gain_db": 0.0} for key in chosen.values()]


def training_probe_specs(rows):
    """Deterministic recording-balanced train-domain probe; never used for selection."""
    import numpy as np

    rng = np.random.default_rng(20261014)
    selected = []
    for corpus, count in (("music_library", 32), ("speech_library", 8)):
        groups = {}
        for r in sorted(rows.values(), key=lambda r: r["row_id"]):
            if r["split"] == "train" and not r["held_out"] and r["corpus"] == corpus:
                groups.setdefault(r["track_id"], []).append(r)
        tracks = sorted(groups)
        for index in rng.permutation(len(tracks))[:count]:
            options = groups[tracks[index]]
            row = options[int(rng.integers(len(options)))]
            selected.append(
                {
                    "row_id": row["row_id"],
                    "start": max(0, (row["length_samples"] - 32768) // 2),
                    "gain_db": 0.0,
                }
            )
    return selected
