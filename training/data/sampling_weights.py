# SPDX-License-Identifier: Apache-2.0
"""Balance recordings independently of mono/stereo role counts or rejected crops."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def recording_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Give each retained recording equal mass while preserving its codec/role weights."""
    totals: dict[str, float] = defaultdict(float)
    raw = [float(row.get("sampling_weight", 1.0)) for row in rows]
    for row, weight in zip(rows, raw, strict=True):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("manifest sampling weights must be finite and positive")
        totals[str(row["track_id"])] += weight
    return [weight / totals[str(row["track_id"])] for row, weight in zip(rows, raw, strict=True)]


def speech_mix_ratios(
    music_rows: Sequence[Mapping[str, Any]], *, synthetic_ratio: float = 0.0
) -> tuple[dict[str, float], dict[str, Any]]:
    """Match speech's draw mass to the smallest labeled real-music genre's mass.

    Count unique training recordings, not role files or codec variants. Unlabeled
    recordings are reported separately and cannot be substituted for a genre.
    Synthetic mass is opt-in; disabled sources are omitted from the final ratios.
    """
    if not 0 <= synthetic_ratio < 1:
        raise ValueError("synthetic_ratio must be in [0, 1)")
    tracks: dict[str, str] = {}
    for row in music_rows:
        if row["split"] != "train":
            continue
        genre = str(row.get("source_metadata", {}).get("genre", "unlabeled")).strip()
        genre = genre or "unlabeled"
        track = str(row["track_id"])
        if track in tracks and tracks[track] != genre:
            raise ValueError(f"inconsistent genre for {track}")
        tracks[track] = genre
    counts: dict[str, int] = defaultdict(int)
    for genre in tracks.values():
        counts[genre] += 1
    labeled = {genre: count for genre, count in counts.items() if genre != "unlabeled"}
    if not labeled:
        raise ValueError("speech ratio requires labeled real music in the training split")
    smallest = min(labeled.values())
    music_mass = 1 - synthetic_ratio
    speech_mass = music_mass * smallest / len(tracks)
    total = 1 + speech_mass
    ratios = {
        "music_library": music_mass / total,
        "speech_library": speech_mass / total,
    }
    if synthetic_ratio:
        ratios["slakh2100"] = synthetic_ratio / total
    report = {
        "training_recordings_by_genre": dict(sorted(counts.items())),
        "smallest_genres": sorted(g for g, count in labeled.items() if count == smallest),
        "speech_ratio": ratios["speech_library"],
        "music_genre_draw_ratios": {
            genre: ratios["music_library"] * count / len(tracks)
            for genre, count in sorted(counts.items())
        },
        "basis": "unique training recordings; codec and channel roles do not multiply mass",
    }
    return ratios, report


def main() -> None:
    """Print acquired genre counts and suggested source ratios for a run config."""
    import argparse
    import json
    from pathlib import Path

    from data.protocol import load_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    args = parser.parse_args()
    ratios, report = speech_mix_ratios(
        load_manifest(args.manifest), synthetic_ratio=args.synthetic_ratio
    )
    print(json.dumps({"train_ratios": ratios, "report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
