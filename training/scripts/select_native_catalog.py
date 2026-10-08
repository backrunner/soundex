# SPDX-License-Identifier: Apache-2.0
"""Combine audited music receipts into a distinct selection and an optional portable index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def select_recordings(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold quality flags and duplicates for review; never count codec variants as songs."""
    selected, held = [], []
    pcm_seen: set[str] = set()
    work_seen: set[str] = set()
    for entry in sorted(entries, key=lambda e: e["id"]):
        quality = entry["source_quality"]
        reasons = list(quality["review_flags"])
        if quality["audio_sha256"] != entry["audio_sha256"]:
            reasons.append("quality_receipt_checksum_mismatch")
        if quality["duration_seconds"] < 30 or quality["sample_rate"] < 44100:
            reasons.append("short_or_low_rate_reference")
        pcm, work = quality["decoded_pcm_sha256"], entry["split_group"]
        if pcm in pcm_seen or work in work_seen:
            reasons.append("duplicate_pcm_or_work")
        if reasons:
            held.append({"id": entry["id"], "reasons": reasons})
            continue
        pcm_seen.add(pcm)
        work_seen.add(work)
        selected.append(entry)
    return selected, held


def portable_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep source attribution and measurements without local paths or private evidence."""
    fields = (
        "id",
        "author",
        "recording_title",
        "album",
        "genre",
        "publisher_genre_tags",
        "genre_basis",
        "source_url",
        "source_page",
        "license_url",
        "audio_sha256",
        "decoded_pcm_sha256",
        "publisher_original_md5",
        "source_bytes",
        "split_group",
        "lossless_origin",
        "source_quality",
    )
    result = {key: entry[key] for key in fields if key in entry}
    quality = entry["source_quality"]
    result.update(
        path="raw/music/" + Path(entry["path"]).name,
        native_sample_rate=quality["sample_rate"],
        native_channels=quality["channels"],
        duration_seconds=quality["duration_seconds"],
    )
    if "publisher_original_md5" not in result and entry.get("file", {}).get("md5"):
        result["publisher_original_md5"] = entry["file"]["md5"]
    if "source_bytes" not in result:
        result["source_bytes"] = Path(entry["path"]).stat().st_size
    return result


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable-index", type=Path)
    parser.add_argument("--minimum-tracks", type=int, default=1000)
    args = parser.parse_args()
    entries = []
    for catalog in args.catalog:
        for line in catalog.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            entry["path"] = str((catalog.parent / entry["path"]).resolve())
            entries.append(entry)
    selected, held = select_recordings(entries)
    report = {
        "input_receipts": len(entries),
        "distinct_music_tracks": len(selected),
        "minimum_requested": args.minimum_tracks,
        "meets_requested_minimum": len(selected) >= args.minimum_tracks,
        "music_hours": sum(e["source_quality"]["duration_seconds"] for e in selected) / 3600,
        "genres": dict(sorted(Counter(e.get("genre", "unlabeled") for e in selected).items())),
        "held_for_review": held,
        "excludes_speech_and_slakh_from_music_count": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["meets_requested_minimum"]:
        raise SystemExit(f"Only {len(selected)} distinct audited tracks; acquire more originals")
    write_jsonl(args.output, selected)
    if args.portable_index:
        write_jsonl(args.portable_index, [portable_entry(e) for e in selected])
    print(f"Selected {len(selected)} distinct audited tracks; held {len(held)} for review")


if __name__ == "__main__":
    main()
