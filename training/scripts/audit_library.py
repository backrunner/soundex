# SPDX-License-Identifier: Apache-2.0
"""Audit source audio; publish accepted entries and a separate review/error report."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.source_quality import inspect_master


def audit_entry(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Return an auditable result, including failures, without silently dropping a file."""
    try:
        quality = inspect_master(root / entry["path"], entry.get("audio_sha256"))
        return {"id": entry["id"], "quality": quality}
    except Exception as error:
        return {"id": entry["id"], "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-duration", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.min_duration < 0:
        parser.error("invalid workers or minimum duration")
    entries = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    identities = [e["id"] for e in entries]
    if len(set(identities)) != len(entries):
        raise ValueError("duplicate recording IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted, reviews, errors = [], [], []
    pcm_seen: set[str] = set()
    works_seen: set[str] = set()
    measurements = []
    with ThreadPoolExecutor(args.workers) as pool:
        results = pool.map(lambda e: audit_entry(e, args.catalog.parent), entries)
        for entry, result in zip(entries, results, strict=True):
            measurements.append(result)
            if "error" in result:
                errors.append(result)
                continue
            quality = result["quality"]
            reasons = list(quality["review_flags"])
            if quality["duration_seconds"] < args.min_duration:
                reasons.append("short_cue")
            if quality["sample_rate"] < 44100:
                reasons.append("low_source_sample_rate")
            work = entry.get("split_group", quality["decoded_pcm_sha256"])
            if quality["decoded_pcm_sha256"] in pcm_seen or work in works_seen:
                reasons.append("duplicate_pcm_or_work")
            if reasons:
                reviews.append({"id": entry["id"], "reasons": reasons})
                continue
            pcm_seen.add(quality["decoded_pcm_sha256"])
            works_seen.add(work)
            accepted.append({**entry, "source_quality": quality})
            print(f"Accepted {len(accepted)}: {entry['id']}", flush=True)
    report = {
        "input_recordings": len(entries),
        "accepted_recordings": len(accepted),
        "accepted_hours": sum(e["source_quality"]["duration_seconds"] for e in accepted) / 3600,
        "review_required": reviews,
        "errors": errors,
        "spectral_metrics_are_not_lossy_history_proof": True,
        "measurements": measurements,
    }
    (args.output_dir / "quality-report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Resolve paths against the input catalog before writing to a different directory.
    for entry in accepted:
        entry["path"] = str((args.catalog.parent / entry["path"]).resolve())
    (args.output_dir / "accepted.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in accepted)
    )
    print(f"Accepted {len(accepted)}; review {len(reviews)}; errors {len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
