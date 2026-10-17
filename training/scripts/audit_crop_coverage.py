#!/usr/bin/env python3
"""Count actual supervised time and sampled diversity, not catalog duration."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from data.causal_study import load_sources, source_origin
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import read


def union_length(intervals):
    end, total = None, 0
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("reversed interval")
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def coverage(specs, rows):
    chosen = {s["row_id"]: rows[s["row_id"]] for s in specs}
    if any(r["split"] != "train" or r["held_out"] for r in chosen.values()):
        raise ValueError("training bank includes held-out audio")
    intervals = defaultdict(list)
    source_intervals = defaultdict(list)
    for s in specs:
        row = rows[s["row_id"]]
        origin = source_origin(row) + s["start"]
        key = (row["source_checksum"], row["channel_role"], row["sample_rate"])
        intervals[key].append((origin + 89 * 128, origin + 32768 - 128))
        source_intervals[row["source_checksum"]].append(
            (
                Fraction(origin + 89 * 128, row["sample_rate"]),
                Fraction(origin + 32768 - 128, row["sample_rate"]),
            )
        )
    return {
        "crops": len(specs),
        "paired_rows": len(chosen),
        "recordings": len({r["track_id"] for r in chosen.values()}),
        "codecs": dict(Counter(r["codec_id"] for r in chosen.values())),
        "channel_views": dict(Counter(r["channel_role"] for r in chosen.values())),
        "unique_supervised_channel_seconds": sum(
            union_length(v) / k[2] for k, v in intervals.items()
        ),
        "unique_supervised_source_seconds": float(
            sum(union_length(v) for v in source_intervals.values())
        ),
        "selected_row_seconds": sum(
            r["length_samples"] / r["sample_rate"] for r in chosen.values()
        ),
        "note": "union by source, channel view and native rate; paired and identity share reference time; mid/side count as separate views, not extra independent songs",
    }


def run(args):
    config = read(args.study / "registration.json")
    manifests = [Path(p) for p in config["input_hashes"] if Path(p).name == "manifest.jsonl"]
    for p in manifests:
        if sha256_file(p) != config["input_hashes"][str(p)]:
            raise ValueError("manifest drift")
    rows, _ = load_sources(manifests)
    report = {}
    for corpus in ("music_library", "speech_library"):
        available = [
            r
            for r in rows.values()
            if r["corpus"] == corpus and r["split"] == "train" and not r["held_out"]
        ]
        report[corpus] = {
            "bank": coverage(
                [s for s in config["bank"] if rows[s["row_id"]]["corpus"] == corpus], rows
            ),
            "available_training_pairs": len(available),
            "available_training_recordings": len({r["track_id"] for r in available}),
            "available_codec_ids": sorted({r["codec_id"] for r in available}),
        }
    if args.output.exists():
        raise ValueError("do not overwrite coverage evidence")
    write_json_report(
        {
            "scope": "P24 finite diagnostic-bank coverage; no training or source changes",
            "registration_sha256": sha256_file(args.study / "registration.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "corpora": report,
        },
        args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
