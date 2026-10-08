# SPDX-License-Identifier: Apache-2.0
"""Join source reviews with signal-audited masters and publish only a complete regional run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.curation import audit_curation
from data.protocol import sha256_file
from scripts.select_native_catalog import select_recordings, write_jsonl
from scripts.wait_for_native_training import read_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="append", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("published catalogs are immutable; choose a new output path")
    entries = [entry for path in args.catalog for entry in read_catalog(path)]
    selected, signal_holds = select_recordings(entries)
    review_rows = [
        json.loads(line) for line in args.reviews.read_text().splitlines() if line.strip()
    ]
    reviews = {row["id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("duplicate source-review recording IDs")
    policy = yaml.safe_load(args.policy.read_text())
    approved, report = audit_curation(selected, reviews, policy, args.reviews.parent)
    report["signal_quality_or_duplicate_holds"] = signal_holds
    report["policy"] = policy
    report["reviews_sha256"] = sha256_file(args.reviews)
    report["policy_sha256"] = sha256_file(args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["ready_for_this_regional_run"]:
        raise SystemExit(
            "Regional/source review incomplete; report written, no training catalog published"
        )
    write_jsonl(args.output, approved)
    report["published_catalog_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Approved {len(approved)} distinct reviewed regional music recordings")


if __name__ == "__main__":
    main()
