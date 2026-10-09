# SPDX-License-Identifier: Apache-2.0
"""Queue fresh training only when distinct, audited native originals reach the target."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.curation import audit_curation
from data.protocol import (
    DatasetPublisher,
    load_manifest,
    load_recipe_from_profile,
    validate_manifest_rows,
)
from scripts.run_native_training import configured_native_corpora, write_progress
from scripts.select_native_catalog import select_recordings, write_jsonl


def read_catalog(path: Path) -> list[dict[str, Any]]:
    """Resolve optional growing receipt files without modifying acquisition state."""
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for entry in entries:
        entry["path"] = str((path.parent / entry["path"]).resolve())
    return entries


def collection_active(directory: Path) -> bool:
    """A stopped/incomplete collector cannot silently satisfy a smaller target."""
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["completed"] + len(progress["errors"]) >= progress["total"]:
            return False
    process = json.loads((directory / "process.json").read_text())
    try:
        os.kill(process["pid"], 0)
    except ProcessLookupError:
        return False
    return True


def verify_supplements(config: Path, processed: Path) -> None:
    """Fail before waiting hours if an enabled supplemental corpus is unavailable."""
    recipe = load_recipe_from_profile(config)
    for corpus in configured_native_corpora(yaml.safe_load(config.read_text())):
        if corpus == "music_library":
            continue
        root = processed / corpus / "processed"
        if not (root / f"{corpus}-{recipe.hash[:16]}").is_dir():
            raise ValueError(f"missing matching prepared supplement: {corpus}")
        with DatasetPublisher(root, corpus, recipe, reuse_existing=True) as publisher:
            rows = load_manifest(publisher.target / "manifest.jsonl")
            validate_manifest_rows(rows, publisher.target, recipe=recipe, audit_files=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-catalog", action="append", type=Path, required=True)
    parser.add_argument("--acquisition-dir", action="append", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--minimum-tracks", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--curation-policy", type=Path)
    parser.add_argument("--source-reviews", type=Path)
    args = parser.parse_args()
    if bool(args.curation_policy) != bool(args.source_reviews):
        parser.error("--curation-policy and --source-reviews must be supplied together")
    if args.minimum_tracks < 1 or not 1 <= args.poll_seconds <= 60:
        parser.error("positive track minimum and poll interval in [1, 60] required")
    job, processed = args.job_dir.resolve(), args.processed_root.resolve()
    job.mkdir(parents=True, exist_ok=True)
    if (job / "progress.json").exists() or (job / "training-run").exists():
        raise ValueError("queue directory already contains a run")
    config = job / "config.yaml"
    config.write_text(yaml.safe_dump(yaml.safe_load(args.config.read_text()), sort_keys=False))
    if args.curation_policy:
        policy_text = args.curation_policy.read_text()
        args.curation_policy = job / "curation-policy.yaml"
        args.curation_policy.write_text(policy_text)
    try:
        write_progress(job, "verify-prepared-supplements")
        verify_supplements(config, processed)
        while True:
            entries = []
            for path in args.base_catalog:
                entries.extend(read_catalog(path))
            for directory in args.acquisition_dir:
                entries.extend(read_catalog(directory / "acquired.jsonl"))
            selected, held = select_recordings(entries)
            coverage = None
            if args.curation_policy:
                rows = [
                    json.loads(line)
                    for line in args.source_reviews.read_text().splitlines()
                    if line.strip()
                ]
                reviews = {row["id"]: row for row in rows}
                if len(reviews) != len(rows):
                    raise ValueError("duplicate source-review recording IDs")
                selected, coverage = audit_curation(
                    selected,
                    reviews,
                    yaml.safe_load(args.curation_policy.read_text()),
                    args.source_reviews.parent.resolve(),
                )
                pending = job / "curation-report.json.tmp"
                pending.write_text(json.dumps(coverage, indent=2) + "\n")
                pending.replace(job / "curation-report.json")
            write_progress(
                job,
                "waiting-for-regional-source-selection"
                if coverage
                else "waiting-for-audited-native-masters",
                distinct_music_recordings=len(selected),
                minimum_required=args.minimum_tracks,
                held_for_review=len(held),
                genres=dict(sorted(Counter(e.get("genre", "unlabeled") for e in selected).items())),
                excludes_speech_and_slakh_from_music_count=True,
                signal_audited_candidates=coverage["signal_audited_candidates"]
                if coverage
                else len(selected),
                regional_coverage=coverage["regions"] if coverage else None,
                coverage_gap_count=len(coverage["coverage_gaps"]) if coverage else 0,
                blocking_gap_count=len(coverage["blocking_gaps"]) if coverage else 0,
            )
            if len(selected) >= args.minimum_tracks and (
                coverage is None or coverage["ready_for_this_regional_run"]
            ):
                break
            if coverage is None and not any(collection_active(d) for d in args.acquisition_dir):
                raise ValueError("acquisition stopped below target; obtain more audited originals")
            time.sleep(args.poll_seconds)
        catalog = job / "music-catalog.jsonl"
        write_jsonl(catalog, selected)
        script = Path(__file__).with_name("run_native_training.py")
        with (job / "pipeline.log").open("w") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--catalog",
                    str(catalog),
                    "--run-dir",
                    str(job / "training-run"),
                    "--processed-root",
                    str(processed),
                    "--config",
                    str(config),
                    "--minimum-tracks",
                    str(args.minimum_tracks),
                    "--project-root",
                    str(args.project_root.resolve()),
                    "--source-commit",
                    args.source_commit,
                    *(
                        [
                            "--curation-policy",
                            str(args.curation_policy.resolve()),
                            "--evidence-root",
                            str(args.source_reviews.parent.resolve()),
                        ]
                        if args.curation_policy
                        else []
                    ),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_progress(
                job,
                "preparation-and-training-pipeline",
                child_pid=child.pid,
                run_progress=str(job / "training-run/progress.json"),
            )
            if child.wait() != 0:
                raise RuntimeError("preparation/training failed; inspect pipeline.log")
        write_progress(job, "complete", run=str(job / "training-run"))
    except Exception as error:
        write_progress(job, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
