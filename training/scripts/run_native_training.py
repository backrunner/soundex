# SPDX-License-Identifier: Apache-2.0
"""Freeze an audited music catalog, reuse matching supplements and start fresh training."""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import yaml


def write_progress(run: Path, stage: str, **fields: Any) -> None:
    value = {
        "stage": stage,
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
        **fields,
    }
    pending = run / "progress.json.tmp"
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(run / "progress.json")
    print(json.dumps(value), flush=True)


def invoke(run: Path, stage: str, script: Path, *args: str) -> None:
    env = os.environ.copy()
    for key in (
        "PATHS_MANIFEST",
        "DATA_ROOT",
        "DATASET_MANIFEST",
        "SOUNDEX_DATA_MANIFEST",
        "MUSDB18_HQ_PATH",
        "SLAKH2100_PATH",
        "MEDLEYDB_PATH",
        "BABYSLAKH_PATH",
        "MUSIC_LIBRARY_PATH",
        "SPEECH_LIBRARY_PATH",
    ):
        env.pop(key, None)
    with (run / f"{stage}.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-u", str(script), *args],
            cwd=run,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        write_progress(run, stage, child_pid=child.pid, log=str(run / f"{stage}.log"))
        if child.wait() != 0:
            raise RuntimeError(f"{stage} failed; inspect {stage}.log")


def freeze_source(project: Path, run: Path, source_commit: str = "HEAD") -> tuple[Path, str]:
    """Use a committed source snapshot rather than changing code during a long run."""
    commit = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "--verify", source_commit + "^{commit}"],
        text=True,
    ).strip()
    archive = subprocess.check_output(["git", "-C", str(project), "archive", commit])
    source = run / "source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    return source / "training", commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--minimum-tracks", type=int, default=1000)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--source-commit", default="HEAD")
    args = parser.parse_args()
    project = (args.project_root or Path(__file__).resolve().parents[2]).resolve()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    try:
        music = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
        if len(music) < args.minimum_tracks:
            raise ValueError("audited music count is below the requested target")
        pcm_seen, works_seen = set(), set()
        for entry in music:
            quality = entry["source_quality"]
            if quality["review_flags"] or quality["audio_sha256"] != entry["audio_sha256"]:
                raise ValueError("catalog contains unaudited or unresolved source quality")
            if quality["sample_rate"] < 44100 or quality["duration_seconds"] < 30:
                raise ValueError("catalog contains a short or low-rate music reference")
            pcm, work = quality["decoded_pcm_sha256"], entry["split_group"]
            if pcm in pcm_seen or work in works_seen:
                raise ValueError("catalog counts duplicate recordings or works")
            pcm_seen.add(pcm)
            works_seen.add(work)
            entry["path"] = str((args.catalog.parent / entry["path"]).resolve())
        training, commit = freeze_source(project, run, args.source_commit)
        sys.path.insert(0, str(training))
        from data.catalog import load_catalog
        from data.protocol import (
            DatasetPublisher,
            load_manifest,
            load_recipe_from_profile,
            sha256_file,
            validate_manifest_rows,
        )
        from data.sampling_weights import speech_mix_ratios

        catalog = run / "music-catalog.jsonl"
        catalog.write_text("".join(json.dumps(e) + "\n" for e in music))
        bound, catalog_hash = load_catalog(catalog)
        config = yaml.safe_load(args.config.read_text())
        processed = args.processed_root.resolve()
        if len(bound) < args.minimum_tracks:
            raise ValueError(
                "bound music sources are below the requested target after deduplication"
            )
        for corpus in ("music_library", "slakh2100", "speech_library"):
            config["data"][corpus + "_path"] = str(processed / corpus)
        profile = run / "config.yaml"
        profile.write_text(yaml.safe_dump(config, sort_keys=False))
        recipe = load_recipe_from_profile(profile)
        write_progress(run, "verify-prepared-supplements", music_recordings=len(bound))
        for corpus in ("slakh2100", "speech_library"):
            root = processed / corpus / "processed"
            version = root / f"{corpus}-{recipe.hash[:16]}"
            if not version.is_dir():
                raise ValueError(f"missing matching prepared supplement: {corpus}")
            with DatasetPublisher(root, corpus, recipe, reuse_existing=True) as publisher:
                rows = load_manifest(publisher.target / "manifest.jsonl")
                validate_manifest_rows(rows, publisher.target, recipe=recipe, audit_files=True)
        receipt = {
            "code_commit": commit,
            "music_catalog_sha256": sha256_file(catalog),
            "bound_catalog_sha256": catalog_hash,
            "music_recordings": len(bound),
            "initialization": "fresh random weights; no legacy checkpoint/teacher",
            "recipe_hash": recipe.hash,
            "supplements": {c: str(processed / c) for c in ("slakh2100", "speech_library")},
        }
        (run / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        invoke(
            run,
            "preprocess-music",
            training / "data/preprocess_library.py",
            "--catalog",
            str(catalog),
            "--corpus",
            "music_library",
            "--output-dir",
            str(processed / "music_library/processed"),
            "--config",
            str(profile),
            "--reuse-existing",
        )
        music_manifest = (
            processed
            / "music_library/processed"
            / (f"music_library-{recipe.hash[:16]}/manifest.jsonl")
        )
        ratios, report = speech_mix_ratios(load_manifest(music_manifest), synthetic_ratio=0.2)
        config["data"]["sampling"]["train_ratios"] = ratios
        config["data"]["sampling"]["val_ratios"] = ratios
        profile.write_text(yaml.safe_dump(config, sort_keys=False))
        (run / "genre-sampling-report.json").write_text(
            json.dumps({"ratios": ratios, **report}, indent=2) + "\n"
        )
        invoke(run, "training", training / "train.py", "--config", str(profile))
        write_progress(run, "complete", checkpoint=str(run / "checkpoints/final-resume.pth"))
    except Exception as error:
        write_progress(run, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
