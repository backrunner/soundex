#!/usr/bin/env python3
"""P63 fresh small-band/global training, with frozen exposure and review barrier.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, make_broad_schedule
from data.causal_study import load_sources
from evaluation.band_training import (
    ARMS,
    SEEDS,
    STEPS,
    export_model,
    freeze_endpoints,
    fresh,
    load,
    trajectory,
)
from evaluation.longform_review import aliases
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_training import objective_value
from evaluation.trainability_audit import assert_reserved_isolation
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_joint_training import finalize, step
from scripts.run_longform_review import checked_receipt
from scripts.run_native_training import freeze_source
from scripts.summarize_guarded_transfer import coverage


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def isolated_rows(rows, allowed_tracks, universe, reserved):
    """Exclude connected work/decoded aliases before drawing any training crop."""
    blocked = set().union(*(aliases(row) for row in reserved))
    previous = -1
    while previous != len(blocked):
        previous = len(blocked)
        for row in universe:
            keys = aliases(row)
            if keys & blocked:
                blocked.update(keys)
    selected = {
        key: row
        for key, row in rows.items()
        if row["track_id"] in allowed_tracks
        and not aliases(row) & blocked
        and row["split"] == "train"
        and not row["held_out"]
    }
    assert_reserved_isolation(list(selected.values()), universe, reserved)
    return selected


def prepare(prior, output):
    if output.exists():
        raise ValueError("registration requires an unused directory; resume with the run command")
    repository = Path(__file__).resolve().parents[2]
    old = read(prior / "registration.json")
    checked_receipt(prior / "complete.json")
    rows, hashes = load_sources(list(map(Path, old["manifests"])))
    old_schedule = read(Path(old["schedule"]))
    allowed = {rows[s["row_id"]]["track_id"] for s in old_schedule["optimization"]["specs"]}
    review_schedule = read(Path(old["review_schedule"]))
    review_rows = [
        rows[review_schedule["specs"][i]["row_id"]]
        for draw in review_schedule["review"]
        for i in draw
    ]
    catalogs = sorted((repository / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repository / "data/catalogs/current-approved.jsonl"
    reserved = [row for path in catalogs for row in jsonl(path)]
    universe = list(rows.values()) + jsonl(catalog)
    selected = isolated_rows(rows, allowed, universe, reserved + review_rows)
    output.mkdir(parents=True)
    source, commit = freeze_source(repository, output)
    for path in (repository / "training").rglob("*.py"):
        relative = path.relative_to(repository / "training")
        if set(relative.parts) & {".venv", "__pycache__", "outputs"}:
            continue
        frozen = source / relative
        if not frozen.is_file() or sha256_file(frozen) != sha256_file(path):
            raise ValueError("commit all training sources before freezing the experiment")

    def copied(path, name):
        path = Path(path)
        destination = output / "assets" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(path)] = hashes[str(destination)] = sha256_file(path)
        return str(destination)

    detector = copied(old["detector"], "soundex-gain-data")
    binary = copied(repository / "target/release/soundex-context-replay", "soundex-context-replay")
    controls = {k: copied(v, k + ".onnx") for k, v in old["controls"].items()}
    p57_endpoints = read(prior / "endpoints.json")
    for label in sorted(p57_endpoints):
        if label.endswith("-full"):
            graph = prior / "exports" / label / "diagnostic.onnx"
            if sha256_file(graph) != read(graph.parent / "report.json")["onnx_sha256"]:
                raise ValueError("P57 reference graph drift")
            controls["p57-" + label] = copied(graph, "p57-" + label + ".onnx")
    assets = output / "assets"
    for rate in (44100, 48000):
        for channels in (1, 2):
            copied(
                Path(old["assets"]) / f"input-{rate}-{channels}.sxa", f"input-{rate}-{channels}.sxa"
            )
    print("preparing 327680 fresh paired crops on reserved-source-isolated TRAIN", flush=True)
    specs, draws, recordings = make_broad_schedule(selected, [], STEPS * 4, 20261063)
    schedule = output / "schedule.json"
    registered({"optimization": {"specs": specs, "draws": draws}}, schedule)
    exposure = coverage(
        [{"samples": {"registered": [{"spec": s, "row": selected[s["row_id"]]} for s in specs]}}],
        "registered",
    )
    exposure["eligible_recordings"] = recordings
    selected_rows = {s["row_id"]: selected[s["row_id"]] for s in specs}
    for row in selected_rows.values():
        for kind in ("clean", "degraded"):
            hashes[str(Path(row["_root"]) / row[kind + "_path"])] = row[kind + "_checksum"]
    tracks = {row["track_id"]: row for row in selected_rows.values()}
    (output / "training-catalog.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tracks.values())
    )
    p55 = Path(old["p55"])
    checked_receipt(p55 / "prepared.json")
    hashes.update(read(p55 / "prepared.json")["files"])
    for path in [
        prior / "registration.json",
        prior / "complete.json",
        prior / "endpoints.json",
        Path(old["schedule"]),
        Path(old["review_schedule"]),
        p55 / "registration.json",
        p55 / "cases.json",
        p55 / "prepared.json",
        catalog,
        *catalogs,
        schedule,
        output / "training-catalog.jsonl",
    ]:
        hashes[str(path)] = sha256_file(path)
    for path in (output / "source").rglob("*"):
        if path.is_file():
            hashes[str(path)] = sha256_file(path)
    config = {
        "format": "soundex-fresh-band-training-p63-v1",
        "repository": str(repository),
        "prior": str(prior),
        "source_commit": commit,
        "frozen_training": str(source),
        "python": str(Path(sys.executable).absolute()),
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("torch", "numpy", "scipy", "soundfile", "onnx", "onnxruntime", "auraloss")
        },
        "hashes": hashes,
        "manifests": old["manifests"],
        "detector": detector,
        "binary": binary,
        "controls": controls,
        "assets": str(assets),
        "p55": old["p55"],
        "parents": {},
        "review_schedule": old["review_schedule"],
        "review_cases": old["review_cases"],
        "schedule": str(schedule),
        "coverage": exposure,
        "evaluation_catalogs": [str(p) for p in catalogs],
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "parameters": {k: sum(p.numel() for p in fresh(k, SEEDS[0]).parameters()) for k in ARMS},
        "optimizer": "Fresh Adam .9/.999 eps1e-8, clip5, all parameters, four batches/update; 256-update linear lr warmup 1e-5 to 1e-4, then cosine to 1e-5 at update16384. Same indexed schedule for all arms/seeds; no hyperparameter selection or post-step adjustment.",
        "objective": "Unchanged common original-reference direct + raw OLA multires spectra, log1p reduction, equal paired/clean identity means. No learned critic/DSP teacher. Observed magnitude and past-only context; 80-hop detached history, original supervised support from frame89.",
        "exposure": "New recording-balanced random source-time/codec/channel draws from P54 optimization recording role; same schedule for all six trajectories. Existing diagnostic 4:1 music/speech mixture, paired and clean identity views. Not whole-recording stereo training or a new regional balance policy.",
        "budget_rationale": "Fresh representations need optimization beyond the old 4096-step tiny-lr continuation. 16384 updates matches the scale of prior full fresh causal studies and quadruples distinct scheduled crop exposure. This is a fixed substantial comparison, not a convergence guarantee. Report final training trend; no midpoint review or data-driven budget shortening.",
        "question": "Does retaining local band state before global readout improve original-reference restoration? Compare 171137 band parameters against 170859 global-width46 and 480809 global-width96. Width46 controls parameter count, not compute/state/head geometry; no pure isolated architecture-causality claim.",
        "preflight": "Two actual optimizer updates per arm/seed on first eight TRAIN draws, then native export/parity on synthetic PCM. Discard diagnostic states and reset fresh for full training; never select rate/arm using these losses. Stop on nonfinite/contract failure only; retain artifacts.",
        "barrier": "Complete all six fixed endpoints and validate step count, signature and hashes before comparative review. Loss telemetry during training is not selection. Exact persistent-Adam resume; no default-weight promotion.",
        "review": "Frozen original-reference raw/native spectral, fidelity and per-recording outcomes; same legacy/identity-DSP/P57 graphs. All P55 20-source160-case stereo examples are exposed rechecks. Source-equal MR, bands, energy, dynamics, absolute PCM and all regressions; no new independent-test or listening claim. All six endpoint native cost grids before completion.",
        "selection": "No automatic deployment winner. Evidence to continue a band direction requires source-equal paired music native MR improvement over both fresh controls in both seeds, reported HF and clean-identity/dynamics tradeoffs, and relative+absolute counterexamples. A mean gain alone is insufficient. Failure does not establish an architecture ceiling.",
        "runtime": "P62 compute feasible but two paced late completions retained. Fixed256-frame buffering excluded from <5ms compute; sustain128-hop deadline separately. No actual-device qualification.",
    }
    registered(config, output / "registration.json")
    registered(
        {"sha256": sha256_file(output / "registration.json")}, output / "registration.receipt.json"
    )
    write_json_report(
        {
            "stage": "registered",
            "input_verification": "required before first training update",
            "coverage": exposure,
        },
        output / "progress.json",
    )
    print(
        "registered",
        {"parameters": config["parameters"], "coverage": exposure, "hashes": len(hashes)},
        flush=True,
    )


def preflight(config, pool, draws, output):
    receipt = output / "preflight-complete.json"
    if receipt.exists():
        checked_receipt(receipt)
        return
    directory = output / "preflight"
    if directory.exists():
        raise ValueError("incomplete preflight retained; diagnose rather than overwrite")
    directory.mkdir()
    items = [pool.select(d) for d in draws[:8]]
    reports = {}
    for seed in config["seeds"]:
        for kind in config["arms"]:
            model = fresh(kind, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            label = f"{seed}-{kind}"
            with torch.no_grad():
                before = float(objective_value(model, items[0], "common"))
            updates = [step(model, items[offset : offset + 4], optimizer) for offset in (0, 4)]
            graph = export_model(model, directory / label / "export", kind)
            parity = native_parity(
                model, graph, Path(config["binary"]), directory / label / "native", 48000
            )
            reports[label] = {
                "updates": updates,
                "first_batch_before": before,
                "native": parity,
                "qualification": False,
            }
            write_json_report(reports[label], directory / label / "result.json")
            print("preflight", label, "complete; weights discarded", flush=True)
    registered(
        {"files": {str(p): sha256_file(p) for p in directory.rglob("*") if p.is_file()}}, receipt
    )


def run(output):
    torch.set_num_threads(1)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "registration.json"
        if sha256_file(path) != read(output / "registration.receipt.json")["sha256"]:
            raise ValueError("registration drift")
        config = read(path)
        if any(importlib.metadata.version(k) != v for k, v in config["packages"].items()):
            raise ValueError("registered package changed")
        catalogs = sorted(
            str(p)
            for p in (Path(config["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")
        )
        if catalogs != config["evaluation_catalogs"]:
            raise ValueError("evaluation reservations changed; audit before resume")
        write_json_report(
            {"stage": "verifying-inputs", "files": len(config["hashes"])}, output / "progress.json"
        )
        verify(config["hashes"])
        if (output / "complete.json").exists():
            checked_receipt(output / "complete.json")
            write_json_report({"stage": "complete", "qualified": False}, output / "progress.json")
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        pool = LazyPool(
            schedule["specs"],
            BoundedEnvelopeClips(
                rows, Path(config["detector"]), output / "feature-cache", capacity=100
            ),
        )
        write_json_report(
            {"stage": "preflight", "input_hashes_verified": len(config["hashes"])},
            output / "progress.json",
        )
        preflight(config, pool, schedule["draws"], output)
        entries = {}
        for seed in config["seeds"]:
            for kind in config["arms"]:
                entries[f"{seed}-{kind}"] = trajectory(
                    config, pool, schedule["draws"], kind, seed, output
                )
        freeze_endpoints(config, entries, output)
        write_json_report(
            {"stage": "reviewing-frozen-endpoints", "trajectories": len(entries)},
            output / "progress.json",
        )
        finalize(config, entries, output, load_model=load, export_model=export_model, study="p63")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if args.prior is None:
                parser.error("prepare requires --prior")
            prepare(args.prior.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except Exception:
        if args.output.is_dir():
            write_json_report(
                {"status": "failed", "traceback": traceback.format_exc()},
                args.output / f"failure-{time.time_ns()}.json",
            )
        raise
