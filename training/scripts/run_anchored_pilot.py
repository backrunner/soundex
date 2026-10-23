#!/usr/bin/env python3
"""P45 fixed-budget small-model optimizer pilot, with resumable frozen endpoints.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import fcntl
import shutil
import time
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool, make_broad_schedule
from data.causal_study import digest, load_sources
from evaluation.anchored_review import assessment, review_case
from evaluation.anchored_training import attempt, observations
from evaluation.batch_update_probe import select_roles
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_guarded_transfer import coverage

STEPS = 128
ARMS = {"stored_adam": None, "spectrum_pcm16": 16, "spectrum_pcm20": 20, "spectrum_pcm24": 24}


def verify(hashes: dict) -> None:
    for name, expected in hashes.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"registered input/source drift: {name}")


def prepare(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if (output / "registration.json").exists():
        registration = read(output / "registration.json")
        verify(registration["hashes"])
        return registration
    config = read(args.study / "registration.json")
    calibration = read(args.calibration / "summary.json")
    if calibration["status"] != "complete":
        raise ValueError("completed P44 calibration required")
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    protected = [*config["development"], *config["shifted"]]
    for report in (calibration, read(args.p41 / "summary.json"), read(args.p43 / "summary.json")):
        protected.extend(
            sample["spec"]
            for cases in report["cases"].values()
            for case in cases
            for samples in case["samples"].values()
            for sample in samples
        )
    specs, draws, populations = make_broad_schedule(rows, protected, 2048, 2026102301)
    schedule = {"specs": specs, "draws": draws}
    groups, excluded = [], set()
    for i in range(STEPS):
        selected = select_roles(schedule, i * len(draws) // STEPS, rows, excluded)
        groups.append(selected[:5])
        excluded.update(selected)
    schedule_path = output / "schedule.json"
    registered(schedule, schedule_path)
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(
            part in {".venv", "tests", "outputs", "__pycache__"}
            for part in p.relative_to(root).parts
        )
    ]
    # Non-Python resources can be imported by frozen evaluation helpers.
    sources += sorted((root / "evaluation").glob("*.json"))
    endpoints = read(args.study / "review-registration.json")
    endpoints = {k: v for k, v in endpoints.items() if k.startswith("control-")}
    hashes = dict(config["input_hashes"])
    for path in [
        *sources,
        schedule_path,
        args.study / "registration.json",
        args.study / "review-registration.json",
        args.calibration / "registration.json",
        args.calibration / "summary.json",
        args.p41 / "summary.json",
        args.p43 / "summary.json",
    ]:
        hashes[str(path.resolve())] = sha256_file(path)
    for endpoint in endpoints.values():
        hashes[endpoint["path"]] = endpoint["sha256"]
    verify(hashes)
    for path in sources:
        destination = output / "source" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    samples = lambda indices: [  # noqa: E731
        {"spec": specs[j], "row": rows[specs[j]["row_id"]]}
        for index in indices
        for j in draws[index]
    ]
    cases = [{"samples": {"fit": samples(g[:4]), "guard": samples(g[4:])}} for g in groups]
    registration = {
        "format": "soundex-anchored-pilot-v1",
        "steps": STEPS,
        "arms": ARMS,
        "scope": "128-attempt method pilot, not complete convergence training or capacity comparison; two frozen small P38 starts, all three precision budgets retained, no winning threshold selected",
        "hashes": hashes,
        "endpoints": endpoints,
        "groups": groups,
        "schedule": str(schedule_path),
        "populations": populations,
        "coverage": {role: coverage(cases, role) for role in ("fit", "guard")},
        "calibration": str((args.calibration / "summary.json").resolve()),
        "sampling": "four fit batches (each 4 music/1 speech + identities) and one separate TRAIN guard batch per attempt, common across arms/seeds; whole-source-time support excludes P44 and prior diagnostic/development support; role recordings can recur across attempts",
        "optimizer_control": "original compressed objective, stored Adam state and learning rate 1e-4, clip5, mean four fit gradients; accumulation differs from original P38 single-batch updates, matched to experimental exposure",
        "candidate": "P43 spectrum-led projected gradient, radii .001 then .0003, recomputed warmup. Fit amplitude AND MR must decrease from current state. Complex/dynamics/budget plus guard amplitude/MR bounded by fixed initial metric +0.1% and numerical allowance. Identity relative scores replaced by per-example fixed-anchor absolute constraints; original total is reported but no longer a gate.",
        "pcm_policy": "ALL identity full-band and paired below4k examples in fit/guard: error RMS <= fixed-start error RMS +2^-bits; output displacement RMS AND peak <=2^-bits. All 16/20/24-bit arms run; these are engineering sensitivity controls, not audibility thresholds. No rolling anchor, no reference clipping, no waveform training target replacing spectra.",
        "selection": "no early best/checkpoint/arm selection. Finish every arm and seed, freeze all endpoints, then review P44 reused TRAIN clips. Final raw review cannot qualify native DSP, simultaneous stereo, runtime or listening. No default weight replacement.",
    }
    registered(registration, output / "registration.json")
    return registration


def train(
    label: str,
    endpoint: dict,
    arm: str,
    bits: int | None,
    config: dict,
    pool: LazyPool,
    schedule: dict,
    output: Path,
) -> Path:
    directory = output / label / arm
    directory.mkdir(parents=True, exist_ok=True)
    latest, final = directory / "latest.pth", directory / "endpoint.pth"
    signature = digest({"registration": config, "label": label, "arm": arm})
    if (directory / "complete.json").exists():
        receipt = read(directory / "complete.json")
        if receipt["signature"] != signature or sha256_file(final) != receipt["sha256"]:
            raise ValueError("completed endpoint drift")
        return final
    initial = torch.load(endpoint["path"], map_location="cpu", weights_only=False)
    state = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if state and state["signature"] != signature:
        raise ValueError("resume signature drift")
    model = load_restoration_head(state or initial).eval()
    anchor = load_restoration_head(initial).eval()
    optimizer = None
    if bits is None:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(copy.deepcopy((state or initial)["optimizer"]))
        if any(group["lr"] != 1e-4 for group in optimizer.param_groups):
            raise ValueError("unexpected stored Adam learning rate")
    history = state["history"] if state else []
    start = state["step"] if state else 0
    if len(history) != start or not 0 <= start <= STEPS:
        raise ValueError("resume step/history mismatch")
    for step in range(start, STEPS):
        started = time.monotonic()
        items = [pool.select(schedule["draws"][i]) for i in config["groups"][step]]
        fixed = observations(anchor, items)
        result = attempt(model, items, fixed, bits, optimizer)
        history.append({"step": step + 1, "seconds": time.monotonic() - started, **result})
        state = {
            "model": model.state_dict(),
            "model_config": initial["model_config"],
            "arm": initial["arm"],
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step + 1,
            "signature": signature,
            "initial_sha256": endpoint["sha256"],
            "history": history,
        }
        if (step + 1) % 8 == 0 or step == start:
            atomic_save(state, latest)
            write_json_report(history, directory / "updates.json")
        info = {
            "stage": "training",
            "endpoint": label,
            "arm": arm,
            "step": step + 1,
            "attempts": STEPS,
            "accepted": sum(h["accepted"] for h in history),
            "last_seconds": history[-1]["seconds"],
            "checkpoint_step": read_checkpoint_step(step, start),
        }
        write_json_report(info, output / "progress.json")
        print(info, flush=True)
    atomic_save(state, final)
    registered({"signature": signature, "sha256": sha256_file(final)}, directory / "complete.json")
    return final


def read_checkpoint_step(step: int, start: int) -> int:
    return max(start + 1, (step + 1) // 8 * 8)


def review(config: dict, endpoints: dict, clips: BoundedEnvelopeClips, output: Path) -> None:
    source = read(Path(config["calibration"]))
    original_cases = next(iter(source["cases"].values()))
    samples = [sample for c in original_cases for sample in c["samples"]["calibration"]]
    pool = LazyPool([s["spec"] for s in samples], clips)
    reports = {}
    for label, endpoint in endpoints.items():
        destination = output / "review" / f"{label}.json"
        if destination.exists():
            report = read(destination)
            if report["checkpoint_sha256"] != endpoint["sha256"]:
                raise ValueError("review endpoint drift")
            reports[label] = report
            continue
        model = load_restoration_head(
            torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        ).eval()
        cases = []
        for i, original in enumerate(original_cases):
            item = pool.select(list(range(i * 5, i * 5 + 5)))
            case = {
                "index": original["index"],
                "samples": original["samples"],
                **review_case(model, item),
            }
            if label in source["cases"]:
                expected = source["cases"][label][i]
                if {k: case[k] for k in expected} != expected:
                    raise ValueError("P44 frozen baseline changed")
            cases.append(case)
        report = {"checkpoint_sha256": endpoint["sha256"], "cases": cases}
        write_json_report(report, destination)
        reports[label] = report
        print("review", label, "complete", flush=True)
    verify(config["hashes"])
    write_json_report(assessment(reports, list(config["endpoints"])), output / "assessment.json")
    write_json_report(
        {
            "status": "complete_raw_review",
            "endpoints": endpoints,
            "review_files": {k: sha256_file(output / "review" / f"{k}.json") for k in reports},
            "limits": "P44 reused TRAIN review; native DSP/stereo/listening/runtime remain untested",
        },
        output / "summary.json",
    )
    write_json_report(
        {"stage": "complete_raw_review", "endpoints": len(endpoints)}, output / "progress.json"
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args)
        paths = [Path(p) for p in config["hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            output / "feature-cache",
            capacity=320,
        )
        schedule = read(Path(config["schedule"]))
        pool, endpoints = LazyPool(schedule["specs"], clips), dict(config["endpoints"])
        for label, endpoint in config["endpoints"].items():
            for arm, bits in ARMS.items():
                path = train(label, endpoint, arm, bits, config, pool, schedule, output)
                endpoints[f"{label}-{arm}"] = {"path": str(path), "sha256": sha256_file(path)}
        verify(config["hashes"])
        registered(endpoints, output / "review-registration.json")
        review(config, endpoints, clips, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "calibration", "p41", "p43", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
