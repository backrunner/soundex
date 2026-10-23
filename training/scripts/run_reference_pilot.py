#!/usr/bin/env python3
"""P46 resumable reference-preservation pilot with a complete endpoint barrier.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import fcntl
import shutil
import time
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.anchored_review import compare
from evaluation.anchored_training import attempt as adam_attempt
from evaluation.anchored_training import observations
from evaluation.reference_pilot_setup import ARMS, STEPS, prepare
from evaluation.reference_review import review_reference
from evaluation.reference_training import attempt
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head
from scripts.run_anchored_pilot import read_checkpoint_step, review, verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered


def preflight(config: dict, pool: LazyPool, schedule: dict, output: Path) -> None:
    destination = output / "preflight.json"
    signature = digest(config)
    if destination.exists():
        report = read(destination)
        if report["signature"] != signature:
            raise ValueError("preflight signature changed")
        if report["status"] == "complete":
            return
        raise ValueError("incomplete preflight retained; inspect before retrying")
    reports = {}
    items = [pool.select(schedule["draws"][i]) for i in config["groups"][0]]
    for label, endpoint in config["endpoints"].items():
        initial = torch.load(endpoint["path"], map_location="cpu", weights_only=False)
        anchor = observations(load_restoration_head(initial).eval(), items)
        model = load_restoration_head(initial).eval()
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
        control = adam_attempt(model, items, anchor, None, optimizer)
        old = read(Path(config["prior"]) / label / "stored_adam/updates.json")[0]
        if control != {k: v for k, v in old.items() if k not in ("step", "seconds")}:
            raise ValueError("P45 first Adam update does not reproduce exactly")
        reports[label] = {"adam_first_update_exact": True, "arms": {}}
        for arm, settings in ARMS.items():
            model = load_restoration_head(initial).eval()
            started = time.monotonic()
            result = attempt(model, items, anchor, **settings)
            reports[label]["arms"][arm] = {"seconds": time.monotonic() - started, "result": result}
            write_json_report(
                {"status": "running", "signature": signature, "reports": reports}, destination
            )
            print("preflight", label, arm, "complete", flush=True)
    verify(config["hashes"])
    write_json_report(
        {
            "status": "complete",
            "signature": signature,
            "reports": reports,
            "scope": "first-batch execution control; no saved weights, tuning or endpoint selection",
        },
        destination,
    )


def train(
    label: str,
    endpoint: dict,
    arm: str,
    settings: dict,
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
    model, anchor = (
        load_restoration_head(state or initial).eval(),
        load_restoration_head(initial).eval(),
    )
    history, start = (state["history"], state["step"]) if state else ([], 0)
    if len(history) != start or not 0 <= start <= STEPS:
        raise ValueError("resume step/history mismatch")
    for step in range(start, STEPS):
        started = time.monotonic()
        items = [pool.select(schedule["draws"][i]) for i in config["groups"][step]]
        fixed = observations(anchor, items)
        result = attempt(model, items, fixed, **settings)
        history.append({"step": step + 1, "seconds": time.monotonic() - started, **result})
        state = {
            "model": model.state_dict(),
            "model_config": initial["model_config"],
            "arm": initial["arm"],
            "optimizer": None,
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


def final_review(config: dict, endpoints: dict, clips: BoundedEnvelopeClips, output: Path) -> None:
    directory = output / "review"
    directory.mkdir(exist_ok=True)
    for label, expected in config["reused_review_files"].items():
        path = Path(config["prior"]) / "review" / f"{label}.json"
        if sha256_file(path) != expected:
            raise ValueError("reused review drift")
        destination = directory / path.name
        if destination.exists() and sha256_file(destination) != expected:
            raise ValueError("local review copy changed")
        if not destination.exists():
            shutil.copy2(path, destination)
    review(config, endpoints, clips, output)
    contrasts = {}
    for label in config["endpoints"]:
        for bits in (16, 20, 24):
            policy = read(directory / f"{label}-reference_policy_pcm{bits}.json")["cases"]
            projected = read(directory / f"{label}-reference_projected_pcm{bits}.json")["cases"]
            old = read(directory / f"{label}-spectrum_pcm{bits}.json")["cases"]
            contrasts[f"{label}-pcm{bits}"] = {
                "projected_against_policy": compare(projected, policy),
                "policy_against_p45_displacement": compare(policy, old),
            }
    write_json_report(contrasts, output / "method-contrasts.json")
    # Original-reference MR relative to untreated input, for every method/role.
    untreated = {}
    for label in endpoints:
        groups = {}
        for case in read(directory / f"{label}.json")["cases"]:
            for i, sample in enumerate(case["samples"]["calibration"]):
                corpus = sample["row"]["corpus"]
                values = groups.setdefault(corpus, {"model": [], "compressed": []})
                values["model"].append(case["paths"]["frozen_model"]["mr"][i])
                values["compressed"].append(case["paths"]["compressed_passthrough"]["mr"][i])
        untreated[label] = {
            corpus: {
                "count": len(v["model"]),
                "model_mean": sum(v["model"]) / len(v["model"]),
                "compressed_mean": sum(v["compressed"]) / len(v["compressed"]),
            }
            for corpus, v in groups.items()
        }
    write_json_report(untreated, output / "untreated-comparison.json")
    write_json_report({"stage": "reference_review"}, output / "progress.json")
    review_reference(config, endpoints, clips, output)
    verify(config["hashes"])
    registered(
        {
            "registration_sha256": sha256_file(output / "registration.json"),
            "files": {
                name: sha256_file(output / name)
                for name in (
                    "review-registration.json",
                    "summary.json",
                    "assessment.json",
                    "method-contrasts.json",
                    "untreated-comparison.json",
                    "reference-review.json",
                )
            },
        },
        output / "complete.json",
    )
    write_json_report(
        {"stage": "complete_raw_and_reference_review", "endpoints": len(endpoints)},
        output / "progress.json",
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior, output)
        paths = [Path(p) for p in config["hashes"]]
        rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
        clips = BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            output / "feature-cache",
            capacity=320,
        )
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(schedule["specs"], clips)
        if (output / "complete.json").exists():
            for name, expected in read(output / "complete.json")["files"].items():
                if sha256_file(output / name) != expected:
                    raise ValueError("completed review changed")
            print("P46 already complete", flush=True)
            return
        write_json_report({"stage": "preflight"}, output / "progress.json")
        preflight(config, pool, schedule, output)
        if args.preflight_only:
            write_json_report({"stage": "preflight_complete"}, output / "progress.json")
            return
        endpoints = dict(config["reused_endpoints"])
        for label, endpoint in config["endpoints"].items():
            for arm, settings in ARMS.items():
                path = train(label, endpoint, arm, settings, config, pool, schedule, output)
                endpoints[f"{label}-{arm}"] = {"path": str(path), "sha256": sha256_file(path)}
        verify(config["hashes"])
        registered(endpoints, output / "review-registration.json")
        write_json_report({"stage": "raw_review"}, output / "progress.json")
        final_review(config, endpoints, clips, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
