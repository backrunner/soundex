#!/usr/bin/env python3
"""P50 bounded continuous high-band adapter training, then frozen raw/native review.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_continuation import ARMS, STEPS, schedule_roles, update
from evaluation.adapter_pilot import assert_frozen, load_endpoint
from evaluation.adapter_runtime import cost, export, native_review
from evaluation.anchored_review import compare, review_case
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from models.spectral_adapter import from_base, load_adapter
from scripts.audit_guarded_transfer import measured
from scripts.run_adapter_pilot import checkpoint, optimizer_for
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_adapter_scale import changes
from scripts.summarize_guarded_transfer import coverage


def prepare(prior, output):
    destination = output / "registration.json"
    if destination.exists():
        config = read(destination)
        if config["prior"] != str(prior):
            raise ValueError("prior changed")
        verify(config["hashes"])
        return config
    recovery = read(prior / "registration.json")
    verify(recovery["hashes"])
    verify(read(prior / "complete.json")["files"])
    calibration = Path(recovery["calibration"])
    old = read(calibration / "registration.json")
    p48 = read(Path(old["prior"]) / "registration.json")
    paths = [Path(p) for p in old["hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    bank = read(Path(old["schedule"]))
    draws = schedule_roles(bank["specs"], rows)
    schedule = {"specs": bank["specs"], **draws}
    cases = {
        role: [
            {
                "index": index,
                "samples": {
                    "calibration": [
                        {"spec": bank["specs"][i], "row": rows[bank["specs"][i]["row_id"]]}
                        for i in draw
                    ]
                },
            }
            for index, draw in enumerate(role_draws)
        ]
        for role, role_draws in draws.items()
    }
    registered(schedule, output / "schedule.json")
    hashes = dict(recovery["hashes"])
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(
            part in {".venv", "__pycache__", "outputs"} for part in p.relative_to(root).parts
        )
    ] + list((root / "evaluation").glob("*.json"))
    for path in sources:
        hashes[str(path)] = sha256_file(path)
        target = output / "source" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (
        prior / "registration.json",
        prior / "complete.json",
        prior / "summary.json",
        output / "schedule.json",
    ):
        hashes[str(path)] = sha256_file(path)
    config = {
        "format": "soundex-adapter-continuation-p50-v1",
        "prior": str(prior),
        "hashes": hashes,
        "starts": old["starts"],
        "steps": STEPS,
        "arms": ARMS,
        "support": "high",
        "scale": 0.001,
        "schedule": str(output / "schedule.json"),
        "cases": cases,
        "coverage": {role: coverage(items, "calibration") for role, items in cases.items()},
        "binary": p48["binary"],
        "assets": p48["assets"],
        "detector": str(next(p for p in paths if p.name == "soundex-gain-data")),
        "manifests": [str(p) for p in paths if p.name == "manifest.jsonl"],
        "objective": old["objective"],
        "optimizer": old["optimizer"],
        "hypothesis": "P49 reset probes cannot establish cumulative gain. Compare persistent Adam moments against per-step reset at identical 0.001 displacement, data, high support and 128 actual steps; original lr=1e-4 proposal scaled after Adam, not a fresh parameter reset.",
        "split": "Connected track_id/source_checksum components, score-independent digest modulo five assigns review. All 512 fit batches are globally separated from all 32 review batches. Existing P45 finite-bank crop support/exclusions preserved. All sources reused TRAIN, not unseen test data; 4 music/1 speech plus identities.",
        "barrier": "Finish all four fixed-budget trajectories and bind every endpoint hash before any review score. No early stopping, guard veto, selected checkpoints or post-hoc tolerance changes. Review frozen starts and endpoints on 32 review batches and 32 evenly spaced fit batches.",
        "native": "Same P48 ABI/model/DSP; export all frozen models and identity-through-DSP control. Eight evenly spaced review batches, mono views, 560 native roles. Four endpoints: 3000 hops at each rate/channel configuration and 1000 paced 48k stereo. No concurrent training/tests during timing, no retry-to-pass. Fixed 256-frame buffer excluded from compute.",
        "decision": "Bounded optimization pilot, not adequate convergence/capacity training, independent generalization, listening or promotion. Report spectra first, phase/dynamics/identity, absolute PCM, all 16/20/24-bit rulers, untreated and native controls. No automatic long training.",
    }
    verify(hashes)
    registered(config, destination)
    return config


def train(seed, entry, arm, config, pool, schedule, output):
    directory = output / "training" / f"{seed}-{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    latest, final = directory / "latest.pth", directory / "endpoint.pth"
    signature = digest({"config": config, "seed": seed, "arm": arm})
    receipt_path = directory / "complete.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["signature"] != signature or receipt["sha256"] != sha256_file(final):
            raise ValueError("completed endpoint drift")
        return final
    initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
    saved = None
    if latest.exists():
        if read(directory / "latest.json")["sha256"] != sha256_file(latest):
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or saved["step"] != len(saved["history"]):
            raise ValueError("resume registration/history drift")
    model = (load_adapter(saved) if saved else from_base(initial, "high")).eval()
    optimizer = optimizer_for(model)
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
    for index in range(len(history), config["steps"]):
        started = time.monotonic()
        if arm == "reset":
            optimizer = optimizer_for(model)
        items = [pool.select(draw) for draw in schedule["fit"][4 * index : 4 * index + 4]]
        result = update(model, items, optimizer)
        history.append({"step": index + 1, "seconds": time.monotonic() - started, **result})
        if index == 0 or (index + 1) % 8 == 0:
            assert_frozen(model, initial)
            atomic_save(checkpoint(model, initial, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, directory / "latest.json")
            write_json_report(history, directory / "updates.json")
            progress = {
                "stage": "training",
                "seed": seed,
                "arm": arm,
                "step": index + 1,
                "steps": config["steps"],
                "seconds_per_step": history[-1]["seconds"],
            }
            write_json_report(progress, output / "progress.json")
            print(progress, flush=True)
    assert_frozen(model, initial)
    atomic_save(checkpoint(model, initial, optimizer, history, signature), final)
    registered({"signature": signature, "sha256": sha256_file(final)}, receipt_path)
    return final


@torch.no_grad()
def raw_review(config, endpoints, pool, schedule, output):
    reports, result = {}, {}
    for role in ("fit", "review"):
        indices = (
            list(range(0, len(schedule[role]), 16))
            if role == "fit"
            else list(range(len(schedule[role])))
        )
        items = [pool.select(schedule[role][i]) for i in indices]
        anchors = {}
        for seed, entry in config["starts"].items():
            model = load_endpoint(entry)
            anchors[seed] = [
                supervised_waves(predict_budget(model, item[0].batch, item[1])[0], item[0].batch)[0]
                for item in items
            ]
        for label, entry in endpoints.items():
            path = output / "review" / role / f"{label}.json"
            receipt = path.with_suffix(".receipt.json")
            if receipt.exists():
                saved = read(path)
                if (
                    read(receipt)["sha256"] != sha256_file(path)
                    or saved["checkpoint_sha256"] != entry["sha256"]
                ):
                    raise ValueError("review receipt drift")
                reports[(role, label)] = saved["cases"]
                continue
            if path.exists():
                raise ValueError("incomplete review retained")
            model = load_endpoint(entry)
            seed = next(s for s in config["starts"] if label == s or label.startswith(s + "-"))
            cases = []
            for j, item in enumerate(items):
                prediction, _ = predict_budget(model, item[0].batch, item[1])
                wave = supervised_waves(prediction, item[0].batch)[0]
                cases.append(
                    {
                        **config["cases"][role][indices[j]],
                        "metrics": measured(model, item),
                        **review_case(model, item),
                        "reference_rulers": {
                            str(bits): pcm_guard([wave], [anchors[seed][j]], [item], bits)
                            for bits in (16, 20, 24)
                        },
                    }
                )
            write_json_report({"checkpoint_sha256": entry["sha256"], "cases": cases}, path)
            registered({"sha256": sha256_file(path)}, receipt)
            reports[(role, label)] = cases
            print("raw", role, label, "complete", flush=True)
        result[role] = {}
        for seed in config["starts"]:
            baseline = reports[(role, seed)]
            for arm in ARMS:
                label = seed + "-" + arm
                cases = reports[(role, label)]
                result[role][label] = {
                    "raw_against_start": compare(cases, baseline),
                    "raw_against_reset": compare(cases, reports[(role, seed + "-reset")]),
                    "metrics": changes(
                        [c["metrics"] for c in cases], [c["metrics"] for c in baseline]
                    ),
                    "ruler_violations": {
                        str(bits): dict(
                            Counter(
                                v["role"]
                                for c in cases
                                for v in c["reference_rulers"][str(bits)]["violations"]
                            )
                        )
                        for bits in (16, 20, 24)
                    },
                }
    write_json_report(result, output / "assessment.json")


def finalize(config, endpoints, pool, schedule, output):
    # The whole endpoint barrier is durable before any raw or native scoring.
    registered(endpoints, output / "review-registration.json")
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
    raw_review(config, endpoints, pool, schedule, output)
    models, costs = {}, []
    for label, entry in endpoints.items():
        model = load_endpoint(entry)
        directory = output / "exports" / label
        export(model, directory)
        models[label] = directory / "diagnostic.onnx"
        if label in config["starts"]:
            continue
        for rate in (44100, 48000):
            for channels in (1, 2):
                row = cost(
                    Path(config["binary"]),
                    models[label],
                    Path(config["assets"]) / f"input-{rate}-{channels}.sxa",
                    output / "native-cost" / label / f"{rate}-{channels}",
                )
                costs.append({"label": label, "rate": rate, "channels": channels, **row})
        row = cost(
            Path(config["binary"]),
            models[label],
            Path(config["assets"]) / "input-48000-2.sxa",
            output / "native-cost" / label / "48000-2-paced",
            paced=True,
        )
        costs.append({"label": label, "rate": 48000, "channels": 2, **row})
        write_json_report(costs, output / "cost-summary.json")
        print("native cost", label, "complete", flush=True)
    identity = load_endpoint(next(iter(config["starts"].values())))
    with torch.no_grad():
        identity.backbone.decoder[-1].weight.zero_()
        identity.backbone.decoder[-1].bias.zero_()
    export(identity, output / "exports/identity_dsp")
    models["identity_dsp"] = output / "exports/identity_dsp/diagnostic.onnx"
    items = [pool.select(draw) for draw in schedule["review"]]
    write_json_report({"stage": "native_quality", "models": len(models)}, output / "progress.json")
    native_review(
        Path(config["binary"]), models, items, config["cases"]["review"], output / "native-review"
    )
    verify(config["hashes"])
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
    files = [
        output / "assessment.json",
        output / "cost-summary.json",
        output / "review-registration.json",
        output / "native-review/summary.json",
        *(output / "review").rglob("*.json"),
        *(output / "exports").rglob("report.json"),
    ]
    registered(
        {
            "registration_sha256": sha256_file(output / "registration.json"),
            "files": {str(p): sha256_file(p) for p in files},
        },
        output / "complete.json",
    )
    write_json_report(
        {"stage": "complete", "endpoints": 4, "steps_each": config["steps"]},
        output / "progress.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), output)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            entries = read(output / "review-registration.json")
            verify({e["path"]: e["sha256"] for e in entries.values()})
            print("P50 complete", flush=True)
            return
        if args.prepare_only:
            print("P50 registered", config["coverage"], flush=True)
            return
        rows, _ = load_sources([Path(p) for p in config["manifests"]])
        clips = BoundedEnvelopeClips(
            rows, Path(config["detector"]), output / "feature-cache", capacity=320
        )
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(schedule["specs"], clips)
        endpoints = dict(config["starts"])
        for seed, entry in config["starts"].items():
            for arm in ARMS:
                path = train(seed, entry, arm, config, pool, schedule, output)
                endpoints[seed + "-" + arm] = {"path": str(path), "sha256": sha256_file(path)}
        finalize(config, endpoints, pool, schedule, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
