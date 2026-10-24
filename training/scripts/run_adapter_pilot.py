#!/usr/bin/env python3
"""P48 matched frozen-feature adapter pilot with a complete endpoint barrier.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import time
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import STEPS, assert_frozen, load_endpoint, prepare, review, step
from evaluation.adapter_runtime import cost, export, native_review
from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head, predict_budget
from models.spectral_adapter import SUPPORTS, from_base, load_adapter
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered


def optimizer_for(model):
    return torch.optim.Adam(model.adapter.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)


def checkpoint(model, initial: dict, optimizer, history: list, signature: str) -> dict:
    return {
        "format": "soundex-spectral-adapter-v1",
        "model": model.state_dict(),
        "base_config": initial["model_config"],
        "support": model.support,
        "optimizer": optimizer.state_dict(),
        "step": len(history),
        "history": history,
        "signature": signature,
    }


def preflight(config: dict, pool, schedule: dict, output: Path) -> None:
    destination = output / "preflight.json"
    if destination.exists():
        old = read(destination)
        if old["signature"] != digest(config):
            raise ValueError("preflight registration drift")
        if not old["throughput_feasible"]:
            raise ValueError("retained preflight throughput failure; training remains blocked")
        for row in old["rows"]:
            if sha256_file(Path(row["onnx"])) != row["export"]["onnx_sha256"]:
                raise ValueError("preflight export drift")
        return
    items = [pool.select(schedule["draws"][i]) for i in config["groups"][0]]
    rows = []
    for seed, entry in config["starts"].items():
        initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
        base = load_restoration_head(initial).eval()
        with torch.no_grad():
            expected, budget = predict_budget(base, items[0][0].batch, items[0][1])
        for support in SUPPORTS:
            model = from_base(initial, support).eval()
            with torch.no_grad():
                actual, auxiliary = predict_budget(model, items[0][0].batch, items[0][1])
            if not torch.equal(expected, actual) or not torch.equal(budget, auxiliary):
                raise ValueError("zero adapter changes start")
            optimizer = optimizer_for(model)
            result = step(model, items, optimizer)
            assert_frozen(model, initial)
            if result["parameter_delta_l2"] == 0:
                raise ValueError("adapter did not learn in real-data preflight")
            directory = output / "preflight" / f"{seed}-{support}"
            directory.mkdir(parents=True, exist_ok=True)
            atomic_save(
                checkpoint(model, initial, optimizer, [result], digest(config)),
                directory / "smoke.pth",
            )
            graph = export(model, directory)
            onnx = directory / "diagnostic.onnx"
            runtime = cost(
                Path(config["binary"]),
                onnx,
                Path(config["assets"]) / "input-48000-2.sxa",
                directory / "native-cost",
            )
            row = {
                "seed": seed,
                "support": support,
                "zero_start_exact": True,
                "base_frozen": True,
                "update": result,
                "export": graph,
                "onnx": str(onnx),
                "runtime": runtime,
            }
            rows.append(row)
            write_json_report({"rows": rows, "stage": "preflight"}, output / "progress.json")
            print("preflight", seed, support, runtime["stages"]["total_ms"], flush=True)
    # Feasibility only: stricter max/streaming qualification is reported, never
    # obtained by selecting a fastest repeat or dropping an OS timing outlier.
    feasible = all(r["runtime"]["stages"]["total_ms"]["p99_ms"] < 128 / 48 for r in rows)
    registered(
        {"signature": digest(config), "rows": rows, "throughput_feasible": feasible}, destination
    )
    if not feasible:
        raise ValueError("preflight throughput infeasible; retain evidence, do not start training")


def train(
    seed: str, entry: dict, support: str, config: dict, pool, schedule: dict, output: Path
) -> Path:
    directory = output / seed / support
    directory.mkdir(parents=True, exist_ok=True)
    latest, final = directory / "latest.pth", directory / "endpoint.pth"
    signature = digest({"registration": config, "seed": seed, "support": support})
    if (directory / "complete.json").exists():
        receipt = read(directory / "complete.json")
        if receipt["signature"] != signature or receipt["sha256"] != sha256_file(final):
            raise ValueError("completed adapter drift")
        return final
    initial = torch.load(entry["path"], map_location="cpu", weights_only=False)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if saved and saved["signature"] != signature:
        raise ValueError("resume signature drift")
    model = (load_adapter(saved) if saved else from_base(initial, support)).eval()
    optimizer = optimizer_for(model)
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        if saved["step"] != len(history):
            raise ValueError("step/history mismatch")
    for index in range(len(history), STEPS):
        started = time.monotonic()
        items = [pool.select(schedule["draws"][i]) for i in config["groups"][index]]
        result = step(model, items, optimizer)
        history.append({"step": index + 1, "seconds": time.monotonic() - started, **result})
        if (index + 1) % 8 == 0 or index == 0:
            assert_frozen(model, initial)
            atomic_save(checkpoint(model, initial, optimizer, history, signature), latest)
            write_json_report(history, directory / "updates.json")
            info = {
                "stage": "training",
                "seed": seed,
                "support": support,
                "step": index + 1,
                "steps": STEPS,
                "last_seconds": history[-1]["seconds"],
            }
            write_json_report(info, output / "progress.json")
            print(info, flush=True)
    assert_frozen(model, initial)
    atomic_save(checkpoint(model, initial, optimizer, history, signature), final)
    registered({"signature": signature, "sha256": sha256_file(final)}, directory / "complete.json")
    return final


def finalize(config: dict, endpoints: dict, clips, output: Path) -> None:
    registered(endpoints, output / "review-registration.json")
    items, originals = review(config, endpoints, clips, output)
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
        print("final native cost", label, "complete", flush=True)
    identity = load_endpoint(next(iter(config["starts"].values())))
    with torch.no_grad():
        identity.backbone.decoder[-1].weight.zero_()
        identity.backbone.decoder[-1].bias.zero_()
    export(identity, output / "exports/identity_dsp")
    models["identity_dsp"] = output / "exports/identity_dsp/diagnostic.onnx"
    write_json_report({"stage": "native_quality", "models": len(models)}, output / "progress.json")
    native_review(Path(config["binary"]), models, items, originals, output / "native-review")
    verify(config["hashes"])
    files = [
        output / "assessment.json",
        output / "review-hashes.json",
        output / "cost-summary.json",
        output / "review-registration.json",
        output / "native-review/summary.json",
    ]
    registered(
        {
            "registration_sha256": sha256_file(output / "registration.json"),
            "files": {str(p): sha256_file(p) for p in files},
        },
        output / "complete.json",
    )
    write_json_report(
        {
            "stage": "complete",
            "trained_endpoints": 4,
            "steps_each": STEPS,
            "native_models": len(models),
        },
        output / "progress.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(
            args.prior.resolve(),
            args.evidence.resolve(),
            args.binary.resolve(),
            args.assets.resolve(),
            output,
        )
        verify(config["hashes"])
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            print("P48 complete", flush=True)
            return
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
        preflight(config, pool, schedule, output)
        if args.preflight_only:
            return
        endpoints = dict(config["starts"])
        for seed, entry in config["starts"].items():
            for support in SUPPORTS:
                path = train(seed, entry, support, config, pool, schedule, output)
                endpoints[f"{seed}-{support}"] = {"path": str(path), "sha256": sha256_file(path)}
        finalize(config, endpoints, clips, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "evidence", "binary", "assets", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
