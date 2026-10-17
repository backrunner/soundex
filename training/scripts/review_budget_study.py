#!/usr/bin/env python3
"""Post-hoc P24 replay recovery; frozen training and original failures stay intact."""

import argparse
import fcntl
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.head_study import review
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from scripts.analyze_head_study import run as analyze
from scripts.run_budget_study import COMPARISONS
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered, verify_receipt
from scripts.run_head_study import progress
from scripts.run_subset_study import Pool


def register(parent, output, observer):
    old = read(parent / "registration.json")
    frozen = parent.parent / "source/training"
    for name, expected in old["source_hashes"].items():
        if sha256_file(frozen / name) != expected:
            raise ValueError("original training source drift")
    endpoints = read(parent / "review-registration.json")
    expected_labels = {
        f"{a}-{s}-{seed}" for a in old["arms"] for s in old["endpoints"] for seed in old["seeds"]
    }
    if set(endpoints) != expected_labels:
        raise ValueError("incomplete frozen endpoint set")
    inputs = dict(old["input_hashes"])
    for arm in old["arms"]:
        for seed in old["seeds"]:
            directory = parent / f"{arm}-{seed}"
            verify_receipt(directory, "files")
            if len(read(directory / "updates.json")) != old["steps"]:
                raise ValueError("incomplete training budget")
            inputs[str(directory / "complete.json")] = sha256_file(directory / "complete.json")
    for value in endpoints.values():
        path = Path(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise ValueError("frozen endpoint drift")
        inputs[str(path)] = value["sha256"]
    for path in (parent / "registration.json", parent / "review-registration.json", observer):
        inputs[str(path)] = sha256_file(path)
    for seed in old["seeds"]:
        path = parent / f"draws-{seed}.json"
        inputs[str(path)] = sha256_file(path)
        registered(read(path), output / path.name)
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    config = {
        **old,
        "format": "soundex-p24-posthoc-runtime-observation-review-v1",
        "scope": "post-hoc replay diagnosis of all frozen P24 endpoints; reused development; original frontend parity failures retained; no promotion",
        "parent": str(parent),
        "observed_binary": str(observer),
        "parity_policy": "original training-proxy errors retained and flagged at 2e-5; additional same-Rust-observation synthesis comparison also requires 2e-5; frozen model predictions and quality references unchanged; not full deployed-model inference validation",
        "input_hashes": inputs,
        "source_hashes": hashes,
    }
    verify_inputs(config)
    registered(config, output / "registration.json")
    registered(endpoints, output / "review-registration.json")
    return config, endpoints


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, endpoints = register(args.parent.resolve(), root, args.observer.resolve())
        if args.prepare_only:
            return
        inputs = [Path(p) for p in config["input_hashes"]]
        rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
        detector = next(p for p in inputs if p.name == "soundex-gain-data")
        parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
        clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
        pools = {name: Pool(config[name], clips) for name in ("development", "shifted")}
        binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
        onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
        observer = Path(config["observed_binary"])
        for label, value in [("initial", None), *endpoints.items()]:
            checkpoint = Path(value["path"]) if value else None
            review(
                label,
                checkpoint,
                value["arm"] if value else None,
                root,
                pools,
                binary,
                onnx,
                config["weight"],
                observed_binary=observer,
            )
            joint_review(label, checkpoint, root, pools, binary, onnx, observed_binary=observer)
            if checkpoint:
                progress(root, stage="export_and_measure", case=label)
                export_and_measure(
                    checkpoint,
                    root / "exports" / label,
                    experimental_unbounded=True,
                    iterations=10000,
                )
        verify_inputs(config)
        parity = []
        for path in sorted((root / "review").glob("*/native/*.json")):
            record = read(path)
            for role in ("paired", "identity"):
                item = record[role]
                parity.append(
                    {
                        "case": path.parent.parent.name,
                        "spec": record["spec"],
                        "role": role,
                        **{
                            k: item[k]
                            for k in (
                                "proxy_max_error",
                                "training_frontend_parity_passed",
                                "runtime_observed_proxy_max_error",
                                "runtime_observed_parity_passed",
                            )
                        },
                    }
                )
        write_json_report(
            {
                "scope": config["parity_policy"],
                "roles": len(parity),
                "original_frontend_failures": sum(
                    not r["training_frontend_parity_passed"] for r in parity
                ),
                "runtime_observed_failures": sum(
                    not r["runtime_observed_parity_passed"] for r in parity
                ),
                "rows": parity,
            },
            root / "frontend-parity.json",
        )
        write_json_report(
            {
                "status": "complete",
                "registration_sha256": sha256_file(root / "registration.json"),
                "review_registration_sha256": sha256_file(root / "review-registration.json"),
                "frontend_parity_sha256": sha256_file(root / "frontend-parity.json"),
                "receipt_hashes": {
                    str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
                },
            },
            root / "summary.json",
        )
        analyze(root, comparisons=COMPARISONS, source=Path(__file__).resolve().parents[1])
        progress(
            root,
            stage="complete",
            original_frontend_failures=sum(
                not r["training_frontend_parity_passed"] for r in parity
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
