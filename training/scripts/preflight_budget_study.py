#!/usr/bin/env python3
"""Train-only P24 integration preflight; never an early quality gate."""

import argparse
from pathlib import Path

import torch

from data.causal_study import digest, load_sources
from data.coverage_study import PoolView
from data.envelope_study import EnvelopeClips
from evaluation.head_export import export_and_measure
from evaluation.joint_review import joint_review
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_budget_study import register
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_head_study import train
from scripts.run_subset_study import Pool


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    full = root / "full-protocol"
    full.mkdir(exist_ok=True)
    config = register(args.parent, full)
    seed = config["seeds"][0]
    original = read(full / f"draws-{seed}.json")["draws"][:16]
    indices = sorted({i for draw in original for i in draw})
    draws = [[indices.index(i) for i in draw] for draw in original]
    bank = [config["bank"][i] for i in indices]
    common = [config["bank"][i] for i in original[0]]
    config.update(
        steps=16,
        endpoints=[16],
        primary_endpoint=16,
        score_interval=16,
        seeds=[seed],
        bank=bank,
        common=common,
        development=common,
        shifted=common,
        schedule_hashes={str(seed): digest(draws)},
        scope="TRAIN ONLY integration, no quality screen; fresh full run discards these weights",
    )
    registered(config, root / "registration.json")
    registered({"draws": draws}, root / f"draws-{seed}.json")
    inputs = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
    detector = next(p for p in inputs if p.name == "soundex-gain-data")
    parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    pool = Pool(bank, clips)
    common = PoolView(pool, common)
    for arm in config["arms"]:
        train(arm, seed, root, config, pool, common)
    binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
    onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
    for arm in config["arms"]:
        checkpoint = root / f"{arm}-{seed}/step-16.pth"
        joint_review(arm, checkpoint, root, {"development": common}, binary, onnx)
        export_and_measure(
            checkpoint, root / "exports" / arm, experimental_unbounded=True, iterations=1000
        )
    verify_inputs(config)
    write_json_report(
        {
            "status": "complete",
            "scope": config["scope"],
            "registration_sha256": sha256_file(root / "registration.json"),
            "receipts": {
                str(p.relative_to(root)): sha256_file(p) for p in root.rglob("complete.json")
            },
        },
        root / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
