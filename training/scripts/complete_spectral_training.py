#!/usr/bin/env python3
"""Finish registered P54 training/review, then time every frozen endpoint once.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

from evaluation.adapter_runtime import cost
from evaluation.reporting import sha256_file, write_json_report
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_spectral_training import run as training_run


def run(args):
    output = args.output.resolve()
    config = read(output / "registration.json")
    prior = read(args.prior.resolve() / "registration.json")
    assets = Path(prior["assets"])
    paths = [Path(__file__), output / "registration.json", Path(config["binary"])]
    paths += [
        assets / f"input-{rate}-{channels}.sxa" for rate in (44100, 48000) for channels in (1, 2)
    ]
    registration = {
        "training_registration_sha256": sha256_file(output / "registration.json"),
        "hashes": {str(p): sha256_file(p) for p in paths},
        "protocol": "After all four training endpoints and raw/native reviews finish: one 3000-measured-hop unpaced run per 44.1/48k mono/stereo case, plus one 1000-hop paced 48k stereo case per endpoint. 52000 measured hops, 100 warmup per run. No training/test work inside this process during timing. Keep every outlier, no retry-to-pass. Diagnostic context replay, not actual callback/worker/device certification. Fixed 256-frame buffer excluded from compute.",
    }
    registered(registration, output / "runtime-registration.json")
    training_run(
        SimpleNamespace(prior=args.prior, output=output, prepare_only=False, calibrate_only=False)
    )
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(registration["hashes"])
        verify(read(output / "complete.json")["files"])
        runtime_receipt = output / "runtime-complete.json"
        if runtime_receipt.exists():
            verify(read(runtime_receipt)["files"])
            return
        endpoints = read(output / "review-registration.json")
        results = {}
        for label in endpoints:
            graph = output / "exports" / label / "diagnostic.onnx"
            for rate in (44100, 48000):
                for channels in (1, 2):
                    name = f"{label}/{rate}-{channels}"
                    results[name] = cost(
                        Path(config["binary"]),
                        graph,
                        assets / f"input-{rate}-{channels}.sxa",
                        output / "native-cost" / name,
                    )
            name = label + "/paced-48000-2"
            results[name] = cost(
                Path(config["binary"]),
                graph,
                assets / "input-48000-2.sxa",
                output / "native-cost" / name,
                paced=True,
            )
        write_json_report(results, output / "cost-summary.json")
        paths = [
            output / "runtime-registration.json",
            output / "cost-summary.json",
            *(p for p in (output / "native-cost").rglob("*") if p.is_file()),
        ]
        registered({"files": {str(p): sha256_file(p) for p in paths}}, runtime_receipt)
        write_json_report(
            {
                "stage": "complete",
                "qualified": False,
                "scope": "All fixed training, diagnostic reviews and timing finished; not independent listening/device qualification.",
            },
            output / "progress.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
