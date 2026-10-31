"""P65 single fixed cost grid; retain every compute and paced failure.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
import platform
import shutil
from pathlib import Path

import numpy as np

from evaluation.adapter_runtime import cost
from evaluation.reporting import sha256_file
from evaluation.rust_bridge import write_sxa
from scripts.run_anchored_pilot import verify
from scripts.run_band_phase_pilot import SEEDS
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def prepare(study, binary, output):
    if output.exists():
        raise ValueError("fresh timing directory required; preserve failures")
    output.mkdir(parents=True)
    frozen = output / "soundex-context-replay"
    shutil.copy2(binary, frozen)
    source = output / "bench_band_phase.py"
    shutil.copy2(Path(__file__), source)
    schedule = []
    labels = [f"{seed}-{arm}" for seed in SEEDS for arm in ("zero", "phase")]
    for rate in (44100, 48000):
        for channels in (1, 2):
            for label in labels:
                schedule.append(dict(label=label, rate=rate, channels=channels, paced=False))
    schedule.extend(dict(label=label, rate=48000, channels=2, paced=True) for label in labels)
    for rate in (44100, 48000):
        time = np.arange(12800) / rate
        for channels in (1, 2):
            waves = [
                sum(np.sin(2 * np.pi * f * time + c * 0.7) for f in (173, 431, 1291, 3917))
                * 0.07
                * (0.7 + 0.3 * np.cos(2 * np.pi * 5 * time + c))
                for c in range(channels)
            ]
            write_sxa(
                output / f"input-{rate}-{channels}.sxa", np.stack(waves, 1).astype(np.float32), rate
            )
    registered(
        {
            "study": str(study),
            "binary": str(frozen),
            "schedule": schedule,
            "scope": "P65 diagnostic model-plus-DSP CPU screen, not production callback/transport or real-device qualification. No retry, no timing during training or quality assessment. Old P63 failures remain valid.",
            "warmup_hops": 100,
            "unpaced_scored_hops_per_case": 3000,
            "paced_scored_hops_per_case": 1000,
            "total_scored_hops": 52000,
            "thread_cpu_probe": False,
            "wake_observations": "all paced starts from new binary; includes backlog and loop overhead",
            "fixed_buffer_frames": 256,
            "compute_ceiling_ms_excluding_buffer": 5,
            "hashes": {str(p): sha256_file(p) for p in output.iterdir() if p.is_file()},
        },
        output / "registration.json",
    )


def run(output):
    config = read(output / "registration.json")
    study = Path(config["study"])
    checked_receipt(study / "complete.json")
    checked_receipt(study / "assessment-receipt.json")
    verify(config["hashes"])
    models = {
        s["label"]: study / "exports" / s["label"] / "diagnostic.onnx" for s in config["schedule"]
    }
    hashes = {str(p): sha256_file(p) for p in models.values()}
    for p in models.values():
        if hashes[str(p)] != read(p.parent / "report.json")["onnx_sha256"]:
            raise ValueError("frozen export mismatch")
    registered(
        {"models": hashes, "machine": platform.platform(), "load_before": os.getloadavg()},
        output / "endpoints.json",
    )
    results = []
    for index, spec in enumerate(config["schedule"]):
        directory = output / f"{index:02d}"
        report = cost(
            Path(config["binary"]),
            models[spec["label"]],
            output / f"input-{spec['rate']}-{spec['channels']}.sxa",
            directory,
            paced=spec["paced"],
        )
        raw = read(directory / "raw.json")
        wake = raw["wake_lateness_ms"][100:]
        if len(wake) != (1000 if spec["paced"] else 0):
            raise ValueError("paced wake observation geometry mismatch")
        results.append(
            {
                **spec,
                "cost": report,
                "wake_max_ms": max(wake, default=0),
                "wake_p99_ms": float(np.percentile(wake, 99)) if wake else 0,
            }
        )
        print(
            index + 1,
            len(config["schedule"]),
            spec,
            report["stages"]["total_ms"]["max_ms"],
            flush=True,
        )
    verify(config["hashes"])
    verify(hashes)
    registered({"scope": config["scope"], "cases": results}, output / "summary.json")
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.suffix in {".json", ".sxa", ".py"}
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.study.resolve(), args.binary.resolve(), args.output.resolve())
    else:
        run(args.output.resolve())
