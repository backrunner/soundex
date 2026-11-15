"""P78 event attribution / P79 late-producer grids on reused assets, not qualification.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import re
import shutil
import subprocess
import traceback
from pathlib import Path

from evaluation.reporting import sha256_file, write_json_report
from evaluation.worker_events import analyze, late_recovery
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_context_worker import summarize
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def prepare(prior, binary, output, *, late=False):
    if output.exists():
        raise ValueError("fresh event audit required")
    output.mkdir(parents=True)
    receipt = prior / ("complete.json" if late else "review-20261116/complete.json")
    checked_receipt(receipt)
    old = read(prior / "registration.json")
    snapshot(output)
    repo = Path(__file__).resolve().parents[2]
    for path in [
        repo / "Cargo.toml",
        repo / "Cargo.lock",
        *(p for p in (repo / "crates").rglob("*") if p.suffix in {".rs", ".toml"}),
    ]:
        destination = output / "source" / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    frozen = output / binary.name
    shutil.copy2(binary, frozen)
    inputs = {name: old["inputs"][name] for name in ("synthetic-44100-2", "music-48000-1")}
    schedule = []
    for index, (name, model) in enumerate(itertools.product(inputs, old["graphs"])):
        policies = ("RT",) if late else (("QOS", "RT") if index % 2 == 0 else ("RT", "QOS"))
        for policy in policies:
            schedule.append({"input": name, "model": model, "policy": policy})
    assets = [
        receipt,
        frozen,
        *(Path(p) for p in inputs.values()),
        *(Path(p) for p in old["graphs"].values()),
        *(p for p in (output / "source").rglob("*") if p.is_file()),
    ]
    registered(
        {
            "format": "soundex-worker-late-p79-v1" if late else "soundex-worker-events-p78-v1",
            "late_producer": late,
            "binary": str(frozen),
            "inputs": inputs,
            "graphs": old["graphs"],
            "schedule": schedule,
            "hashes": {str(p): sha256_file(p) for p in assets},
            "protocol": (
                "Eight once-only fault cases: all four P74 parents, inherited synthetic44.1k stereo and stored48k mono mid music, RT requested. Each1100 callbacks, delay callback256 by2 periods and640 by4 periods; retain original nominal clock and all following catch-up windows. Require observed faults/misses/resets and subsequent fully warmed acceptance, finite complete output at fixed256-frame delay, and protected dry alignment after two accepted warmup outputs within existing2e-6 functional tolerance. Keep every normal software-screen failure separately. No timing retries or concurrent training/build. This is fault handling, not normal availability or device qualification."
                if late
                else "16 once-only traced cases: all4 frozen P74 parents, synthetic44.1k stereo and stored48k mono mid music view from P77; QOS/RT order alternates. 1100 callbacks each, retain every startup/miss/reset event and original thresholds. No concurrent build/training, no retries. New instrumentation can perturb timings; this cannot erase or rescore the P77 failure."
            ),
            "analysis": "Reconcile per-input enqueue/reject, take/start/finish/terminal, history reset/warmup, callback consume/decision with final post-join counters. Attribute every missed presentation using explicit input sequence and wall-time intervals; preserve publication/consumer ambiguity. No thread-CPU or OS-cause claim. Retain CPU/RSS including process startup/IO and all raw audio.",
            "scope": "Reused diagnostic assets, narrower rate/channel coverage than P77, no physical stereo music/device/listening or policy superiority. No qualification or model change.",
        },
        output / "registration.json",
    )


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "cases").exists():
        raise ValueError("existing event runs retained")
    for index, case in enumerate(config["schedule"]):
        directory = output / "cases" / f"{index:02d}"
        directory.mkdir(parents=True)
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-l",
                config["binary"],
                config["graphs"][case["model"]],
                config["inputs"][case["input"]],
                str(directory / "output.sxa"),
                str(directory / "raw.json"),
                case["policy"],
                "TRACE-LATE" if config.get("late_producer") else "TRACE",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (directory / "time.txt").write_text(result.stderr)
        registered(
            {"returncode": result.returncode, "stdout": result.stdout}, directory / "exit.json"
        )
        print("event case", index, "exit", result.returncode, flush=True)
    # Finish the declared schedule before reviewing any new timing scores.
    results = []
    for index, case in enumerate(config["schedule"]):
        directory = output / "cases" / f"{index:02d}"
        report = read(directory / "raw.json")
        attribution = analyze(report)
        result = {
            "index": index,
            **case,
            **summarize(report),
            "returncode": read(directory / "exit.json")["returncode"],
            "rt_accepted": report["time_constraint_accepted"],
            "attribution": attribution,
        }
        if config.get("late_producer"):
            result["fault_recovery"] = late_recovery(report)
        resources = (directory / "time.txt").read_text()
        memory = re.search(r"(\d+)\s+maximum resident set size", resources)
        cpu = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", resources)
        if memory is None or cpu is None or float(cpu[1]) <= 0:
            raise ValueError("missing process resource evidence")
        result.update(
            peak_rss_bytes=int(memory[1]),
            process_one_core_percent=100 * (float(cpu[2]) + float(cpu[3])) / float(cpu[1]),
        )
        registered(result, directory / "assessment.json")
        results.append(result)
    verify(config["hashes"])
    registered(
        {"cases": results, "qualified": False, "scope": config["scope"]}, output / "assessment.json"
    )
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.log" and "__pycache__" not in p.parts
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--late-producer",
        action="store_true",
        help="prepare P79 fixed producer stalls from a completed P78 event audit",
    )
    args = parser.parse_args()
    fresh = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.binary.resolve(),
                args.output.resolve(),
                late=args.late_producer,
            )
        else:
            run(args.output.resolve())
    except BaseException:
        failure = args.output / "failure.json"
        if args.output.exists() and not failure.exists() and (args.mode == "run" or fresh):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
