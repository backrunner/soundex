#!/usr/bin/env python3
"""Same-binary QoS versus audio time-constraint scheduling, frozen P32 endpoints."""

import argparse
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.bench_context_deployment import summarize
from scripts.run_consistency_followup import read, registered


def run(args):
    if platform.system() != "Darwin":
        raise ValueError("macOS scheduling diagnostic only")
    study, output, binary = args.study.resolve(), args.output.resolve(), args.binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read(study / "registration.json")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("completed study required")
    endpoints = read(study / "review-registration.json")
    labels = [
        f"{arm}-{config['steps']}-{seed}"
        for seed in config["seeds"]
        for arm in ("control", "mr_add")
    ]
    audio = study / "native-cost/input-48000-2.sxa"
    hashes = {
        str(p): sha256_file(p)
        for p in [binary, audio, Path(__file__), study / "review-registration.json"]
    }
    models = {}
    for label in labels:
        value = endpoints[label]
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("checkpoint drift")
        model = study / "exports" / label / "diagnostic.onnx"
        hashes[str(model)] = read(model.parent / "report.json")["onnx_sha256"]
        if sha256_file(model) != hashes[str(model)]:
            raise ValueError("export drift")
        models[label] = model
    schedule = [
        {"label": label, "time_constraint": mode}
        for i, label in enumerate(labels)
        for mode in ((False, True) if i % 2 == 0 else (True, False))
    ]
    registered(
        {
            "scope": "same binary, paced stereo 48k, QoS requested in both arms; opt-in audio time-constraint only difference; no training during test; uncontrolled background load retained; not device qualification",
            "hashes": hashes,
            "schedule": schedule,
            "warmup_hops": 100,
            "measured_hops": 10000,
            "selection": "all four control/mr_add endpoints; no best-case exclusion or retry",
            "allocation": "both modes preallocate timing vectors before paced loop",
        },
        output / "registration.json",
    )
    rows = []
    for i, spec in enumerate(schedule):
        stem = output / f"{i:02d}"
        report = stem.with_suffix(".json")
        if report.exists():
            raise ValueError("refuse to overwrite timing evidence")
        command = [
            "/usr/bin/time",
            "-l",
            str(binary),
            str(models[spec["label"]]),
            str(audio),
            str(stem.with_suffix(".sxa")),
            str(report),
            "101",
        ]
        load = os.getloadavg()
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "SOUNDEX_DIAGNOSTIC_QOS": "1",
                "SOUNDEX_DIAGNOSTIC_PACED": "1",
                "SOUNDEX_DIAGNOSTIC_TIME_CONSTRAINT": str(int(spec["time_constraint"])),
            },
        )
        stem.with_suffix(".time.txt").write_text(result.stderr)
        raw = read(report)
        if raw["time_constraint_requested"] != spec["time_constraint"] or not raw["paced"]:
            raise ValueError("scheduling configuration drift")
        row = {
            **spec,
            **summarize(raw, result.stderr),
            "time_constraint_applied": raw["time_constraint_applied"],
            "host_load_before": load,
            "host_load_after": os.getloadavg(),
            "raw_sha256": sha256_file(report),
        }
        if not row["finite"]:
            raise ValueError("nonfinite benchmark output")
        rows.append(row)
        write_json_report(
            {"status": "running", "completed": len(rows), "total": len(schedule), "rows": rows},
            output / "progress.json",
        )
        print(
            spec,
            "applied",
            row["time_constraint_applied"],
            row["stages"]["total_ms"],
            "late",
            row["paced_late_hops"],
            flush=True,
        )
    parity = {}
    for label in labels:
        indices = [i for i, spec in enumerate(schedule) if spec["label"] == label]
        a, ar = read_sxa(output / f"{indices[0]:02d}.sxa")
        b, br = read_sxa(output / f"{indices[1]:02d}.sxa")
        if ar != br or a.shape != b.shape:
            raise ValueError("scheduling changed audio timeline")
        parity[label] = float(np.max(np.abs(a - b)))
        if parity[label] > 2e-5:
            raise ValueError("scheduling changed audio beyond parity tolerance")
    for path, expected in hashes.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("benchmark input drift")
    write_json_report(
        {
            "status": "complete",
            "registration_sha256": sha256_file(output / "registration.json"),
            "rows": rows,
            "scheduling_audio_max_error": parity,
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
