#!/usr/bin/env python3
"""Isolated-process experimental full compute chain, with raw timing evidence."""

import argparse
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np

from evaluation.head_export import distribution, stream_cost
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import write_sxa
from scripts.run_consistency_followup import read, registered


def summarize(report, stderr):
    values = np.asarray(report["timings"])[100:]
    period = 128000 / report["sample_rate"]
    memory = re.search(r"(\d+)\s+maximum resident set size", stderr)
    cpu = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", stderr)
    stages = {k: distribution(values[:, j]) for j, k in enumerate(report["timing_columns"])}
    stages["total_ms"].update(stream_cost(values[:, 3].tolist(), period))
    stages["total_ms"]["over_5ms"] = int((values[:, 3] >= 5).sum())
    stages["total_ms"]["service_over_hop"] = int((values[:, 3] >= period).sum())
    return {
        "measured_hops": len(values),
        "stages": stages,
        "cold_first_hop_ms": report["timings"][0][-1],
        "load_ms": report["load_ms"],
        "process_peak_rss_bytes": int(memory[1]) if memory else None,
        "cpu_user_seconds": float(cpu[2]) if cpu else None,
        "cpu_system_seconds": float(cpu[3]) if cpu else None,
        "process_wall_seconds": float(cpu[1]) if cpu else None,
        "paced_late_hops": sum(x > 0 for x in report["paced_finish_lateness_ms"][100:]),
        "paced_max_lateness_ms": max(report["paced_finish_lateness_ms"][100:], default=0),
        "qos_applied": report["qos_applied"],
        "finite": report["finite"],
        "peak": report["output_peak"],
        "active_hops": report["active_hops"],
        "inference_hops": report["inference_hops"],
    }


def run(args):
    root, binary, study = args.output.resolve(), args.binary.resolve(), args.study.resolve()
    root.mkdir(parents=True, exist_ok=True)
    labels = sorted(k for k in read(study / "review-registration.json") if "-16192-" in k)
    if len(labels) != 8:
        raise ValueError("expected all eight fixed P24 final endpoints")
    schedule = []
    for rate in (44100, 48000):
        for channels in (1, 2):
            for label in labels:
                schedule.append(
                    dict(label=label, rate=rate, channels=channels, paced=False, round=0)
                )
    for repeat in (1, 2):
        for label in labels[repeat:] + labels[:repeat]:
            schedule.append(dict(label=label, rate=48000, channels=2, paced=False, round=repeat))
    for label in labels:
        if label.startswith(("control-", "budget-")):
            schedule.append(dict(label=label, rate=48000, channels=2, paced=True, round=0))
    models = {label: study / "exports" / label / "diagnostic.onnx" for label in labels}
    hashes = {str(p): sha256_file(p) for p in [binary, Path(__file__), *models.values()]}
    for path in models.values():
        if hashes[str(path)] != read(path.parent / "report.json")["onnx_sha256"]:
            raise ValueError("frozen model drift")
    registered(
        {
            "scope": "diagnostic CPU full compute chain; no callback/transport/device certification",
            "schedule": schedule,
            "hashes": hashes,
            "machine": platform.platform(),
            "warmup_hops": 100,
            "measured_hops": 10000,
            "qos": "USER_INITIATED requested; no time-constraint scheduling",
            "stimulus": "deterministic 100-hop low-band multitone with time-varying envelope, independently phased physical channels; repeated without reset; synthetic performance only",
        },
        root / "registration.json",
    )
    for rate in (44100, 48000):
        t = np.arange(12800) / rate
        for channels in (1, 2):
            waves = []
            for c in range(channels):
                wave = (
                    sum(np.sin(2 * np.pi * f * t + c * 0.7) for f in (173, 431, 1291, 3917)) * 0.07
                )
                wave *= 0.7 + 0.3 * np.cos(2 * np.pi * 5 * t + c)
                waves.append(wave)
            write_sxa(
                root / f"input-{rate}-{channels}.sxa", np.stack(waves, 1).astype(np.float32), rate
            )
    results = []
    for i, spec in enumerate(schedule):
        destination = root / f"{i:02d}.json"
        if destination.exists():
            raise ValueError("do not overwrite timing evidence; choose a new directory")
        stem = root / f"{i:02d}"
        command = [
            str(binary),
            str(models[spec["label"]]),
            str(root / f"input-{spec['rate']}-{spec['channels']}.sxa"),
            str(stem.with_suffix(".sxa")),
            str(destination),
            "101",
        ]
        if platform.system() == "Darwin":
            command = ["/usr/bin/time", "-l", *command]
        load_before = os.getloadavg()
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "SOUNDEX_DIAGNOSTIC_QOS": "1",
                "SOUNDEX_DIAGNOSTIC_PACED": str(int(spec["paced"])),
            },
        )
        stem.with_suffix(".time.txt").write_text(result.stderr)
        measurement = summarize(read(destination), result.stderr)
        results.append(
            {
                **spec,
                **measurement,
                "raw_sha256": sha256_file(destination),
                "host_load_before": load_before,
                "host_load_after": os.getloadavg(),
            }
        )
        write_json_report(
            {
                "status": "running",
                "completed": len(results),
                "total": len(schedule),
                "rows": results,
            },
            root / "progress.json",
        )
        print(i + 1, "/", len(schedule), spec, measurement["stages"]["total_ms"], flush=True)
    for p, digest in hashes.items():
        if sha256_file(Path(p)) != digest:
            raise ValueError("benchmark input/source drift")
    write_json_report(
        {
            "status": "complete",
            "rows": results,
            "registration_sha256": sha256_file(root / "registration.json"),
        },
        root / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
