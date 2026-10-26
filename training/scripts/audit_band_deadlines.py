#!/usr/bin/env python3
"""P63 bounded P62 CPU/wake probe; never replace the original timing failures."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from evaluation.adapter_runtime import environment
from evaluation.head_export import distribution
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.bench_context_deployment import summarize
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered


def run(study, output):
    if output.exists():
        raise ValueError("retain every previous timing; choose an unused directory")
    verify({str(study / p): h for p, h in read(study / "receipt.json")["artifacts"].items()})
    output.mkdir(parents=True)
    binary, audio = study / "soundex-context-replay", study / "input-48000-2.sxa"
    models = {k: study / "exports" / k / "diagnostic.onnx" for k in ("global", "band")}
    schedule = []
    for round_index, labels in enumerate((("global", "band"), ("band", "global"))):
        for kind in labels:
            for enabled in (False, True) if round_index == 0 else (True, False):
                schedule.append({"round": round_index, "kind": kind, "thread_cpu": enabled})
    hashes = {str(p): sha256_file(p) for p in (binary, audio, Path(__file__), *models.values())}
    registered(
        {
            "scope": "fixed eight-run probe-on/off attribution; no concurrent SoundEx training/build/tests; ordinary host load uncontrolled; no retry or replacement of P62 failures",
            "schedule": schedule,
            "hashes": hashes,
            "warmup_hops": 100,
            "scored_hops_per_run": 1000,
            "policy": "same P62 binary/graphs/PCM/QoS; CPU probe off/on order reversed in round two; probe overhead reported, never subtracted; no time-constraint scheduling",
        },
        output / "registration.json",
    )
    rows, references = [], {}
    for index, spec in enumerate(schedule):
        prefix = output / f"{index:02d}"
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-l",
                str(binary),
                str(models[spec["kind"]]),
                str(audio),
                str(prefix.with_suffix(".sxa")),
                str(prefix.with_suffix(".json")),
                "11",
            ],
            env={
                **environment(paced=True),
                "SOUNDEX_DIAGNOSTIC_THREAD_CPU": str(int(spec["thread_cpu"])),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        prefix.with_suffix(".stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError("instrumented native run failed; retain all artifacts")
        raw = read(prefix.with_suffix(".json"))
        row = {**spec, **summarize(raw, result.stderr)}
        if (
            row["measured_hops"] != 1000
            or not row["finite"]
            or raw["thread_cpu_requested"] != spec["thread_cpu"]
        ):
            raise ValueError("probe geometry/policy/nonfinite failure")
        if spec["thread_cpu"]:
            cpu = np.asarray(raw["cpu_timings"], dtype=float)[100:]
            if (
                cpu.shape != (1000, 3)
                or not np.isfinite(cpu).all()
                or len(raw["wake_lateness_ms"]) != 1100
            ):
                raise ValueError("invalid CPU/wake trace")
            row["cpu_stages"] = {
                name: distribution(cpu[:, j]) for j, name in enumerate(raw["cpu_timing_columns"])
            }
            row["wake_lateness_ms"] = distribution(raw["wake_lateness_ms"][100:])
            row["probe_wall_overhead_ms"] = distribution(
                np.asarray(raw["cpu_clock_calibration"])[:, 0]
            )
            row["outliers"] = [
                {
                    "absolute_hop": j,
                    "wake_ms": raw["wake_lateness_ms"][j],
                    "compute_ms": raw["timings"][j][-1],
                    "cpu": raw["cpu_timings"][j],
                    "finish_late_ms": raw["paced_finish_lateness_ms"][j],
                }
                for j in range(100, 1100)
                if raw["paced_finish_lateness_ms"][j] > 0 or raw["timings"][j][-1] >= 128000 / 48000
            ]
        pcm, rate = read_sxa(prefix.with_suffix(".sxa"))
        if rate != 48000 or pcm.shape != (12800, 2):
            raise ValueError("PCM timeline drift")
        reference = references.setdefault(spec["kind"], pcm)
        row["pcm_max_error"] = float(np.max(np.abs(pcm - reference)))
        if row["pcm_max_error"] > 2e-5:
            raise ValueError("probe changes output PCM")
        rows.append(row)
        write_json_report({"status": "running", "rows": rows}, output / "progress.json")
        print(index + 1, spec, "late", row["paced_late_hops"], flush=True)
    verify(hashes)
    write_json_report(
        {"status": "complete", "rows": rows, "original_p62_failures_retained": True},
        output / "summary.json",
    )
    registered(
        {"files": {str(p): sha256_file(p) for p in output.rglob("*") if p.is_file()}},
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
