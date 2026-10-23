#!/usr/bin/env python3
"""P40 paired CPU-clock on/off diagnostic; preserve every timing failure."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from scripts.bench_context_deployment import summarize
from scripts.run_consistency_followup import read, registered


def run(args: argparse.Namespace) -> None:
    study, output, binary = (p.resolve() for p in (args.study, args.output, args.binary))
    if platform.system() != "Darwin":
        raise ValueError("macOS /usr/bin/time diagnostic required")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("complete all registered endpoints before timing")
    if output.exists():
        raise ValueError("choose a new output directory; never overwrite timing evidence")
    endpoints = read(study / "review-registration.json")
    labels = sorted(endpoints)
    if len(labels) != 4:
        raise ValueError("this balanced four-period diagnostic requires four endpoints")
    models = {label: study / "exports" / label / "diagnostic.onnx" for label in labels}
    audio = study / "native-cost/input-48000-2.sxa"
    reference_root = args.reference.resolve()
    reference_registration = read(reference_root / "registration.json")
    references_by_label = {}
    for index, spec in enumerate(reference_registration["schedule"]):
        references_by_label.setdefault(spec["label"], reference_root / f"{index:02d}.sxa")
    hashes = {
        str(p): sha256_file(p)
        for p in (binary, audio, Path(__file__), study / "review-registration.json")
    }
    for label, endpoint in endpoints.items():
        checkpoint = Path(endpoint["path"])
        if sha256_file(checkpoint) != endpoint["sha256"]:
            raise ValueError("checkpoint drift")
        hashes[str(checkpoint)] = endpoint["sha256"]
        model = models[label]
        hashes[str(model)] = read(model.parent / "report.json")["onnx_sha256"]
        if sha256_file(model) != hashes[str(model)]:
            raise ValueError("export drift")
    for path in references_by_label.values():
        hashes[str(path)] = sha256_file(path)
    # Two rounds reverse both endpoint and probe order. Every endpoint receives
    # off/on then on/off in separate fresh processes, without selecting a best run.
    schedule = []
    for round_index, ordered in enumerate((labels, list(reversed(labels)))):
        for label in ordered:
            for enabled in (False, True) if round_index == 0 else (True, False):
                schedule.append({"round": round_index, "label": label, "thread_cpu": enabled})
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "crates/soundex-core/src/bin/soundex-context-replay.rs",
        "crates/soundex-core/src/bin/diagnostic_timing/mod.rs",
        "crates/soundex-core/Cargo.toml",
        "Cargo.lock",
    ):
        path = repo / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(path)] = sha256_file(path)
    frozen_binary = output / "binaries" / binary.name
    frozen_binary.parent.mkdir()
    shutil.copy2(binary, frozen_binary)
    binary = frozen_binary
    hashes[str(binary)] = sha256_file(binary)
    registered(
        {
            "scope": "P40 paired probe-on/off synthetic paced CPU diagnostic; no concurrent SoundEx training/build during measurements; no imposed load; ordinary OS/application activity uncontrolled; not callback/device qualification",
            "schedule": schedule,
            "hashes": hashes,
            "machine": platform.platform(),
            "warmup_hops": 100,
            "measured_hops_per_run": 10000,
            "policy": "USER_INITIATED QoS; no time-constraint policy; current fusion/native detector",
            "acceptance": "retain every run, service >= hop and >=5 ms counts, paced finish lateness, enclosing wall/thread CPU gaps and output parity; no overhead subtraction or retry-to-pass",
            "reference_pcm": {k: str(v) for k, v in references_by_label.items()},
        },
        output / "registration.json",
    )
    rows, references, parity = [], {}, {}
    for i, spec in enumerate(schedule):
        stem = output / f"{i:02d}"
        before = os.getloadavg()
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-l",
                str(binary),
                str(models[spec["label"]]),
                str(audio),
                str(stem.with_suffix(".sxa")),
                str(stem.with_suffix(".json")),
                "101",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "SOUNDEX_DIAGNOSTIC_THREAD_CPU": str(int(spec["thread_cpu"])),
                "SOUNDEX_DIAGNOSTIC_QOS": "1",
                "SOUNDEX_DIAGNOSTIC_PACED": "1",
                "SOUNDEX_DIAGNOSTIC_TIME_CONSTRAINT": "0",
                "SOUNDEX_DIAGNOSTIC_FUSION": "current",
                "SOUNDEX_DIAGNOSTIC_CONDITION": "native",
            },
        )
        stem.with_suffix(".time.txt").write_text(result.stderr)
        raw = read(stem.with_suffix(".json"))
        if (
            not raw["paced"]
            or raw["time_constraint_requested"]
            or raw["thread_cpu_requested"] != spec["thread_cpu"]
        ):
            raise ValueError("timing policy drift")
        row = {
            **spec,
            **summarize(raw, result.stderr),
            "host_load_before": before,
            "host_load_after": os.getloadavg(),
            "raw_sha256": sha256_file(stem.with_suffix(".json")),
        }
        if spec["thread_cpu"]:
            from evaluation.head_export import distribution

            cpu = np.asarray(raw["cpu_timings"], dtype=float)[100:]
            if cpu.shape != (10000, 3) or not np.isfinite(cpu).all() or (cpu[:, 1] < 0).any():
                raise ValueError("invalid thread CPU observations")
            row["cpu_stages"] = {
                k: distribution(cpu[:, j]) for j, k in enumerate(raw["cpu_timing_columns"])
            }
            row["probe_wall_overhead_ms"] = distribution(
                np.asarray(raw["cpu_clock_calibration"])[:, 0]
            )
            row["wake_lateness_ms"] = distribution(raw["wake_lateness_ms"][100:])
            total = np.asarray(raw["timings"])[100:, 3]
            late = np.asarray(raw["paced_finish_lateness_ms"])[100:]
            outliers = np.flatnonzero((total >= 128000 / 48000) | (late > 0))
            row["outliers"] = [
                {
                    "measured_hop": int(j),
                    "stage_wall_ms": raw["timings"][j + 100],
                    "cpu": cpu[j].tolist(),
                    "wake_lateness_ms": raw["wake_lateness_ms"][j + 100],
                    "finish_lateness_ms": float(late[j]),
                }
                for j in outliers
            ]
        if not row["finite"]:
            raise ValueError("nonfinite output")
        pcm, rate = read_sxa(stem.with_suffix(".sxa"))
        if rate != 48000:
            raise ValueError("timeline drift")
        label = spec["label"]
        if label not in references:
            references[label], reference_rate = read_sxa(references_by_label[label])
            if reference_rate != rate:
                raise ValueError("reference rate drift")
        if references[label].shape != pcm.shape:
            raise ValueError("output shape drift")
        error = float(np.max(np.abs(references[label] - pcm)))
        parity[label] = max(parity.get(label, 0), error)
        if error > 2e-5:
            raise ValueError("output parity failure")
        rows.append(row)
        write_json_report(
            {"status": "running", "rows": rows, "completed": len(rows), "total": len(schedule)},
            output / "progress.json",
        )
        print(
            i + 1, spec, row["stages"]["total_ms"], "paced late", row["paced_late_hops"], flush=True
        )
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("source/input drift")
    write_json_report(
        {
            "status": "complete",
            "rows": rows,
            "audio_max_error": parity,
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "binary", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
