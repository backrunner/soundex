#!/usr/bin/env python3
"""Counterbalance endpoint order without retrying or excluding timing failures."""

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
    # Williams order: each endpoint occupies each position once, with every
    # ordered predecessor pair represented once inside the four periods.
    orders = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    schedule = [
        {"round": r, "position": pos, "label": labels[i]}
        for r, order in enumerate(orders)
        for pos, i in enumerate(order)
    ]
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    registered(
        {
            "scope": "balanced-order same-binary synthetic paced CPU diagnostic; no training; background activity uncontrolled; not hardware callback or device qualification",
            "schedule": schedule,
            "hashes": hashes,
            "machine": platform.platform(),
            "warmup_hops": 100,
            "measured_hops_per_run": 10000,
            "policy": "USER_INITIATED QoS; no time-constraint policy; current fusion/native detector",
            "acceptance": "retain every run, service >= hop and >=5 ms counts, paced finish lateness, stage costs and output parity; no retry-to-pass",
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
                "SOUNDEX_DIAGNOSTIC_QOS": "1",
                "SOUNDEX_DIAGNOSTIC_PACED": "1",
                "SOUNDEX_DIAGNOSTIC_TIME_CONSTRAINT": "0",
                "SOUNDEX_DIAGNOSTIC_FUSION": "current",
                "SOUNDEX_DIAGNOSTIC_CONDITION": "native",
            },
        )
        stem.with_suffix(".time.txt").write_text(result.stderr)
        raw = read(stem.with_suffix(".json"))
        if not raw["paced"] or raw["time_constraint_requested"]:
            raise ValueError("timing policy drift")
        row = {
            **spec,
            **summarize(raw, result.stderr),
            "host_load_before": before,
            "host_load_after": os.getloadavg(),
            "raw_sha256": sha256_file(stem.with_suffix(".json")),
        }
        if not row["finite"]:
            raise ValueError("nonfinite output")
        pcm, rate = read_sxa(stem.with_suffix(".sxa"))
        if rate != 48000:
            raise ValueError("timeline drift")
        label = spec["label"]
        if label not in references:
            references[label] = pcm
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
    for name in ("study", "output", "binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
