#!/usr/bin/env python3
"""P62 fixed nonzero prototype feasibility; never a model-quality evaluation.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import torch

from evaluation.adapter_runtime import cost, environment, export
from evaluation.head_export import cartesian
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.band_tcn import make_prototype
from models.envelope_probe import context_features
from scripts.audit_context_deployment import unpack_trace
from scripts.run_consistency_followup import read


@torch.no_grad()
def native_parity(model, graph, binary, directory, rate):
    directory.mkdir(parents=True)
    signal = (0.2 * np.sin(np.arange(128 * 100) * 0.17)).astype(np.float32)
    outputs, errors = [], []
    for label, pcm, channels in (
        ("mono", signal, 1),
        ("stereo", np.stack((np.zeros_like(signal), signal), 1), 2),
    ):
        source, destination, report_path = [
            directory / f"{label}.{suffix}" for suffix in ("input.sxa", "output.sxa", "json")
        ]
        write_sxa(source, pcm, rate)
        result = subprocess.run(
            [
                str(binary),
                str(graph),
                str(source),
                str(destination),
                str(report_path),
                "1",
                "trace",
            ],
            env=environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        (directory / f"{label}.stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError("native parity failed; retain trace and stderr")
        audio, actual_rate = read_sxa(destination)
        if actual_rate != rate or audio.shape != (len(signal), channels):
            raise ValueError("native PCM timeline mismatch")
        trace = unpack_trace(read(report_path), channels)
        context = context_features(torch.from_numpy(pcm.reshape(-1, channels).T.copy()), "past")
        torch.testing.assert_close(trace["context"], context, atol=3e-6, rtol=2e-5)
        prediction, _ = model(
            trace["features"], trace["context"], trace["condition"], model.initial_state(channels)
        )
        observed, expected = [
            cartesian(v.numpy())[:, 80:] for v in (trace["prediction"], prediction)
        ]
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=1e-4)
        errors.append(float(np.max(np.abs(observed - expected))))
        outputs.append(audio)
    np.testing.assert_allclose(outputs[0][:, 0], outputs[1][:, 1], atol=2e-6, rtol=1e-5)
    quiet_peak = float(np.max(np.abs(outputs[1][:, 0])))
    if quiet_peak >= 1e-9:
        raise ValueError("silent physical channel contaminated")
    report = {"rate": rate, "cartesian_max_errors": errors, "silent_channel_peak": quiet_peak}
    write_json_report(report, directory / "report.json")
    return report


def run(args):
    torch.set_num_threads(1)
    root, binary = args.output.resolve(), args.binary.resolve()
    if root.exists():
        raise ValueError("retain existing attempts; choose an unused output directory")
    root.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        if set(relative.parts) & {".venv", "outputs", "__pycache__"}:
            continue
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for pattern in ("*.json", "*.toml", "requirements*.txt", "constraints*.txt", "configs/*.yaml"):
        for path in source.glob(pattern):
            destination = root / "source" / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    repository = source.parent
    rust_sources = [repository / "Cargo.toml", repository / "Cargo.lock"]
    rust_sources.extend(
        p for p in (repository / "crates").rglob("*") if p.suffix in {".rs", ".toml"}
    )
    for path in rust_sources:
        destination = root / "source-rust" / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    frozen_binary = root / binary.name
    shutil.copy2(binary, frozen_binary)
    protocol = {
        "study": "p62-band-tcn-feasibility",
        "seed": 62,
        "training_updates": 0,
        "scope": "fresh deterministic nonzero heads; feasibility only, no audio quality ranking",
        "models": ["global", "band"],
        "token_width": 24,
        "bands": 8,
        "compute_limit_ms": 5,
        "hop_frames": 128,
        "fixed_buffer_frames": 256,
        "warmup_hops": 100,
        "scored_hops_per_unpaced_case": 3000,
        "scored_hops_per_paced_case": 1000,
        "rates": [44100, 48000],
        "channels": [1, 2],
        "paced": "48000 Hz stereo, one run per model",
        "rule": "retain every timing; no retry-to-pass; synthetic pacing is not device certification",
        "machine": platform.platform(),
        "processor": platform.machine(),
        "source_hashes": {
            str(p.relative_to(root)): sha256_file(p)
            for directory in (root / "source", root / "source-rust")
            for p in directory.rglob("*")
            if p.is_file()
        },
        "binary_sha256": sha256_file(frozen_binary),
    }
    write_json_report(protocol, root / "registration.json")
    graphs, exports, parity = {}, {}, {}
    for kind in protocol["models"]:
        model = make_prototype(kind, nonzero=True)
        directory = root / "exports" / kind
        report = export(model, directory)
        graph_path = directory / "diagnostic.onnx"
        graph = onnx.load(graph_path)
        properties = {p.key: p.value for p in graph.metadata_props}
        properties.update({"soundex.study": protocol["study"], "soundex.prototype": kind})
        onnx.helper.set_model_props(graph, properties)
        onnx.save(graph, graph_path)
        report.update(
            onnx_sha256=sha256_file(graph_path),
            state_values=model.state_size,
            study=protocol["study"],
        )
        write_json_report(report, directory / "report.json")
        torch.save(
            {"prototype": kind, "seed": 62, "nonzero_stress": True, "model": model.state_dict()},
            directory / "fixture.pth",
        )
        graphs[kind], exports[kind] = graph_path, report
        parity[kind] = [
            native_parity(
                model, graph_path, frozen_binary, root / "parity" / kind / str(rate), rate
            )
            for rate in protocol["rates"]
        ]
        print(f"{kind}: export and native parity complete", flush=True)
    schedule = []
    for rate in protocol["rates"]:
        for channels in protocol["channels"]:
            t = np.arange(12800) / rate
            waves = [
                sum(np.sin(2 * np.pi * f * t + c * 0.7) for f in (173, 431, 1291, 3917))
                * 0.07
                * (0.7 + 0.3 * np.cos(2 * np.pi * 5 * t + c))
                for c in range(channels)
            ]
            write_sxa(
                root / f"input-{rate}-{channels}.sxa", np.stack(waves, 1).astype(np.float32), rate
            )
            for kind in protocol["models"]:
                schedule.append(dict(kind=kind, rate=rate, channels=channels, paced=False))
    schedule.extend(dict(kind=k, rate=48000, channels=2, paced=True) for k in protocol["models"])
    write_json_report(
        {
            "schedule": schedule,
            "graphs": {k: sha256_file(p) for k, p in graphs.items()},
            "inputs": {p.name: sha256_file(p) for p in root.glob("input-*.sxa")},
        },
        root / "timing-registration.json",
    )
    results = []
    for index, spec in enumerate(schedule):
        before = os.getloadavg()
        result = cost(
            frozen_binary,
            graphs[spec["kind"]],
            root / f"input-{spec['rate']}-{spec['channels']}.sxa",
            root / "timing" / f"{index:02d}",
            paced=spec["paced"],
        )
        results.append(
            {**spec, **result, "host_load_before": before, "host_load_after": os.getloadavg()}
        )
        print(f"timing {index + 1}/{len(schedule)}: {spec}", flush=True)
    summary = {
        "status": "complete",
        "training_updates": 0,
        "exports": exports,
        "native_parity": parity,
        "timing": results,
    }
    summary["all_compute_under_5ms"] = all(
        r["stages"]["total_ms"]["over_5ms"] == 0 for r in results
    )
    summary["all_service_under_hop"] = all(
        r["stages"]["total_ms"]["service_over_hop"] == 0 for r in results
    )
    summary["all_paced_completions_on_time"] = all(
        r["paced_late_hops"] == 0 for r in results if r["paced"]
    )
    write_json_report(summary, root / "summary.json")
    write_json_report(
        {
            "artifacts": {
                str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()
            }
        },
        root / "receipt.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    run(parser.parse_args())
