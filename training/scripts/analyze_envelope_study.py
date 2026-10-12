#!/usr/bin/env python3
"""Recording-paired P5 comparisons and diagnostic ONNX/CPU measurements.

Run only after training has finished: avoids benchmarking against an active run.
This amplitude-head contract is intentionally different from the audio runtime.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from evaluation.reporting import bootstrap_mean_ci, sha256_file, write_json_report
from models.envelope_probe import EnvelopeProbe, StreamingContext

KEYS = (
    "total",
    "amplitude",
    "temporal",
    "high_sc",
    "high_flux",
    "high_energy_error_db",
    "band_0_4000_sc",
)


def paired_comparison(candidate, control):
    baseline = {r["row_id"]: r for r in control["rows"]}
    if set(baseline) != {r["row_id"] for r in candidate["rows"]}:
        raise ValueError("unpaired evaluation")
    output = {}
    for corpus in ("music_library", "speech_library"):
        selected = [r for r in candidate["rows"] if r["corpus"] == corpus]
        result = {}
        for key in KEYS:
            groups = {}
            for row in selected:
                groups.setdefault(row["track_id"], []).append(
                    row["metrics"][key] - baseline[row["row_id"]]["metrics"][key]
                )
            values = [float(np.mean(v)) for v in groups.values()]
            result[key] = {
                "candidate_minus_control": float(np.mean(values)),
                "recording_ci95": bootstrap_mean_ci(values, seed=20261012),
                "recordings": len(values),
                "rows": len(selected),
            }
        output[corpus] = result
    return output


def export_probe(checkpoint_path, destination):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["format"] != "envelope-probe-v1":
        raise ValueError("not an amplitude probe")
    model = EnvelopeProbe().eval()
    model.load_state_dict(checkpoint["model"])
    generator = torch.Generator().manual_seed(20261012)
    inputs = (
        torch.rand(1, 1, 129, generator=generator),
        torch.rand(1, 1, 1538, generator=generator),
        torch.tensor([[[1.0, 0.5, 1.0]]]),
        model.initial_state(1),
    )
    names = ("magnitude", "context", "condition", "state")
    stage = destination.with_suffix(".partial.onnx")
    torch.onnx.export(
        model,
        inputs,
        stage,
        input_names=list(names),
        output_names=["amplitude", "next_state"],
        opset_version=17,
        dynamic_axes={name: {0: "batch"} for name in (*names, "amplitude", "next_state")},
        dynamo=False,
    )
    artifact = onnx.load(stage)
    onnx.helper.set_model_props(
        artifact,
        {
            "soundex.experimental_contract": "envelope-probe-v1",
            "soundex.production_compatible": "false",
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    onnx.checker.check_model(artifact)
    onnx.save(artifact, stage)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(stage), options, providers=["CPUExecutionProvider"])
    max_error = 0.0
    with torch.no_grad():
        for batch in (1, 2):
            state = model.initial_state(batch)
            runtime_state = state.numpy()
            for hop in range(80):
                mag = torch.rand(batch, 1, 129, generator=generator) * (0 if hop < 2 else 1)
                context = torch.rand(batch, 1, 1538, generator=generator)
                condition = torch.tensor([[[44100 / 48000, 0.5, 1.0]]]).expand(batch, -1, -1)
                expected, state = model(mag, context, condition, state)
                actual, runtime_state = session.run(
                    None,
                    dict(
                        zip(
                            names,
                            [mag.numpy(), context.numpy(), condition.numpy(), runtime_state],
                            strict=True,
                        )
                    ),
                )
                np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=3e-6)
                np.testing.assert_allclose(runtime_state, state.numpy(), rtol=5e-5, atol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(actual - expected.numpy()))))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 2_000_000 or stage.stat().st_size >= 8 * 1024 * 1024:
        raise ValueError("diagnostic model exceeds project size budget")
    stage.replace(destination)
    return session, {
        "parameters": parameters,
        "bytes": destination.stat().st_size,
        "onnx_sha256": sha256_file(destination),
        "max_amplitude_error": max_error,
        "production_compatible": False,
    }


def benchmark(session, arm, seconds):
    results = []
    for rate in (44100, 48000):
        for channels in (1, 2):
            frontend = StreamingContext(channels, arm)
            state = np.zeros((channels, 9072), dtype=np.float32)
            window = torch.hann_window(256)
            t = torch.arange(128 * 64) / rate
            wave = (
                0.1 * torch.sin(2 * torch.pi * 440 * t) + 0.01 * torch.sin(2 * torch.pi * 10000 * t)
            ).repeat(channels, 1)
            condition = np.tile(
                np.array([[[rate / 48000, 16000 / (rate / 2), 1.0]]], dtype=np.float32),
                (channels, 1, 1),
            )
            durations, features, inference = [], [], []
            start = time.monotonic()
            cpu_start = time.process_time()
            hop = 0
            with torch.no_grad():
                while time.monotonic() - start < seconds:
                    begin = time.perf_counter_ns()
                    context = frontend.step(wave[:, (hop % 64) * 128 : (hop % 64 + 1) * 128])
                    magnitude = torch.fft.rfft(frontend.history[:, -256:] * window).abs()[:, None]
                    inputs = {
                        "magnitude": magnitude.numpy(),
                        "context": context.numpy(),
                        "condition": condition,
                        "state": state,
                    }
                    feature_end = time.perf_counter_ns()
                    amplitude, state = session.run(None, inputs)
                    end = time.perf_counter_ns()
                    if not np.isfinite(amplitude).all() or not np.isfinite(state).all():
                        raise ValueError("nonfinite benchmark output")
                    if hop >= 96:
                        durations.append((end - begin) / 1e6)
                        features.append((feature_end - begin) / 1e6)
                        inference.append((end - feature_end) / 1e6)
                    hop += 1
            cpu = time.process_time() - cpu_start
            elapsed = time.monotonic() - start
            results.append(
                {
                    "sample_rate": rate,
                    "channels": channels,
                    "measured_hops": len(durations),
                    "wall_seconds": elapsed,
                    "busy_loop_cpu_seconds": cpu,
                    "estimated_one_core_cpu_percent_at_audio_rate": cpu / (hop * 128 / rate) * 100,
                    "feature_p99_ms": float(np.percentile(features, 99)),
                    "inference_p99_ms": float(np.percentile(inference, 99)),
                    "combined_p99_ms": float(np.percentile(durations, 99)),
                    "combined_max_ms": max(durations),
                    "compute_over_hop_budget_count": sum(d > 128000 / rate for d in durations),
                }
            )
    # Peak includes imports, export and parity: do not label it isolated ORT RSS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "cases": results,
        "process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "scope": "unpaced synthetic CPU diagnostic incl. Python FFT frontend; excludes detector, DSP, adapter, device; RSS includes export; not realtime acceptance",
    }


def main(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("training must finish before analysis/benchmark")
    config = json.loads((root / "config.json").read_text())
    report = {"scope": "raw amplitude diagnostic, reused development recordings", "seeds": {}}
    dry = json.loads((root / "compressed-reference.json").read_text())
    for seed in config["seeds"]:
        arms = {}
        for arm in config["arms"]:
            directory = root / f"{arm}-{seed}"
            receipt = json.loads((directory / "complete.json").read_text())
            if sha256_file(directory / "evaluation.json") != receipt["evaluation_sha256"]:
                raise ValueError("evaluation receipt changed")
            arms[arm] = json.loads((directory / "evaluation.json").read_text())
        comparisons = {}
        for control in ("short", "tiled"):
            for split in ("validation", "identity", "train_probe"):
                comparisons[f"past_minus_{control}_{split}"] = paired_comparison(
                    arms["past"][split], arms[control][split]
                )
            comparisons[f"ablated_{control}_minus_past"] = paired_comparison(
                arms["past"]["ablation_" + control], arms["past"]["validation"]
            )
        for arm in config["arms"]:
            comparisons[arm + "_minus_compressed"] = paired_comparison(arms[arm]["validation"], dry)
        report["seeds"][str(seed)] = comparisons
    write_json_report(report, root / "comparisons.json")
    if args.benchmark:
        # Predeclared seed 1 past, no benchmark-driven model selection.
        directory = root / f"past-{config['seeds'][0]}"
        session, exported = export_probe(directory / "best.pth", directory / "diagnostic.onnx")
        write_json_report(
            {"export": exported, "benchmark": benchmark(session, "past", args.seconds)},
            root / "diagnostic-cost.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    options = parser.parse_args()
    if options.seconds < 1:
        parser.error("benchmark case must run at least one second")
    main(options)
