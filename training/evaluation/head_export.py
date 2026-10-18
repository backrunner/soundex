"""Diagnostic one-hop ONNX for P20. Context/detector inputs are not a runtime ABI."""

import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from artifact_contract import MAX_ARTIFACT_BYTES, MAX_PARAMETERS
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_probe import StreamingContext
from models.restoration_head import load_restoration_head
from scripts.run_consistency_followup import complete, read, verify_receipt

NAMES = ("features", "context", "condition", "state", "output", "next_state")


def session_for(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def cartesian(features):
    return np.power(10.0, features[:, 0] / 20) * np.exp(1j * features[:, 1])


def verify(model, session):
    rng = np.random.default_rng(20261014)
    maximum, state_maximum = 0.0, 0.0
    for rate in (44100, 48000):
        for channels in (1, 2):
            state = np.zeros((channels, model.state_size), np.float32)
            reference_state = torch.from_numpy(state.copy())
            for hop in range(80):
                if hop == 70:
                    state.fill(0)
                    reference_state.zero_()
                x = rng.normal(size=(channels, 2, 1, 129)).astype(np.float32)
                x[:, 0] = x[:, 0] * 10 - 40
                context = np.abs(rng.normal(size=(channels, 1, 1538))).astype(np.float32)
                condition = np.tile(np.array([rate / 48000, 0.6, 1], np.float32), (channels, 1, 1))
                if hop < 4:
                    x[0, 0], x[0, 1], context[0] = -200, 0, 0
                with torch.no_grad():
                    expected, reference_state = model(
                        torch.from_numpy(x),
                        torch.from_numpy(context),
                        torch.from_numpy(condition),
                        reference_state,
                    )
                actual, state = session.run(
                    None, dict(zip(NAMES[:4], (x, context, condition, state), strict=True))
                )
                a, b = cartesian(actual), cartesian(expected.numpy())
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(state, reference_state.numpy(), atol=2e-5, rtol=2e-5)
                maximum = max(maximum, float(np.max(np.abs(a - b))))
                state_maximum = max(
                    state_maximum, float(np.max(np.abs(state - reference_state.numpy())))
                )
    return {
        "cartesian_max_error": maximum,
        "state_max_error": state_maximum,
        "sequential_hops": 320,
        "reset": True,
        "silence": True,
    }


def export_graph(model, path, *, experimental_unbounded=False):
    args = (
        torch.full((2, 2, 1, 129), -40.0),
        torch.ones(2, 1, 1538),
        torch.ones(2, 1, 3),
        model.initial_state(2),
    )
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=list(NAMES[:4]),
        output_names=list(NAMES[4:]),
        dynamic_axes={name: {0: "channels"} for name in NAMES},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.helper.set_model_props(
        graph,
        {
            "soundex.experimental": "p20-context-head-diagnostic",
            "soundex.deployable": "false",
            "soundex.head": model.arm,
            "soundex.width": str(model.width),
            "soundex.residual_slope": str(model.residual_slope),
            "soundex.generation_budget": str(model.generation_budget).lower(),
            "soundex.budget_scope": model.budget_scope,
            "soundex.decoder_normalization": str(model.decoder_normalization).lower(),
            "soundex.phase_input": str(model.phase_input).lower(),
        },
    )
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, path)
    if not experimental_unbounded and path.stat().st_size >= MAX_ARTIFACT_BYTES:
        raise ValueError("ONNX budget exceeded")
    return session_for(path)


def distribution(values):
    return {
        name: float(value)
        for name, value in zip(
            ("p50_ms", "p95_ms", "p99_ms", "max_ms"),
            (*np.percentile(values, [50, 95, 99]), max(values)),
            strict=True,
        )
    }


def stream_cost(values, period_ms=128 / 48):
    """Replay measured service times on a fixed arrival clock; not device evidence."""
    backlog, maximum, misses = 0.0, 0.0, 0
    for value in values:
        completion = backlog + value
        misses += completion > period_ms
        backlog = max(0.0, completion - period_ms)
        maximum = max(maximum, backlog)
    return {
        "mean_ms": float(np.mean(values)),
        "virtual_max_backlog_ms": maximum,
        "virtual_final_backlog_ms": backlog,
        "virtual_deadline_misses": misses,
        "period_ms": period_ms,
    }


def measure(model, session, iterations=1000):
    results = {}
    for channels in (1, 2):
        frontend = StreamingContext(channels, "past")
        pcm = torch.full((channels, 128), 0.01)
        feed = {
            "features": np.full((channels, 2, 1, 129), -40, np.float32),
            "context": np.ones((channels, 1, 1538), np.float32),
            "condition": np.ones((channels, 1, 3), np.float32),
            "state": np.zeros((channels, model.state_size), np.float32),
        }
        inference, context_times, combined = [], [], []
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        for i in range(iterations + 100):
            start = time.perf_counter()
            feed["context"] = frontend.step(pcm).numpy()
            middle = time.perf_counter()
            _, feed["state"] = session.run(None, feed)
            end = time.perf_counter()
            if i >= 100:
                context_times.append((middle - start) * 1000)
                inference.append((end - middle) * 1000)
                combined.append((end - start) * 1000)
        results[str(channels)] = {
            "inference": distribution(inference),
            "python_frontend": distribution(context_times),
            "combined": {**distribution(combined), **stream_cost(combined)},
            "measured_hops": iterations,
            "combined_over_48k_hop": sum(v > 128 / 48 for v in combined),
            "cpu_seconds_including_warmup": time.process_time() - cpu_start,
            "wall_seconds_including_warmup": time.perf_counter() - wall_start,
            "state_bytes": int(feed["state"].nbytes),
        }
    return results


def export_and_measure(checkpoint, directory, *, experimental_unbounded=False, iterations=1000):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(checkpoint)
    policy = {"experimental_unbounded": experimental_unbounded, "iterations": iterations}
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        previous = read(directory / "report.json")
        if (
            previous.get(
                "measurement_policy", {"experimental_unbounded": False, "iterations": 1000}
            )
            != policy
        ):
            raise ValueError("export measurement policy drift")
        if previous["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("export checkpoint drift")
        return
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_restoration_head(state).eval()
    parameters = sum(p.numel() for p in model.parameters())
    if not experimental_unbounded and parameters > MAX_PARAMETERS:
        raise ValueError("parameter budget exceeded")
    path = directory / "diagnostic.onnx"
    session = export_graph(model, path, experimental_unbounded=experimental_unbounded)
    report = {
        "checkpoint_sha256": checkpoint_hash,
        "measurement_policy": policy,
        "onnx_sha256": sha256_file(path),
        "parameters": parameters,
        "model_config": state.get("model_config", {"arm": state["arm"]}),
        "onnx_bytes": path.stat().st_size,
        "parity": verify(model, session),
        "cost": measure(model, session, iterations),
        "isolated_memory": json.loads(
            subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("onnx_resources.py")), str(path)],
                text=True,
            )
        ),
        "machine": platform.platform(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "process_peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "whole study process high-water mark; bytes on macOS, KiB on Linux; not incremental model RAM",
        "scope": "CPU ORT one-hop graph plus Python causal FFT frontend; no pacing, callback, device or full Rust deadline certification; diagnostic context ABI only",
        "lookahead_samples": 0,
        "hop_samples": 128,
    }
    write_json_report(report, directory / "report.json")
    complete(directory)
