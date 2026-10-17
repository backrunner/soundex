#!/usr/bin/env python3
"""P25 diagnosis: all fixed final P24 endpoints, real Rust input -> ORT -> DSP.

No quality selection, training, threshold search or production promotion.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.absolute_audio import compare_pcm
from evaluation.head_export import cartesian
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.budget_objective import budget_target
from models.causal_objective import render
from models.envelope_probe import probe_inputs
from models.restoration_head import load_restoration_head, predict_budget, predict_head
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered
from scripts.run_subset_study import Pool


def unpack_trace(report, channels=1):
    traces = report["traces"]
    if report["channels"] != channels or report["hop_samples"] != 128 or not report["finite"]:
        raise ValueError("invalid runtime report")
    result = {}
    for name, shape in (
        ("features", (channels, 2, 1, 129)),
        ("prediction", (channels, 2, 1, 129)),
        ("context", (channels, 1, 1538)),
        ("condition", (channels, 1, 3)),
    ):
        values = [torch.tensor(t[name], dtype=torch.float32).reshape(shape) for t in traces]
        result[name] = torch.cat(values, dim=2 if name in {"features", "prediction"} else 1)
        if not torch.isfinite(result[name]).all():
            raise ValueError("nonfinite runtime trace")
    return result


def error_rms(a, b):
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=np.float64) - b))))


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    root, output, binary = args.study.resolve(), args.output.resolve(), args.binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read(root / "registration.json")
    if read(root / "summary.json")["status"] != "complete":
        raise ValueError("completed parent required")
    # Parent validation uses its own frozen source, not this new audit's code.
    for name, expected in config["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("parent input drift")
    for name, expected in config["source_hashes"].items():
        if sha256_file(root.parent / "source/training" / name) != expected:
            raise ValueError("parent source drift")
    paths = {
        k: v
        for k, v in read(root / "review-registration.json").items()
        if f"-{config['primary_endpoint']}-" in k
    }
    if len(paths) != len(config["arms"]) * len(config["seeds"]):
        raise ValueError("incomplete final endpoint set")
    inputs = {
        str(binary): sha256_file(binary),
        str(root / "summary.json"): sha256_file(root / "summary.json"),
    }
    for label, value in paths.items():
        inputs[value["path"]] = value["sha256"]
        onnx = root / "exports" / label / "diagnostic.onnx"
        inputs[str(onnx)] = read(onnx.parent / "report.json")["onnx_sha256"]
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not set(p.relative_to(source).parts) & {".venv", "tests", "__pycache__", "outputs"}
    }
    registration = {
        "scope": "post-hoc reused development; all final endpoints; no model selection or release",
        "input_hashes": inputs,
        "source_hashes": hashes,
        "cases": paths,
        "specs": config["development"],
        "policy": "all 89 pairs and both roles; same warmup/crop; no threshold exclusions; zero budget is an oracle ablation not a proposed runtime gate",
    }
    verify_inputs(registration)
    registered(registration, output / "registration.json")
    old_inputs = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in old_inputs if p.name == "manifest.jsonl"])
    detector = next(p for p in old_inputs if p.name == "soundex-gain-data")
    parent = next(p for p in old_inputs if "rendered-p10" in str(p) and p.name == "config.json")
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    pool = Pool(config["development"], clips)
    for label, value in sorted(paths.items()):
        records = []
        destination = output / f"{label}.json"
        if destination.exists():
            raise ValueError("use a new audit directory; do not overwrite completed evidence")
        model = load_restoration_head(
            torch.load(value["path"], map_location="cpu", weights_only=False)
        ).eval()
        onnx = root / "exports" / label / "diagnostic.onnx"
        for i, row in enumerate(pool.rows):
            objective, context = pool.select([i])
            b, region = objective.batch, objective.region
            old_prediction = predict_head(model, b, context)
            old_pcm = objective.render(old_prediction).numpy()
            budget_info = None
            if model.generation_budget:
                _, budget = predict_budget(model, b, context)
                target = budget_target(b)
                # Diagnostic only: zero the declared budget pathway, leaving weights intact.
                # Global scope retains local-power units; residual scope removes all units.
                hook = model.budget_head.register_forward_hook(
                    lambda _m, _a, out: torch.full_like(out, -torch.inf)
                )
                try:
                    zero_pcm = objective.render(predict_head(model, b, context)).numpy()
                finally:
                    hook.remove()
                budget_info = (
                    budget[:, 9:].mean(1).tolist(),
                    target[:, 9:].mean(1).tolist(),
                    zero_pcm,
                )
            for role, name in enumerate(("paired", "identity")):
                batch = {k: v[role : role + 1] for k, v in b.items()}
                rate = int(batch["rate"][0, 0])
                with tempfile.TemporaryDirectory(prefix="soundex-context-audit-") as tmp:
                    work = Path(tmp)
                    write_sxa(work / "input.sxa", batch["pcm"][0].numpy(), rate)
                    subprocess.run(
                        [
                            str(binary),
                            str(onnx),
                            str(work / "input.sxa"),
                            str(work / "output.sxa"),
                            str(work / "report.json"),
                            "1",
                            "trace",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    audio, actual_rate = read_sxa(work / "output.sxa")
                    report = read(work / "report.json")
                if actual_rate != rate or audio.shape != (batch["pcm"].shape[-1], 1):
                    raise ValueError("output timeline mismatch")
                trace = unpack_trace(report)
                _, _, condition = probe_inputs(batch, context[role : role + 1])
                condition_error = float((trace["condition"] - condition).abs().max())
                if condition_error > 1e-6:
                    raise ValueError(f"detector/clock mismatch: {condition_error}")
                runtime_batch = {**batch, "x": trace["features"]}
                torch_runtime = predict_head(model, runtime_batch, trace["context"])
                a, expected = (
                    cartesian(trace["prediction"].numpy()),
                    cartesian(torch_runtime.numpy()),
                )
                same_input_error = float(np.max(np.abs(a[:, 89:] - expected[:, 89:])))
                synthesized = render(
                    trace["prediction"],
                    trace["features"],
                    batch["mask"],
                    batch["active"],
                    batch["gate"],
                    batch["dry"],
                    limiter_mode="smooth-knee",
                )[0][0].numpy()
                parity = float(np.max(np.abs(synthesized[region] - audio[region, 0])))
                # Keep discrepancies explicit; do not quietly relax synthesis tolerance.
                clean = batch["clean_wave"][0, region].numpy()
                baseline = objective.baseline[role, region].numpy()
                record = {
                    "index": i,
                    "spec": pool.specs[i],
                    "track_id": row["track_id"],
                    "corpus": row["corpus"],
                    "role": name,
                    "condition_max_error": condition_error,
                    "context_relative_rms_error": error_rms(
                        trace["context"].numpy(), context[role : role + 1].numpy()
                    )
                    / max(error_rms(context[role : role + 1].numpy(), 0), 1e-20),
                    "same_input_ort_cartesian_max_error": same_input_error,
                    "synthesis_max_error": parity,
                    "synthesis_parity_passed": parity <= 2e-5,
                    "native_vs_clean": compare_pcm(audio[region, 0], clean, rate),
                    "native_vs_identity_dsp": compare_pcm(audio[region, 0], baseline, rate),
                    "native_vs_old_proxy": compare_pcm(
                        audio[region, 0], old_pcm[role, region], rate
                    ),
                    "active_fraction": float(batch["active"][0, 89:].float().mean()),
                }
                if budget_info:
                    record["budget"] = {
                        "scope": model.budget_scope,
                        "predicted_band_mean": budget_info[0][role],
                        "teacher_band_mean": budget_info[1][role],
                        "zero_global_budget_vs_identity_dsp": compare_pcm(
                            budget_info[2][role, region], baseline, rate
                        )["error"],
                        "zero_global_budget_vs_clean": compare_pcm(
                            budget_info[2][role, region], clean, rate
                        )["error"],
                    }
                records.append(record)
            if (i + 1) % 16 == 0:
                print(label, i + 1, "/", len(pool.rows), flush=True)
        write_json_report({"checkpoint_sha256": value["sha256"], "rows": records}, destination)
        print("complete", label, len(records), flush=True)
    verify_inputs(registration)
    if any(sha256_file(source / p) != v for p, v in hashes.items()):
        raise ValueError("audit source drift")
    write_json_report(
        {
            "status": "complete",
            "roles": len(paths) * len(pool.rows) * 2,
            "registration_sha256": sha256_file(output / "registration.json"),
            "reports": {p.name: sha256_file(p) for p in output.glob("*-16192-*.json")},
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    run(parser.parse_args())
