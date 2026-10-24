"""P48 nonzero ONNX and actual Rust model-plus-DSP checks.

SPDX-License-Identifier: Apache-2.0
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import onnx
import torch

from evaluation.fidelity_calibration import assess_wave
from evaluation.head_export import export_graph, verify
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from scripts.bench_context_deployment import summarize
from scripts.run_consistency_followup import read


def export(model, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "diagnostic.onnx"
    digest = hashlib.sha256(getattr(model, "support", "base_control").encode())
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    signature = digest.hexdigest()
    if (directory / "report.json").exists():
        report = read(directory / "report.json")
        if report["model_signature"] != signature or report["onnx_sha256"] != sha256_file(path):
            raise ValueError("existing export drift")
        return report
    if path.exists():
        raise ValueError("incomplete export retained; do not overwrite")
    session = export_graph(model, path)
    parity = verify(model, session)
    del session
    graph = onnx.load(path)
    properties = {p.key: p.value for p in graph.metadata_props}
    properties.update(
        {
            "soundex.study": "p48-frozen-feature-adapter",
            "soundex.adapter_support": getattr(model, "support", "base_control"),
            "soundex.lookahead_samples": "0",
            "soundex.hop_samples": "128",
            "soundex.state_clock": "past-and-current-per-physical-channel",
        }
    )
    onnx.helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, path)
    memory = read_resource(path)
    report = {
        "model_signature": signature,
        "onnx_sha256": sha256_file(path),
        "onnx_bytes": path.stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parity": parity,
        "isolated_memory": memory,
    }
    write_json_report(report, directory / "report.json")
    return report


def read_resource(path: Path) -> dict:
    import json

    return json.loads(
        subprocess.check_output(
            [sys.executable, str(Path(__file__).with_name("onnx_resources.py")), str(path)],
            text=True,
        )
    )


def environment(*, paced=False) -> dict:
    return {
        **os.environ,
        "SOUNDEX_DIAGNOSTIC_QOS": "1",
        "SOUNDEX_DIAGNOSTIC_PACED": str(int(paced)),
        "SOUNDEX_DIAGNOSTIC_TIME_CONSTRAINT": "0",
        "SOUNDEX_DIAGNOSTIC_THREAD_CPU": "0",
        "SOUNDEX_DIAGNOSTIC_FUSION": "current",
        "SOUNDEX_DIAGNOSTIC_CONDITION": "native",
    }


def cost(binary: Path, model: Path, audio: Path, output: Path, *, paced=False) -> dict:
    if (output / "report.json").exists():
        report = read(output / "report.json")
        if (
            report["model_sha256"] != sha256_file(model)
            or report["audio_sha256"] != sha256_file(audio)
            or report["paced"] != paced
            or report["raw_sha256"] != sha256_file(output / "raw.json")
        ):
            raise ValueError("recorded native cost drift")
        return report
    if output.exists() and any(output.iterdir()):
        raise ValueError("incomplete native timing retained; do not retry-to-pass")
    output.mkdir(parents=True, exist_ok=True)
    repeats = 11 if paced else 31  # Registered 100-hop stimulus: 100 warmup + 1000/3000 scored.
    result = subprocess.run(
        [
            "/usr/bin/time",
            "-l",
            str(binary),
            str(model),
            str(audio),
            str(output / "output.sxa"),
            str(output / "raw.json"),
            str(repeats),
        ],
        env=environment(paced=paced),
        check=False,
        capture_output=True,
        text=True,
    )
    (output / "time.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"native runtime failed ({result.returncode}): {result.stderr}")
    raw = read(output / "raw.json")
    report = summarize(raw, result.stderr)
    if not report["finite"] or report["measured_hops"] != (1000 if paced else 3000):
        raise ValueError("native timing geometry/finite failure")
    report.update(
        {
            "paced": paced,
            "model_sha256": sha256_file(model),
            "audio_sha256": sha256_file(audio),
            "raw_sha256": sha256_file(output / "raw.json"),
            "scope": "Rust causal frontend + ORT + DSP; synthetic pacing, no actual audio callback/device certification; fixed adapter buffering 256 frames excluded from compute",
        }
    )
    write_json_report(report, output / "report.json")
    return report


def native_review(binary: Path, models: dict, items: list, originals: list, output: Path) -> dict:
    """Registered eight evenly spaced batches, mono views only; retain all PCM.

    PCM targets/timeline match the existing Rust context review. This does not
    relabel independent channel views as simultaneous stereo music evidence.
    """
    results = {}
    for label, model in models.items():
        path = output / label / "report.json"
        if path.exists():
            previous = read(path)
            if previous["model_sha256"] != sha256_file(model):
                raise ValueError("native quality export drift")
            for name, expected in previous["artifacts"].items():
                if sha256_file(Path(name)) != expected:
                    raise ValueError("native quality artifact drift")
            results[label] = {"report": str(path), "sha256": sha256_file(path)}
            continue
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("incomplete native review retained; choose a separate retry directory")
        cases = []
        for index in range(0, len(items), 4):
            objective = items[index][0]
            batch, region = objective.batch, objective.region
            rendered = []
            for role in range(len(batch["pcm"])):
                destination = output / label / f"{index:02d}-{role:02d}"
                destination.mkdir(parents=True, exist_ok=True)
                rate = int(batch["rate"][role, 0])
                write_sxa(destination / "input.sxa", batch["pcm"][role].numpy(), rate)
                result = subprocess.run(
                    [
                        str(binary),
                        str(model),
                        str(destination / "input.sxa"),
                        str(destination / "output.sxa"),
                        str(destination / "runtime.json"),
                        "1",
                    ],
                    env=environment(),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                (destination / "stderr.txt").write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(
                        f"native quality failed ({result.returncode}): {result.stderr}"
                    )
                wave, actual_rate = read_sxa(destination / "output.sxa")
                runtime = read(destination / "runtime.json")
                if (
                    actual_rate != rate
                    or wave.shape != (batch["pcm"].shape[-1], 1)
                    or not runtime["finite"]
                ):
                    raise ValueError("native output timeline/finite failure")
                rendered.append(torch.from_numpy(wave[:, 0]))
            wave = torch.stack(rendered)
            cases.append(
                {
                    "index": originals[index]["index"],
                    "samples": originals[index]["samples"],
                    "identity": batch["identity"].bool().tolist(),
                    "native": assess_wave(
                        wave[:, region], batch["clean_wave"][:, region], batch["rate"]
                    ),
                    "untreated": assess_wave(
                        batch["dry"][:, region], batch["clean_wave"][:, region], batch["rate"]
                    ),
                }
            )
        artifacts = {str(p): sha256_file(p) for p in path.parent.rglob("*") if p.is_file()}
        write_json_report(
            {"model_sha256": sha256_file(model), "cases": cases, "artifacts": artifacts}, path
        )
        results[label] = {"report": str(path), "sha256": sha256_file(path)}
        print("native review", label, "complete", flush=True)
    write_json_report(results, output / "summary.json")
    return results
