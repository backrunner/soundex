#!/usr/bin/env python3
"""Frozen P32 stage attribution using actual Rust frontend/ORT predictions.

No DSP override is deployed. Unlimited stages are diagnostic counterfactuals.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.absolute_audio import compare_pcm
from evaluation.multires_review import scores, summarize
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.causal_objective import render
from models.complex_synthesis import complex_features, synthesize
from scripts.audit_context_deployment import unpack_trace
from scripts.run_consistency_followup import read, registered


def stage_waves(prediction, batch):
    final, fused = render(
        prediction,
        batch["x"],
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )
    return {
        "raw_ola": synthesize(complex_features(prediction)),
        "fused_ola": fused,
        # Actual deployment limits each branch BEFORE crossfading. This diagnostic
        # leaves both branches unlimited; it is not an alternate limiter ordering.
        "gated_unlimited": batch["dry"] * (1 - batch["gate"]) + fused * batch["gate"],
        "limited_output": final,
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    study, output, binary = args.study.resolve(), args.output.resolve(), args.binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read(study / "registration.json")
    if read(study / "progress.json")["stage"] != "complete":
        raise ValueError("completed frozen study required")
    endpoints = read(study / "review-registration.json")
    paths = [Path(p) for p in config["input_hashes"]]
    inputs = {**config["input_hashes"], str(binary): sha256_file(binary)}
    for name in ["registration.json", "review-registration.json", "spectrum-review/summary.json"]:
        inputs[str(study / name)] = sha256_file(study / name)
    for label, value in endpoints.items():
        inputs[value["path"]] = value["sha256"]
        onnx = study / "exports" / label / "diagnostic.onnx"
        inputs[str(onnx)] = read(onnx.parent / "report.json")["onnx_sha256"]
    source = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not set(p.relative_to(source).parts) & {".venv", "outputs", "tests", "__pycache__"}
    }
    registration = {
        "scope": "post-hoc stage diagnosis on all fitted P32 crops/endpoints, original unclipped clean reference; not an optimization or deployment change",
        "stages": ["raw_ola", "fused_ola", "gated_unlimited", "limited_output", "native_output"],
        "reference": "original clean for paired AND identity",
        "input_hashes": inputs,
        "source_hashes": sources,
        "checkpoints": endpoints,
        "specs": config["specs"],
        "loss": config["multires_loss"],
    }
    for name, expected in inputs.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("input drift")
    registered(registration, output / "registration.json")
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    pool = LazyPool(
        config["specs"],
        BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            Path(config["cache_path"]),
        ),
    )
    summaries, hashes = {}, {}
    for label in endpoints:
        destination = output / f"{label}.json"
        if destination.exists():
            raise ValueError("do not overwrite stage evidence")
        records, parity = [], []
        for index, row in enumerate(pool.rows):
            objective, _ = pool.select([index])
            for role, name in enumerate(("paired", "identity")):
                batch = {k: v[role : role + 1] for k, v in objective.batch.items()}
                rate = int(batch["rate"][0, 0])
                with tempfile.TemporaryDirectory(prefix="soundex-stage-") as tmp:
                    work = Path(tmp)
                    write_sxa(work / "input.sxa", batch["pcm"][0].numpy(), rate)
                    subprocess.run(
                        [
                            str(binary),
                            str(study / "exports" / label / "diagnostic.onnx"),
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
                    trace = unpack_trace(read(work / "report.json"))
                if actual_rate != rate or audio.shape != (batch["pcm"].shape[-1], 1):
                    raise ValueError("native timeline mismatch")
                stages = stage_waves(trace["prediction"], {**batch, "x": trace["features"]})
                stages["native_output"] = torch.from_numpy(audio.T.copy())
                region = objective.region
                error = float(
                    (stages["limited_output"][:, region] - stages["native_output"][:, region])
                    .abs()
                    .max()
                )
                parity.append(error)
                if error > 2e-5:
                    raise ValueError("native final-stage parity failure")
                for stage, wave in stages.items():
                    records.append(
                        {
                            "spec": pool.specs[index],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "role": name,
                            "stage": stage,
                            "score": scores(wave, batch, region)[0],
                            "pcm": compare_pcm(
                                wave[0, region].numpy(),
                                batch["clean_wave"][0, region].numpy(),
                                rate,
                            ),
                        }
                    )
        summaries[label] = summarize(records)
        write_json_report(
            {
                "registration_sha256": sha256_file(output / "registration.json"),
                "checkpoint_sha256": endpoints[label]["sha256"],
                "native_roles": len(parity),
                "synthesis_max_error": max(parity),
                "summary": summaries[label],
                "rows": records,
            },
            destination,
        )
        hashes[destination.name] = sha256_file(destination)
        print("stage audit complete", label, "parity", max(parity), flush=True)
    for name, expected in inputs.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("input drift")
    for name, expected in sources.items():
        if sha256_file(source / name) != expected:
            raise ValueError("source drift")
    write_json_report(
        {
            "status": "complete",
            "registration_sha256": sha256_file(output / "registration.json"),
            "reports": hashes,
            "cases": summaries,
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
