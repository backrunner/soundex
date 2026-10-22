#!/usr/bin/env python3
"""P37 frozen P36 weights: independent output-fusion and condition interventions.

All reused recording-isolated clips and both seeds, no optimization, promotion,
threshold fitting, or inference-sensitivity claims about retrained models.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.absolute_audio import compare_pcm
from evaluation.fusion_ablation import CASES, render_ablation
from evaluation.multires_review import scores
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.complex_synthesis import complex_features, synthesize
from scripts.audit_context_deployment import unpack_trace
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import read, registered


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['corpus']}/{row['role']}/{row['stage']}"].append(row)
    summary = {}
    for name, group in groups.items():
        metrics = {}
        for metric in ("total", "amplitude", "log_shape", "envelope", "dynamics", "pcm_rms"):
            tracks = defaultdict(list)
            for row in group:
                value = (
                    row["pcm"]["error"]["rms"]
                    if metric == "pcm_rms"
                    else row["score"]["total"]
                    if metric == "total"
                    else row["score"]["components"][metric]
                )
                tracks[row["track_id"]].append(value)
            records = {k: statistics.mean(v) for k, v in tracks.items()}
            metrics[metric] = {"mean": statistics.mean(records.values()), "recordings": records}
        summary[name] = metrics
    return summary


def comparisons(cases):
    output = {}
    for label, variants in cases.items():
        control = variants["current--native"]
        output[label] = {}
        for variant, groups in variants.items():
            result = {}
            for group, metrics in groups.items():
                result[group] = {}
                for metric, value in metrics.items():
                    old = control[group][metric]
                    if value["recordings"].keys() != old["recordings"].keys():
                        raise ValueError("recording population mismatch")
                    delta = [v - old["recordings"][k] for k, v in value["recordings"].items()]
                    result[group][metric] = {
                        "control": old["mean"],
                        "candidate": value["mean"],
                        "delta_pct": (value["mean"] / old["mean"] - 1) * 100
                        if old["mean"]
                        else None,
                        "median_difference": statistics.median(delta),
                        "wins": sum(v < 0 for v in delta),
                        "losses": sum(v > 0 for v in delta),
                    }
            output[label][variant] = result
    return output


@torch.no_grad()
def run(
    args, *, endpoint_arms=("compressed",), cases=CASES, expected_endpoints=2, require_complete=True
):
    torch.set_num_threads(1)
    study, root = args.study.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "progress.json").exists():
        raise ValueError("do not overwrite prior audit; choose a new output directory")
    cfg = read(study / "registration.json")
    verify_inputs(cfg)
    if require_complete and read(study / "progress.json")["stage"] != "complete":
        raise ValueError("completed frozen parent required")
    endpoints = {
        k: v
        for k, v in read(study / "review-registration.json").items()
        if v["arm"] in endpoint_arms
    }
    if len(endpoints) != expected_endpoints:
        raise ValueError("all requested frozen endpoints required")
    source = Path(__file__).resolve().parents[1]
    repository = Path(cfg.get("repository_root", source.parent))
    sources = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not set(p.relative_to(source).parts) & {".venv", "tests", "outputs", "__pycache__"}
    }
    resources = {"parity_policy.json": sha256_file(source / "parity_policy.json")}
    binary = root / "binaries" / "soundex-context-replay"
    binary.parent.mkdir(exist_ok=True)
    if not binary.exists():
        shutil.copy2(args.binary, binary)
    hashes = {
        str(binary): sha256_file(args.binary),
        str(study / "registration.json"): sha256_file(study / "registration.json"),
    }
    for label, value in endpoints.items():
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("checkpoint drift")
        onnx = study / "exports" / label / "diagnostic.onnx"
        hashes[str(onnx)] = read(onnx.parent / "report.json")["onnx_sha256"]
    indices = [0, 10] if args.preflight else list(range(len(cfg["specs"])))
    registration = {
        "scope": cfg.get("fusion_review_scope", __doc__),
        "parent": str(study),
        "preflight": args.preflight,
        "cases": cases,
        "endpoints": endpoints,
        "indices": indices,
        "hashes": hashes,
        "source_hashes": sources,
        "resource_hashes": resources,
        "rust_source_hashes": {
            str(p.relative_to(repository)): sha256_file(p)
            for p in (repository / "crates").rglob("*.rs")
        },
        "cargo_lock_sha256": sha256_file(repository / "Cargo.lock"),
        "reference": "original unclipped clean for both roles",
        "selection": "all cases frozen before review; no automatic promotion or formal retraining selection; preserve raw/native spectral, dynamics, identity, band PCM and every recording",
        "condition_scope": "fixed cutoff=.5 and active=1 from first warmup hop; input metadata only, unchanged runtime detector/gate; out-of-distribution sensitivity, not no-condition retraining",
        "fusion_scope": "magnitude-only retains magnitude blend; phase-only retains phase blend; unfused retains detector/gate; safety-only forces wet after 80-hop warmup; default untouched",
    }
    registered(registration, root / "registration.json")
    for name, digest in (sources | resources).items():
        destination = root / "source" / "training" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source / name, destination)
        if sha256_file(destination) != digest:
            raise ValueError("snapshot drift")
    paths = [Path(p) for p in cfg["input_hashes"]]
    metadata, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    pool = LazyPool(
        cfg["specs"],
        BoundedEnvelopeClips(
            metadata,
            next(p for p in paths if p.name == "soundex-gain-data"),
            Path(cfg["cache_path"]),
        ),
    )
    interventions = cases
    cases, evidence = {}, {}
    for label in endpoints:
        cases[label] = {}
        prediction_hashes = {}
        for fusion, condition in interventions:
            variant = fusion + "--" + condition
            path = root / label / f"{variant}.json"
            records, parity, signatures = [], [], {}
            for index in indices:
                obj, _ = pool.select([index])
                row = pool.rows[index]
                for role, role_name in enumerate(("paired", "identity")):
                    batch = {k: v[role : role + 1] for k, v in obj.batch.items()}
                    rate = int(batch["rate"][0, 0])
                    with tempfile.TemporaryDirectory(prefix="soundex-fusion-") as td:
                        work = Path(td)
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
                            env={
                                **os.environ,
                                "SOUNDEX_DIAGNOSTIC_FUSION": fusion,
                                "SOUNDEX_DIAGNOSTIC_CONDITION": condition,
                                "SOUNDEX_DIAGNOSTIC_PACED": "0",
                                "SOUNDEX_DIAGNOSTIC_TIME_CONSTRAINT": "0",
                            },
                        )
                        native_report = read(work / "report.json")
                        native, actual_rate = read_sxa(work / "output.sxa")
                        trace = unpack_trace(native_report)
                    if (
                        native_report["fusion_ablation"] != fusion
                        or native_report["condition_ablation"] != condition
                        or actual_rate != rate
                        or native.shape != (batch["pcm"].shape[-1], 1)
                    ):
                        raise ValueError("native intervention/timeline mismatch")
                    if condition in {"fixed-cutoff", "fixed-detector"} and not torch.all(
                        trace["condition"][..., 1] == 0.5
                    ):
                        raise ValueError("cutoff intervention mismatch")
                    if condition in {"active-one", "fixed-detector"} and not torch.all(
                        trace["condition"][..., 2] == 1
                    ):
                        raise ValueError("active intervention mismatch")
                    import hashlib

                    signature = hashlib.sha256(trace["prediction"].numpy().tobytes()).hexdigest()
                    key = f"{index}/{role}"
                    signatures[key] = signature
                    if condition == "native":
                        if fusion == "current":
                            prediction_hashes[key] = signature
                        elif prediction_hashes[key] != signature:
                            raise ValueError("output ablation changed model prediction")
                    batch = {**batch, "x": trace["features"]}
                    proxy, _ = render_ablation(trace["prediction"], batch, fusion)
                    wave = torch.from_numpy(native.T.copy())
                    error = float((proxy[:, obj.region] - wave[:, obj.region]).abs().max())
                    parity.append(error)
                    if error > 2e-5:
                        raise ValueError(
                            f"native proxy failed: {label}/{variant}/{index}/{role}: {error}"
                        )
                    for stage, output in (
                        ("raw_ola", synthesize(complex_features(trace["prediction"]))),
                        ("native_output", wave),
                    ):
                        records.append(
                            {
                                "spec": pool.specs[index],
                                "track_id": row["track_id"],
                                "corpus": row["corpus"],
                                "role": role_name,
                                "stage": stage,
                                "score": scores(output, batch, obj.region)[0],
                                "pcm": compare_pcm(
                                    output[0, obj.region].numpy(),
                                    batch["clean_wave"][0, obj.region].numpy(),
                                    rate,
                                ),
                            }
                        )
            summary = aggregate(records)
            cases[label][variant] = summary
            write_json_report(
                {
                    "registration_sha256": sha256_file(root / "registration.json"),
                    "checkpoint_sha256": endpoints[label]["sha256"],
                    "native_roles": len(parity),
                    "proxy_max_error": max(parity),
                    "prediction_hashes": signatures,
                    "summary": summary,
                    "rows": records,
                },
                path,
            )
            evidence[str(path)] = sha256_file(path)
            write_json_report(
                {
                    "stage": "reviewing",
                    "completed_cases": len(evidence),
                    "total_cases": len(interventions) * len(endpoints),
                    "label": label,
                    "variant": variant,
                },
                root / "progress.json",
            )
            print(label, variant, "complete", flush=True)
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("evidence drift")
    for name, digest in (sources | resources).items():
        if sha256_file(source / name) != digest:
            raise ValueError("executed source drift")
    verify_inputs(cfg)
    write_json_report(
        {
            "scope": registration["scope"],
            "status": "complete",
            "promotion": False,
            "registration_sha256": sha256_file(root / "registration.json"),
            "cases": cases,
            "comparisons": comparisons(cases),
            "evidence": evidence,
        },
        root / "assessment.json",
    )
    write_json_report(
        {"stage": "complete", "preflight": args.preflight, "completed_cases": len(evidence)},
        root / "progress.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
