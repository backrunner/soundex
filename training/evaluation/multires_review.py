"""Common spectrum-first scores, including actual Rust frontend -> ORT -> DSP."""

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.complex_synthesis import complex_features, synthesize
from models.multires_spectrum import spectral_loss
from models.restoration_head import load_restoration_head, predict_head
from scripts.run_capacity_study import verify_inputs
from scripts.run_consistency_followup import complete, read, registered, verify_receipt


def scores(wave, batch, region):
    result = spectral_loss(wave[:, region], batch["clean_wave"][:, region], batch["rate"])
    return [
        {
            "total": float(result["per_example"][i]),
            "components": {k: float(v[i]) for k, v in result["components"].items()},
            "resolutions": [
                {
                    "fft_size": r["fft_size"],
                    "bands": [
                        {"hz": b["hz"], **{k: float(v[i]) for k, v in b.items() if k != "hz"}}
                        for b in r["bands"]
                    ],
                }
                for r in result["resolutions"]
            ],
        }
        for i in range(len(wave))
    ]


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['corpus']}/{r['role']}/{r['stage']}"].append(r)
    result = {}
    for name, rows in groups.items():
        tracks = defaultdict(list)
        for row in rows:
            tracks[row["track_id"]].append(row["score"]["total"])
        result[name] = {
            "clip_mean": float(np.mean([r["score"]["total"] for r in rows])),
            "recording_mean": float(np.mean([np.mean(v) for v in tracks.values()])),
            "components": {
                k: float(np.mean([r["score"]["components"][k] for r in rows]))
                for k in rows[0]["score"]["components"]
            },
            "recordings": len(tracks),
            "clips": len(rows),
        }
    return result


@torch.no_grad()
def review(study, binary):
    torch.set_num_threads(1)
    study, binary = study.resolve(), binary.resolve()
    config = read(study / "registration.json")
    verify_inputs(config)
    endpoints = read(study / "review-registration.json")
    expected = {f"{a}-{config['steps']}-{s}" for a in config["arms"] for s in config["seeds"]}
    if set(endpoints) != expected:
        raise ValueError("all registered endpoints required")
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    pool = LazyPool(
        config["specs"],
        BoundedEnvelopeClips(
            rows,
            next(p for p in paths if p.name == "soundex-gain-data"),
            Path(config["cache_path"]),
        ),
    )
    output = study / "spectrum-review"
    output.mkdir(exist_ok=True)
    registration = {
        "scope": config["review_scope"],
        "registration_sha256": sha256_file(study / "registration.json"),
        "binary_sha256": sha256_file(binary),
        "loss": config["multires_loss"],
        "checkpoints": endpoints,
    }
    registered(registration, output / "registration.json")
    summaries = {}
    for label, value in {"initial": None, **endpoints}.items():
        directory = output / label
        directory.mkdir(exist_ok=True)
        if (directory / "complete.json").exists():
            verify_receipt(directory, "files")
            report = read(directory / "report.json")
            if report["registration_sha256"] != sha256_file(output / "registration.json"):
                raise ValueError("spectrum receipt drift")
            summaries[label] = report["summary"]
            continue
        model = None
        if value:
            if sha256_file(Path(value["path"])) != value["sha256"]:
                raise ValueError("endpoint drift")
            model = load_restoration_head(
                torch.load(value["path"], map_location="cpu", weights_only=False)
            ).eval()
            onnx = study / "exports" / label / "diagnostic.onnx"
            if sha256_file(onnx) != read(onnx.parent / "report.json")["onnx_sha256"]:
                raise ValueError("export drift")
        records = []
        for index, row in enumerate(pool.rows):
            objective, context = pool.select([index])
            batch, region = objective.batch, objective.region
            predicted = predict_head(model, batch, context) if model else batch["x"]
            waves = {
                "raw_ola": synthesize(complex_features(predicted)),
                "input": batch["dry"],
                "identity_dsp": torch.from_numpy(
                    np.load(study / "joint-baseline" / f"{index:03d}.npy", allow_pickle=False)
                ),
            }
            if model:
                native = []
                for role in range(2):
                    rate = int(batch["rate"][role, 0])
                    with tempfile.TemporaryDirectory(prefix="soundex-multires-") as tmp:
                        work = Path(tmp)
                        write_sxa(work / "input.sxa", batch["pcm"][role].numpy(), rate)
                        subprocess.run(
                            [
                                str(binary),
                                str(onnx),
                                str(work / "input.sxa"),
                                str(work / "output.sxa"),
                                str(work / "report.json"),
                                "1",
                            ],
                            check=True,
                            capture_output=True,
                        )
                        wave, actual_rate = read_sxa(work / "output.sxa")
                        runtime = read(work / "report.json")
                    if (
                        actual_rate != rate
                        or wave.shape != (batch["pcm"].shape[-1], 1)
                        or not runtime["finite"]
                    ):
                        raise ValueError("invalid native timeline or output")
                    native.append(torch.from_numpy(wave[:, 0]))
                waves["native_output"] = torch.stack(native)
            else:
                waves["native_output"] = waves["identity_dsp"]
            for stage, wave in waves.items():
                for role, score in zip(
                    ("paired", "identity"), scores(wave, batch, region), strict=True
                ):
                    records.append(
                        {
                            "spec": pool.specs[index],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "role": role,
                            "stage": stage,
                            "score": score,
                        }
                    )
        summaries[label] = summarize(records)
        write_json_report(
            {
                "registration_sha256": sha256_file(output / "registration.json"),
                "checkpoint_sha256": value["sha256"] if value else None,
                "summary": summaries[label],
                "rows": records,
            },
            directory / "report.json",
        )
        complete(directory)
        print("multires review complete", label, flush=True)
    verify_inputs(config)
    write_json_report(
        {
            "status": "complete",
            "scope": config["review_scope"],
            "cases": summaries,
            "registration_sha256": sha256_file(output / "registration.json"),
            "reports": {label: sha256_file(output / label / "report.json") for label in summaries},
            "additional_native_roles": len(endpoints) * len(pool.rows) * 2,
        },
        output / "summary.json",
    )
