#!/usr/bin/env python3
"""Post-hoc P23 absolute-error/stage audit; never selects or changes a model."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.absolute_audio import BANDS, compare_pcm
from evaluation.head_study import predict
from evaluation.loss_scale import spectral_parts, summarize_parts
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from models.restoration_head import load_restoration_head
from scripts.audit_render_stages import stages
from scripts.run_consistency_followup import read, registered
from scripts.run_subset_study import Pool


def reference_rows(batch, specs, rows):
    amplitude = torch.pow(10.0, batch["y"][:, 0, 89:] / 20)
    records = []
    for i, spec in enumerate(specs):
        rate = int(batch["rate"][i, 0])
        hz = torch.arange(129) * rate / 256
        global_rms = float(amplitude[i].square().mean().sqrt())
        ratios = {
            f"{low}_{high}": float(
                amplitude[i, :, (hz >= low) & (hz < high)].square().mean().sqrt()
            )
            / max(global_rms, 1e-20)
            for low, high in BANDS
        }
        records.append(
            {
                "spec": spec,
                "track_id": rows[spec["row_id"]]["track_id"],
                "corpus": rows[spec["row_id"]]["corpus"],
                "spectral_band_to_global": ratios,
            }
        )
    return records


def native_audio(features, batch, objective, binary, onnx):
    waves, errors = [], []
    expected = objective.render(features).numpy()
    with tempfile.TemporaryDirectory(prefix="soundex-p23-audit-") as tmp:
        work = Path(tmp)
        for role in range(len(features)):
            rate = int(batch["rate"][role, 0])
            write_sxa(work / "input.sxa", batch["pcm"][role].numpy(), rate)
            (work / "features.json").write_text(
                json.dumps(
                    features[role : role + 1].permute(2, 0, 1, 3).reshape(-1).tolist(),
                    allow_nan=False,
                )
            )
            subprocess.run(
                [
                    str(binary),
                    str(onnx),
                    str(work / "input.sxa"),
                    str(work / "features.json"),
                    str(work / "output.sxa"),
                ],
                check=True,
                capture_output=True,
                env={**os.environ, "SOUNDEX_EVAL_LIMITER_MODE": "smooth-knee"},
            )
            wave, actual_rate = read_sxa(work / "output.sxa")
            if (
                wave.shape != (expected.shape[-1], 1)
                or actual_rate != rate
                or not np.isfinite(wave).all()
            ):
                raise ValueError("native shape/rate/finite failure")
            error = float(
                np.max(np.abs(wave[objective.region, 0] - expected[role, objective.region]))
            )
            if error > 2e-5:
                raise ValueError("native parity failure")
            waves.append(wave[:, 0])
            errors.append(error)
    return np.stack(waves), errors


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    root, output = args.study.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read(root / "registration.json")
    if read(root / "summary.json")["status"] != "complete":
        raise ValueError("full study must finish before audit")
    paths = read(root / "review-registration.json")
    paths = {k: v for k, v in paths.items() if f"-{config['primary_endpoint']}-" in k}
    for value in paths.values():
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("checkpoint drift")
    for name, expected in config["input_hashes"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("input drift")
    source = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(source)): sha256_file(p)
        for p in source.rglob("*.py")
        if not set(p.relative_to(source).parts) & {".venv", "tests", "__pycache__", "outputs"}
    }
    registered(
        {
            "scope": "post-hoc all-development absolute PCM and stage audit; no loss/runtime changes or threshold selection; short audio exports are not a listening result",
            "study_sha256": sha256_file(root / "summary.json"),
            "checkpoints": paths,
            "source_hashes": hashes,
            "fft_size": 4096,
            "dbfs_reference": "PCM magnitude 1; unit peak sine RMS=-3.0103 dBFS",
            "native_examples": "known main outlier, known secondary outlier, music and speech reference-highband medians",
        },
        output / "registration.json",
    )
    inputs = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in inputs if p.name == "manifest.jsonl"])
    detector = next(p for p in inputs if p.name == "soundex-gain-data")
    parent = next(p for p in inputs if "rendered-p10" in str(p) and p.name == "config.json")
    clips = EnvelopeClips(rows, detector, Path(read(parent)["cache_path"]))
    binary = next(p for p in inputs if p.name == "soundex-prediction-replay")
    onnx = next(p for p in inputs if p.name == "replay-contract.onnx")
    references = {}
    for group in ("bank", "development", "shifted"):
        refs = []
        for start in range(0, len(config[group]), 32):
            specs = config[group][start : start + 32]
            refs += reference_rows(clips.batch(specs), specs, clips.rows)
        references[group] = refs
        print("reference_scan", group, len(refs), flush=True)
    write_json_report(references, output / "reference-coverage.json")
    pool = Pool(config["development"], clips)
    selected = {
        i
        for i, s in enumerate(pool.specs)
        if s["row_id"] in {"d2599868e6a472eff47c1674", "90729d18b315fc317cc35b29"}
    }
    for corpus in ("music_library", "speech_library"):
        ordered = sorted(
            (i for i, r in enumerate(pool.rows) if r["corpus"] == corpus),
            key=lambda i: references["development"][i]["spectral_band_to_global"]["12000_16000"],
        )
        selected.add(ordered[len(ordered) // 2])
    registered(
        {"indices": sorted(selected), "specs": [pool.specs[i] for i in sorted(selected)]},
        output / "native-selection.json",
    )
    baseline_cache = []
    for i in range(len(pool.specs)):
        objective, _ = pool.select([i])
        baseline_cache.append(stages(objective.batch["x"], objective.batch))
    records = []
    for label, value in sorted(paths.items()):
        model = load_restoration_head(
            torch.load(value["path"], map_location="cpu", weights_only=False)
        ).eval()
        proxy = read(root / "review" / label / "proxy.json")["groups"]["development"]["rows"]
        for i, row in enumerate(pool.rows):
            objective, context = pool.select([i])
            batch, region = objective.batch, objective.region
            prediction = predict(model, objective, context)
            result = objective(prediction)
            if abs(float(result["total"]) - proxy[i]["metrics"]["total"]) > 2e-5:
                raise ValueError("rehydrated objective drift")
            candidate, prelimit = stages(prediction, batch)
            base, base_prelimit = baseline_cache[i]
            output_wave = objective.render(prediction).numpy()
            native_errors = None
            if i in selected:
                output_wave, native_errors = native_audio(
                    prediction, batch, objective, binary, onnx
                )
                baseline_native, base_errors = native_audio(
                    batch["x"], batch, objective, binary, onnx
                )
            for role, role_name in enumerate(("paired", "identity")):
                rate = int(batch["rate"][role, 0])
                clean = batch["clean_wave"][role, region].numpy()
                baseline = objective.baseline[role, region].numpy()
                wave = output_wave[role, region]
                stage_parts = {}
                for stage, spectrum in candidate.items():
                    stage_parts[stage] = summarize_parts(
                        spectral_parts(spectrum[role], base[stage][role], rate)
                    )
                record = {
                    "label": label,
                    "index": i,
                    "spec": pool.specs[i],
                    "corpus": row["corpus"],
                    "track_id": row["track_id"],
                    "role": role_name,
                    "rate": rate,
                    "samples": len(wave),
                    "absolute_vs_clean": compare_pcm(wave, clean, rate),
                    "absolute_change_vs_identity_dsp": compare_pcm(wave, baseline, rate),
                    "baseline_vs_clean": compare_pcm(baseline, clean, rate),
                    "prelimit_change": compare_pcm(
                        prelimit[role, region].numpy(), base_prelimit[role, region].numpy(), rate
                    )["error"],
                    "limiter_change": compare_pcm(
                        wave,
                        (
                            batch["dry"][role, region] * (1 - batch["gate"][role, region])
                            + prelimit[role, region] * batch["gate"][role, region]
                        ).numpy(),
                        rate,
                    )["error"],
                    "stage_change_spectral_parts": stage_parts,
                    "loss": proxy[i]["metrics"],
                    "cutoff_hz_mean": float(batch["cutoff"][role, 89:].mean()),
                    "active_fraction": float(batch["active"][role, 89:].float().mean()),
                    "gate_min": float(batch["gate"][role, region].min()),
                    "native_proxy_max_error": None
                    if native_errors is None
                    else native_errors[role],
                }
                if i in selected:
                    record["baseline_native_proxy_max_error"] = base_errors[role]
                    directory = output / "audio" / f"case-{i:03d}-{role_name}"
                    directory.mkdir(parents=True, exist_ok=True)
                    for name, samples in (
                        ("clean", clean),
                        ("input", batch["dry"][role, region].numpy()),
                        ("identity-dsp", baseline_native[role, region]),
                        (label, wave),
                        (label + "-difference", wave - baseline_native[role, region]),
                    ):
                        sf.write(directory / f"{name}.wav", samples, rate, subtype="FLOAT")
                records.append(record)
        print("complete", label, "pairs", len(pool.specs), flush=True)
        write_json_report({"records": records}, output / "measurements.json")
    for name, expected in hashes.items():
        if sha256_file(source / name) != expected:
            raise ValueError("audit source drift")
    write_json_report(
        {
            "status": "complete",
            "records": len(records),
            "native_candidate_roles": sum(r["native_proxy_max_error"] is not None for r in records),
            "native_baseline_roles": sum(r["native_proxy_max_error"] is not None for r in records),
            "measurements_sha256": sha256_file(output / "measurements.json"),
            "reference_coverage_sha256": sha256_file(output / "reference-coverage.json"),
            "audio": "equal original gain, aligned supervised crop, FLOAT WAV; difference files are unamplified; not a listening/ABX result",
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
