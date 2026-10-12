#!/usr/bin/env python3
"""Diagnostic amplitude/phase combinations through actual Rust DSP, not deployment.

Amplitude is predicted offline. P3 phase is frozen. No candidate is selected by
these results and no clean information enters inference or the input detector.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from export_study import make_model
from models.causal_objective import render
from models.envelope_probe import context_features, predict_probe
from models.factorized_envelope import ARMS, BURN, make_factor_model
from scripts.run_causal_study import prediction
from scripts.run_weak_band_study import freeze_binary


def hybrid_features(amplitude, phase, original, burn=BURN):
    features = torch.stack((20 * amplitude.clamp_min(1e-10).log10(), phase), dim=1)
    # All variants share the same startup prediction, irrespective of head type.
    return torch.cat((original[:, :, :burn], features[:, :, burn:]), dim=2)


def run(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    args.phase_study = args.phase_study.resolve()
    config = json.loads((root / "config.json").read_text())
    if json.loads((root / "summary.json").read_text())["status"] != "complete":
        raise ValueError("finish frozen training before DSP review")
    binary = freeze_binary(
        args.replay_binary,
        root / "binaries/soundex-prediction-replay",
        sha256_file(args.replay_binary),
    )
    paths = {}
    for seed in config["seeds"]:
        paths[f"p3-{seed}"] = args.phase_study / f"control-{seed}/best.pth"
        for arm in ARMS:
            paths[f"{arm}-{seed}"] = root / f"{arm}-{seed}/best.pth"
    registration = {
        "scope": "reused development; offline predicted features through native DSP; not realtime deployment or listening",
        "checkpoints": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()},
        "phase_onnx": {
            str(seed): sha256_file(paths[f"p3-{seed}"].with_suffix(".onnx"))
            for seed in config["seeds"]
        },
        "binary_sha256": sha256_file(binary),
        "script_sha256": sha256_file(Path(__file__)),
        "specs": config["validation_specs"],
        "burn_hops": BURN,
        "scored_start_sample": (BURN + 9) * 128,
        "scored_end_sample": 32768 - 128,
        "gain": "model",
        "limiter": "smooth-knee",
        "selection_changed": False,
    }
    registered = root / "dsp-review-registration.json"
    if registered.exists() and json.loads(registered.read_text()) != registration:
        raise ValueError("DSP review registration changed")
    write_json_report(registration, registered)
    rows, hashes = load_sources([Path(p) for p in config["manifests"]])
    if hashes != config["manifests"]:
        raise ValueError("manifests changed")
    clips = EnvelopeClips(rows, root / "binaries/soundex-gain-data", root / "cache")
    records = []
    region = slice((BURN + 9) * 128, 32768 - 128)
    with torch.no_grad(), tempfile.TemporaryDirectory(prefix="soundex-p6-replay-") as tmp:
        work = Path(tmp)
        for seed in config["seeds"]:
            phase_model = make_model("tcn").eval()
            phase_model.load_state_dict(
                torch.load(paths[f"p3-{seed}"], map_location="cpu", weights_only=False)["model"]
            )
            models = {}
            for arm in ARMS:
                model = make_factor_model(arm).eval()
                model.load_state_dict(
                    torch.load(paths[f"{arm}-{seed}"], map_location="cpu", weights_only=False)[
                        "model"
                    ]
                )
                models[arm] = model
            for index, spec in enumerate(config["validation_specs"]):
                row = rows[spec["row_id"]]
                batch = clips.batch([spec])
                context = context_features(batch["pcm"], "past")
                phase_features = prediction(phase_model, "tcn", batch)
                base = torch.cat((batch["x"][:, :, :BURN], phase_features[:, :, BURN:]), dim=2)
                variants = {"p3_original": base, "identity_dsp": batch["x"]}
                for arm, model in models.items():
                    amplitude = predict_probe(model, batch, context, burn=BURN)
                    variants[arm + "_p3_phase"] = hybrid_features(
                        amplitude, phase_features[:, 1], batch["x"]
                    )
                    if arm == "factorized":
                        variants["factorized_input_phase"] = hybrid_features(
                            amplitude, batch["x"][:, 1], batch["x"]
                        )
                rate = int(batch["rate"][0, 0])
                clean, dry = [batch[key][0, region].numpy() for key in ("clean_wave", "dry")]
                write_sxa(work / "input.sxa", batch["pcm"][0].numpy(), rate)
                for name, features in variants.items():
                    proxy, _ = render(
                        features,
                        batch["x"],
                        batch["mask"],
                        batch["active"],
                        batch["gate"],
                        batch["dry"],
                        limiter_mode="smooth-knee",
                    )
                    (work / "features.json").write_text(
                        json.dumps(
                            features.permute(2, 0, 1, 3).reshape(-1).tolist(), allow_nan=False
                        )
                    )
                    subprocess.run(
                        [
                            str(binary),
                            str(paths[f"p3-{seed}"].with_suffix(".onnx")),
                            str(work / "input.sxa"),
                            str(work / "features.json"),
                            str(work / "output.sxa"),
                        ],
                        check=True,
                        capture_output=True,
                        env={**os.environ, "SOUNDEX_EVAL_LIMITER_MODE": "smooth-knee"},
                    )
                    actual, actual_rate = read_sxa(work / "output.sxa")
                    if (
                        actual_rate != rate
                        or actual.shape != (32768, 1)
                        or not np.isfinite(actual).all()
                    ):
                        raise ValueError("invalid native replay output")
                    error = float(np.max(np.abs(actual[region, 0] - proxy[0, region].numpy())))
                    if error > 2e-5:
                        raise ValueError(f"native DSP proxy discrepancy: {error}")
                    metrics = evaluate_signal_triplet(
                        clean,
                        dry,
                        actual[region, 0],
                        sample_rate=rate,
                        cutoff_hz=row["measured_cutoff_hz"],
                    )
                    fixed = evaluate_signal_triplet(
                        clean, dry, actual[region, 0], sample_rate=rate, cutoff_hz=4500.0
                    )
                    records.append(
                        {
                            "seed": seed,
                            "variant": name,
                            "row_id": row["row_id"],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "proxy_max_error": error,
                            "metrics": metrics,
                            "fixed_4khz_metrics": fixed,
                        }
                    )
                if index % 16 == 0:
                    print(json.dumps({"seed": seed, "clips": index + 1}), flush=True)
        write_json_report({"registration": registration, "rows": records}, root / "dsp-review.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase-study", required=True, type=Path)
    parser.add_argument("--replay-binary", required=True, type=Path)
    run(parser.parse_args())
