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
from data.weak_band_study import identity_specs
from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa, write_sxa
from export_study import make_model
from models.causal_objective import render
from models.conditioned_shape import make_shape_model
from models.envelope_objective import amplitude_metrics
from models.envelope_probe import context_features, predict_probe
from models.factorized_envelope import BURN, make_factor_model
from models.linear_objective import make_linear_model
from models.rendered_probe import make_rendered_model
from models.residual_energy import make_energy_model
from scripts.analyze_envelope_study import paired_comparison
from scripts.run_causal_study import prediction
from scripts.run_weak_band_study import freeze_binary
from train import spectral_features


def hybrid_features(amplitude, phase, original, burn=BURN):
    features = torch.stack((20 * amplitude.clamp_min(1e-10).log10(), phase), dim=1)
    # All variants share the same startup prediction, irrespective of head type.
    return torch.cat((original[:, :, :burn], features[:, :, burn:]), dim=2)


def aligned_band_metrics(features, target, actual, rate):
    """Compare raw and synthesized amplitudes on identical FFT windows and bands.

    Native output already has the 128-sample synthesis delay represented by the
    left pad used to construct target features. Do not pad the output again.
    Keep only windows wholly inside the common scoring region.
    """
    output = spectral_features(actual[:, : 32768 - 128], 256, 128)
    start, stop = BURN + 9, output.shape[2]
    reference = torch.pow(10.0, target[:, 0, start:stop] / 20)
    raw = torch.pow(10.0, features[:, 0, start:stop] / 20)
    reconstructed = torch.pow(10.0, output[:, 0, start:stop] / 20)
    return (
        amplitude_metrics(raw, reference, rate, burn=0),
        amplitude_metrics(reconstructed, reference, rate, burn=0),
    )


def run(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    args.phase_study = args.phase_study.resolve()
    config = json.loads((root / "config.json").read_text())
    specs = config["validation_specs"]
    input_groups = ("paired", "identity") if config.get("dsp_identity_controls") else ("paired",)
    if config.get("dsp_identity_controls"):
        specs = specs + identity_specs(specs)
    arms = config["arms"]
    candidate = config.get("benchmark_candidate", "factorized")
    factory = (
        make_energy_model if config["format"] == "soundex-energy-study-v1" else make_factor_model
    )
    if config["format"] == "soundex-linear-study-v1":
        factory = make_linear_model
    if config["format"] == "soundex-shape-study-v1":
        factory = make_shape_model
    if config["format"] == "soundex-rendered-study-v1":
        factory = make_rendered_model
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
        for arm in arms:
            paths[f"{arm}-{seed}"] = root / f"{arm}-{seed}" / args.checkpoint_name
    registration = {
        "scope": "reused development; offline predicted features through native DSP; not realtime deployment or listening",
        "checkpoints": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in paths.items()},
        "phase_onnx": {
            str(seed): sha256_file(paths[f"p3-{seed}"].with_suffix(".onnx"))
            for seed in config["seeds"]
        },
        "binary_sha256": sha256_file(binary),
        "script_sha256": sha256_file(Path(__file__)),
        "specs": specs,
        "input_groups": list(input_groups),
        "burn_hops": BURN,
        "scored_start_sample": (BURN + 9) * 128,
        "scored_end_sample": 32768 - 128,
        "gain": "model",
        "limiter": "smooth-knee",
        "selection_changed": args.checkpoint_name != "best.pth",
        "checkpoint_name": args.checkpoint_name,
    }
    registered = root / f"{args.report_prefix}-review-registration.json"
    if registered.exists() and json.loads(registered.read_text()) != registration:
        raise ValueError("DSP review registration changed")
    write_json_report(registration, registered)
    rows, hashes = load_sources([Path(p) for p in config["manifests"]])
    if hashes != config["manifests"]:
        raise ValueError("manifests changed")
    clips = EnvelopeClips(
        rows, root / "binaries/soundex-gain-data", config.get("cache_path", root / "cache")
    )
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
            for arm in arms:
                model = factory(arm).eval()
                model.load_state_dict(
                    torch.load(paths[f"{arm}-{seed}"], map_location="cpu", weights_only=False)[
                        "model"
                    ]
                )
                models[arm] = model
            for index, spec in enumerate(specs):
                row = clips.rows[spec["row_id"]]
                batch = clips.batch([spec])
                context = context_features(batch["pcm"], "past")
                phase_features = prediction(phase_model, "tcn", batch)
                base = torch.cat((batch["x"][:, :, :BURN], phase_features[:, :, BURN:]), dim=2)
                variants = {"p3_original": base, "identity_dsp": batch["x"]}
                for arm, model in models.items():
                    amplitude = predict_probe(model, batch, context, burn=BURN)
                    input_phase = config.get("deployment_phase") == "input"
                    primary = batch["x"][:, 1] if input_phase else phase_features[:, 1]
                    suffix = "_input_phase" if input_phase else "_p3_phase"
                    variants[arm + suffix] = hybrid_features(amplitude, primary, batch["x"])
                    if arm == candidate:
                        alternate = phase_features[:, 1] if input_phase else batch["x"][:, 1]
                        other = "_p3_phase" if input_phase else "_input_phase"
                        variants[candidate + other] = hybrid_features(
                            amplitude, alternate, batch["x"]
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
                    raw_bands, output_bands = aligned_band_metrics(
                        features, batch["y"], torch.from_numpy(actual[:, 0])[None], batch["rate"]
                    )
                    records.append(
                        {
                            "seed": seed,
                            "variant": name,
                            "input_group": "identity"
                            if spec["row_id"].endswith(":identity")
                            else "paired",
                            "row_id": row["row_id"],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "proxy_max_error": error,
                            "metrics": metrics,
                            "fixed_4khz_metrics": fixed,
                            "raw_fixed_band_metrics": raw_bands,
                            "output_fixed_band_metrics": output_bands,
                        }
                    )
                if index % 16 == 0:
                    print(json.dumps({"seed": seed, "clips": index + 1}), flush=True)
        write_json_report(
            {"registration": registration, "rows": records},
            root / f"{args.report_prefix}-review.json",
        )
    summary = {}
    keys = (
        "enhanced_high_spectral_convergence",
        "enhanced_high_flux_error",
        "enhanced_high_band_energy_error_db",
        "delta_si_sdr_db",
    )
    for seed in config["seeds"]:
        by_metric_region = {}
        for metric_region, input_group in (
            (region, group)
            for region in (
                "metrics",
                "fixed_4khz_metrics",
                "raw_fixed_band_metrics",
                "output_fixed_band_metrics",
            )
            for group in input_groups
        ):
            groups = {}
            for record in records:
                if record["seed"] == seed and record["input_group"] == input_group:
                    groups.setdefault(record["variant"], {"rows": []})["rows"].append(
                        {**record, "metrics": record[metric_region]}
                    )
            comparisons = {}
            contrasts = (
                ("direct_factored_loss_p3_phase", "direct_legacy_p3_phase"),
                ("factorized_p3_phase", "direct_factored_loss_p3_phase"),
                ("factorized_p3_phase", "p3_original"),
                ("factorized_p3_phase", "identity_dsp"),
                ("factorized_input_phase", "factorized_p3_phase"),
            )
            if config["format"] == "soundex-energy-study-v1":
                contrasts = (
                    (candidate + "_p3_phase", "factorized_p3_phase"),
                    (candidate + "_p3_phase", "p3_original"),
                    (candidate + "_p3_phase", "identity_dsp"),
                    (candidate + "_input_phase", candidate + "_p3_phase"),
                )
            if config["format"] == "soundex-linear-study-v1":
                contrasts = (
                    (candidate + "_p3_phase", "energy_control_p3_phase"),
                    (candidate + "_p3_phase", "p3_original"),
                    (candidate + "_p3_phase", "identity_dsp"),
                    (candidate + "_input_phase", candidate + "_p3_phase"),
                )
            if config["format"] == "soundex-shape-study-v1":
                contrasts = (
                    *(
                        (a + "_p3_phase", b + "_p3_phase")
                        for a, b in config["experiment"]["contrasts"]
                    ),
                    (candidate + "_p3_phase", "p3_original"),
                    (candidate + "_p3_phase", "identity_dsp"),
                    (candidate + "_input_phase", candidate + "_p3_phase"),
                )
            if config["format"] == "soundex-rendered-study-v1":
                contrasts = (
                    *(
                        (a + "_input_phase", b + "_input_phase")
                        for a, b in config["experiment"]["contrasts"]
                    ),
                    (candidate + "_input_phase", "identity_dsp"),
                    (candidate + "_input_phase", "p3_original"),
                    (candidate + "_p3_phase", candidate + "_input_phase"),
                )
            for treatment, control in contrasts:
                comparisons[treatment + "_minus_" + control] = paired_comparison(
                    groups[treatment],
                    groups[control],
                    keys=("high_sc", "high_flux", "high_energy_error_db", "band_0_4000_sc")
                    if metric_region.endswith("band_metrics")
                    else keys,
                )
            key = metric_region if input_group == "paired" else metric_region + "_identity"
            by_metric_region[key] = comparisons
        summary[str(seed)] = by_metric_region
    write_json_report(
        {"scope": registration["scope"], "seeds": summary},
        root / f"{args.report_prefix}-comparisons.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-name",
        choices=("best.pth", "latest.pth"),
        default="best.pth",
        help="latest is for nonzero diagnostic replay only; never changes selection",
    )
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase-study", required=True, type=Path)
    parser.add_argument("--replay-binary", required=True, type=Path)
    parser.add_argument(
        "--report-prefix", choices=("dsp", "dsp-aligned", "dsp-latest"), default="dsp"
    )
    run(parser.parse_args())
