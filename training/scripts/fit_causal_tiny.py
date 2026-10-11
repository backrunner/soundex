"""P0 learnability diagnostic; fixed training clips, never a generalization result.

Uses production input-only detector trajectories, V5 Model-gain reconstruction,
64 real burn-in hops and 128 supervised hops. V5 has no limiter or sample gate;
its reconstructed spectral-convergence mask is time-averaged. Actual Rust output
is scored separately. This deliberately balanced tiny set is not P2 sampling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from evaluation.metrics import evaluate_signal_triplet
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import RustStreamEvaluator, write_sxa
from export_causal import export_causal
from models.causal_restorer import CausalRestorer
from models.losses import GeneratorLoss, build_missing_band_mask
from train import spectral_features

BURN = 64
FRAMES = 192
SAMPLES = FRAMES * 128


def select_clips(manifests):
    selected = []
    for manifest in manifests:
        # Two distinct musical sources/rates, plus one mono speech source/rate.
        wanted = (44100, 48000) if "music_library" in str(manifest) else (48000,)
        sources = set()
        for rate in wanted:
            for line in manifest.open():
                row = json.loads(line)
                if (
                    row["split"] != "train"
                    or row["held_out"]
                    or row["sample_rate"] != rate
                    or row["codec_id"] != "mp3-cbr-64"
                    or row["source_id"] in sources
                    or row["length_samples"] < SAMPLES
                ):
                    continue
                clean, sr = sf.read(manifest.parent / row["clean_path"], dtype="float32")
                degraded, dr = sf.read(manifest.parent / row["degraded_path"], dtype="float32")
                if clean.ndim != 1 or degraded.shape != clean.shape or sr != rate or dr != rate:
                    raise ValueError("unaligned mono-role pair")
                clean, degraded = clean[:SAMPLES], degraded[:SAMPLES]
                if not np.isfinite(clean).all() or not np.isfinite(degraded).all():
                    raise ValueError("nonfinite source")
                # Avoid silence or effectively absent targets in a learnability probe.
                spectrum = np.abs(np.fft.rfft(clean)) ** 2
                freqs = np.fft.rfftfreq(len(clean), 1 / sr)
                ratio = spectrum[freqs > 8000].sum() / max(spectrum.sum(), 1e-30)
                if np.sqrt(np.mean(clean**2)) < 0.02 or ratio < 1e-4:
                    continue
                for role in ("clean", "degraded"):
                    if (
                        sha256_file(manifest.parent / row[f"{role}_path"])
                        != row[f"{role}_checksum"]
                    ):
                        raise ValueError(f"changed {role} audio: {row['row_id']}")
                selected.append((row, clean, degraded))
                sources.add(row["source_id"])
                break
            else:
                raise ValueError(f"no active train example for {manifest} at {rate}")
    return selected


def make_case(selected, task, detector):
    clean = np.stack([c for _, c, _ in selected])
    if task == "identity":
        degraded = clean.copy()
    elif task == "lowpass":
        # Ideal whole-clip filter only for this diagnostic corruption, not runtime.
        degraded = np.stack(
            [
                np.fft.irfft(
                    np.fft.rfft(c) * (np.fft.rfftfreq(len(c), 1 / r["sample_rate"]) < 8000),
                    n=len(c),
                ).astype(np.float32)
                for r, c, _ in selected
            ]
        )
    else:
        degraded = np.stack([d for _, _, d in selected])
    rates = torch.tensor([r["sample_rate"] for r, _, _ in selected], dtype=torch.float32)
    masks = []
    active = []
    with tempfile.TemporaryDirectory(prefix="soundex-tiny-") as directory:
        root = Path(directory)
        for index, rate in enumerate(rates.int().tolist()):
            write_sxa(root / "input.sxa", degraded[index], rate)
            write_sxa(root / "clean.sxa", clean[index], rate)
            subprocess.run(
                [
                    str(detector),
                    str(root / "input.sxa"),
                    str(root / "clean.sxa"),
                    str(root / "trace.json"),
                    "1",
                ],
                check=True,
                capture_output=True,
            )
            trace = json.loads((root / "trace.json").read_text())[:FRAMES]
            if len(trace) != FRAMES or any(r["hop"] != i for i, r in enumerate(trace)):
                raise ValueError("detector time alignment mismatch")
            cutoff = torch.tensor([r["cutoff_hz"] for r in trace])
            masks.append(
                build_missing_band_mask(
                    cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
                )[:, 0, 0]
            )
            active.append(sum(r["active"] for r in trace[BURN:]) / (FRAMES - BURN))
    # Zero left padding is exactly the production first analysis frame.
    x, y = [
        spectral_features(F.pad(torch.from_numpy(a), (128, 0)), 256, 128) for a in (degraded, clean)
    ]
    return x, y, rates[:, None], torch.stack(masks)[:, None], clean, degraded, active


def run(args):
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = select_clips(args.manifest)
    # Bind the exact already audited manifest set and architecture-probe provenance.
    manifest_hashes = {str(p.resolve()): sha256_file(p) for p in args.manifest}
    if json.loads((args.probe / "probe.json").read_text())["manifests"] != manifest_hashes:
        raise ValueError("tiny manifest set differs from the provenance probe")
    probe = torch.load(args.probe / "gru/untrained.pth", map_location="cpu", weights_only=True)
    report = {
        "diagnostic_only": True,
        "independent_test": False,
        "listening_performed": False,
        "steps": args.steps,
        "burn_in_hops": BURN,
        "supervised_hops": FRAMES - BURN,
        "source_sampling": "balanced fixed tiny training clips; not production sampling",
        "objective": "GeneratorLoss V5 defaults, FFT256/H128, steady_state, GAN=0, Model gain",
        "proxy_limits": ["no sample gate/limiter", "SC reconstruction mask time-averaged"],
        "detector_sha256": sha256_file(args.detector),
        "manifests": manifest_hashes,
        "selected": [
            {
                k: r[k]
                for k in (
                    "row_id",
                    "source_id",
                    "split",
                    "sample_rate",
                    "codec_id",
                    "clean_checksum",
                    "degraded_checksum",
                )
            }
            for r, _, _ in selected
        ],
        "predeclared_gate": {
            "nonidentity_loss_ratio_max": 0.6,
            "nonidentity_raw_hfsc_ratio_max": 0.8,
            "identity_loss_ratio_max": 0.6,
        },
        "runs": [],
    }
    write_json_report(report, output / "report.json")
    criterion = GeneratorLoss(
        fft_size=256,
        hop_size=128,
        objective_version=5,
        waveform_region="steady_state",
        adversarial_weight=0,
        feature_matching_weight=0,
    )
    evaluator = RustStreamEvaluator(args.stream_binary)
    for task in ("identity", "lowpass", "codec"):
        x, y, rate, mask, clean, degraded, active = make_case(selected, task, args.detector)
        for kind in ("gru", "tcn"):
            root = output / f"{task}-{kind}"
            root.mkdir(exist_ok=True)
            torch.manual_seed(20261011)
            model = CausalRestorer(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

            def predict(model=model, x=x, rate=rate):
                with torch.no_grad():
                    _, state = model(x[:, :, :BURN], model.initial_state(len(x)), rate)
                return model(x[:, :, BURN:], state.detach(), rate)[0]

            def score(predict=predict, x=x, y=y, mask=mask):
                with torch.no_grad():
                    return {
                        k: float(v)
                        for k, v in criterion(
                            predict(), y[:, :, BURN:], x[:, :, BURN:], mask[:, :, BURN:]
                        ).items()
                    }

            initial = score()
            started = time.monotonic()
            history = [{"step": 0, **initial}]
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(predict(), y[:, :, BURN:], x[:, :, BURN:], mask[:, :, BURN:])
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError("nonfinite tiny loss")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                if step % 32 == 0 or step == args.steps:
                    values = score()
                    history.append({"step": step, "gradient_norm": float(norm), **values})
                    print(
                        json.dumps(
                            {"task": task, "kind": kind, "step": step, "loss": values["total"]}
                        ),
                        flush=True,
                    )
            final = score()
            checkpoint = root / "last.pth"
            torch.save(
                {
                    "format": "soundex-causal-study-v1",
                    "kind": kind,
                    "width": model.width,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "provenance": probe["provenance"],
                    "config": {
                        "diagnostic_only": True,
                        "task": task,
                        "steps": args.steps,
                        "burn_in_hops": BURN,
                        "seed": 20261011,
                    },
                },
                checkpoint,
            )
            model_path = root / "last.onnx"
            export_causal(checkpoint, model_path)
            metrics = []
            for i, (row, _, _) in enumerate(selected):
                result = evaluator.process(
                    model_path,
                    degraded[i],
                    row["sample_rate"],
                    mode="offline",
                    high_band_gain="model",
                )
                # Same fully supported region after burn-in + gate attack, before tail.
                start, stop = (BURN + 8) * 128, SAMPLES - 256
                metrics.append(
                    evaluate_signal_triplet(
                        clean[i, start:stop],
                        degraded[i, start:stop],
                        result.audio[start:stop, 0],
                        sample_rate=row["sample_rate"],
                        cutoff_hz=8000 if task != "codec" else row["measured_cutoff_hz"],
                    )
                )
            passed = final["total"] <= initial["total"] * 0.6
            if task != "identity":
                passed &= (
                    final["high_band_spectral_convergence"]
                    <= initial["high_band_spectral_convergence"] * 0.8
                )
            record = {
                "task": task,
                "kind": kind,
                "initial": initial,
                "final": final,
                "history": history,
                "seconds": time.monotonic() - started,
                "finite_gradients": True,
                "learnability_gate_passed": bool(passed),
                "actual_rust_metrics": metrics,
                "detector_active_ratio": active,
                "checkpoint_sha256": sha256_file(checkpoint),
                "onnx_sha256": sha256_file(model_path),
            }
            report["runs"].append(record)
            write_json_report(report, output / "report.json")
    report["all_learnability_gates_passed"] = all(
        r["learnability_gate_passed"] for r in report["runs"]
    )
    write_json_report(report, output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args)
