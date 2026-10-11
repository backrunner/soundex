"""Fit a bounded ten-coefficient DSP calibration using train-only Rust observations."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.protocol import load_manifest, sha256_file
from evaluation.rust_bridge import write_sxa


def is_inner_validation(track: str) -> bool:
    return int(hashlib.sha256(("gain-inner-v1:" + track).encode()).hexdigest()[:8], 16) % 5 == 0


def recording_weights(tracks: np.ndarray, corpora: np.ndarray) -> np.ndarray:
    result = np.zeros(len(tracks), dtype=np.float64)
    for corpus, fraction in [("music_library", 253 / 254), ("speech_library", 1 / 254)]:
        selected = corpora == corpus
        ids, counts = np.unique(tracks[selected], return_counts=True)
        if not len(ids):
            raise ValueError("both reviewed music and speech are required")
        for track, count in zip(ids, counts, strict=True):
            result[selected & (tracks == track)] = fraction / len(ids) / count
    return result


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float) -> np.ndarray:
    # Huber IRLS; normalization makes the penalty independent of frame count.
    current = weights.copy()
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 1e-8
    coefficients = np.zeros(x.shape[1])
    for _ in range(8):
        coefficients = np.linalg.solve(
            x.T @ (current[:, None] * x) + regularizer, x.T @ (current * y)
        )
        residual = np.abs(y - x @ coefficients)
        current = weights * np.minimum(1.0, 3.0 / np.maximum(residual, 1e-8))
    if not np.isfinite(coefficients).all() or np.max(np.abs(coefficients)) > 64:
        raise ValueError("fit exceeds runtime coefficient contract")
    return coefficients


def collect(row: dict, binary: Path, context: int) -> tuple[np.ndarray, np.ndarray, dict]:
    root = Path(row["_root"])
    for variant in ["clean", "degraded"]:
        if sha256_file(root / row[variant + "_path"]) != row[variant + "_checksum"]:
            raise ValueError("audio checksum mismatch")
    clean, rate = sf.read(root / row["clean_path"], dtype="float32", always_2d=True)
    degraded, degraded_rate = sf.read(root / row["degraded_path"], dtype="float32", always_2d=True)
    if clean.shape != degraded.shape or rate != degraded_rate or rate != row["sample_rate"]:
        raise ValueError("unaligned audio")
    # Retain the beginning so observed state never depends on unseen/future frames.
    clean = clean[: 4 * rate]
    degraded = degraded[: 4 * rate]
    with tempfile.TemporaryDirectory(prefix="soundex-gain-fit-") as tmp:
        p = Path(tmp)
        write_sxa(p / "input.sxa", degraded, rate)
        write_sxa(p / "clean.sxa", clean, rate)
        subprocess.run(
            [
                str(binary),
                str(p / "input.sxa"),
                str(p / "clean.sxa"),
                str(p / "trace.json"),
                str(context),
            ],
            check=True,
            capture_output=True,
        )
        trace = json.loads((p / "trace.json").read_text())
    valid = [
        r
        for r in trace
        if r["active"]
        and r["hop"] >= 32
        and r["hop"] % 8 == 0
        and r["hop"] * 128 < len(clean)
        and r["edge_rms"] > 1e-4
        and r["clean_high_rms"] > 1e-7
    ]
    x = np.asarray([r["features"] for r in valid], dtype=np.float64).reshape(-1, 10)
    y = np.asarray([20 * np.log10(r["clean_high_rms"] / r["edge_rms"]) + 6 for r in valid]).clip(
        -24, 24
    )
    receipt = {
        k: row[k]
        for k in [
            "row_id",
            "track_id",
            "corpus",
            "split",
            "sample_rate",
            "codec_id",
            "clean_checksum",
            "degraded_checksum",
            "measured_cutoff_hz",
        ]
    }
    receipt.update(
        frames=len(valid),
        mean_runtime_cutoff_hz=float(np.mean([r["cutoff_hz"] for r in trace])),
        inner_validation=is_inner_validation(row["track_id"]),
    )
    return x, y, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, choices=[1, 9], default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = []
    manifests = []
    for path in args.manifest:
        rows = load_manifest(path)
        train_tracks = {r["track_id"] for r in rows if r["split"] == "train"}
        other_tracks = {r["track_id"] for r in rows if r["split"] != "train"}
        if train_tracks & other_tracks:
            raise ValueError("recording split leakage")
        manifests.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for track in sorted(train_tracks):
            for rate in [44100, 48000]:
                candidates = sorted(
                    (
                        r
                        for r in rows
                        if r["track_id"] == track
                        and r["split"] == "train"
                        and r["sample_rate"] == rate
                    ),
                    key=lambda r: hashlib.sha256(
                        ("gain-select-v1:" + r["row_id"]).encode()
                    ).hexdigest(),
                )
                # Two codec families per recording/rate when available.
                codecs = set()
                for row in candidates:
                    if row["codec_id"] in codecs:
                        continue
                    selected.append({**row, "_root": str(path.resolve().parent)})
                    codecs.add(row["codec_id"])
                    if len(codecs) == 2:
                        break
    args.output.mkdir(parents=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, value in enumerate(
            pool.map(lambda r: collect(r, args.binary.resolve(), args.context), selected), 1
        ):
            results.append(value)
            if i % 100 == 0:
                print(f"Collected {i}/{len(selected)}", flush=True)
    x = np.concatenate([r[0] for r in results])
    y = np.concatenate([r[1] for r in results])
    tracks = np.concatenate([np.repeat(r[2]["track_id"], len(r[0])) for r in results])
    corpora = np.concatenate([np.repeat(r[2]["corpus"], len(r[0])) for r in results])
    inner = np.asarray([is_inner_validation(t) for t in tracks])
    train_weights = recording_weights(tracks[~inner], corpora[~inner])
    valid_weights = recording_weights(tracks[inner], corpora[inner])
    scores = []
    for penalty in [0.001, 0.01, 0.1, 1.0]:
        coefficients = fit_ridge(x[~inner], y[~inner], train_weights, penalty)
        prediction = (x[inner] @ coefficients).clip(-6, 6)
        scores.append(
            {
                "penalty": penalty,
                "weighted_mae_db": float(valid_weights @ np.abs(prediction - y[inner])),
                "coefficients": coefficients.tolist(),
            }
        )
    choice = min(scores, key=lambda r: r["weighted_mae_db"])
    coefficients = fit_ridge(x, y, recording_weights(tracks, corpora), choice["penalty"]).astype(
        np.float32
    )
    artifact = dict(
        schema_version=1,
        feature_contract="soundex-edge-gain-v1",
        fft_size=256,
        hop_size=128,
        crossover_width_hz=1000,
        base_ratio_db=-6,
        correction_limit_db=6,
        coefficients=coefficients.tolist(),
    )
    (args.output / "calibration.json").write_text(json.dumps(artifact, indent=2) + "\n")
    np.savez_compressed(
        args.output / "observations.npz",
        features=x,
        labels=y,
        tracks=tracks,
        corpora=corpora,
        inner_validation=inner,
    )
    report = dict(
        schema_version=1,
        model="bounded linear edge target",
        fitted_parameters=10,
        frames=len(y),
        recordings=len(set(tracks)),
        context_frames=args.context,
        selection="train-only recording-disjoint inner validation MAE; refit selected penalty on all sampled training recordings; no development audio used",
        manifests=manifests,
        collector_sha256=sha256_file(args.binary),
        source_git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        baseline_inner_mae_db=float(valid_weights @ np.abs(y[inner])),
        candidates=scores,
        selected_penalty=choice["penalty"],
        calibration_sha256=sha256_file(args.output / "calibration.json"),
        rows=[r[2] for r in results],
    )
    (args.output / "fit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "frames",
                    "recordings",
                    "baseline_inner_mae_db",
                    "selected_penalty",
                    "calibration_sha256",
                ]
            }
        ),
        flush=True,
    )
    print("inner validation", scores, flush=True)


if __name__ == "__main__":
    main()
