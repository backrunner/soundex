# SPDX-License-Identifier: Apache-2.0
"""Audit numerical parity on validation mixtures without redistributing audio."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import load_checkpoint
from evaluation.parity import _resolve_rust_binary, _write_sxt
from evaluation.reporting import write_json_report
from export_onnx import _build_generator, artifact_metadata, sha256_file, validate_onnx_contract
from parity_metrics import PARITY_POLICY, POLICY_SHA256, compare_features, validate_features


def audit(checkpoint_path: Path, model_path: Path, output: Path) -> dict[str, object]:
    """Sample 32 frames per validation track, eight tracks per sample rate."""
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    validate_onnx_contract(model_path, artifact_metadata(checkpoint, checkpoint_hash))
    model = _build_generator(checkpoint)
    binary = _resolve_rust_binary(None)
    fft_size = int(checkpoint["feature_contract"]["fft_size"])
    hop_size = int(checkpoint["feature_contract"]["hop_size"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options)
    rows = []
    manifest_hashes = []
    for record in checkpoint["data"]["manifests"]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("validation manifest no longer matches the checkpoint")
        manifest_hashes.append(record["sha256"])
        rows.extend(
            (path.parent, row)
            for row in (json.loads(line) for line in path.read_text().splitlines())
            if row["split"] == "validation"
        )
    random.Random(1307).shuffle(rows)
    tracks: dict[int, set[str]] = {44100: set(), 48000: set()}
    measurements = []
    rust_measurements = []
    failures = []
    frame_count = 0
    failed_batches = 0
    window = torch.hann_window(fft_size)
    with tempfile.TemporaryDirectory(prefix="soundex-validation-parity-") as directory:
        input_path = Path(directory) / "input.sxt"
        expected_path = Path(directory) / "expected.sxt"
        for parent, row in rows:
            rate, track = int(row["sample_rate"]), row["track_id"]
            if rate not in tracks or len(tracks[rate]) >= 8 or track in tracks[rate]:
                continue
            audio_path = parent / row["degraded_path"]
            if sha256_file(audio_path) != row["degraded_checksum"]:
                raise ValueError("validation audio checksum mismatch")
            audio, actual_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            if actual_rate != rate or not np.isfinite(audio).all() or len(audio) < fft_size:
                raise ValueError("invalid validation audio")
            spectrum = torch.stft(
                torch.from_numpy(audio[:, 0].copy()),
                n_fft=fft_size,
                hop_length=hop_size,
                window=window,
                center=False,
                return_complex=True,
            )
            features = torch.stack(
                (20 * spectrum.abs().clamp_min(1e-10).log10(), torch.angle(spectrum)), 0
            ).permute(2, 0, 1)[:, :, None, :]
            if len(features) < 32:
                continue
            positions = np.linspace(0, len(features) - 1, 32, dtype=int)
            for index in range(0, 32, 2):
                tensor = features[positions[index : index + 2]].contiguous()
                with torch.no_grad():
                    expected = model(tensor).numpy()
                actual = session.run(None, {"input_features": tensor.numpy()})[0]
                metrics = compare_features(expected, actual)
                measurements.append(metrics)
                frame_count += len(tensor)
                failed = False
                try:
                    validate_features(expected, actual, label="validation-music")
                except RuntimeError as error:
                    failures.append(str(error))
                    failed = True
                _write_sxt(input_path, tensor)
                _write_sxt(expected_path, torch.from_numpy(expected))
                result = subprocess.run(
                    [str(binary), str(model_path.resolve()), str(input_path), str(expected_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                if result.returncode != 0:
                    failures.append("Rust: " + result.stderr.strip())
                    failed = True
                else:
                    report = json.loads(result.stdout.splitlines()[-1])
                    if report["policy"] != PARITY_POLICY:
                        raise ValueError("Rust utility uses a stale parity policy")
                    rust_measurements.append(report["metrics"])
                failed_batches += int(failed)
            tracks[rate].add(track)
            if all(len(group) == 8 for group in tracks.values()):
                break
    if not all(len(group) == 8 for group in tracks.values()):
        raise ValueError("audit requires eight distinct validation tracks per sample rate")
    report = {
        "audit_kind": "diagnostic-validation-numerical-parity",
        "artifact_sha256": sha256_file(model_path),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hashes,
        "policy": PARITY_POLICY,
        "policy_sha256": POLICY_SHA256,
        "seed": 1307,
        "tracks_per_sample_rate": {str(rate): len(group) for rate, group in tracks.items()},
        "unique_track_count": len(set().union(*tracks.values())),
        "frame_count": frame_count,
        "batch_size": 2,
        "failed_batches": failed_batches,
        "failure_examples": failures[:5],
        "max_metrics": {key: max(row[key] for row in measurements) for key in measurements[0]},
        "rust_max_metrics": {
            key: max(row[key] for row in rust_measurements) for key in rust_measurements[0]
        }
        if rust_measurements
        else {},
        "source": "Degraded validation mixtures; no training/test rows used. Track names, paths and audio are not distributed.",
        "scope": "Python and Rust CPU ORT versus PyTorch features and normalized inverse FFT, before deployed DSP; not model-quality or real-time evidence.",
        "torch": torch.__version__,
        "ort": ort.__version__,
    }
    write_json_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.model, args.output)
    print(
        f"Validation numerical audit: frames={report['frame_count']}, failed_batches={report['failed_batches']}"
    )


if __name__ == "__main__":
    main()
