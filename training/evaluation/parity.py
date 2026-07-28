"""Generate artifact-bound PyTorch/ORT/Rust parity evidence."""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from evaluation.reporting import sha256_file, write_json_report
from export_onnx import _build_generator, artifact_metadata, validate_onnx_contract
from export_validation import deterministic_parity_inputs, validate_ort_parity

PARITY_EVIDENCE_SCHEMA_VERSION = 1
PARITY_SUITE = "pytorch-ort-rust-v1"


def generate_parity_evidence(
    *,
    checkpoint_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    rust_binary: str | Path | None = None,
) -> dict[str, object]:
    """Run every plan-004 parity case and atomically write passing evidence."""
    checkpoint_file = Path(checkpoint_path).resolve()
    model_file = Path(model_path).resolve()
    checkpoint = load_checkpoint(checkpoint_file)
    checkpoint_hash = sha256_file(checkpoint_file)
    artifact_hash = sha256_file(model_file)
    validate_onnx_contract(model_file, artifact_metadata(checkpoint, checkpoint_hash))
    generator = _build_generator(checkpoint)
    ort_max, ort_mean = validate_ort_parity(generator, model_file)
    feature = checkpoint["feature_contract"]
    parity_inputs = deterministic_parity_inputs(
        fft_size=int(feature["fft_size"]),
        hop_size=int(feature["hop_size"]),
    )
    binary = _resolve_rust_binary(rust_binary)
    case_names: list[str] = []
    with tempfile.TemporaryDirectory(prefix="soundex-parity-") as directory:
        root = Path(directory)
        for case_name, input_tensor in parity_inputs:
            with torch.no_grad():
                expected = generator(input_tensor)
            input_path = root / f"{case_name}.input.sxt"
            expected_path = root / f"{case_name}.expected.sxt"
            _write_sxt(input_path, input_tensor)
            _write_sxt(expected_path, expected)
            result = subprocess.run(
                [str(binary), str(model_file), str(input_path), str(expected_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Rust parity case {case_name!r} failed: {detail}")
            case_names.append(case_name)
    evidence: dict[str, object] = {
        "schema_version": PARITY_EVIDENCE_SCHEMA_VERSION,
        "suite": PARITY_SUITE,
        "passed": True,
        "artifact_sha256": artifact_hash,
        "checkpoint_sha256": checkpoint_hash,
        "training_manifest_corpora": sorted(
            {str(manifest["corpus"]) for manifest in checkpoint["data"]["manifests"]}
        ),
        "cases": case_names,
        "pytorch_ort_max_absolute_error": ort_max,
        "pytorch_ort_max_mean_error": ort_mean,
        "rust_tolerance": 1e-5,
    }
    write_json_report(evidence, output_path)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rust-binary", type=Path)
    args = parser.parse_args()
    evidence = generate_parity_evidence(
        checkpoint_path=args.checkpoint,
        model_path=args.model,
        output_path=args.output,
        rust_binary=args.rust_binary,
    )
    print(
        f"Parity evidence: {args.output} "
        f"artifact={evidence['artifact_sha256']} cases={len(evidence['cases'])}"
    )


def _resolve_rust_binary(path: str | Path | None) -> Path:
    repository = Path(__file__).resolve().parents[2]
    configured = path or os.environ.get("SOUNDEX_MODEL_PARITY")
    binary = Path(configured) if configured else repository / "target/debug/soundex-model-parity"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    result = subprocess.run(
        ["cargo", "build", "--locked", "-p", "soundex-core", "--bin", "soundex-model-parity"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not binary.is_file():
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to build Rust parity utility: {detail}")
    return binary


def _write_sxt(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    if array.ndim != 4:
        raise ValueError("SXT1 parity tensors must have rank four")
    header = b"SXT1" + struct.pack("<4I", *array.shape)
    path.write_bytes(header + array.tobytes(order="C"))


if __name__ == "__main__":
    main()
