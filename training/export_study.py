"""Export P2 checkpoints without mislabeling them as legacy resumable checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from artifact_contract import MAX_ARTIFACT_BYTES, MAX_PARAMETERS
from checkpoint import feature_contract_from_config
from data.causal_study import digest
from evaluation.reporting import sha256_file
from export_causal import export_causal
from export_onnx import _embed_metadata, _export_graph, artifact_metadata, validate_onnx_contract
from export_validation import validate_ort_parity
from models.causal_restorer import CausalRestorer
from models.generator import SoundExGenerator

LEGACY_CONFIG = {
    "channels": [24, 48, 96, 96],
    "bottleneck_blocks": 2,
    "expand_ratio": 4,
    "cross_stream_interactions": False,
    "circular_phase_features": False,
    "spectral_refiner": {"width": 128, "representation": "gain_shape", "bins": 129},
}


def make_model(kind):
    if kind == "legacy":
        return SoundExGenerator(**LEGACY_CONFIG)
    return CausalRestorer(kind)


def export_study(checkpoint_path, output_path):
    checkpoint_path, output_path = Path(checkpoint_path), Path(output_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["format"] != "soundex-p2-v1":
        raise ValueError("not a P2 checkpoint")
    if state["config"]["objective_version"] != "causal-reconstruction-v1":
        raise ValueError("unknown objective")
    if state["kind"] != "legacy":
        return export_causal(checkpoint_path, output_path)
    model = make_model("legacy").eval()
    model.load_state_dict(state["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) > MAX_PARAMETERS:
        raise ValueError("model too large")
    config = {
        "model": {"generator": LEGACY_CONFIG},
        "audio": {
            "fft_size": 256,
            "hop_size": 128,
            "sample_rates": [44100, 48000],
            "crossover_width_hz": 1000,
        },
        "training": {"context_frames": 1, "sequence_warmup_frames": 64},
        "data": {"recipe": {"sample_rates": [44100, 48000]}},
    }
    feature = feature_contract_from_config(config)
    metadata = artifact_metadata(
        {
            "feature_contract": feature,
            "model": {"architecture": {"name": "soundex-generator", "major": 1, "minor": 3}},
            "hashes": {"resolved_config_sha256": digest(state["config"])},
            "data": {
                "recipe_sha256": state["provenance"]["data_recipe_sha256"],
                "manifest_set_sha256": state["provenance"]["manifest_set_sha256"],
            },
        },
        sha256_file(checkpoint_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(".partial.onnx")
    try:
        _export_graph(model, staged, (2, 1, 129))
        _embed_metadata(staged, metadata)
        validate_onnx_contract(staged, metadata)
        validate_ort_parity(model, staged)
        if staged.stat().st_size >= MAX_ARTIFACT_BYTES:
            raise ValueError("artifact too large")
        staged.replace(output_path)
    finally:
        staged.unlink(missing_ok=True)
    return metadata
