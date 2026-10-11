"""Prepare a matched long-sequence ablation of multiresolution reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import feature_contract_from_config, load_checkpoint
from models.losses import GeneratorLoss


def prepare(parent_path: Path, destination: Path) -> dict:
    parent_path = parent_path.resolve()
    parent = load_checkpoint(parent_path)
    config = deepcopy(parent["resolved_config"])
    spec = config["model"]["generator"].get("spectral_refiner") or {}
    if spec.get("history_frames", 0) or spec.get("representation") != "gain_shape":
        raise ValueError("study requires the reviewed frame-local gain_shape parent")
    if (config["audio"]["fft_size"], config["audio"]["hop_size"]) != (256, 128):
        raise ValueError("study requires FFT256/hop128")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    spec["history_frames"] = 8
    config["training"].update(
        batch_size=8,
        max_epochs=4,
        learning_rate=2e-5,
        warmup_steps=64,
        sequence_warmup_frames=8,
        context_frames=9,
        trainable_scope="all",
        batch_norm_statistics="frozen",
        validation_interval_epochs=1,
    )
    config["data"]["sampling"]["samples_per_epoch"] = 1024
    config["training"]["objective"].update(
        version=7,
        causal_sequence_frames=32,
        reconstruction_warmup_frames=32,
        adversarial_weight=0.0,
        feature_matching_weight=0.0,
        waveform_region="steady_state",
        multires_spectral_weight=0.5,
        multires_energy_weight=1.0,
        multires_temporal_weight=0.25,
        multires_low_weight=1.0,
    )
    configs = {}
    for name in ("long-control", "multires"):
        current = deepcopy(config)
        if name == "long-control":
            for key in (
                "multires_spectral_weight",
                "multires_energy_weight",
                "multires_temporal_weight",
                "multires_low_weight",
            ):
                current["training"]["objective"][key] = 0.0
        feature_contract_from_config(current)
        GeneratorLoss.from_config(current["training"]["objective"], fft_size=256, hop_size=128)
        configs[name] = current
    receipt = {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "recipe_sha256": parent["data"]["recipe_sha256"],
        "manifest_set_sha256": parent["data"]["manifest_set_sha256"],
        "epochs": 4,
        "updates": 512,
        "supervised_frames_per_arm": 131072,
        "model_warmup_frames": 8,
        "dsp_warmup_frames": 32,
        "supervised_frames": 32,
        "selection": "Use fixed epoch4/final checkpoints; compare both with common CPU FP32 full v7 loss before Rust scoring. Training totals with different weights are not comparable.",
        "scope": "Development-only experiment, no new heldout claims or automatic promotion.",
        "variants": {},
    }
    destination.mkdir(parents=True)
    for name, current in configs.items():
        path = destination / name / "config.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        receipt["variants"][name] = {
            "config_path": str(path.resolve()),
            "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (destination / "study.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output_dir), indent=2))
