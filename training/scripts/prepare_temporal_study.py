"""Prepare matched frame/causal continuations from an audited frame-refiner parent."""

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
from configuration import load_config
from export_onnx import _build_generator


def prepare(parent_path: Path, destination: Path) -> dict:
    parent_path = parent_path.resolve()
    parent = load_checkpoint(parent_path)
    config = deepcopy(parent["resolved_config"])
    spec = config["model"]["generator"].get("spectral_refiner") or {}
    if (config["audio"]["fft_size"], config["audio"]["hop_size"]) != (256, 128):
        raise ValueError("temporal study requires FFT256/hop128")
    if not spec or spec.get("history_frames", 0) != 0:
        raise ValueError("temporal study requires a frame-refiner parent")
    profile = load_config(
        Path(__file__).resolve().parents[1] / "configs/deployment_continuation.yaml"
    )
    config["training"] = deepcopy(profile["training"])
    config["training"]["sequence_warmup_frames"] = 8
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing study: {destination}")
    # Validate all profiles before creating a partially configured study.
    configs = {}
    for name, history in (("frame-control", 0), ("causal", 8)):
        current = deepcopy(config)
        if history:
            current["model"]["generator"]["spectral_refiner"]["history_frames"] = history
        current["training"]["context_frames"] = history + 1
        feature_contract_from_config(current)
        configs[name] = current
    parent_count = _build_generator(parent).count_parameters()
    width = int(spec["width"])
    counts = {"frame-control": parent_count, "causal": parent_count + width * 10 + width**2 + width}
    if max(counts.values()) > 2_000_000:
        raise ValueError("temporal study exceeds the parameter budget")
    destination.mkdir(parents=True)
    receipt = {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "recipe_sha256": parent["data"]["recipe_sha256"],
        "manifest_set_sha256": parent["data"]["manifest_set_sha256"],
        "max_epochs_per_arm": 8,
        "sequence_warmup_frames": 8,
        "supervised_frames": 8,
        "objective_version": 6,
        "selection": {
            "checkpoint": "lowest validation V6 total; high-band magnitude breaks ties",
            "candidate": "lowest CPU FP32 V6 total between the two completed arms on identical 8-warmup/8-supervised crops",
            "heldout": "freeze hashes before historical held-out recheck; no retuning",
            "promotion": "quality, listening and realtime gates remain separate; no automatic promotion",
        },
        "variants": {},
    }
    for name, current in configs.items():
        path = destination / name / "config.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        receipt["variants"][name] = {
            "config_path": str(path.resolve()),
            "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters": counts[name],
            "context_frames": current["training"]["context_frames"],
        }
    (destination / "study.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output_dir), indent=2))
