"""Prepare matched representation/capacity experiments from a reviewed parent.

Run from training/: python scripts/prepare_architecture_study.py --help
This writes configurations and a pre-scoring selection rule, not model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint import load_checkpoint
from configuration import load_config
from models.generator import SoundExGenerator

VARIANTS = {
    "polar-small": {"width": 128, "representation": "polar", "bins": 129},
    "shape-small": {"width": 128, "representation": "gain_shape", "bins": 129},
    "shape-large": {"width": 512, "representation": "gain_shape", "bins": 129},
}


def prepare(parent_path: Path, destination: Path) -> dict:
    parent_path = parent_path.resolve()
    parent = load_checkpoint(parent_path)
    profile = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "deployment_continuation.yaml"
    )
    config = deepcopy(parent["resolved_config"])
    if (config["audio"]["fft_size"], config["audio"]["hop_size"]) != (256, 128):
        raise ValueError("this study requires FFT256/hop128")
    if config["model"]["generator"].get("spectral_refiner") is not None:
        raise ValueError("this study requires a parent without a spectral refiner")
    config["training"] = deepcopy(profile["training"])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing study: {destination}")
    destination.mkdir(parents=True)
    receipt = {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "recipe_sha256": parent["data"]["recipe_sha256"],
        "manifest_set_sha256": parent["data"]["manifest_set_sha256"],
        "selection": {
            "checkpoint": "lowest validation v6 total; high-band magnitude breaks ties",
            "candidate": "lowest CPU FP32 validation v6 total among all completed arms",
            "heldout": "freeze candidate ONNX hash before held-out recheck; no retuning",
            "promotion": "quality, listening and realtime gates remain independent; no automatic promotion",
        },
        "max_epochs_per_arm": 8,
        "variants": {},
    }
    # A fresh control allows exact comparisons on machines where older runs differ.
    for name, spec in {"control": None, **VARIANTS}.items():
        current = deepcopy(config)
        if spec is not None:
            current["model"]["generator"]["spectral_refiner"] = spec
        gen_config = current["model"]["generator"]
        model = SoundExGenerator(
            channels=gen_config["channels"],
            bottleneck_blocks=gen_config["bottleneck_blocks"],
            expand_ratio=gen_config["expand_ratio"],
            cross_stream_interactions=gen_config.get("cross_stream_interactions", False),
            circular_phase_features=gen_config.get("circular_phase_features", False),
            spectral_refiner=spec,
        )
        count = model.count_parameters()
        if count > 2_000_000:
            raise ValueError(f"{name} exceeds the parameter budget: {count}")
        path = destination / name / "config.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        receipt["variants"][name] = {
            "config_path": str(path.resolve()),
            "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters": count,
            "spectral_refiner": spec,
        }
    (destination / "study.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
