"""Publish versioned Slakh2100 rendered-mixture data (never stems/MIDI)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from .audio_prep import process_mixture_file, validate_recipe_encoders
    from .protocol import (
        DataRecipe,
        DatasetPublisher,
        assert_deployment_coverage,
        assign_hashed_split,
        canonical_track_id,
        load_recipe_from_profile,
    )
except ImportError:
    from audio_prep import process_mixture_file, validate_recipe_encoders
    from protocol import (
        DataRecipe,
        DatasetPublisher,
        assert_deployment_coverage,
        assign_hashed_split,
        canonical_track_id,
        load_recipe_from_profile,
    )

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
CORPUS = "slakh2100"
_SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "eval": "test",
}
_SKIP_SPLITS = frozenset({"omitted", "omit", "exclude", "excluded"})
_MIX_NAMES = ("mix.flac", "mix.wav", "mixture.flac", "mixture.wav")


def find_mixture(track_dir: Path) -> Path | None:
    """Find only a rendered full mix directly under a track directory."""
    for name in _MIX_NAMES:
        candidate = track_dir / name
        if candidate.is_file():
            return candidate
    return None


def _metadata_split(track_dir: Path) -> str | None:
    metadata_path = track_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    for key in ("split", "subset", "set"):
        value = metadata.get(key)
        if isinstance(value, str):
            lower = value.lower()
            if lower in _SKIP_SPLITS:
                return "skip"
            if lower in _SPLIT_MAP:
                return _SPLIT_MAP[lower]
    return None


def iter_tracks(data_root: Path, recipe: DataRecipe) -> list[tuple[str, str, Path]]:
    """Preserve official splits or hash a flat layout into three roles."""
    policy = recipe.split_policy(CORPUS)
    records: list[tuple[str, str, Path]] = []
    official_dirs = [
        child
        for child in sorted(data_root.iterdir())
        if child.is_dir() and child.name.lower() in _SPLIT_MAP
    ]
    if official_dirs and bool(policy["preserve_official"]):
        for split_dir in official_dirs:
            split = _SPLIT_MAP[split_dir.name.lower()]
            for track_dir in sorted(split_dir.iterdir()):
                mixture = find_mixture(track_dir) if track_dir.is_dir() else None
                if mixture is not None:
                    records.append((split, canonical_track_id(CORPUS, track_dir.name), mixture))
        return records

    for track_dir in sorted(data_root.iterdir()):
        if not track_dir.is_dir() or track_dir.name.lower() in _SKIP_SPLITS:
            continue
        mixture = find_mixture(track_dir)
        if mixture is None:
            continue
        metadata_split = _metadata_split(track_dir)
        if metadata_split == "skip":
            continue
        track_id = canonical_track_id(CORPUS, track_dir.name)
        split = metadata_split or assign_hashed_split(track_id, recipe.seed, policy)
        records.append((split, track_id, mixture))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Slakh2100 full mixes")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Slakh2100 root not found: {args.data_root}")
    recipe = load_recipe_from_profile(args.config)
    tracks = iter_tracks(args.data_root, recipe)
    if not tracks:
        raise FileNotFoundError(f"No rendered mix tracks found under {args.data_root}")

    with DatasetPublisher(
        args.output_dir, CORPUS, recipe, reuse_existing=args.reuse_existing
    ) as publisher:
        if publisher.reused:
            print(f"Verified existing dataset version: {publisher.target}")
            return
        validate_recipe_encoders(recipe)
        rows: list[dict[str, object]] = []
        for split, track_id, mixture in tracks:
            generated = process_mixture_file(
                mixture,
                publisher,
                corpus=CORPUS,
                corpus_version="Slakh2100-redux",
                track_id=track_id,
                split=split,
                recipe=recipe,
            )
            rows.extend(generated)
            print(f"  [{split}] {mixture.parent.name}: {len(generated)} pairs")
        assert_deployment_coverage(rows, recipe)
        target = publisher.publish(rows)
    print(f"Published {len(rows)} pairs: {target}")


if __name__ == "__main__":
    main()
