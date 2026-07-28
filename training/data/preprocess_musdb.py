"""Publish versioned MUSDB18-HQ full-mixture data (never stems)."""

from __future__ import annotations

import argparse
from pathlib import Path

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
CORPUS = "musdb18_hq"


def iter_tracks(data_root: Path, recipe: DataRecipe) -> list[tuple[str, str, Path]]:
    """Return stable (split, track_id, mixture) records from official folders."""
    policy = recipe.split_policy(CORPUS)
    records: list[tuple[str, str, Path]] = []
    for official_split in ("train", "test"):
        subset = data_root / official_split
        if not subset.is_dir():
            continue
        for track_dir in sorted(subset.iterdir()):
            mixture = track_dir / "mixture.wav"
            if not track_dir.is_dir() or not mixture.is_file():
                continue
            track_id = canonical_track_id(CORPUS, track_dir.name)
            if official_split == "test" and bool(policy["preserve_official_test"]):
                split = "test"
            else:
                split = assign_hashed_split(
                    track_id,
                    recipe.seed,
                    policy,
                    allowed_splits=("train", "validation"),
                )
            records.append((split, track_id, mixture))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MUSDB18-HQ mixtures")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"MUSDB18-HQ root not found: {args.data_root}")
    recipe = load_recipe_from_profile(args.config)
    tracks = iter_tracks(args.data_root, recipe)
    if not tracks:
        raise FileNotFoundError(f"No train/test mixture.wav tracks found under {args.data_root}")

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
                corpus_version="MUSDB18-HQ",
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
