"""Publish versioned MedleyDB full-mix data (never stems/raw tracks)."""

from __future__ import annotations

import argparse
import re
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
CORPUS = "medleydb"
_SKIP_DIR_EXACT = frozenset({"stems", "stem", "raw", "audio_stems", "instrumental"})
_SKIP_DIR_SUFFIXES = ("_stems", "_raw", "-stems", "-raw")
_MIX_NAME = re.compile(r".+_MIX\.wav$", re.IGNORECASE)


def is_under_stem_tree(path: Path, root: Path) -> bool:
    """Return whether a path is nested in a stems/raw directory."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    for part in relative.parts[:-1]:
        lower = part.lower()
        if lower in _SKIP_DIR_EXACT or lower.endswith(_SKIP_DIR_SUFFIXES):
            return True
    return False


def find_mix_files(data_root: Path) -> list[Path]:
    """Discover only official ``*_MIX.wav`` files outside stem/raw trees."""
    root = data_root.resolve()
    mixes = [
        path.resolve()
        for path in root.rglob("*.wav")
        if path.is_file() and _MIX_NAME.fullmatch(path.name) and not is_under_stem_tree(path, root)
    ]
    return sorted(set(mixes), key=lambda path: str(path).casefold())


def iter_tracks(data_root: Path, recipe: DataRecipe) -> list[tuple[str, str, Path]]:
    """Hash canonical identities and collapse duplicate V1/V2 mix releases."""
    policy = recipe.split_policy(CORPUS)
    by_identity: dict[str, Path] = {}
    for mixture in find_mix_files(data_root):
        track_id = canonical_track_id(CORPUS, mixture.stem)
        by_identity.setdefault(track_id, mixture)
    return [
        (
            assign_hashed_split(track_id, recipe.seed, policy),
            track_id,
            mixture,
        )
        for track_id, mixture in sorted(by_identity.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MedleyDB full mixes")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"MedleyDB root not found: {args.data_root}")
    recipe = load_recipe_from_profile(args.config)
    tracks = iter_tracks(args.data_root, recipe)
    if not tracks:
        raise FileNotFoundError(f"No *_MIX.wav tracks found under {args.data_root}")

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
                corpus_version="MedleyDB-1+2",
                track_id=track_id,
                split=split,
                recipe=recipe,
            )
            rows.extend(generated)
            print(f"  [{split}] {mixture.name}: {len(generated)} pairs")
        assert_deployment_coverage(rows, recipe)
        target = publisher.publish(rows)
    print(f"Published {len(rows)} pairs: {target}")


if __name__ == "__main__":
    main()
