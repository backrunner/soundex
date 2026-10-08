# SPDX-License-Identifier: Apache-2.0
"""Generate codec-degraded pairs from audio directories or optional source catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.audio_prep import process_mixture_file, validate_recipe_encoders
from data.catalog import load_catalog, scan_audio_directory
from data.protocol import (
    DatasetPublisher,
    assert_deployment_coverage,
    assign_hashed_split,
    load_manifest,
    load_recipe_from_profile,
)

CORPUS = "music_library"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/music_library.yaml"


def preprocess(
    catalog: Path | None,
    output: Path,
    config: Path,
    *,
    data_root: Path | None = None,
    reuse_existing: bool = False,
    corpus: str = CORPUS,
) -> Path:
    """Record source metadata and split groups before creating codec variants."""
    if (catalog is None) == (data_root is None):
        raise ValueError("provide exactly one catalog or audio directory")
    records, catalog_hash = (
        load_catalog(catalog, corpus=corpus)
        if catalog is not None
        else scan_audio_directory(data_root, corpus=corpus)
    )
    recipe = load_recipe_from_profile(config)
    policy = recipe.split_policy(corpus)
    with DatasetPublisher(output, corpus, recipe, reuse_existing=reuse_existing) as publisher:
        if publisher.reused:
            rows = load_manifest(publisher.target / "manifest.jsonl")
            if any(row.get("catalog_sha256") != catalog_hash for row in rows):
                raise ValueError("existing dataset uses a different recording/source catalog")
            return publisher.target
        validate_recipe_encoders(recipe)
        rows = []
        for record in records:
            # A version/work group stays together even when represented by multiple recordings.
            group = record["source_metadata"]["split_group"]
            split = assign_hashed_split(f"{corpus}:{group}", recipe.seed, policy)
            generated = process_mixture_file(
                record["audio"],
                publisher,
                corpus=corpus,
                corpus_version=f"audio-catalog-{catalog_hash[:16]}",
                track_id=record["track_id"],
                split=split,
                recipe=recipe,
            )
            for row in generated:
                row["source_metadata"] = record["source_metadata"]
                row["catalog_sha256"] = catalog_hash
                row["split_group"] = group
            rows.extend(generated)
            print(f"[{split}] {record['track_id']}: {len(generated)} pairs")
        # Stereo release evidence is evaluated separately, not required of each input corpus.
        assert_deployment_coverage(rows, recipe, require_stereo=False)
        (publisher.staging / "source-credits.json").write_text(
            json.dumps(
                {
                    "catalog_sha256": catalog_hash,
                    "recordings": [
                        {"track_id": record["track_id"], **record["source_metadata"]}
                        for record in records
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        target = publisher.publish(rows)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalog", type=Path)
    source.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--corpus", choices=(CORPUS, "speech_library"), default=CORPUS)
    args = parser.parse_args()
    print(
        preprocess(
            args.catalog,
            args.output_dir,
            args.config,
            data_root=args.data_root,
            reuse_existing=args.reuse_existing,
            corpus=args.corpus,
        )
    )


if __name__ == "__main__":
    main()
