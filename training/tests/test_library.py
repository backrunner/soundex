# SPDX-License-Identifier: Apache-2.0
"""Check diverse audio ingestion, source bindings and disjoint codec-pair splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from configuration import load_config
from data import audio_prep, preprocess_library
from data.catalog import load_catalog, scan_audio_directory
from data.protocol import assign_hashed_split, load_manifest, load_recipe_from_profile, sha256_file


def recording(tmp_path: Path) -> tuple[Path, dict]:
    audio = tmp_path / "recording.wav"
    samples = np.sin(np.arange(44100) * (2 * np.pi * 1100 / 44100)) * 0.1
    sf.write(audio, samples, 44100, subtype="PCM_24")
    proof = tmp_path / "grant.txt"
    proof.write_text("Test-created composition and recording dedicated under CC0.\n")
    row = {
        "id": "owned-test-recording",
        "path": audio.name,
        "audio_sha256": sha256_file(audio),
        "evidence_path": proof.name,
        "evidence_sha256": sha256_file(proof),
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "author": "Synthetic test creator",
        "source_url": "https://example.test/recording",
        "reviewed_by": "Test reviewer",
        "reviewed": True,
        "rights_coverage": "composition-and-recording",
        "split_group": "test-work",
    }
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(row) + "\n")
    return catalog, row


def test_catalog_metadata_is_optional_and_declared_hashes_bind_files(tmp_path: Path) -> None:
    catalog, row = recording(tmp_path)
    records, checksum = load_catalog(catalog)
    assert len(records) == 1 and len(checksum) == 64
    minimal = {"id": row["id"], "path": row["path"]}
    catalog.write_text(json.dumps(minimal) + "\n")
    records, _ = load_catalog(catalog)
    assert "license" not in records[0]["source_metadata"]
    # The importer records a declaration without pretending to approve its legal scope.
    catalog.write_text(json.dumps({**minimal, "license": "custom-permission"}) + "\n")
    records, _ = load_catalog(catalog)
    assert records[0]["source_metadata"]["license"] == "custom-permission"
    for patch, message in (
        ({"audio_sha256": "0" * 64}, "source audio checksum"),
        ({"evidence_sha256": "0" * 64}, "evidence checksum"),
    ):
        catalog.write_text(json.dumps({**row, **patch}) + "\n")
        with pytest.raises(ValueError, match=message):
            load_catalog(catalog)
    catalog.write_text(json.dumps(row) + "\n" + json.dumps({**row, "id": "duplicate"}) + "\n")
    assert len(load_catalog(catalog)[0]) == 1


def test_folder_ingestion_keeps_original_rate_and_deduplicates_copies(tmp_path: Path) -> None:
    _catalog, row = recording(tmp_path)
    audio = tmp_path / row["path"]
    sf.write(audio, np.ones(1600) * 0.1, 16000, subtype="PCM_16")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "copy.wav").write_bytes(audio.read_bytes())
    records, _ = scan_audio_directory(tmp_path)
    assert len(records) == 1
    metadata = records[0]["source_metadata"]
    assert metadata["original_sample_rate"] == 16000
    assert metadata["split_group"] == sha256_file(audio)
    assert "license" not in metadata


def test_different_albums_can_share_recording_filenames(tmp_path: Path) -> None:
    for index in range(2):
        folder = tmp_path / f"album-{index}"
        folder.mkdir()
        sf.write(folder / "track01.wav", np.ones(1600) * (0.1 + index * 0.1), 16000)
    records, _ = scan_audio_directory(tmp_path)
    assert len(records) == 2
    assert len({row["track_id"] for row in records}) == 2


def test_dotted_work_numbers_do_not_collapse_recording_ids(tmp_path: Path) -> None:
    for index in range(2):
        sf.write(
            tmp_path / f"Bach-Bwv.988-{index:02d}.flac",
            np.ones(1600) * (0.1 + index * 0.1),
            16000,
        )
    records, _ = scan_audio_directory(tmp_path)
    assert len(records) == 2
    assert len({row["track_id"] for row in records}) == 2


def test_lossy_content_cannot_be_hidden_by_a_wav_extension(tmp_path: Path) -> None:
    audio = tmp_path / "fake-master.wav"
    sf.write(audio, np.sin(np.arange(1600)) * 0.1, 16000, format="OGG", subtype="VORBIS")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({"id": "fake-master", "path": audio.name}) + "\n")
    with pytest.raises(ValueError, match="lossless PCM/FLAC"):
        load_catalog(catalog)


@pytest.mark.parametrize("channels", [1, 2])
def test_library_publishes_source_metadata_and_grouped_codec_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    config_path = tmp_path / "config.yaml"
    config = load_config(Path(__file__).parents[1] / "configs/music_library.yaml")
    config["data"]["recipe"]["segment"] = {"duration_seconds": [0.02, 0.02], "count_per_track": 1}
    config_path.write_text(yaml.safe_dump(config))
    recipe = load_recipe_from_profile(config_path)
    policy = recipe.split_policy("music_library")
    groups = {}
    for index in range(200):
        group = f"test-work-{index}"
        split = assign_hashed_split(f"music_library:{group}", recipe.seed, policy)
        groups.setdefault(split, group)
    assert set(groups) == {"train", "validation", "test"}
    records = []
    for index, group in enumerate(
        [groups["train"], groups["train"], groups["validation"], groups["test"]]
    ):
        folder = tmp_path / f"source-{index}"
        folder.mkdir()
        _catalog, row = recording(folder)
        time = np.arange(4410) / 44100
        left = 0.1 * np.sin(time * 2 * np.pi * (1100 + 100 * index))
        right = 0.08 * np.sin(time * 2 * np.pi * (7300 + 100 * index))
        audio = left if channels == 1 else np.column_stack((left, right))
        sf.write(folder / row["path"], audio, 44100, subtype="PCM_24")
        row.update(
            {
                "id": f"test-{index}",
                "path": f"source-{index}/recording.wav",
                "evidence_path": f"source-{index}/grant.txt",
                "split_group": group,
                "audio_sha256": sha256_file(folder / "recording.wav"),
            }
        )
        records.append(row)
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("\n".join(json.dumps(row) for row in records))

    # Test source/metadata/codec/manifest plumbing without making restoration claims.
    def process(*args, **kwargs):
        return audio_prep.process_mixture_file(*args, **kwargs, encoder=lambda audio, *_args: audio)

    monkeypatch.setattr(preprocess_library, "process_mixture_file", process)
    monkeypatch.setattr(preprocess_library, "validate_recipe_encoders", lambda _recipe: None)
    target = preprocess_library.preprocess(catalog, tmp_path / "processed", config_path)
    rows = load_manifest(target / "manifest.jsonl")
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    by_group = {}
    for row in rows:
        by_group.setdefault(row["split_group"], set()).add(row["split"])
        assert row["source_checksum"] == row["source_metadata"]["audio_sha256"]
        assert len(row["catalog_sha256"]) == 64
        assert not (row["split"] == "train" and row["held_out"])
    assert all(len(splits) == 1 for splits in by_group.values())
    credits = json.loads((target / "source-credits.json").read_text())
    assert len(credits["recordings"]) == 4
    assert (
        preprocess_library.preprocess(
            catalog, tmp_path / "processed", config_path, reuse_existing=True
        )
        == target
    )
    records[0]["author"] = "Changed credit"
    catalog.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ValueError, match="different recording/source catalog"):
        preprocess_library.preprocess(
            catalog, tmp_path / "processed", config_path, reuse_existing=True
        )
