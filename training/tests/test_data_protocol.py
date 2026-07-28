"""Tests for versioned, split-safe dataset generation and loading."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader

from data import audio_prep
from data.audio_prep import (
    align_codec_pair,
    derive_channel_roles,
    measure_cutoff_hz,
    process_mixture_file,
)
from data.dataset import (
    DatasetSourceSpec,
    SoundExDataset,
    create_balanced_dataloaders,
    set_dataset_epoch,
)
from data.preprocess_medleydb import find_mix_files
from data.preprocess_medleydb import iter_tracks as iter_medley_tracks
from data.preprocess_musdb import iter_tracks as iter_musdb_tracks
from data.preprocess_slakh import iter_tracks as iter_slakh_tracks
from data.protocol import (
    DATA_SCHEMA_VERSION,
    DataProtocolError,
    DataRecipe,
    DatasetPublisher,
    SegmentPolicy,
    assert_deployment_coverage,
    assign_hashed_split,
    canonical_track_id,
    load_recipe_from_profile,
    recipe_from_mapping,
    sha256_file,
    validate_data_config,
    validate_manifest_rows,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def recipe() -> DataRecipe:
    return load_recipe_from_profile(CONFIG_DIR / "default.yaml")


def _write_manifest_row(
    publisher: DatasetPublisher,
    recipe: DataRecipe,
    *,
    split: str,
    track_id: str,
    suffix: str,
    length: int = 256,
    sample_rate: int = 44_100,
) -> dict[str, object]:
    row_id = f"row-{suffix}"
    directory = publisher.staging / "audio" / split
    directory.mkdir(parents=True, exist_ok=True)
    time = np.arange(length, dtype=np.float32) / sample_rate
    clean = (0.25 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
    degraded = (clean * 0.9).astype(np.float32)
    clean_relative = Path("audio") / split / f"{row_id}_clean.wav"
    degraded_relative = Path("audio") / split / f"{row_id}_degraded.wav"
    clean_path = publisher.staging / clean_relative
    degraded_path = publisher.staging / degraded_relative
    sf.write(clean_path, clean, sample_rate, subtype="FLOAT")
    sf.write(degraded_path, degraded, sample_rate, subtype="FLOAT")
    codec = next(spec for spec in recipe.codecs if not spec.held_out)
    return {
        "row_id": row_id,
        "schema_version": DATA_SCHEMA_VERSION,
        "recipe_hash": recipe.hash,
        "corpus": "musdb18_hq",
        "corpus_version": "fixture",
        "track_id": track_id,
        "split": split,
        "source_id": track_id,
        "source_path": f"fixture/{track_id}",
        "source_checksum": "0" * 64,
        "sample_rate": sample_rate,
        "source_channels": 2,
        "channel_role": "mid",
        "channel_weight": 0.4,
        "codec_id": codec.id,
        "codec": codec.codec,
        "encoder": codec.encoder,
        "codec_mode": codec.mode,
        "codec_setting": codec.setting,
        "codec_weight": codec.weight,
        "sampling_weight": codec.weight * 0.4,
        "held_out": False,
        "clean_path": clean_relative.as_posix(),
        "degraded_path": degraded_relative.as_posix(),
        "start_sample": 123,
        "length_samples": length,
        "alignment_offset_samples": 17,
        "residual_alignment_samples": 0,
        "measured_cutoff_hz": 12_000.0,
        "clean_checksum": sha256_file(clean_path),
        "degraded_checksum": sha256_file(degraded_path),
    }


def _changed_recipe() -> DataRecipe:
    with (CONFIG_DIR / "default.yaml").open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    changed = copy.deepcopy(profile["data"]["recipe"])
    changed["segment"]["count_per_track"] += 1
    return recipe_from_mapping(changed)


def test_both_profiles_use_the_same_strict_recipe(recipe: DataRecipe) -> None:
    titan_recipe = load_recipe_from_profile(CONFIG_DIR / "titan_xp.yaml")

    assert titan_recipe.hash == recipe.hash
    assert recipe.sample_rates == (44_100, 48_000)
    assert {(codec.codec, codec.mode) for codec in recipe.codecs} >= {
        ("mp3", "cbr"),
        ("mp3", "vbr"),
        ("aac", "cbr"),
        ("vorbis", "vbr"),
    }
    assert any(codec.held_out for codec in recipe.codecs)


def test_unknown_data_key_reports_its_full_path() -> None:
    with (CONFIG_DIR / "default.yaml").open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    profile["data"]["recipe"]["alignment"]["mystery"] = 3

    with pytest.raises(DataProtocolError, match=r"data\.recipe\.alignment\.mystery"):
        validate_data_config(profile["data"])


def test_track_splits_are_stable_and_identity_based(recipe: DataRecipe) -> None:
    policy = recipe.split_policy("medleydb")
    identities = [
        canonical_track_id("medleydb", f"Artist_Song_{index}_MIX.wav") for index in range(200)
    ]
    first = {
        identity: assign_hashed_split(identity, recipe.seed, policy) for identity in identities
    }
    second = {
        identity: assign_hashed_split(identity, recipe.seed, policy)
        for identity in reversed(identities)
    }

    assert first == second
    assert canonical_track_id("medleydb", "Artist_Song_MIX.wav") == canonical_track_id(
        "medleydb", "Artist_Song"
    )
    sets = {
        split: {track for track, role in first.items() if role == split}
        for split in ("train", "validation", "test")
    }
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])


def test_manifest_validation_rejects_track_leakage(tmp_path: Path, recipe: DataRecipe) -> None:
    with DatasetPublisher(tmp_path, "musdb18_hq", recipe) as publisher:
        train = _write_manifest_row(
            publisher,
            recipe,
            split="train",
            track_id="musdb18_hq:same-song",
            suffix="train",
        )
        validation = _write_manifest_row(
            publisher,
            recipe,
            split="validation",
            track_id="musdb18_hq:same-song",
            suffix="validation",
        )

        with pytest.raises(DataProtocolError, match="track split leakage"):
            validate_manifest_rows(
                [train, validation], publisher.staging, recipe=recipe, audit_files=True
            )


def test_atomic_versions_do_not_merge_stale_files(tmp_path: Path, recipe: DataRecipe) -> None:
    with DatasetPublisher(tmp_path, "musdb18_hq", recipe) as first:
        first_row = _write_manifest_row(
            first,
            recipe,
            split="train",
            track_id="musdb18_hq:first",
            suffix="first",
        )
        first_target = first.publish([first_row])
    stale = first_target / "stale_degraded.wav"
    stale.write_bytes(b"stale")

    changed = _changed_recipe()
    with DatasetPublisher(tmp_path, "musdb18_hq", changed) as second:
        second_row = _write_manifest_row(
            second,
            changed,
            split="train",
            track_id="musdb18_hq:second",
            suffix="second",
        )
        second_target = second.publish([second_row])

    assert first_target != second_target
    assert stale.is_file()
    assert not (second_target / stale.name).exists()
    with pytest.raises(FileExistsError):
        DatasetPublisher(tmp_path, "musdb18_hq", recipe)
    reused = DatasetPublisher(tmp_path, "musdb18_hq", recipe, reuse_existing=True)
    assert reused.reused and reused.target == first_target


def test_interrupted_staging_never_publishes(tmp_path: Path, recipe: DataRecipe) -> None:
    publisher: DatasetPublisher | None = None
    with (
        pytest.raises(RuntimeError, match="interrupt"),
        DatasetPublisher(tmp_path, "slakh2100", recipe) as active,
    ):
        publisher = active
        (active.staging / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupt")

    assert publisher is not None
    assert not publisher.target.exists()
    assert not list(tmp_path.glob(".staging-*"))


@pytest.mark.parametrize("signal_kind", ["impulse", "noise", "tone"])
def test_fake_codec_delay_is_recovered_within_one_sample(
    signal_kind: str, recipe: DataRecipe
) -> None:
    length = 16_384
    delay = 137
    generator = np.random.default_rng(7)
    if signal_kind == "impulse":
        signal = np.zeros(length, dtype=np.float32)
        signal[2048] = 1.0
    elif signal_kind == "noise":
        signal = generator.normal(0.0, 0.1, length).astype(np.float32)
    else:
        time = np.arange(length, dtype=np.float32) / 44_100.0
        signal = (0.2 * np.sin(2.0 * np.pi * 997.0 * time)).astype(np.float32)
        signal[2048] += 0.75
    clean = np.column_stack((signal, np.roll(signal, 11))).astype(np.float32)
    degraded = np.pad(clean, ((delay, 0), (0, 0)))

    aligned = align_codec_pair(clean, degraded, 44_100, recipe)

    assert abs(aligned.offset_samples - delay) <= 1
    assert abs(aligned.residual_offset_samples) <= 1
    np.testing.assert_allclose(aligned.clean, aligned.degraded, atol=1e-6)


def test_stereo_is_encoded_before_role_derivation() -> None:
    left = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    right = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    clean = np.column_stack((left, right))
    degraded = clean * 0.5
    weights = {"mono": 1.0, "left": 0.2, "right": 0.2, "mid": 0.4, "side": 0.2}

    roles = derive_channel_roles(clean, degraded, weights)

    assert set(roles) == {"left", "right", "mid", "side"}
    np.testing.assert_allclose(roles["left"][1], degraded[:, 0])
    np.testing.assert_allclose(roles["side"][1], (degraded[:, 0] - degraded[:, 1]) / np.sqrt(2.0))


def test_cutoff_measurement_observes_the_full_segment() -> None:
    sample_rate = 44_100
    time = np.arange(16_384, dtype=np.float64) / sample_rate
    low_only = 0.2 * np.sin(2.0 * np.pi * 440.0 * time)
    late_high = low_only.copy()
    late_high[8192:] += 0.1 * np.sin(2.0 * np.pi * 12_000.0 * time[8192:])

    assert measure_cutoff_hz(low_only.astype(np.float32), sample_rate) < 2_000.0
    assert measure_cutoff_hz(late_high.astype(np.float32), sample_rate) > 11_000.0


def test_deployment_coverage_requires_every_validation_and_test_stratum(
    recipe: DataRecipe,
) -> None:
    rows = [
        {
            "split": split,
            "codec_id": codec.id,
            "sample_rate": sample_rate,
            "channel_role": role,
        }
        for split in ("validation", "test")
        for codec in recipe.codecs
        for sample_rate in recipe.sample_rates
        for role in ("left", "right", "mid", "side")
    ]

    assert_deployment_coverage(rows, recipe)
    incomplete = [row for row in rows if row["codec_id"] != recipe.codecs[-1].id]
    with pytest.raises(DataProtocolError, match="missing codec strata"):
        assert_deployment_coverage(incomplete, recipe)


def test_fake_codec_generation_publishes_the_complete_matrix(
    tmp_path: Path, recipe: DataRecipe, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_recipe = replace(recipe, segment=SegmentPolicy((0.01, 0.01), 1))
    source = tmp_path / "source.wav"
    sf.write(source, np.ones((32, 2), dtype=np.float32) * 0.1, 44_100, subtype="FLOAT")

    def fake_load_audio(_path: Path, sample_rate: int) -> np.ndarray:
        time = np.arange(1024, dtype=np.float32) / sample_rate
        return np.column_stack(
            (
                0.2 * np.sin(2.0 * np.pi * 440.0 * time),
                0.2 * np.sin(2.0 * np.pi * 659.0 * time),
            )
        ).astype(np.float32)

    def fake_encoder(audio: np.ndarray, _sample_rate: int, _codec: object) -> np.ndarray:
        return np.pad(audio, ((5, 0), (0, 0)))

    monkeypatch.setattr(audio_prep, "load_audio", fake_load_audio)
    with DatasetPublisher(tmp_path / "published", "musdb18_hq", short_recipe) as publisher:
        rows: list[dict[str, object]] = []
        for split in ("validation", "test"):
            rows.extend(
                process_mixture_file(
                    source,
                    publisher,
                    corpus="musdb18_hq",
                    corpus_version="fixture",
                    track_id=f"musdb18_hq:{split}",
                    split=split,
                    recipe=short_recipe,
                    encoder=fake_encoder,
                )
            )
        assert_deployment_coverage(rows, short_recipe)
        target = publisher.publish(rows)

    assert target.is_dir()
    assert {int(row["sample_rate"]) for row in rows} == {44_100, 48_000}
    assert {str(row["channel_role"]) for row in rows} == {"left", "right", "mid", "side"}
    assert {str(row["codec_id"]) for row in rows} == {codec.id for codec in short_recipe.codecs}
    assert all(abs(int(row["alignment_offset_samples"]) - 5) <= 1 for row in rows)


def test_manifest_loader_ignores_strays_and_is_worker_deterministic(
    tmp_path: Path, recipe: DataRecipe
) -> None:
    with DatasetPublisher(tmp_path, "musdb18_hq", recipe) as publisher:
        rows = [
            _write_manifest_row(
                publisher,
                recipe,
                split=split,
                track_id=f"musdb18_hq:{split}",
                suffix=split,
            )
            for split in ("train", "validation", "test")
        ]
        target = publisher.publish(rows)
    sf.write(target / "stray_degraded.wav", np.ones(32, dtype=np.float32), 44_100)
    validation = SoundExDataset(
        target,
        "validation",
        segment_length=128,
        sample_rates=recipe.sample_rates,
        augment=False,
    )

    assert len(validation) == 1
    first = validation[0]
    second = validation[0]
    assert torch.equal(first["degraded"], second["degraded"])
    assert first["metadata"]["track_id"] == "musdb18_hq:validation"
    assert first["metadata"]["corpus"] == "musdb18_hq"
    assert first["metadata"]["crop_start_sample"] == 64

    single_worker = next(iter(DataLoader(validation, batch_size=1, num_workers=0)))
    two_workers = next(iter(DataLoader(validation, batch_size=1, num_workers=2)))
    assert torch.equal(single_worker["degraded"], two_workers["degraded"])
    assert torch.equal(single_worker["clean"], two_workers["clean"])

    (target / str(rows[2]["clean_path"])).unlink()
    training = SoundExDataset(
        target,
        "train",
        segment_length=128,
        sample_rates=recipe.sample_rates,
        augment=False,
    )
    assert training[0]["metadata"]["split"] == "train"
    with pytest.raises(FileNotFoundError):
        SoundExDataset(target, "test", augment=False)

    (target / str(rows[1]["clean_path"])).unlink()
    with pytest.raises(FileNotFoundError):
        SoundExDataset(target, "validation", augment=False)


def test_training_augmentation_varies_by_epoch_but_is_reproducible(
    tmp_path: Path, recipe: DataRecipe
) -> None:
    with DatasetPublisher(tmp_path, "musdb18_hq", recipe) as publisher:
        row = _write_manifest_row(
            publisher,
            recipe,
            split="train",
            track_id="musdb18_hq:epoch-crop",
            suffix="epoch-crop",
            length=4096,
        )
        target = publisher.publish([row])
    first = SoundExDataset(target, "train", segment_length=512, augment=True, seed=17)
    second = SoundExDataset(target, "train", segment_length=512, augment=True, seed=17)

    set_dataset_epoch(first, 3)
    set_dataset_epoch(second, 3)
    epoch_three = first[0]
    repeated = second[0]
    assert epoch_three["metadata"]["crop_start_sample"] == repeated["metadata"]["crop_start_sample"]
    assert torch.equal(epoch_three["degraded"], repeated["degraded"])

    set_dataset_epoch(first, 4)
    epoch_four = first[0]
    assert (
        epoch_four["metadata"]["crop_start_sample"] != epoch_three["metadata"]["crop_start_sample"]
    )
    assert not torch.equal(epoch_four["degraded"], epoch_three["degraded"])


def test_validation_source_quota_is_exact_and_fails_closed(
    tmp_path: Path, recipe: DataRecipe
) -> None:
    with DatasetPublisher(tmp_path, "musdb18_hq", recipe) as publisher:
        rows = [
            _write_manifest_row(
                publisher,
                recipe,
                split=split,
                track_id=f"musdb18_hq:quota-{split}",
                suffix=f"quota-{split}",
            )
            for split in ("train", "validation")
        ]
        target = publisher.publish(rows)
    source = DatasetSourceSpec(
        name="musdb18_hq",
        label="MUSDB fixture",
        manifest_path=target / "manifest.jsonl",
        kind="real",
    )
    sampling = {
        "train_ratios": {"musdb18_hq": 1.0},
        "val_ratios": {"musdb18_hq": 1.0},
        "validation_source_quotas": {"musdb18_hq": 1},
        "samples_per_epoch": 1,
    }

    _, _, summary = create_balanced_dataloaders(
        [source],
        batch_size=1,
        segment_length=128,
        sample_rates=recipe.sample_rates,
        num_workers=0,
        sampling_config=sampling,
    )

    assert summary["validation_source_counts"] == {"musdb18_hq": 1}
    assert summary["validation_row_ids"] == ["row-quota-validation"]
    sampling["validation_source_quotas"]["musdb18_hq"] = 2
    with pytest.raises(ValueError, match="quota 2 exceeds 1 rows"):
        create_balanced_dataloaders(
            [source],
            batch_size=1,
            segment_length=128,
            sample_rates=recipe.sample_rates,
            num_workers=0,
            sampling_config=sampling,
        )


def test_mix_discovery_preserves_upstream_roles_and_excludes_stems(
    tmp_path: Path, recipe: DataRecipe
) -> None:
    for split in ("train", "test"):
        track = tmp_path / "musdb" / split / f"Song-{split}"
        track.mkdir(parents=True)
        (track / "mixture.wav").touch()
    musdb = iter_musdb_tracks(tmp_path / "musdb", recipe)
    assert {split for split, _track, _path in musdb if "test" in _track} == {"test"}

    for split in ("train", "validation", "test", "omitted"):
        track = tmp_path / "slakh" / split / f"Track-{split}"
        track.mkdir(parents=True)
        (track / "mix.flac").touch()
    slakh = iter_slakh_tracks(tmp_path / "slakh", recipe)
    assert {split for split, _track, _path in slakh} == {"train", "validation", "test"}
    assert all("omitted" not in track for _split, track, _path in slakh)

    mix_a = tmp_path / "medley" / "V1" / "Artist_Song_MIX.wav"
    mix_b = tmp_path / "medley" / "V2" / "Artist_Song_MIX.wav"
    stem = tmp_path / "medley" / "Artist_Song_STEMS" / "Fake_MIX.wav"
    for path in (mix_a, mix_b, stem):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert set(find_mix_files(tmp_path / "medley")) == {mix_a.resolve(), mix_b.resolve()}
    medley = iter_medley_tracks(tmp_path / "medley", recipe)
    assert len(medley) == 1
