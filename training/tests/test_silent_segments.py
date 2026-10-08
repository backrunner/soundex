"""Silent crops and mono-derived side channels must not abort corpus preparation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data import audio_prep
from data.audio_prep import SilentAudioError, process_mixture_file, validate_pair_samples
from data.protocol import DataProtocolError, DatasetPublisher, load_recipe_from_profile


@pytest.mark.parametrize("identical_channels", [False, True])
def test_silent_crops_are_omitted_and_active_roles_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identical_channels: bool
) -> None:
    recipe = load_recipe_from_profile(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    recipe = replace(recipe, sample_rates=(44100,), codecs=(recipe.codecs[0],))
    clean = np.random.default_rng(17).normal(0.0, 0.1, (1024, 2)).astype(np.float32)
    clean[:256] = 0.0
    if identical_channels:
        clean[:, 1] = clean[:, 0]
    source = tmp_path / "mixture.wav"
    sf.write(source, clean, 44100, subtype="FLOAT")
    monkeypatch.setattr(audio_prep, "deterministic_segments", lambda *args: [(0, 256), (512, 256)])

    with DatasetPublisher(tmp_path / "processed", "musdb18_hq", recipe) as publisher:
        rows = process_mixture_file(
            source,
            publisher,
            corpus="musdb18_hq",
            corpus_version="fixture",
            track_id="musdb18_hq:silence-fixture",
            split="train",
            recipe=recipe,
            encoder=lambda audio, sample_rate, codec: audio * 0.9,
        )
        target = publisher.publish(rows)

    expected_roles = {"left", "right", "mid"}
    if not identical_channels:
        expected_roles.add("side")
    assert {row["channel_role"] for row in rows} == expected_roles
    assert len(rows) == len(expected_roles)
    assert all(row["start_sample"] == 512 for row in rows)
    for row in rows:
        validate_pair_samples(
            sf.read(target / str(row["clean_path"]))[0],
            sf.read(target / str(row["degraded_path"]))[0],
        )


def test_silent_pair_errors_are_distinct_from_corrupt_audio() -> None:
    silence = np.zeros(256, dtype=np.float32)
    with pytest.raises(SilentAudioError, match="silent"):
        validate_pair_samples(silence, silence)
    for degraded, message in [
        (np.full(256, np.nan), "non-finite"),
        (np.full(256, 9.0), "peak"),
        (np.zeros(128), "shape mismatch"),
    ]:
        with pytest.raises(DataProtocolError, match=message) as caught:
            validate_pair_samples(silence, degraded)
        assert not isinstance(caught.value, SilentAudioError)


def test_wholly_silent_source_is_audited_without_aborting_preparation(tmp_path: Path) -> None:
    recipe = load_recipe_from_profile(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    recipe = replace(recipe, sample_rates=(44100,), codecs=(recipe.codecs[0],))
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros((1024, 1), dtype=np.float32), 44100, subtype="FLOAT")
    with DatasetPublisher(tmp_path / "processed", "slakh2100", recipe) as publisher:
        rows = process_mixture_file(
            source,
            publisher,
            corpus="slakh2100",
            corpus_version="fixture",
            track_id="slakh2100:silent-fixture",
            split="train",
            recipe=recipe,
            encoder=lambda audio, sample_rate, codec: audio.copy(),
        )
        assert not rows
        audits = list((publisher.staging / "rejections").glob("*.json"))
        assert len(audits) == 1
        assert "silent" in json.loads(audits[0].read_text())[0]["reason"]
