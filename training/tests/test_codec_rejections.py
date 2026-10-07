"""Only alignment-invalid degradations are excluded, with an auditable record."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data import audio_prep
from data.audio_prep import CodecAlignmentError, align_codec_pair, process_mixture_file
from data.protocol import DataProtocolError, DatasetPublisher, load_recipe_from_profile


def test_alignment_check_still_rejects_excess_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = load_recipe_from_profile(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    delays = iter([0, 3])
    monkeypatch.setattr(audio_prep, "estimate_delay_samples", lambda *a, **k: next(delays))
    clean = np.ones((1024, 2), dtype=np.float32) * 0.1
    with pytest.raises(CodecAlignmentError, match="3 exceeds tolerance 1"):
        align_codec_pair(clean, clean, 44100, recipe)


@pytest.mark.parametrize("corrupt", [False, True])
def test_failed_codec_is_audited_while_corrupt_audio_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    recipe = load_recipe_from_profile(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    recipe = replace(recipe, sample_rates=(44100,), codecs=recipe.codecs[:2])
    source = tmp_path / "mixture.wav"
    clean = np.random.default_rng(7).normal(0.0, 0.1, (1024, 2)).astype(np.float32)
    sf.write(source, clean, 44100, subtype="FLOAT")
    original = audio_prep.align_codec_pair
    calls = 0

    def alignment(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if corrupt:
                raise DataProtocolError("non-finite samples")
            raise CodecAlignmentError("residual codec alignment 3 exceeds tolerance 1")
        return original(*args, **kwargs)

    monkeypatch.setattr(audio_prep, "align_codec_pair", alignment)
    with DatasetPublisher(tmp_path / "processed", "musdb18_hq", recipe) as publisher:
        if corrupt:
            with pytest.raises(DataProtocolError, match="non-finite"):
                process_mixture_file(
                    source,
                    publisher,
                    corpus="musdb18_hq",
                    corpus_version="fixture",
                    track_id="musdb18_hq:codec-fixture",
                    split="train",
                    recipe=recipe,
                    encoder=lambda audio, sample_rate, codec: audio * 0.9,
                )
            return
        rows = process_mixture_file(
            source,
            publisher,
            corpus="musdb18_hq",
            corpus_version="fixture",
            track_id="musdb18_hq:codec-fixture",
            split="train",
            recipe=recipe,
            encoder=lambda audio, sample_rate, codec: audio * 0.9,
        )
        assert rows and {row["codec_id"] for row in rows} == {recipe.codecs[1].id}
        target = publisher.publish(rows)
    audits = list((target / "rejections").glob("*.json"))
    assert len(audits) == 1
    rejection = json.loads(audits[0].read_text())[0]
    assert rejection["codec_id"] == recipe.codecs[0].id
    assert rejection["sample_rate"] == 44100
    assert "3 exceeds tolerance 1" in rejection["reason"]
