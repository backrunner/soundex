from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from data.broad_coverage import (
    BoundedEnvelopeClips,
    LazyPool,
    allowed_starts,
    assert_isolated,
    make_broad_schedule,
    protected_intervals,
    repair_finite_bank,
)
from data.causal_study import SAMPLES
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file
from scripts.audit_crop_coverage import coverage
from scripts.run_subset_study import Pool


def row(key, corpus="music_library", **extra):
    return {
        "row_id": key,
        "track_id": key,
        "source_checksum": key,
        "start_sample": 0,
        "sample_rate": 48000,
        "length_samples": SAMPLES * 5,
        "corpus": corpus,
        "split": "train",
        "held_out": False,
        "sampling_weight": 1.0,
        "codec_id": "test",
        "channel_role": "mono",
        **extra,
    }


def test_source_time_exclusion_crosses_codecs_views_rates_and_alignment():
    a = row("a", start_sample=48000, alignment_offset_samples=-480)
    b = row("b", source_checksum="a", sample_rate=44100, channel_role="right", codec_id="other")
    rows = {"a": a, "b": b}
    protected = protected_intervals([{"row_id": "a", "start": 0}], rows)
    assert protected["a"][0][0] == Fraction(101, 100)
    spans = allowed_starts(b, protected)
    for start in range(b["length_samples"] - SAMPLES + 1):
        left, right = Fraction(start, 44100), Fraction(start + SAMPLES, 44100)
        p, q = protected["a"][0]
        assert any(a <= start <= z for a, z in spans) == (right <= p or left >= q)
    with pytest.raises(ValueError, match="overlap"):
        assert_isolated([{"row_id": "b", "start": 44100}], rows, protected)
    assert allowed_starts(row("short", length_samples=12), protected) == []


def test_broad_draws_reproducible_balanced_and_strictly_training_only():
    rows = {}
    for corpus, count in (("music_library", 7), ("speech_library", 3)):
        for i in range(count):
            key = f"{corpus}{i}"
            for codec in range(3):
                name = f"{key}-{codec}"
                rows[name] = row(
                    name, corpus, track_id=key, source_checksum=key, codec_id=str(codec)
                )
    rows["test"] = row("test", split="test", held_out=True)
    first = next(iter(rows))
    probes = [{"row_id": first, "start": SAMPLES}]
    a, draws, eligible = make_broad_schedule(rows, probes, 150, 7)
    assert (a, draws, eligible) == make_broad_schedule(rows, probes, 150, 7)
    assert a != make_broad_schedule(rows, probes, 150, 8)[0]
    assert eligible == {"music_library": 7, "speech_library": 3}
    tracks = Counter()
    for draw in draws:
        selected = [rows[a[i]["row_id"]] for i in draw]
        assert [r["corpus"] for r in selected] == ["music_library"] * 4 + ["speech_library"]
        assert len({r["track_id"] for r in selected}) == 5
        tracks.update(r["track_id"] for r in selected)
    assert (
        max(v for k, v in tracks.items() if k.startswith("music"))
        - min(v for k, v in tracks.items() if k.startswith("music"))
        <= 1
    )
    assert set(r["codec_id"] for r in [rows[s["row_id"]] for s in a]) == {"0", "1", "2"}
    bad = dict(rows)
    bad["leak"] = row("leak", track_id=rows[first]["track_id"], split="test", held_out=True)
    with pytest.raises(ValueError, match="split leakage"):
        make_broad_schedule(bad, probes, 1, 7)


def test_finite_repairs_preserve_valid_crops_and_unique_indices_without_target_access():
    rows = {"a": row("a")}
    protected = protected_intervals([{"row_id": "a", "start": 0}], rows)
    specs = [{"row_id": "a", "start": s, "gain_db": 0.0} for s in (SAMPLES, 10, 20, SAMPLES + 2)]
    bank, repairs = repair_finite_bank(specs, rows, protected)
    assert [s["start"] for s in bank] == [SAMPLES, SAMPLES + 1, SAMPLES + 3, SAMPLES + 2]
    assert [r["index"] for r in repairs] == [1, 2]
    assert specs[1]["start"] == 10
    assert_isolated(bank, rows, protected)


def test_coverage_deduplicates_codec_channel_and_rate_for_source_time():
    rows = {"a": row("a"), "b": row("b", source_checksum="a", channel_role="right")}
    specs = [{"row_id": key, "start": 0} for key in rows]
    report = coverage(specs * 2, rows)
    assert report["unique_supervised_source_seconds"] == pytest.approx((SAMPLES - 90 * 128) / 48000)
    assert report["unique_supervised_channel_seconds"] == pytest.approx(
        2 * report["unique_supervised_source_seconds"]
    )


REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "target/release/soundex-gain-data"
if not BINARY.exists():
    BINARY = REPO / "target/debug/soundex-gain-data"


@pytest.mark.skipif(not BINARY.exists(), reason="build soundex-gain-data for PCM integration")
def test_lazy_batches_match_eager_objective_with_silence_mono_and_bounded_cache(tmp_path):
    torch.set_num_threads(1)
    rows, specs = {}, []
    for i, rate in enumerate((44100, 48000, 48000)):
        wave = (
            np.sin(np.arange(SAMPLES + 256) * 0.27) * (0 if i == 0 else 1e-5 if i == 1 else 0.2)
        ).astype(np.float32)
        name = str(i)
        for role, pcm in (("clean", wave), ("degraded", wave * 0.8)):
            sf.write(tmp_path / f"{name}-{role}.wav", pcm, rate, subtype="FLOAT")
        rows[name] = row(
            name,
            sample_rate=rate,
            length_samples=len(wave),
            _root=str(tmp_path),
            recipe_hash="fixture",
            **{f"{role}_path": f"{name}-{role}.wav" for role in ("clean", "degraded")},
            **{
                f"{role}_checksum": sha256_file(tmp_path / f"{name}-{role}.wav")
                for role in ("clean", "degraded")
            },
        )
        specs.append({"row_id": name, "start": 128, "gain_db": 0.0})
    eager = Pool(specs, EnvelopeClips(rows, BINARY, tmp_path / "eager"))
    clips = BoundedEnvelopeClips(rows, BINARY, tmp_path / "lazy", capacity=2)
    lazy = LazyPool(specs, clips)
    for indices in ([0, 1, 2], [2, 0], [1], [0, 1, 2]):
        a, ac = eager.select(indices)
        b, bc = lazy.select(indices)
        torch.testing.assert_close(ac, bc, atol=0, rtol=0)
        for key in a.batch:
            torch.testing.assert_close(a.batch[key], b.batch[key], atol=0, rtol=0)
        for key, value in a(a.batch["x"]).items():
            torch.testing.assert_close(value, b(b.batch["x"])[key], atol=0, rtol=0)
        assert len(clips.memory) <= 2
    assert not list((tmp_path / "lazy").glob("*.npz"))
    # A new reader must detect a changed source before using a disk or RAM result.
    (tmp_path / "0-clean.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed clean PCM"):
        BoundedEnvelopeClips(rows, BINARY, tmp_path / "eager").get(specs[0])
