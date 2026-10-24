"""Long-form evaluation isolation, timeline, weighting and blinded-pack contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from evaluation.longform_listening import write_trial
from evaluation.longform_review import (
    aggregate,
    aligned_output,
    publisher_evidence,
    raw_from_trace,
    score_audio,
    select_sources,
    source_candidates,
)
from scripts.run_longform_review import endpoint_barrier


def test_transitive_source_aliases_excluded(tmp_path):
    history = tmp_path / "manifest.jsonl"
    history.write_text(json.dumps({"source_checksum": "known"}) + "\n")
    rows = [
        {"id": "a", "audio_sha256": "known", "split_group": "same-work"},
        {"id": "b", "audio_sha256": "different", "split_group": "same-work"},
    ]
    candidates, excluded = source_candidates(rows[::-1], [history])
    assert not candidates
    assert {r["reason"] for r in excluded} == {"historical_source_or_work_alias"}


def test_selection_deterministic_and_author_bounded():
    rows = [
        {
            "id": str(i),
            "audio_sha256": str(i),
            "split_group": str(i),
            "author": f"author-{i % 3}",
            "genre": "rock" if i % 2 else "folk",
        }
        for i in range(20)
    ]
    selected = select_sources(rows, per_genre=4, per_author=2)
    assert selected == select_sources(rows[::-1], per_genre=4, per_author=2)
    assert all(
        sum(r["author"] == author for r in selected) <= 2 for author in {r["author"] for r in rows}
    )
    assert len(selected) == 6


def test_publisher_evidence_rejects_license_and_original_drift(tmp_path):
    path = tmp_path / "publisher.json"
    meta = {
        "metadata": {"licenseurl": "https://creativecommons.org/licenses/by/4.0/"},
        "files": [{"name": "song.flac", "source": "original", "md5": "123"}],
    }
    path.write_text(json.dumps(meta))
    row = {
        "id": "album/song.flac",
        "license_url": "http://creativecommons.org/licenses/by/4.0/",
        "source_metadata_path": str(path),
        "publisher_original_md5": "123",
        "source_page": "https://example.com/album",
    }
    assert publisher_evidence(row)["license"] == "/licenses/by/4.0"
    with pytest.raises(ValueError):
        publisher_evidence(
            {**row, "license_url": "https://creativecommons.org/licenses/by-nc/4.0/"}
        )
    with pytest.raises(ValueError, match="checksum"):
        publisher_evidence({**row, "publisher_original_md5": "changed"})


def test_sources_are_not_reweighted_by_clip_duplication():
    rows = []
    for path, values in [
        ("native/legacy", {"a": 1.0, "b": 1.0}),
        ("native/new", {"a": 0.5, "b": 2.0}),
    ]:
        for source, score in values.items():
            rows.extend(
                {"path": path, "role": "paired", "source": source, "scores": {"mr": score}}
                for _ in range(8 if source == "a" else 1)
            )
    result = aggregate(rows)
    assert result["groups"]["native/new/paired"]["source_mean"]["mr"] == 1.25
    assert result["groups"]["native/new/paired"]["clip_mean"]["mr"] < 1.0
    assert (
        result["comparisons"]["native/new/paired versus native/legacy"]["mr"]["mean_delta"] == 0.25
    )


def test_ola_trace_alignment_and_channel_independence():
    torch.set_num_threads(1)
    rng = np.random.default_rng(8)
    wave = rng.normal(0, 0.03, (128 * 24, 2)).astype(np.float32)
    wave[:, 1] *= 0.1
    samples = torch.from_numpy(wave.T.copy())
    windows = torch.nn.functional.pad(samples, (128, 0)).unfold(-1, 256, 128)
    spectrum = torch.fft.rfft(windows * torch.hann_window(256))
    features = torch.stack((20 * spectrum.abs().clamp_min(1e-10).log10(), spectrum.angle()), dim=2)
    traces = [{"prediction": frame.reshape(-1).tolist()} for frame in features.permute(1, 0, 2, 3)]
    raw = raw_from_trace(traces, 2)
    np.testing.assert_allclose(aligned_output(raw, 256, 1024), wave[256:1280], atol=2e-7)
    with pytest.raises(ValueError):
        aligned_output(raw, len(raw) - 128, 256)


def test_spectrum_reference_zero_absolute_rms_and_stereo_error():
    torch.set_num_threads(1)
    rate = 48000
    time = np.arange(rate * 10) / rate
    left = 0.05 * np.sin(2 * np.pi * 18000 * time)
    wave = np.stack((left, -left), axis=1).astype(np.float32)
    correct = score_audio(wave, wave, rate)
    assert abs(correct["mr"]) < 1e-8
    assert correct["full_rms"] == 0
    swapped = score_audio(wave[:, ::-1].copy(), wave, rate)
    assert abs(swapped["mr"]) < 1e-8  # Magnitudes cannot establish phase/stereo fidelity.
    assert swapped["side_error_rms"] > 0.07
    assert swapped["mid_error_rms"] == 0
    assert swapped["band16000_reference_rms"] == pytest.approx(0.05 / np.sqrt(2), rel=1e-6)
    with pytest.raises(ValueError, match="ten finite"):
        score_audio(wave[:4000], wave[:4000], rate)


def test_endpoint_barrier_never_reads_models_while_training(tmp_path):
    assert (
        endpoint_barrier({"p54": str(tmp_path), "models": {"a": "/missing/model"}}, tmp_path)
        is None
    )
    (tmp_path / "complete.json").write_text('{"files":{}}')
    assert endpoint_barrier({"p54": str(tmp_path)}, tmp_path) is None


def test_pair_hashes_are_rechecked_after_waiting(tmp_path, monkeypatch):
    from scripts import run_longform_review as runner

    state = {"waited": False, "reviewed": False}
    monkeypatch.setattr(runner, "prepare", lambda _: {"hashes": {}})
    monkeypatch.setattr(runner, "prepare_pairs", lambda *_: [])
    monkeypatch.setattr(runner, "endpoint_barrier", lambda *_: {} if state["waited"] else None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: state.update(waited=True))
    monkeypatch.setattr(runner, "verify", lambda _: None)

    def check(path):
        assert state["waited"] and path.name == "prepared.json"
        raise ValueError("PCM changed while waiting")

    monkeypatch.setattr(runner, "checked_receipt", check)
    monkeypatch.setattr(runner, "review", lambda *_: state.update(reviewed=True))
    with pytest.raises(ValueError, match="PCM changed"):
        runner.run(SimpleNamespace(output=tmp_path, prepare_only=False, wait=True))
    assert not state["reviewed"]


def test_listening_pack_common_gain_and_hidden_labels(tmp_path):
    rate = 48000
    time = np.arange(rate * 10) / rate
    reference = np.sin(2 * np.pi * 1000 * time).astype(np.float32)[:, None]
    trial, secret = write_trial(
        tmp_path / "trial-01",
        reference,
        {"candidate-private": reference * 2, "untreated-private": reference * 0.5},
        rate,
        np.random.default_rng(7),
    )
    assert "private" not in json.dumps(trial)
    assert set(secret["mapping"].values()) == {
        "candidate-private",
        "untreated-private",
        "hidden_reference",
        "anchor_3500hz",
    }
    assert secret["common_gain"] == pytest.approx(0.445)
    candidate = next(k for k, v in secret["mapping"].items() if v == "candidate-private")
    actual, _ = sf.read(tmp_path / "trial-01" / f"{candidate}.wav", always_2d=True)
    ref, _ = sf.read(tmp_path / "trial-01" / "reference.wav", always_2d=True)
    np.testing.assert_allclose(actual, ref * 2, atol=3e-7)


def test_template_has_no_network_or_answer_key():
    text = (Path(__file__).parents[1] / "evaluation/listening.html").read_text()
    assert "fetch(" not in text and "XMLHttpRequest" not in text
    assert "listening-key" not in text
    assert "__TRIAL_DATA__" in text
