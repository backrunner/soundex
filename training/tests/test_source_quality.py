# SPDX-License-Identifier: Apache-2.0
"""Source audits catch damaged masters and preserve legitimate mono/stereo signals."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.source_quality import inspect_master


def test_corrupt_bytes_and_checksum_fail(tmp_path: Path) -> None:
    path = tmp_path / "broken.flac"
    path.write_bytes(b"not an audio file")
    with pytest.raises(sf.LibsndfileError):
        inspect_master(path)
    sf.write(path, np.sin(np.arange(44100) * 0.1) * 0.1, 44100)
    with pytest.raises(ValueError, match="checksum"):
        inspect_master(path, "0" * 64)


@pytest.mark.parametrize("channels", [1, 2])
def test_quiet_tonal_music_is_not_declared_lossy(tmp_path: Path, channels: int) -> None:
    path = tmp_path / "music.wav"
    tone = (np.sin(np.arange(44100) * (2 * np.pi * 440 / 44100)) * 0.02)[:, None]
    samples = tone if channels == 1 else np.c_[tone, -tone]
    sf.write(path, samples, 44100, subtype="PCM_24")
    result = inspect_master(path)
    assert result["channels"] == channels
    assert result["frames"] == len(samples)
    assert result["review_flags"] == []
    assert 300 < result["bandwidth_99_9_hz"] < 1000
    assert not result["mastering_history_verified_by_spectrum"]


def test_full_scale_runs_cross_decode_block_boundary(tmp_path: Path) -> None:
    path = tmp_path / "clipped.wav"
    samples = np.sin(np.arange(70000) * 0.1).astype(np.float32) * 0.1
    samples[65000:66000] = 1.0
    sf.write(path, samples, 48000, subtype="PCM_16")
    result = inspect_master(path)
    assert result["longest_full_scale_run"] == 1000
    assert "sustained_full_scale_samples" in result["review_flags"]


def test_nonfinite_float_samples_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.wav"
    samples = np.zeros(80000, dtype=np.float32)
    samples[-1] = np.nan
    sf.write(path, samples, 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        inspect_master(path)


def test_silent_recording_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(44100), 44100)
    with pytest.raises(ValueError, match="silent"):
        inspect_master(path)
