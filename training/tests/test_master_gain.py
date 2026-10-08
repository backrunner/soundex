# SPDX-License-Identifier: Apache-2.0
"""Gain-only normalization preserves float transients, stereo ratios and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.master_gain import attenuate_float_master


@pytest.mark.parametrize("channels,subtype", [(1, "DOUBLE"), (2, "FLOAT")])
def test_over_unity_float_is_attenuated_without_clipping(
    tmp_path: Path, channels: int, subtype: str
) -> None:
    source, target = tmp_path / "original.wav", tmp_path / "derivative.wav"
    wave = 0.4 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    wave[1000:1002] = [1.08, -1.02]
    samples = wave[:, None] if channels == 1 else np.stack((wave, -0.6 * wave), axis=1)
    sf.write(source, samples, 44100, subtype=subtype)
    original_bytes = source.read_bytes()
    checksum = hashlib.sha256(original_bytes).hexdigest()
    receipt = attenuate_float_master(source, target, source_sha256=checksum)
    original, rate = sf.read(source, dtype="float64", always_2d=True)
    actual, derivative_rate = sf.read(target, dtype="float64", always_2d=True)
    assert source.read_bytes() == original_bytes
    assert actual.shape == original.shape and rate == derivative_rate == 44100
    np.testing.assert_allclose(
        actual,
        original * receipt["common_gain"],
        rtol=0,
        atol=receipt["rounding_error_budget"],
    )
    assert np.max(np.abs(actual)) < 1
    assert receipt["source_quality"]["review_flags"] == []
    assert receipt["derivative_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert receipt["maximum_absolute_rounding_error"] <= receipt["rounding_error_budget"]


def test_mismatched_original_hash_never_publishes_derivative(tmp_path: Path) -> None:
    source, target = tmp_path / "original.wav", tmp_path / "derivative.wav"
    sf.write(source, np.sin(np.arange(44100)) * 1.08, 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="checksum"):
        attenuate_float_master(source, target, source_sha256="0" * 64)
    assert not target.exists()
