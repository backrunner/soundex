"""P70 independent rulers and controlled high-band replacement geometry."""

import numpy as np
import pytest

from scripts.audit_baseline_metrics import controls, independent


@pytest.mark.parametrize("rate", [44100, 48000])
def test_reference_stationarity_half_error_and_matched_energy_negative(rate):
    time = np.arange(rate) / rate
    target = np.stack(
        [0.2 * np.sin(2 * np.pi * 440 * time), 0.1 * np.sin(2 * np.pi * 330 * time)], axis=1
    )
    high = 0.03 * np.sin(2 * np.pi * 11000 * time)[:, None]
    target += high
    dry = target - high
    fixtures = controls(dry, target, rate, 70)
    scores = {k: independent(v, target, rate) for k, v in fixtures.items()}
    untreated = independent(dry, target, rate)
    assert all(v == 0 for v in independent(target, target, rate).values())
    assert scores["correct_hf"]["numpy_hf_error_rms"] < 1e-12
    assert scores["half_correct_hf"]["numpy_hf_error_rms"] == pytest.approx(
        untreated["numpy_hf_error_rms"] / 2
    )
    assert scores["same_energy_scrambled_hf"]["numpy_hf_energy_error_db"] < 1e-10
    assert scores["same_energy_scrambled_hf"]["numpy_hf_error_rms"] > 1e-3
    for label, wave in fixtures.items():
        assert np.isfinite(wave).all(), label
        assert wave.shape == dry.shape
        assert all(np.isfinite(v) for v in scores[label].values())


def test_silent_reference_and_channels_remain_finite():
    silent = np.zeros((48000, 2))
    assert all(v == 0 for v in independent(silent, silent, 48000).values())
    for wave in controls(silent, silent, 48000, 70).values():
        assert not wave.any()
