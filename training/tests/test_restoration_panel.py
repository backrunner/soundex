"""Analytic calibration and recording-paired statistics for P71."""

import numpy as np
import pytest

from evaluation.restoration_panel import paired_recordings, panel


@pytest.mark.parametrize("rate", [44100, 48000])
def test_shape_separates_gain_from_frequency_and_reference_is_zero(rate):
    t = np.arange(rate) / rate
    y = (0.1 * np.sin(2 * np.pi * 10000 * t))[:, None]
    ref, gain, wrong = [
        panel(x, y, rate)["metrics"]
        for x in (y, y * 2, (0.1 * np.sin(2 * np.pi * 11000 * t))[:, None])
    ]
    assert ref["hf_sc"] == ref["hf_shape"] == ref["hf_complex"] == 0
    assert gain["hf_shape"] < 1e-20
    assert gain["hf_envelope"] == pytest.approx(1)
    assert gain["hf_sc"] == pytest.approx(1)
    assert wrong["hf_shape"] > 0.8


def test_transients_weak_energy_silence_and_stereo_are_distinct():
    rate = 48000
    t = np.arange(rate) / rate
    env = np.exp(-(((t - 0.25) / 0.015) ** 2)) + np.exp(-(((t - 0.7) / 0.03) ** 2))
    y = (0.1 * env * np.sin(2 * np.pi * 11000 * t))[:, None]
    shifted = np.roll(y, 2400, axis=0)
    a = panel(shifted, y, rate)["metrics"]
    b = panel(shifted * 1e-6, y * 1e-6, rate)["metrics"]
    assert a["hf_dynamics"] > 0
    assert b["hf_sc"] == pytest.approx(a["hf_sc"])
    assert b["hf_magnitude_error_rms"] == pytest.approx(a["hf_magnitude_error_rms"] * 1e-6)
    silent = panel(y * 0, y * 0, rate)["metrics"]
    assert silent["hf_sc"] is None and silent["hf_magnitude_error_rms"] == 0
    added = panel(y, y * 0, rate)["metrics"]
    assert added["hf_sc"] is None and added["hf_magnitude_error_rms"] > 0
    stereo = np.concatenate((y, -y), axis=1)
    swapped = panel(stereo[:, ::-1], stereo, rate)["metrics"]
    assert swapped["hf_sc"] == 0
    assert swapped["side_complex"] == pytest.approx(2)


def test_recordings_receive_equal_weight_and_missing_cases_fail():
    rows = [
        {"source": src, "case": i, "path": path, "scores": {"m": val}}
        for src, i, control, candidate in (("a", 0, 1, 0), ("a", 1, 3, 4), ("b", 0, 6, 4))
        for path, val in (("control", control), ("candidate", candidate))
    ]
    result = paired_recordings(rows, "candidate", "control")["m"]
    assert result["control_mean"] == 4 and result["candidate_mean"] == 3
    assert result["source_deltas"] == {"a": 0, "b": -2}
    assert result["paired_source_bootstrap_95"] == [-2, 0]
    with pytest.raises(ValueError, match="coverage"):
        paired_recordings(rows[:-1], "candidate", "control")
    with pytest.raises(ValueError, match="duplicate"):
        paired_recordings(rows + rows[:1], "candidate", "control")


def test_undefined_relative_error_does_not_hide_silent_source_absolute_error():
    rows = [
        {
            "source": "silent",
            "case": 0,
            "path": path,
            "scores": {"relative": None, "absolute": value},
        }
        for path, value in (("c", 1e-8), ("b", 0))
    ]
    out = paired_recordings(rows, "c", "b")
    assert out["relative"]["sources"] == 0
    assert out["absolute"]["adverse_sources"] == ["silent"]
