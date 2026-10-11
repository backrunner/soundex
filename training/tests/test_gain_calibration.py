"""Calibration fits only train recordings with the required corpus proportions."""

import numpy as np

from scripts.fit_gain_calibration import fit_ridge, is_inner_validation, recording_weights


def test_recording_weights_do_not_favor_long_recordings_or_speech_count():
    tracks = np.asarray(["a", "a", "b", "s", "t", "t"])
    corpora = np.asarray(["music_library"] * 3 + ["speech_library"] * 3)
    w = recording_weights(tracks, corpora)
    np.testing.assert_allclose(w.sum(), 1)
    np.testing.assert_allclose(w[:2].sum(), w[2])
    np.testing.assert_allclose(w[3:].sum(), 1 / 254)
    assert is_inner_validation("a") == is_inner_validation("a")


def test_robust_fit_learns_input_conditioned_target_with_outliers():
    rng = np.random.default_rng(52)
    x = np.column_stack([np.ones(400), rng.normal(size=(400, 9))])
    y = 2 * x[:, 1] - x[:, 2]
    contaminated = y.copy()
    contaminated[::20] += 20
    coefficients = fit_ridge(x, contaminated, np.full(400, 1 / 400), 0.01)
    prediction = (x @ coefficients).clip(-6, 6)
    assert np.mean(abs(prediction - y)) < 0.3
    assert np.mean(abs(prediction - y)) < np.mean(abs(y)) / 5
