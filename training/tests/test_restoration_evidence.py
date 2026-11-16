"""Known-answer P81 controls, independent of candidate scores."""

import copy

import numpy as np
import pytest

from evaluation.restoration_evidence import (
    describe_comparison,
    direction,
    extended_panel,
    spectral_dynamics,
    strict_paired_recordings,
)
from evaluation.restoration_panel import _band, paired_recordings, panel


def rows():
    return [
        {"source": source, "case": case, "path": path, "scores": {"mr": value, "hf_sc": value}}
        for source, case, before, after in (("a", 0, 1, 0), ("a", 1, 3, 4), ("b", 0, 6, 4))
        for path, value in (("old", before), ("new", after))
    ]


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_additive_panel_preserves_history_and_calibrated_pcm(rate, channels):
    rng = np.random.default_rng(81)
    reference = rng.normal(0, 0.02, (16495, channels))
    wave = reference * 0.5
    old, new = panel(wave, reference, rate), extended_panel(wave, reference, rate)
    for key, value in old["metrics"].items():
        assert new["metrics"][key] == value
    for size, scores in old["resolutions"].items():
        assert all(new["resolutions"][size][k] == v for k, v in scores.items())
    assert new["analysis_coverage"]["8192"]["unused_tail_samples"] == 111
    assert new["metrics"]["full_pcm_error_rms"] == np.sqrt(np.mean((wave - reference) ** 2))
    same = extended_panel(reference, reference, rate)
    assert same["metrics"]["hf_spectral_dynamics"] == 0
    assert same["metrics"]["full_pcm_error_rms"] == 0


def test_squared_hellinger_and_within_band_motion_are_not_envelope_distance():
    a = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    b = np.array([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    old = _band(a, b)
    assert old["dynamics"] == old["envelope"] == 0
    assert spectral_dynamics(a, b)["spectral_dynamics"] == pytest.approx(np.sqrt(4 / 3))
    c = np.tile([0.5, np.sqrt(0.75)], (1, 3, 1))
    assert _band(c, b)["shape"] == pytest.approx(0.5)  # H^2, not sqrt(0.5).


def test_window_endpoint_blind_spot_keeps_full_pcm_error():
    reference = np.zeros((16384, 1))
    wave = reference.copy()
    wave[0] = 0.25  # Every left-aligned periodic Hann has zero weight here.
    result = extended_panel(wave, reference, 48000)
    assert result["metrics"]["hf_magnitude_error_rms"] == 0
    assert result["metrics"]["full_pcm_error_rms"] == 0.25 / 128


def test_fp32_ulp_is_resolved_below_historical_score_epsilon_ruler():
    reference = (0.125 * (-1.0) ** np.arange(16384))[:, None]
    control = reference.astype(np.float32) / 2
    candidate = np.nextafter(control, reference.astype(np.float32))
    before = panel(control, reference, 48000)["metrics"]["hf_sc"]
    after = panel(candidate, reference, 48000)["metrics"]["hf_sc"]
    analytical_delta = -float(np.spacing(np.float32(0.0625))) / 0.125
    assert after - before == pytest.approx(analytical_delta, rel=0, abs=1e-14)
    assert 0 < before - after < 8 * np.finfo(np.float32).eps * before


def test_gain_phase_silence_and_weak_controls_remain_separate():
    reference = (0.125 * (-1.0) ** np.arange(16384))[:, None]
    gain, phase = [
        extended_panel(x, reference, 48000)["metrics"] for x in (reference / 2, -reference)
    ]
    assert gain["hf_shape"] < 1e-20 and gain["hf_sc"] == pytest.approx(0.5)
    assert phase["hf_sc"] == 0 and phase["hf_complex"] == pytest.approx(2)
    weak = extended_panel(reference * 0.5e-6, reference * 1e-6, 48000)["metrics"]
    assert weak["hf_sc"] == pytest.approx(gain["hf_sc"])
    assert weak["full_pcm_error_rms"] == pytest.approx(gain["full_pcm_error_rms"] * 1e-6)
    silence = extended_panel(reference, reference * 0, 48000)["metrics"]
    assert silence["hf_sc"] is None and silence["hf_magnitude_error_rms"] > 0


@pytest.mark.parametrize("rate", [44100, 48000])
def test_one_sided_hann_rms_calibration_and_stereo_cancellation(rate):
    nyquist = (0.125 * (-1.0) ** np.arange(16384))[:, None]
    measured = extended_panel(nyquist, nyquist, rate)
    for scores in measured["resolutions"].values():
        assert scores["hf_reference_rms"] == pytest.approx(0.125, rel=1e-14)
    stereo = np.concatenate((nyquist, -nyquist), axis=1)
    result = extended_panel(stereo[:, ::-1], stereo, rate)["metrics"]
    assert result["hf_sc"] == 0
    assert result["mid_sc"] is None
    assert result["side_complex"] == pytest.approx(2)


def test_undefined_excerpt_excludes_entire_source_but_retains_absolute_errors():
    data = [
        {
            "source": source,
            "case": case,
            "path": path,
            "scores": {"relative": relative, "absolute": absolute},
        }
        for source, case, before, after in (
            ("a", 0, (2, 1), (1, 0.5)),
            ("a", 1, (None, 0), (None, 0.1)),
            ("b", 0, (3, 1), (2, 0.5)),
        )
        for path, (relative, absolute) in (("old", before), ("new", after))
    ]
    out = strict_paired_recordings(data, "new", "old")
    assert out["relative"]["undefined_sources"] == ["a"]
    assert out["relative"]["source_deltas"] == {"b": -1}
    assert out["absolute"]["sources"] == 2
    assert direction(out["relative"]) == "undefined"


def test_strict_pairing_keeps_numbers_and_rejects_dropped_schema():
    assert strict_paired_recordings(rows(), "new", "old") == paired_recordings(rows(), "new", "old")
    bad = rows()
    del bad[0]["scores"]["hf_sc"]
    assert "hf_sc" not in paired_recordings(bad, "new", "old")  # Retain old behavior as evidence.
    with pytest.raises(ValueError, match="schema"):
        strict_paired_recordings(bad, "new", "old")
    with pytest.raises(ValueError, match="distinct"):
        strict_paired_recordings(rows(), "old", "old")
    with pytest.raises(ValueError, match="coverage"):
        strict_paired_recordings(rows()[:-1], "new", "old")
    with pytest.raises(ValueError, match="duplicate"):
        strict_paired_recordings(rows() + rows()[:1], "new", "old")


def test_nonfinite_cannot_hide_behind_undefined_partner():
    bad = rows()
    bad[0]["scores"]["mr"], bad[1]["scores"]["mr"] = None, float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        strict_paired_recordings(bad, "new", "old")


def summary(delta, *, sources=4):
    return {
        "sources": sources,
        "undefined_sources": [],
        "mean_delta": delta,
        "paired_source_bootstrap_95": [delta, delta],
        "source_deltas": {str(i): delta for i in range(sources)},
    }


def test_spectral_progress_and_coherent_tradeoff_are_separate_not_promotion():
    values = {"mr": summary(-1e-9), "hf_sc": summary(-2e-9), "realized_complex": summary(1e-8)}
    result = describe_comparison(values)
    assert result["spectral_evidence"] == "decrease_on_both_primary_rulers"
    assert result["fidelity_increases"] == ["realized_complex"]
    assert result["qualified"] is False
    assert "hf_spectral_dynamics" in result["missing_metrics"]
    values["mr"]["paired_source_bootstrap_95"] = [-3e-9, 1e-9]
    assert direction(values["mr"]) == "unresolved"
    assert describe_comparison(values)["spectral_evidence"] == "mixed_or_partial"


def test_missing_undefined_single_source_and_no_change_are_distinct():
    assert direction(None) == "missing"
    assert direction(summary(0)) == "unchanged"
    assert direction(summary(-1, sources=1)) == "insufficient_sources"
    invalid = summary(-1)
    invalid["mean_delta"] = float("nan")
    assert direction(invalid) == "invalid"
    incomplete = copy.deepcopy(summary(-1))
    incomplete["undefined_sources"] = ["silent"]
    assert direction(incomplete) == "undefined"
