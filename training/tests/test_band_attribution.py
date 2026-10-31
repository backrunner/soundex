"""Independent mathematical controls for the P64 diagnostic rulers."""

import numpy as np
import pytest

from evaluation.band_attribution import deadline_accounting, spectral_error_components


@pytest.mark.parametrize("length", [4800, 4801])
@pytest.mark.parametrize("channels", [1, 2])
def test_parseval_random_aligned_channels(length, channels):
    rng = np.random.default_rng(4)
    target = rng.normal(size=(length, channels)) * 0.1
    wave = rng.normal(size=(length, channels)) * 0.1
    result = spectral_error_components(wave, target, 48000)
    assert result["full_error_power"] == pytest.approx(np.mean((wave - target) ** 2), rel=1e-12)


def test_amplitude_and_quadrature_have_distinct_exact_contributions():
    t = np.arange(4800) / 48000
    target = np.sin(2 * np.pi * 10000 * t)[:, None]
    gain = spectral_error_components(2 * target, target, 48000)
    phase = spectral_error_components(np.cos(2 * np.pi * 10000 * t)[:, None], target, 48000)
    assert gain["hf8k_magnitude_error_power"] == pytest.approx(0.5)
    assert gain["hf8k_phase_interaction_power"] < 1e-24
    assert phase["hf8k_magnitude_error_power"] < 1e-24
    assert phase["hf8k_phase_interaction_power"] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1.0])
def test_silence_identity_and_empty_reference(scale):
    wave = np.full((64, 2), scale)
    identity = spectral_error_components(wave, wave, 48000)
    assert identity["full_error_power"] == 0
    result = spectral_error_components(wave, np.zeros_like(wave), 48000)
    assert result["band0_magnitude_error_power"] == pytest.approx(scale**2, abs=1e-30)
    assert result["band0_phase_interaction_power"] == 0


def test_nyquist_and_channel_isolation():
    target = np.zeros((64, 2))
    wave = target.copy()
    wave[:, 1] = (-1.0) ** np.arange(64)
    result = spectral_error_components(wave, target, 48000)
    assert result["band16000_error_power"] == pytest.approx(0.5)
    assert result["band0_error_power"] == 0


def test_deadline_accounting_retains_fast_late_and_slow_calls():
    raw = {
        "timings": [[0, 0, 0, 0], [0.1, 0.3, 0.1, 0.5], [0.1, 5.8, 0.1, 6.0]],
        "paced_finish_lateness_ms": [0, 0.2, 4],
        "warmup_hops": 1,
        "paced": True,
        "hop_samples": 128,
        "sample_rate": 48000,
        "thread_cpu_requested": False,
    }
    result = deadline_accounting(raw, scored_start=1)
    assert result["late_hops"] == 2
    assert result["late_with_service_fitting_period"] == 1
    assert result["late_with_previous_on_time_and_service_fitting"] == 1
    assert result["over_5ms"] == 1
    assert result["outliers"][0]["arrival_to_finish_minus_stages_ms"] == pytest.approx(
        0.2 + 128 / 48 - 0.5
    )


def test_reject_nonfinite_or_wrong_geometry():
    with pytest.raises(ValueError):
        spectral_error_components(np.full((10, 1), np.nan), np.zeros((10, 1)), 48000)
    with pytest.raises(ValueError):
        spectral_error_components(np.zeros((10, 1)), np.zeros((10, 2)), 48000)


def test_relative_study_root_keeps_logical_receipt_symlink_paths(tmp_path, monkeypatch):
    from scripts import audit_band_attribution

    study = tmp_path / "study"
    physical = tmp_path / "internal"
    study.mkdir()
    physical.mkdir()
    (study / "complete.json").write_text("{}")
    (physical / "complete.json").write_text("{}")
    (study / "longform-recheck").symlink_to(physical, target_is_directory=True)
    visited = []

    class EndPreflight(Exception):
        pass

    def capture(path, receipt):
        visited.append(path)
        assert (
            path
            == study
            / ("endpoints.json", "cost-summary.json", "longform-recheck/complete.json")[
                len(visited) - 1
            ]
        )
        if len(visited) == 3:
            raise EndPreflight

    monkeypatch.setattr(audit_band_attribution, "bound", capture)
    with pytest.raises(EndPreflight):
        audit_band_attribution.run(study / ".." / "study", tmp_path / "result")
    assert len(visited) == 3
    assert not (tmp_path / "result").exists()


def test_long_interleaved_reduction_stays_within_original_numeric_bound():
    rng = np.random.default_rng(20261101)
    reference = rng.normal(size=(441000, 2)) * [0.1, 1e-4]
    prediction = rng.normal(size=(441000, 2)) * [0.02, 1e-5]
    result = spectral_error_components(prediction, reference, 44100)
    assert result["pcm_error_power"] == pytest.approx(result["full_error_power"], rel=1e-13)


def test_benchmark_warmup_is_separate_from_model_history():
    raw = {
        "timings": [[0, 0, 0, 0.1]] * 1100,
        "paced_finish_lateness_ms": [0.0] * 1100,
        "warmup_hops": 80,
        "paced": True,
        "hop_samples": 128,
        "sample_rate": 48000,
        "thread_cpu_requested": False,
    }
    raw["paced_finish_lateness_ms"][90] = 0.5
    raw["paced_finish_lateness_ms"][100] = 0.25
    result = deadline_accounting(raw)
    assert result["measured_hops"] == 1000
    assert result["late_hops"] == 1
    assert result["outliers"][0]["hop"] == 100


def test_phase_factorization_separates_magnitude_and_angular_changes():
    from evaluation.band_attribution import phase_change_factors

    t = np.arange(4800) / 48000
    reference = np.sin(2 * np.pi * 10000 * t)[:, None]
    control = np.cos(2 * np.pi * 10000 * t)[:, None]
    amplitude = phase_change_factors(control * 2, control, reference, 48000)
    assert amplitude["hf8k_magnitude_factor"] == pytest.approx(1.0)
    assert abs(amplitude["hf8k_angular_factor"]) < 1e-20
    angle = phase_change_factors(-reference, control, reference, 48000)
    assert angle["hf8k_angular_factor"] == pytest.approx(1.0)
    assert abs(angle["hf8k_magnitude_factor"]) < 1e-12
    zero = phase_change_factors(
        np.zeros_like(reference), np.zeros_like(reference), reference, 48000
    )
    assert all(v == 0 for v in zero.values())
