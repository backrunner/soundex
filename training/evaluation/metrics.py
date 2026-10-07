"""Finite, alignment-strict objective audio metrics."""

from __future__ import annotations

import math

import numpy as np


def evaluate_signal_triplet(
    clean: np.ndarray,
    degraded: np.ndarray,
    enhanced: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float,
    crossover_width_hz: float = 1000.0,
    fft_size: int = 256,
    hop_size: int = 128,
) -> dict[str, float]:
    """Compare aligned degraded/enhanced mono signals against clean audio."""
    clean, degraded, enhanced = _aligned_mono(clean, degraded, enhanced)
    clean_magnitude = _stft_magnitude(clean, fft_size, hop_size)
    degraded_magnitude = _stft_magnitude(degraded, fft_size, hop_size)
    enhanced_magnitude = _stft_magnitude(enhanced, fft_size, hop_size)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    high_start = max(0.0, float(cutoff_hz) - float(crossover_width_hz) / 2.0)
    split_bin = int(np.searchsorted(frequencies, high_start, side="left"))
    split_bin = min(max(split_bin, 1), len(frequencies) - 1)
    high_mask = np.arange(len(frequencies)) >= split_bin
    low_mask = ~high_mask

    baseline_high = _lsd(degraded_magnitude, clean_magnitude, high_mask)
    enhanced_high = _lsd(enhanced_magnitude, clean_magnitude, high_mask)
    baseline_full = _lsd(degraded_magnitude, clean_magnitude, None)
    enhanced_full = _lsd(enhanced_magnitude, clean_magnitude, None)
    baseline_low = _lsd(degraded_magnitude, clean_magnitude, low_mask)
    enhanced_low = _lsd(enhanced_magnitude, clean_magnitude, low_mask)
    metrics = {
        "baseline_full_lsd_db": baseline_full,
        "enhanced_full_lsd_db": enhanced_full,
        "delta_full_lsd_db": enhanced_full - baseline_full,
        "baseline_high_lsd_db": baseline_high,
        "enhanced_high_lsd_db": enhanced_high,
        "delta_high_lsd_db": enhanced_high - baseline_high,
        "baseline_low_lsd_db": baseline_low,
        "enhanced_low_lsd_db": enhanced_low,
        "delta_low_lsd_db": enhanced_low - baseline_low,
        "low_band_preservation_lsd_db": _lsd(enhanced_magnitude, degraded_magnitude, low_mask),
        "baseline_spectral_convergence": _spectral_convergence(
            degraded_magnitude, clean_magnitude, None
        ),
        "enhanced_spectral_convergence": _spectral_convergence(
            enhanced_magnitude, clean_magnitude, None
        ),
        "baseline_high_band_energy_error_db": _energy_error_db(
            degraded_magnitude, clean_magnitude, high_mask
        ),
        "enhanced_high_band_energy_error_db": _energy_error_db(
            enhanced_magnitude, clean_magnitude, high_mask
        ),
        "baseline_si_sdr_db": scale_invariant_sdr(degraded, clean),
        "enhanced_si_sdr_db": scale_invariant_sdr(enhanced, clean),
    }
    metrics["delta_si_sdr_db"] = metrics["enhanced_si_sdr_db"] - metrics["baseline_si_sdr_db"]
    _require_finite(metrics)
    return metrics


def evaluate_stereo_image(
    clean: np.ndarray,
    degraded: np.ndarray,
    enhanced: np.ndarray,
    *,
    fft_size: int = 256,
    hop_size: int = 128,
) -> dict[str, float]:
    """Measure stereo correlation, inter-channel phase, and image width change."""
    signals = []
    for name, value in (("clean", clean), ("degraded", degraded), ("enhanced", enhanced)):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{name} must have shape [frames, 2]")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite samples")
        signals.append(array)
    clean, degraded, enhanced = signals
    if clean.shape != degraded.shape or clean.shape != enhanced.shape:
        raise ValueError("stereo clean, degraded, and enhanced signals must be exactly aligned")

    clean_correlation = _correlation(clean[:, 0], clean[:, 1])
    degraded_correlation = _correlation(degraded[:, 0], degraded[:, 1])
    enhanced_correlation = _correlation(enhanced[:, 0], enhanced[:, 1])
    clean_width = _width_db(clean)
    degraded_width = _width_db(degraded)
    enhanced_width = _width_db(enhanced)
    clean_phase, clean_phase_weight = _cross_phase(clean, fft_size, hop_size)
    degraded_phase, _ = _cross_phase(degraded, fft_size, hop_size)
    enhanced_phase, _ = _cross_phase(enhanced, fft_size, hop_size)
    metrics = {
        "clean_interchannel_correlation": clean_correlation,
        "degraded_interchannel_correlation": degraded_correlation,
        "enhanced_interchannel_correlation": enhanced_correlation,
        "baseline_correlation_error": abs(degraded_correlation - clean_correlation),
        "enhanced_correlation_error": abs(enhanced_correlation - clean_correlation),
        "clean_image_width_db": clean_width,
        "degraded_image_width_db": degraded_width,
        "enhanced_image_width_db": enhanced_width,
        "baseline_image_width_error_db": abs(degraded_width - clean_width),
        "enhanced_image_width_error_db": abs(enhanced_width - clean_width),
        "baseline_interchannel_phase_error_rad": _circular_phase_error(
            degraded_phase, clean_phase, clean_phase_weight
        ),
        "enhanced_interchannel_phase_error_rad": _circular_phase_error(
            enhanced_phase, clean_phase, clean_phase_weight
        ),
    }
    _require_finite(metrics)
    return metrics


def scale_invariant_sdr(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return finite SI-SDR with explicit silence and perfect-match behavior."""
    predicted, target = _aligned_mono(predicted, target)
    predicted = predicted.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    target_energy = float(np.dot(target, target))
    predicted_energy = float(np.dot(predicted, predicted))
    epsilon = 1e-20
    if target_energy <= epsilon:
        return 0.0 if predicted_energy <= epsilon else -120.0
    projection = target * (float(np.dot(predicted, target)) / target_energy)
    signal_energy = float(np.dot(projection, projection))
    noise = predicted - projection
    noise_energy = float(np.dot(noise, noise))
    if noise_energy <= epsilon:
        return 120.0
    if signal_energy <= epsilon:
        return -120.0
    return float(np.clip(10.0 * math.log10(signal_energy / noise_energy), -120.0, 120.0))


def _aligned_mono(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(signal, dtype=np.float64) for signal in signals)
    if not arrays:
        raise ValueError("at least one signal is required")
    expected_length = len(arrays[0])
    for array in arrays:
        if array.ndim != 1:
            raise ValueError("metric inputs must be mono one-dimensional arrays")
        if len(array) != expected_length:
            raise ValueError("metric inputs must have exactly equal lengths")
        if not np.isfinite(array).all():
            raise ValueError("metric input contains non-finite samples")
    return arrays


def _stft_magnitude(signal: np.ndarray, fft_size: int, hop_size: int) -> np.ndarray:
    if fft_size < 2 or hop_size < 1 or hop_size > fft_size:
        raise ValueError("invalid STFT size or hop")
    if len(signal) < fft_size:
        raise ValueError(f"metric input requires at least {fft_size} samples")
    window = np.hanning(fft_size + 1)[:-1]
    frame_count = 1 + (len(signal) - fft_size) // hop_size
    frames = np.stack(
        [signal[index * hop_size : index * hop_size + fft_size] for index in range(frame_count)]
    )
    return np.abs(np.fft.rfft(frames * window, axis=1))


def _lsd(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is not None:
        predicted = predicted[:, mask]
        target = target[:, mask]
    predicted_db = 20.0 * np.log10(np.maximum(predicted, 1e-10))
    target_db = 20.0 * np.log10(np.maximum(target, 1e-10))
    return float(np.mean(np.sqrt(np.mean(np.square(predicted_db - target_db), axis=1))))


def _spectral_convergence(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray | None
) -> float:
    if mask is not None:
        predicted = predicted[:, mask]
        target = target[:, mask]
    numerator = float(np.linalg.norm(predicted - target))
    denominator = float(np.linalg.norm(target))
    if denominator <= 1e-20:
        return 0.0 if numerator <= 1e-20 else 1e6
    return numerator / denominator


def _energy_error_db(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    predicted_energy = float(np.mean(np.square(predicted[:, mask])))
    target_energy = float(np.mean(np.square(target[:, mask])))
    return abs(10.0 * math.log10((predicted_energy + 1e-20) / (target_energy + 1e-20)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-20:
        return 1.0 if np.linalg.norm(left - right) <= 1e-20 else 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _width_db(stereo: np.ndarray) -> float:
    mid = (stereo[:, 0] + stereo[:, 1]) / math.sqrt(2.0)
    side = (stereo[:, 0] - stereo[:, 1]) / math.sqrt(2.0)
    mid_rms = math.sqrt(float(np.mean(np.square(mid))))
    side_rms = math.sqrt(float(np.mean(np.square(side))))
    return float(np.clip(20.0 * math.log10((side_rms + 1e-10) / (mid_rms + 1e-10)), -120, 120))


def _cross_phase(stereo: np.ndarray, fft_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray]:
    left = _stft_complex(stereo[:, 0], fft_size, hop_size)
    right = _stft_complex(stereo[:, 1], fft_size, hop_size)
    cross_spectrum = left * np.conjugate(right)
    return np.angle(cross_spectrum), np.abs(cross_spectrum)


def _stft_complex(signal: np.ndarray, fft_size: int, hop_size: int) -> np.ndarray:
    if len(signal) < fft_size:
        raise ValueError(f"metric input requires at least {fft_size} samples")
    window = np.hanning(fft_size + 1)[:-1]
    frame_count = 1 + (len(signal) - fft_size) // hop_size
    frames = np.stack(
        [signal[index * hop_size : index * hop_size + fft_size] for index in range(frame_count)]
    )
    return np.fft.rfft(frames * window, axis=1)


def _circular_phase_error(actual: np.ndarray, expected: np.ndarray, weights: np.ndarray) -> float:
    difference = np.angle(np.exp(1j * (actual - expected)))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-20:
        return 0.0
    return float(np.sum(np.abs(difference) * weights) / weight_sum)


def _require_finite(metrics: dict[str, float]) -> None:
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("metric computation produced a non-finite value")
