"""Offline diagnostic PCM measurements; no perceptual threshold or runtime filter."""

import numpy as np

BANDS = ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001))


def db_power(power):
    # JSON represents exact silence as null, never as an invented noise floor.
    return 10 * float(np.log10(power)) if power > 0 else None


def pcm_stats(wave):
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("expected nonempty finite mono PCM")
    power = float(np.mean(wave**2))
    return {
        "rms": power**0.5,
        "rms_dbfs": db_power(power),
        "peak": float(np.max(np.abs(wave))),
        "peak_step": float(np.max(np.abs(np.diff(wave)))) if wave.size > 1 else 0.0,
    }


def band_powers(wave, rate, size=4096):
    """Hann-periodogram one-sided power, Parseval/window-energy calibrated.

    Unit-amplitude DC has power 1; a unit-peak stationary sine has RMS -3.01
    dBFS. Complete frames only; this is offline analysis, not processing latency.
    """
    wave = np.asarray(wave, dtype=np.float64)
    pcm_stats(wave)
    if len(wave) < size or size < 4 or size % 2 or rate not in (44100, 48000):
        raise ValueError("unsupported analysis shape or sample rate")
    frames = np.lib.stride_tricks.sliding_window_view(wave, size)[:: size // 4]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    power[:, 1:-1] *= 2
    power = power.mean(axis=0) / (size * np.sum(window**2))
    hz = np.fft.rfftfreq(size, 1 / rate)
    return {f"{low}_{high}": float(power[(hz >= low) & (hz < high)].sum()) for low, high in BANDS}


def compare_pcm(output, reference, rate):
    output, reference = [np.asarray(v, dtype=np.float64) for v in (output, reference)]
    if output.shape != reference.shape:
        raise ValueError("unaligned PCM")
    error = output - reference
    stats = pcm_stats(error)
    reference_stats = pcm_stats(reference)
    stats["error_to_reference_rms"] = (
        stats["rms"] / reference_stats["rms"] if reference_stats["rms"] else None
    )
    powers = [band_powers(w, rate) for w in (reference, output, error)]
    return {
        "error": stats,
        "reference": reference_stats,
        "output": pcm_stats(output),
        "bands": {
            band: {
                name: {"rms": values[band] ** 0.5, "rms_dbfs": db_power(values[band])}
                for name, values in zip(("reference", "output", "error"), powers, strict=True)
            }
            for band in powers[0]
        },
    }
