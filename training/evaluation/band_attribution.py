"""P64 exact PCM spectral-error decomposition and retained deadline accounting.

These diagnostic rulers do not replace the registered MR loss or qualify audio.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

BANDS = ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, np.inf))


def spectral_error_components(wave, reference, rate):
    """Parseval power averaged across physical channels, without weak-bin thresholds.

    |Y-T|² = (|Y|-|T|)² + |Y||T| |unit(Y)-unit(T)|².
    The second term depends on BOTH amplitude and phase; it is not phase-only loss.
    Rectangular whole-excerpt FFT is a diagnostic, not a causal feature or new loss.
    """
    wave, reference = np.asarray(wave, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if wave.shape != reference.shape or wave.ndim != 2 or len(wave) < 2:
        raise ValueError("aligned nonempty frames/channels required")
    if rate <= 32000 or wave.shape[1] not in (1, 2):
        raise ValueError("supported rate and physical channels required")
    if not np.isfinite(wave).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite PCM")
    count = len(wave)
    y, t = np.fft.rfft(wave, axis=0) / count, np.fft.rfft(reference, axis=0) / count
    ym, tm = np.abs(y), np.abs(t)
    yu = np.divide(y, ym, out=np.zeros_like(y), where=ym != 0)
    tu = np.divide(t, tm, out=np.zeros_like(t), where=tm != 0)
    terms = {
        "reference_power": tm**2,
        "prediction_power": ym**2,
        "magnitude_error_power": (ym - tm) ** 2,
        "phase_interaction_power": ym * tm * np.abs(yu - tu) ** 2,
        "error_power": np.abs(y - t) ** 2,
    }
    sides = np.full(len(y), 2.0)
    sides[0] = 1.0
    if count % 2 == 0:
        sides[-1] = 1.0
    hz = np.fft.rfftfreq(count, 1 / rate)
    result = {}
    for low, high in (*BANDS, (8000, np.inf)):
        prefix = "hf8k" if (low, high) == (8000, np.inf) else f"band{low}"
        mask = (hz >= low) & (hz < high)
        values = {
            k: float((v[mask] * sides[mask, None]).sum() / wave.shape[1]) for k, v in terms.items()
        }
        residual = (
            values["error_power"]
            - values["magnitude_error_power"]
            - values["phase_interaction_power"]
        )
        bound = 128 * np.finfo(float).eps * (values["reference_power"] + values["prediction_power"])
        if abs(residual) > bound:
            raise ValueError("complex decomposition numerical identity failed")
        result.update({prefix + "_" + k: v for k, v in values.items()})
        result[prefix + "_error_rms"] = float(np.sqrt(values["error_power"]))
        result[prefix + "_reference_rms"] = float(np.sqrt(values["reference_power"]))
        result[prefix + "_identity_residual"] = residual
    full = sum(result[f"band{low}_error_power"] for low, _ in BANDS)
    pcm = float(np.mean((wave - reference) ** 2))
    scale = float(np.mean(wave**2) + np.mean(reference**2))
    if abs(full - pcm) > 128 * np.finfo(float).eps * scale:
        raise ValueError("Parseval identity failed")
    result.update(full_error_power=full, pcm_error_power=pcm)
    return result


def deadline_accounting(raw, *, scored_start=100):
    """Account existing paced traces without inferring unmeasured CPU/wake causes."""
    timing = np.asarray(raw["timings"], dtype=float)
    late = np.asarray(raw["paced_finish_lateness_ms"], dtype=float)
    warmup = scored_start  # Benchmark exclusion is 100; model history warmup is 80.
    period = raw["hop_samples"] * 1000 / raw["sample_rate"]
    if not raw["paced"] or timing.shape != (len(late), 4) or not 0 <= warmup < len(late):
        raise ValueError("invalid paced trace geometry")
    if (
        not np.isfinite(timing).all()
        or not np.isfinite(late).all()
        or np.any(timing < 0)
        or np.any(late < 0)
    ):
        raise ValueError("invalid timing values")
    rows = []
    for i in range(warmup, len(late)):
        if late[i] <= 0:
            continue
        service = timing[i, -1]
        rows.append(
            {
                "hop": i,
                "stages_ms": timing[i].tolist(),
                "finish_late_ms": float(late[i]),
                "previous_finish_late_ms": float(late[i - 1]) if i else 0.0,
                "arrival_to_finish_minus_stages_ms": float(late[i] + period - service),
                "service_fits_period": bool(service < period),
            }
        )
    return {
        "period_ms": period,
        "measured_hops": len(late) - warmup,
        "late_hops": len(rows),
        "late_with_service_fitting_period": sum(r["service_fits_period"] for r in rows),
        "late_with_previous_on_time_and_service_fitting": sum(
            r["service_fits_period"] and r["previous_finish_late_ms"] == 0 for r in rows
        ),
        "over_5ms": int((timing[warmup:, -1] > 5).sum()),
        "thread_cpu_measured": raw["thread_cpu_requested"],
        "outliers": rows,
        "scope": "Outside-stage time includes backlog, sleep/wakeup delay and unmeasured loop overhead; stage wall time may itself include descheduling. No causal OS attribution or subtraction from deadline failures.",
    }


def phase_change_factors(candidate, control, reference, rate):
    """Symmetric two-factor accounting of phase-interaction POWER change.

    Separate changing predicted magnitude from changing unit-vector distance.
    This is algebra on frozen outputs, not an oracle output or causal intervention.
    Unit(0)=0 is only a decomposition convention; phase at zero is undefined.
    """
    waves = [np.asarray(x, dtype=np.float64) for x in (candidate, control, reference)]
    if any(x.shape != waves[0].shape for x in waves) or waves[0].ndim != 2:
        raise ValueError("aligned physical channels required")
    if len(waves[0]) < 2 or waves[0].shape[1] not in (1, 2) or rate <= 32000:
        raise ValueError("invalid PCM geometry")
    if any(not np.isfinite(x).all() for x in waves):
        raise ValueError("nonfinite PCM")
    n, channels = waves[0].shape
    c, b, t = [np.fft.rfft(x, axis=0) / n for x in waves]
    cm, bm, tm = np.abs(c), np.abs(b), np.abs(t)
    cu, bu, tu = [
        np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x) != 0) for x in (c, b, t)
    ]
    dc, db = np.abs(cu - tu) ** 2, np.abs(bu - tu) ** 2
    values = {
        "magnitude_factor": tm * (cm - bm) * (dc + db) / 2,
        "angular_factor": tm * (cm + bm) * (dc - db) / 2,
        "interaction_delta": tm * (cm * dc - bm * db),
    }
    sides = np.full(len(c), 2.0)
    sides[0] = 1
    if n % 2 == 0:
        sides[-1] = 1
    hz = np.fft.rfftfreq(n, 1 / rate)
    result = {}
    for low, high in (*BANDS, (8000, np.inf)):
        prefix = "hf8k" if (low, high) == (8000, np.inf) else f"band{low}"
        mask = (hz >= low) & (hz < high)
        terms = {
            k: float((v[mask] * sides[mask, None]).sum() / channels) for k, v in values.items()
        }
        scale = float(((cm[mask] + bm[mask]) * tm[mask] * sides[mask, None]).sum() / channels)
        if (
            abs(terms["interaction_delta"] - terms["magnitude_factor"] - terms["angular_factor"])
            > 128 * np.finfo(float).eps * scale
        ):
            raise ValueError("phase interaction factorization failed")
        result.update({prefix + "_" + k: v for k, v in terms.items()})
    return result
