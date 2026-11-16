"""P81 additive spectral diagnostics and non-binary evidence interpretation.

Historical P71 scores and P76--P80 gates remain reproducible. No training loss,
perceptual threshold, candidate acceptance or default promotion is defined here.
SPDX-License-Identifier: Apache-2.0
"""

import math

import numpy as np

from evaluation.restoration_panel import BANDS, SIZES, paired_recordings, panel

VERSION = "soundex-restoration-evidence-v2"
PRIMARY = ("mr", "hf_sc")
SUPPORTING = ("hf_shape", "hf_envelope", "hf_dynamics", "hf_spectral_dynamics")
FIDELITY = ("realized_complex", "hf8k_rms", "low8k_rms", "hop_boundary_difference_rms")


def spectral_dynamics(a, b):
    """Error of signed bin-magnitude changes, not positive-only onset flux.

    Inputs have (..., frame, frequency) geometry and calibrated PCM RMS units.
    Normalization uses the complete reference magnitude norm, as in P71 envelope
    dynamics. It is a difference per analysis hop, not a derivative per second.
    """
    ma, mb = np.abs(a), np.abs(b)
    change = np.diff(ma - mb, axis=-2)
    norm = np.linalg.norm(mb)
    return {
        "spectral_dynamics": float(np.linalg.norm(change) / norm) if norm > 1e-12 else None,
        "spectral_dynamics_error_rms": float(np.sqrt(np.mean(np.square(change).sum(-1)))),
    }


def extended_panel(wave, reference, rate):
    """P71 scores unchanged, plus within-band dynamics and complete PCM support.

    The historical `shape` field is *squared* Hellinger, with an explicit missing
    prediction convention. No square root or retrospective score rewrite occurs.
    Full PCM RMS covers the unwindowed endpoints/tail that Hann STFTs can miss.
    """
    result = panel(wave, reference, rate)  # Also validates all input geometry.
    wave, reference = np.asarray(wave, np.float64), np.asarray(reference, np.float64)
    coverage = {}
    for size in SIZES:
        hop = size // 4
        window = np.hanning(size + 1)[:-1]
        hz = np.fft.rfftfreq(size, 1 / rate)
        sides = np.full(len(hz), 2.0)
        sides[0] = sides[-1] = 1
        scale = np.sqrt(sides / (size * np.square(window).sum()))
        spectra = []
        for pcm in (wave, reference):
            frames = np.lib.stride_tricks.sliding_window_view(pcm.T, size, axis=-1)
            spectra.append(np.fft.rfft(frames[:, ::hop] * window, axis=-1) * scale)
        a, b = spectra
        values = {}
        for prefix, low, high in (
            ("hf", 8000, float("inf")),
            *((f"band{low}", low, high) for low, high in BANDS),
        ):
            mask = (hz >= low) & (hz < high)
            values.update(
                {
                    prefix + "_" + k: v
                    for k, v in spectral_dynamics(a[..., mask], b[..., mask]).items()
                }
            )
        if wave.shape[1] == 2:
            for prefix, sign in (("mid", 1), ("side", -1)):
                aa, bb = [(z[0:1] + sign * z[1:2]) / 2 for z in spectra]
                values.update(
                    {
                        prefix + "_" + k: v
                        for k, v in spectral_dynamics(
                            aa[..., hz >= 8000], bb[..., hz >= 8000]
                        ).items()
                    }
                )
        result["resolutions"][str(size)].update(values)
        coverage[str(size)] = {
            "frames": a.shape[-2],
            "hop_samples": hop,
            "hop_seconds": hop / rate,
            "unused_tail_samples": len(wave) - ((a.shape[-2] - 1) * hop + size),
            "first_sample_window_weight": float(window[0]),
        }
    # Only average newly added spectral metrics; existing numbers stay exact.
    for key in values:
        series = [v[key] for v in result["resolutions"].values()]
        result["metrics"][key] = (
            float(np.mean(series)) if all(v is not None for v in series) else None
        )
    result["metrics"]["full_pcm_error_rms"] = float(np.sqrt(np.mean(np.square(wave - reference))))
    result.update(version=VERSION, analysis_coverage=coverage, shape_definition="squared_hellinger")
    return result


def strict_paired_recordings(records, candidate, control, *, seed=71, resamples=2000):
    """Reject schema loss instead of silently intersecting available metrics.

    Evaluate mono/stereo and paired/identity strata explicitly. Optional stereo
    fields must not silently disappear in a mixed-geometry aggregate. `None` is
    an explicitly undefined relative metric, not a missing field or nonfinite PCM.
    """
    if candidate == control:
        raise ValueError("distinct named candidate and control required")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 2:
        raise ValueError("at least two bootstrap resamples required")
    schema = None
    for row in records:
        if row["path"] not in (candidate, control):
            continue
        if not isinstance(row["scores"], dict) or not row["scores"]:
            raise ValueError("nonempty metric schema required")
        keys = set(row["scores"])
        if schema is None:
            schema = keys
        if keys != schema:
            raise ValueError(
                "metric schema differs; stratify or represent undefined fields explicitly"
            )
        for value in row["scores"].values():
            if value is not None and (
                isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value)
            ):
                raise ValueError("nonfinite or nonnumeric score")
    return paired_recordings(records, candidate, control, seed=seed, resamples=resamples)


def direction(value):
    """Descriptive recording-bootstrap direction, never an audibility test.

    A score computed on actual FP32 PCM in FP64 is not uncertain by `8*eps32*score`.
    Neither proxy/native discrepancy nor machine epsilon supplies a bound on the
    difference of two native scores. Numeric resolution needs separate controls.
    """
    if not value:
        return "missing"
    if value.get("undefined_sources") or value.get("sources", 0) < 1:
        return "undefined"
    delta, interval = value.get("mean_delta"), value.get("paired_source_bootstrap_95")
    if (
        delta is None
        or not math.isfinite(delta)
        or not isinstance(interval, list)
        or len(interval) != 2
        or not all(math.isfinite(x) for x in interval)
        or interval[0] > interval[1]
    ):
        return "invalid"
    source_deltas = value.get("source_deltas", {})
    if len(source_deltas) != value["sources"] or not all(
        math.isfinite(x) for x in source_deltas.values()
    ):
        return "invalid"
    if all(x == 0 for x in source_deltas.values()) and delta == 0 and interval == [0, 0]:
        return "unchanged"
    if value["sources"] < 2:
        return "insufficient_sources"
    if interval[1] < 0 and delta < 0:
        return "decrease"
    if interval[0] > 0 and delta > 0:
        return "increase"
    return "unresolved"


def describe_comparison(values):
    """Keep spectrum, coherence and qualification as distinct questions."""
    metrics = {key: direction(values.get(key)) for key in PRIMARY + SUPPORTING + FIDELITY}
    primary = [metrics[k] for k in PRIMARY]
    if all(x == "decrease" for x in primary):
        spectral = "decrease_on_both_primary_rulers"
    elif all(x == "increase" for x in primary):
        spectral = "increase_on_both_primary_rulers"
    elif any(x in ("missing", "undefined", "invalid", "insufficient_sources") for x in primary):
        spectral = "incomplete"
    elif "increase" in primary or "decrease" in primary:
        spectral = "mixed_or_partial"
    elif all(x == "unchanged" for x in primary):
        spectral = "unchanged"
    else:
        spectral = "unresolved"
    return {
        "spectral_evidence": spectral,
        "metric_directions": metrics,
        "fidelity_increases": [k for k in FIDELITY if metrics[k] == "increase"],
        "missing_metrics": [k for k, v in metrics.items() if v == "missing"],
        "numerical_resolution": "not_established_by_bootstrap_or_machine_epsilon",
        "practical_or_audible_relevance": "not_established",
        "qualified": False,
    }
