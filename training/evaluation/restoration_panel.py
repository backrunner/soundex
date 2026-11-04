"""P71 independent spectral panel and recording-clustered paired uncertainty.

Diagnostics, not a trained perceptual score or a replacement training loss.
SPDX-License-Identifier: Apache-2.0
"""

from collections import defaultdict

import numpy as np

VERSION = "soundex-restoration-panel-v1"
SIZES = (512, 2048, 8192)
BANDS = ((8000, 12000), (12000, 16000), (16000, float("inf")))


def _ratio(numerator, denominator, *, floor=1e-12):
    # Numerical zero only, not a perceptual audibility threshold.
    return float(numerator / denominator) if denominator > floor else None


def _band(a, b):
    # Each channel/frame vector has one-sided, window-energy-normalized units.
    ma, mb = np.abs(a), np.abs(b)
    ea, eb = np.linalg.norm(ma, axis=-1), np.linalg.norm(mb, axis=-1)
    ref = np.linalg.norm(mb)
    pa = ma / np.maximum(ea[..., None], 1e-30)
    pb = mb / np.maximum(eb[..., None], 1e-30)
    # Hellinger distance between normalized power spectra: gain-independent.
    # A missing predicted frame receives distance 1 when reference is nonzero.
    shape = 0.5 * np.square(pa - pb).sum(-1)
    shape = np.where((ea == 0) & (eb > 0), 1.0, shape)
    weight = np.square(eb)
    delta_a, delta_b = np.diff(ea, axis=-1), np.diff(eb, axis=-1)
    return {
        "sc": _ratio(np.linalg.norm(ma - mb), ref),
        "shape": _ratio(np.sum(shape * weight), np.sum(weight), floor=1e-24),
        "envelope": _ratio(np.linalg.norm(ea - eb), np.linalg.norm(eb)),
        "dynamics": _ratio(np.linalg.norm(delta_a - delta_b), np.linalg.norm(eb)),
        "complex": _ratio(np.linalg.norm(a - b), ref),
        "magnitude_error_rms": float(np.sqrt(np.mean(np.square(ma - mb).sum(-1)))),
        "reference_rms": float(np.sqrt(np.mean(np.square(eb)))),
        "envelope_error_rms": float(np.sqrt(np.mean(np.square(ea - eb)))),
        "dynamics_error_rms": float(np.sqrt(np.mean(np.square(delta_a - delta_b)))),
    }


def panel(wave, reference, rate):
    """Aligned physical channels; analysis windows add no deployment lookahead."""
    wave, reference = np.asarray(wave, np.float64), np.asarray(reference, np.float64)
    if (
        rate not in (44100, 48000)
        or wave.shape != reference.shape
        or wave.ndim != 2
        or wave.shape[1] not in (1, 2)
        or len(wave) < 2 * max(SIZES)
        or not np.isfinite(wave).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("finite aligned mono/stereo at 44.1/48 kHz with enough context required")
    resolutions = []
    for size in SIZES:
        window = np.hanning(size + 1)[:-1]
        hz = np.fft.rfftfreq(size, 1 / rate)
        sides = np.full(len(hz), 2.0)
        sides[0] = sides[-1] = 1
        scale = np.sqrt(sides / (size * np.square(window).sum()))

        def spectrum(pcm, size=size, window=window, scale=scale):
            frames = np.lib.stride_tricks.sliding_window_view(pcm.T, size, axis=-1)
            return np.fft.rfft(frames[:, :: size // 4] * window, axis=-1) * scale

        a, b = spectrum(wave), spectrum(reference)
        scores = {"hf_" + k: v for k, v in _band(a[..., hz >= 8000], b[..., hz >= 8000]).items()}
        for low, high in BANDS:
            mask = (hz >= low) & (hz < high)
            scores.update(
                {f"band{low}_{k}": v for k, v in _band(a[..., mask], b[..., mask]).items()}
            )
        if wave.shape[1] == 2:
            for label, sign in (("mid", 1), ("side", -1)):
                aa, bb = [(z[0:1] + sign * z[1:2]) / 2 for z in (a, b)]
                scores.update(
                    {
                        f"{label}_{k}": v
                        for k, v in _band(aa[..., hz >= 8000], bb[..., hz >= 8000]).items()
                    }
                )
        resolutions.append(scores)
    means = {
        k: (
            float(np.mean([r[k] for r in resolutions]))
            if all(r[k] is not None for r in resolutions)
            else None
        )
        for k in resolutions[0]
    }
    return {"metrics": means, "resolutions": dict(zip(map(str, SIZES), resolutions, strict=True))}


def paired_recordings(records, candidate, control, *, seed=71, resamples=2000):
    """Pair by source/case before aggregation; keep all adverse sources.

    CIs describe recording resampling of this panel, not artist independence,
    multiplicity-adjusted significance or inference to an unobserved population.
    """
    paths = {candidate: {}, control: {}}
    for row in records:
        if row["path"] in paths:
            key = (row["source"], row["case"])
            if key in paths[row["path"]]:
                raise ValueError("duplicate paired case")
            paths[row["path"]][key] = row["scores"]
    if not paths[candidate] or paths[candidate].keys() != paths[control].keys():
        raise ValueError("candidate/control case coverage must match exactly")
    metrics = set.intersection(*(set(v) for p in paths.values() for v in p.values()))
    result = {}
    for metric in sorted(metrics):
        groups = defaultdict(list)
        undefined = set()
        for key, after in paths[candidate].items():
            a, b = after[metric], paths[control][key][metric]
            if a is None or b is None:
                undefined.add(key[0])
            elif not np.isfinite(a) or not np.isfinite(b):
                raise ValueError("nonfinite score")
            else:
                groups[key[0]].append((a, b))
        # Do not silently use only a source's favorable/defined excerpts.
        grouped = {k: np.mean(v, axis=0) for k, v in sorted(groups.items()) if k not in undefined}
        if not grouped:
            result[metric] = {"sources": 0, "undefined_sources": sorted(undefined)}
            continue
        values = np.asarray(list(grouped.values()))
        delta = values[:, 0] - values[:, 1]
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(delta), size=(resamples, len(delta)))
        ci = np.quantile(delta[sampled].mean(1), (0.025, 0.975)).tolist()
        result[metric] = {
            "sources": len(delta),
            "undefined_sources": sorted(undefined),
            "candidate_mean": float(values[:, 0].mean()),
            "control_mean": float(values[:, 1].mean()),
            "mean_delta": float(delta.mean()),
            "paired_source_bootstrap_95": ci,
            "source_deltas": dict(zip(grouped, map(float, delta), strict=True)),
            "adverse_sources": [k for k, d in zip(grouped, delta, strict=True) if d > 0],
        }
    return result
