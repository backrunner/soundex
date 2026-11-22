"""P107 missing-band rulers: score only spectra the codec actually removed.

The P71/P81 panel scores fixed >=8 kHz bands, which mostly contain retained
spectra for high-bitrate codecs. Here the scored region is derived per clip and
channel from the untreated input versus the aligned lossless reference. All
constants are registered before scoring; none is fitted to a known recording.
No training loss, acceptance threshold or audibility claim is defined here.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

VERSION = "soundex-missing-band-v2"
FFT_SIZE = 2048
HOP = 512
MASK_FLOOR_HZ = 4000.0  # Lower search limit; codec low-pass/holes of interest lie above.
MASK_DB = -10.0  # Primary: long-term input power this far below the reference.
MASK_SENSITIVITY_DB = (-6.0, -20.0)  # Reported alongside, never substituted.
REFERENCE_FLOOR_REL = 1e-10  # Ignore bins 100 dB below the clip's loudest bin.
ACTIVE_FRAME_MS = 1e-9  # -90 dB re unit PCM mean square in the missing band.
ACTIVE_BAND_MS = 1e-10  # -100 dB re unit PCM mean square per ERB band.
BIN_FLOOR = 1e-13  # Per-bin power floor for log ratios (-130 dB).
BAND_FLOOR = 1e-12  # Per-band power floor for log ratios (-120 dB).
ERB_STEP = 0.5  # Band width in ERB-number units (Glasberg & Moore 1990).
MIN_CORR_FRAMES = 8


def erb_number(hz):
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(hz, np.float64))


def calibrated_stft(pcm, rate, size=FFT_SIZE, hop=HOP):
    """[channel, frame, bin] with one-sided window-energy PCM mean-square units.

    sum(|X|^2) over bins equals the windowed frame's PCM mean square (Parseval),
    matching the P71 normalization. Returns spectra and bin frequencies.
    """
    pcm = np.asarray(pcm, np.float64)
    if pcm.ndim != 2 or pcm.shape[0] < size:
        raise ValueError("expected [samples, channels] audio longer than one window")
    window = np.hanning(size + 1)[:-1]
    hz = np.fft.rfftfreq(size, 1 / rate)
    sides = np.full(len(hz), 2.0)
    sides[0] = sides[-1] = 1
    scale = np.sqrt(sides / (size * np.square(window).sum()))
    frames = np.lib.stride_tricks.sliding_window_view(pcm.T, size, axis=-1)[:, ::hop]
    return np.fft.rfft(frames * window, axis=-1) * scale, hz


def missing_mask(dry_power, ref_power, hz, threshold_db=MASK_DB, *, floor=BIN_FLOOR, rule="median"):
    """Boolean [channel, bin]: bins the codec removed.

    `dry_power`/`ref_power` are [channel, frame, bin] powers. The primary rule is
    the per-bin median over frames of the per-frame dB ratio: codecs drop high
    bands intermittently, and a long-term mean is dominated by the few frames
    that keep them (P107 v2 diagnosis). `rule="mean"` keeps the v1/v2 long-term
    mean definition as a sensitivity field. Bins whose reference is ~absent are
    excluded: nothing was there to remove.
    """
    ref = ref_power.mean(1)
    present = ref > REFERENCE_FLOOR_REL * ref.max(-1, keepdims=True)
    if rule == "median":
        ratio = np.median(10 * np.log10((dry_power + floor) / (ref_power + floor)), axis=1)
    elif rule == "mean":
        ratio = 10 * np.log10((dry_power.mean(1) + 1e-30) / (ref + 1e-30))
    else:
        raise ValueError("unknown mask rule")
    return (hz >= MASK_FLOOR_HZ) & present & (ratio < threshold_db)


def missing_region(row, share=0.9):
    """(start, stop) bins: lowest masked start whose span up to the highest masked
    bin is >= `share` masked, else None. Bins above the highest masked bin are
    excluded because a reference already at its noise floor is not missing."""
    index = np.nonzero(row)[0]
    if not len(index):
        return None
    stop = int(index[-1]) + 1
    span = row[:stop]
    tail = np.cumsum(span[::-1])[::-1] / np.arange(stop, 0, -1)
    hits = np.nonzero((tail >= share) & span)[0]
    return (int(hits[0]), stop) if len(hits) else None


def cutoff_hz(mask, hz, share=0.9):
    """Start frequency of each channel's contiguous missing region, else None."""
    regions = [missing_region(row, share) for row in mask]
    return [float(hz[r[0]]) if r else None for r in regions]


def erb_bands(hz, mask_row, step=ERB_STEP):
    """Lists of bin indices for consecutive half-ERB bands over masked bins."""
    index = np.nonzero(mask_row)[0]
    if not len(index):
        return []
    edges = np.floor(erb_number(hz[index]) / step).astype(int)
    return [index[edges == e] for e in np.unique(edges)]


def _corr(a, b):
    if np.std(b) == 0:
        return None
    if np.std(a) == 0:
        return 0.0  # A constant candidate carries no temporal information.
    return float(np.corrcoef(a, b)[0, 1])


def channel_scores(xa, xb, hz, mask_row):
    """Scores for one channel; xa candidate, xb reference, [frame, bin] complex."""
    pa, pb = np.square(np.abs(xa)), np.square(np.abs(xb))
    keep = ~mask_row & (hz >= 50)
    preserved = {
        "preserved_complex_rel": float(
            np.linalg.norm(xa[:, keep] - xb[:, keep]) / max(np.linalg.norm(xb[:, keep]), 1e-12)
        ),
        "preserved_lsd_db": float(
            np.mean(
                np.sqrt(
                    np.mean(
                        np.square(
                            10 * np.log10((pa[:, keep] + BIN_FLOOR) / (pb[:, keep] + BIN_FLOOR))
                        ),
                        -1,
                    )
                )
            )
        ),
    }
    undefined = dict.fromkeys(
        (
            "missing_lsd_db",
            "missing_erb_rmse_db",
            "missing_erb_bias_db",
            "missing_temporal_corr",
            "missing_energy_ratio_db",
        )
    )
    if not mask_row.any():
        return {**undefined, **preserved, "missing_active_frames": 0}
    ea, eb = pa[:, mask_row].sum(-1), pb[:, mask_row].sum(-1)
    active = eb > ACTIVE_FRAME_MS
    if not active.any():
        return {**undefined, **preserved, "missing_active_frames": 0}
    ratio_bins = 10 * np.log10(
        (pa[active][:, mask_row] + BIN_FLOOR) / (pb[active][:, mask_row] + BIN_FLOOR)
    )
    bands = erb_bands(hz, mask_row)
    ba = np.stack([pa[:, b].sum(-1) for b in bands], -1)[active]
    bb = np.stack([pb[:, b].sum(-1) for b in bands], -1)[active]
    band_on = bb > ACTIVE_BAND_MS
    band_db = 10 * np.log10((ba + BAND_FLOOR) / (bb + BAND_FLOOR))
    corrs, weights = [], []
    if active.sum() >= MIN_CORR_FRAMES:
        for j in range(len(bands)):
            c = _corr(np.log10(ba[:, j] + BAND_FLOOR), np.log10(bb[:, j] + BAND_FLOOR))
            if c is not None:
                corrs.append(c)
                weights.append(bb[:, j].sum())
    return {
        "missing_lsd_db": float(np.mean(np.sqrt(np.mean(np.square(ratio_bins), -1)))),
        "missing_erb_rmse_db": float(np.sqrt(np.mean(np.square(band_db[band_on]))))
        if band_on.any()
        else None,
        "missing_erb_bias_db": float(np.mean(band_db[band_on])) if band_on.any() else None,
        "missing_temporal_corr": float(np.average(corrs, weights=weights)) if corrs else None,
        "missing_energy_ratio_db": float(
            10 * np.log10((ea[active].sum() + BAND_FLOOR) / (eb[active].sum() + BAND_FLOOR))
        ),
        **preserved,
        "missing_active_frames": int(active.sum()),
    }


def score(candidate, reference, dry, rate):
    """Channel-mean scores of `candidate` against `reference`.

    `dry` (the untreated codec output on the same timeline) defines the mask only.
    All three arrays are [samples, channels] and must be time-aligned.
    """
    if not (np.shape(candidate) == np.shape(reference) == np.shape(dry)):
        raise ValueError("candidate, reference and dry input must share geometry")
    xa, hz = calibrated_stft(candidate, rate)
    xb, _ = calibrated_stft(reference, rate)
    xd, _ = calibrated_stft(dry, rate)
    pd, pb = np.square(np.abs(xd)), np.square(np.abs(xb))
    mask = missing_mask(pd, pb, hz)
    rows = [channel_scores(xa[c], xb[c], hz, mask[c]) for c in range(len(mask))]
    result = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        if key == "missing_active_frames":
            result[key] = int(sum(values))
        else:
            defined = [v for v in values if v is not None]
            # A channel without a missing band does not dilute the other's score.
            result[key] = float(np.mean(defined)) if defined else None
    variants = [(f"mask{int(-db)}", missing_mask(pd, pb, hz, db)) for db in MASK_SENSITIVITY_DB]
    variants.append(("meanmask", missing_mask(pd, pb, hz, rule="mean")))
    for suffix, alt in variants:
        alt_rows = [channel_scores(xa[c], xb[c], hz, alt[c]) for c in range(len(alt))]
        for key in ("missing_lsd_db", "missing_erb_rmse_db", "missing_erb_bias_db"):
            defined = [r[key] for r in alt_rows if r[key] is not None]
            result[f"{key}_{suffix}"] = float(np.mean(defined)) if defined else None
    cuts = [c for c in cutoff_hz(mask, hz) if c is not None]
    result["mask_cutoff_hz"] = float(np.mean(cuts)) if cuts else None
    result["mask_missing_bins"] = int(mask.sum())
    return result
