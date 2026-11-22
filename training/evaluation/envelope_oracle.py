"""P107 reference-assisted ceilings for parametric high-band synthesis.

These signals use the lossless reference and are *not* deployable models. They
measure how much of the missing-band error a causal per-frame parameterization
could remove on the deployment STFT grid (FFT 256, hop 128): a copy-up patch of
the retained band, scaled per frame to the reference envelope. They run offline
in Python without the Rust limiter; identity DSP is reported separately.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

from evaluation.missing_band import erb_number, missing_mask, missing_region

FFT_SIZE = 256
HOP = 128
MASK_DB = -10.0
PATCH_FLOOR_HZ = 4000.0
REGION_SHARE = 0.75  # Masked share locating the top missing region on the coarse grid.
MAX_GAIN = 1e3  # Amplitude cap; a silent source band cannot be raised arbitrarily.


def _window():
    # Periodic Hann for analysis and synthesis, as in soundex-dsp/src/stft.rs.
    # (P107 v2 used sqrt-Hann: its sidelobes leaked low-band power over the
    # 60-90 dB weaker high band and emptied the deployment-grid mask.)
    return np.hanning(FFT_SIZE + 1)[:-1]


def stft(pcm):
    """[channel, frame, bin]; frames start at -HOP so every sample has two windows."""
    pcm = np.asarray(pcm, np.float64)
    padded = np.pad(pcm.T, ((0, 0), (HOP, FFT_SIZE)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE, axis=-1)[:, ::HOP]
    return np.fft.rfft(frames * _window(), axis=-1)


def istft(spec, length):
    """Weighted overlap-add normalized by accumulated squared windows (Rust WOLA)."""
    window = _window()
    frames = np.fft.irfft(spec, FFT_SIZE, axis=-1) * window
    channels, count = spec.shape[:2]
    out = np.zeros((channels, (count - 1) * HOP + FFT_SIZE))
    weight = np.zeros(out.shape[1])
    for t in range(count):
        out[:, t * HOP : t * HOP + FFT_SIZE] += frames[:, t]
        weight[t * HOP : t * HOP + FFT_SIZE] += np.square(window)
    out /= np.maximum(weight, 1e-12)
    return out[:, HOP : HOP + length].T


def grid_mask(xd, xr, rate):
    """[channel, bin] missing bins on the deployment grid, same rule as the ruler."""
    hz = np.fft.rfftfreq(FFT_SIZE, 1 / rate)
    pd, pr = np.square(np.abs(xd)), np.square(np.abs(xr))
    # Unnormalized FFT power: a floor relative to the reference keeps dB ratios finite.
    floor = 1e-13 * max(float(pr.mean(1).max()), 1e-30)
    return missing_mask(pd, pr, hz, MASK_DB, floor=floor), hz


def copy_up(x, c, hz, stop=None):
    """Translate the retained band [lo, c) into [c, stop) with hop-consistent phase."""
    lo = max(int(np.searchsorted(hz, PATCH_FLOOR_HZ)), 1)
    width = c - lo
    if width < 1:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    t = np.arange(x.shape[0])[:, None]
    for k in range(c, x.shape[1] if stop is None else stop):
        source = lo + (k - c) % width
        shift = k - source
        # A shift of s bins advances phase by 2*pi*s*HOP/FFT_SIZE per frame.
        out[:, k] = x[:, source] * np.exp(1j * np.pi * shift * t[:, 0] * 2 * HOP / FFT_SIZE)
    return out


def bands(hz, index, step):
    """Groups of the edited bins `index`: ERB-number width `step`, or one if None."""
    if step is None:
        return [index]
    edges = np.floor(erb_number(hz[index]) / step).astype(int)
    return [index[edges == e] for e in np.unique(edges)]


def shape_to(candidate, target, groups):
    """Per-frame band power of `candidate` scaled to `target`'s, [frame, bin]."""
    out = candidate.copy()
    for g in groups:
        ec = np.square(np.abs(candidate[:, g])).sum(-1)
        et = np.square(np.abs(target[:, g])).sum(-1)
        gain = np.where(ec > 1e-24, np.sqrt(et / np.maximum(ec, 1e-24)), 0.0)
        out[:, g] *= np.minimum(gain, MAX_GAIN)[:, None]
    return out


ORACLES = (
    "oracle_patch_level",  # Copy-up, one per-frame level for the whole missing band.
    "oracle_patch_erb",  # Copy-up, per-frame half-ERB envelope.
    "oracle_noise_erb",  # Seeded complex noise, per-frame half-ERB envelope.
    "oracle_patch_binmag",  # Reference per-bin magnitude, copy-up phase.
    "oracle_complex",  # Reference complex bins: pipeline sanity ceiling.
)


def render(dry, reference, rate, kind, seed=107):
    """Oracle output on the dry timeline, [samples, channels]."""
    if kind not in ORACLES:
        raise ValueError("unknown oracle")
    xd, xr = stft(dry), stft(reference)
    mask, hz = grid_mask(xd, xr, rate)
    out = xd.copy()
    rng = np.random.default_rng(seed)
    for ch in range(xd.shape[0]):
        region = missing_region(mask[ch], REGION_SHARE)
        if region is None:
            continue
        c, stop = region
        # Edit only masked bins of the top region; retained bins inside it already
        # match the reference, and masked holes below it are left untreated.
        edit = c + np.nonzero(mask[ch, c:stop])[0]
        src, ref = xd[ch], xr[ch]
        if kind == "oracle_complex":
            out[ch][:, edit] = ref[:, edit]
            continue
        if kind == "oracle_noise_erb":
            base = rng.standard_normal(src.shape) + 1j * rng.standard_normal(src.shape)
        else:
            base = copy_up(src, c, hz, stop)
        if kind == "oracle_patch_binmag":
            out[ch][:, edit] = np.abs(ref[:, edit]) * np.exp(1j * np.angle(base[:, edit]))
            continue
        step = None if kind == "oracle_patch_level" else 0.5
        out[ch][:, edit] = shape_to(base, ref, bands(hz, edit, step))[:, edit]
    return istft(out, len(dry))
