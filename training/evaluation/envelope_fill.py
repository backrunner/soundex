"""P108 offline proxy of a deployable envelope-driven copy-up fill.

Per physical channel and causal frame: the Rust bandwidth detector (ported
exactly: peak-60 dB top-down scan, alpha 0.1, 0.85 Nyquist, 10-frame hangover,
silence reset) gives a cutoff; above it, a hop-coherent copy-up of the 4 kHz
to cutoff band is added with energy max(predicted - present, 0) per band (or
bin). Input is never attenuated. Synthesis is Rust-matching Hann WOLA. This is
a Python proxy; native Rust correspondence is a separate, required check.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import torch

from models.envelope_predictor import (
    BANDS,
    CALIBRATION,
    FFT_SIZE,
    HOP,
    LOW_HZ,
    band_index,
    band_matrix,
    bin_hz,
    features,
    high_bins,
    stft_power,
)

THRESHOLD_DB = -60.0
ALPHA = 0.1
MIN_RATIO = 0.85
HANGOVER = 10
MAX_GAIN = 1e3


def detector(spec, frames_pcm, rate):
    """Per-frame (smoothed bandwidth Hz, needs_enhancement), as soundex-dsp."""
    log_mag = 20 * np.log10(np.maximum(np.abs(spec), 1e-10))
    resolution = (rate / 2) / (log_mag.shape[1] - 1)
    smoothed, hang = 0.0, 0
    bw, needs = np.zeros(len(spec)), np.zeros(len(spec), bool)
    for t, row in enumerate(log_mag):
        above = np.nonzero(row[1:] > row.max() + THRESHOLD_DB)[0]
        detected = (above[-1] + 1) * resolution if len(above) else 0.0
        smoothed = detected if smoothed == 0.0 else ALPHA * detected + (1 - ALPHA) * smoothed
        bw[t] = smoothed
        if np.all(np.abs(frames_pcm[t]) <= 1e-10):
            smoothed, hang = 0.0, 0
            continue
        need = smoothed / (rate / 2) < MIN_RATIO
        if need:
            hang = HANGOVER
        elif hang > 0:
            hang -= 1
            need = True
        needs[t] = need
    return bw, needs


def wola(spec, length):
    """Inverse of `stft_power` framing with Hann synthesis / sum(w^2)."""
    w = np.hanning(FFT_SIZE + 1)[:-1]
    frames = np.fft.irfft(spec, FFT_SIZE, axis=-1) * w
    out = np.zeros(len(spec) * HOP + FFT_SIZE)
    weight = np.zeros_like(out)
    for t in range(len(spec)):
        out[t * HOP : t * HOP + FFT_SIZE] += frames[t]
        weight[t * HOP : t * HOP + FFT_SIZE] += w * w
    out = out / np.maximum(weight, 1e-12)
    pcm = out[FFT_SIZE - HOP :][:length]
    return np.pad(pcm, (0, length - len(pcm)))


def predict(model, power, rate):
    """Absolute log10 power per output unit for every frame (causal GRU)."""
    x, ref = features(power, rate)
    with torch.no_grad():
        y, _ = model(torch.as_tensor(x[None], dtype=torch.float32))
    return y[0].double().numpy() + ref[:, None]


def fill_channel(pcm, rate, mode, model=None, reference=None):
    """One physical channel. mode: 'band' | 'bin' | 'band_deficit' (model) or
    'oracle_band' (reference). Level modes fill max(predicted - present, 0);
    'band_deficit' predicts the missing band energy itself and adds it directly."""
    pcm = np.asarray(pcm, np.float64)
    power, spec = stft_power(pcm)
    padded = np.concatenate([np.zeros(FFT_SIZE - HOP), pcm, np.zeros(len(spec) * HOP - len(pcm))])
    frames_pcm = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::HOP][: len(spec)]
    bw, needs = detector(spec, frames_pcm, rate)
    if mode == "oracle_band":
        ref_power, _ = stft_power(reference)
        target = np.log10(ref_power @ band_matrix(rate) + 1e-30)
    elif mode in ("band", "bin", "band_deficit"):
        target = predict(model, power, rate)
    else:
        raise ValueError("unknown fill mode")
    hz = bin_hz(rate)
    resolution = (rate / 2) / (len(hz) - 1)
    lo = max(int(np.searchsorted(hz, LOW_HZ)), 1)
    ids = band_index(rate)
    counts = np.bincount(ids[ids >= 0], minlength=BANDS)
    hb = high_bins(rate)
    out = spec.copy()
    for t in np.nonzero(needs)[0]:
        cut = int(np.clip(round(bw[t] / resolution), 1, len(hz) - 1))
        start = cut + 1
        width = start - lo
        if width < 1 or start >= len(hz):
            continue
        ks = np.arange(start, len(hz))
        source = lo + (ks - start) % width
        patch = spec[t, source] * np.exp(1j * np.pi * (ks - source) * t * 2 * HOP / FFT_SIZE)
        p_pow = np.square(np.abs(patch)) * CALIBRATION
        present = power[t, ks]
        if mode == "bin":
            want = 10 ** target[t, ks - hb[0]]  # ks >= start > lo == hb[0].
            fill = np.maximum(want - present, 0)
            gain = np.sqrt(fill / np.maximum(p_pow, 1e-30))
        else:
            gain = np.zeros(len(ks))
            kb = ids[ks]
            for b in np.unique(kb[kb >= 0]):
                sel = kb == b
                share = sel.sum() / max(counts[b], 1)  # Part of the band above the cutoff.
                want = 10 ** target[t, b] * share
                fill = want if mode == "band_deficit" else max(want - present[sel].sum(), 0.0)
                gain[sel] = np.sqrt(fill / max(p_pow[sel].sum(), 1e-30))
        out[t, ks] += np.minimum(gain, MAX_GAIN) * patch
    return wola(out, len(pcm)), {"active_frames": int(needs.sum()), "frames": len(needs)}


def render(pcm, rate, mode, model=None, reference=None):
    """[samples, channels]; each physical channel keeps independent state."""
    pcm = np.asarray(pcm, np.float64)
    outs, info = [], []
    for c in range(pcm.shape[1]):
        ref = None if reference is None else np.asarray(reference, np.float64)[:, c]
        y, i = fill_channel(pcm[:, c], rate, mode, model, ref)
        outs.append(y)
        info.append(i)
    return np.stack(outs, 1), info
