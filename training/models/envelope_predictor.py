"""P108 causal high-band envelope predictor and matched per-bin control.

Both arms share one GRU backbone, the deployment STFT (FFT 256, hop 128,
periodic Hann) and an L1 dB objective against the original lossless reference.
They differ only in the output parameterization: half-ERB band energies above
4 kHz ("band") or per-bin energies above 4 kHz ("bin"). Outputs drive an
additive copy-up fill (`evaluation.envelope_fill`); no phase is predicted.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import torch
from torch import nn

FFT_SIZE = 256
HOP = 128
LOW_HZ = 4000.0
ERB_STEP = 0.5
MAX_HZ = 24000.0
# Calibrated one-sided PCM mean-square units (P71/P107): 2 / (N * sum(hann^2)).
CALIBRATION = 2.0 / (FFT_SIZE * np.square(np.hanning(FFT_SIZE + 1)[:-1]).sum())
LOG_FLOOR = 1e-12  # -120 dB re unit PCM mean square, inside log10.
ACTIVE = 1e-10  # Reference band/bin power that enters supervision (-100 dB).
ARMS = ("band", "bin")


def erb_number(hz):
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(hz, np.float64))


def bin_hz(rate):
    return np.fft.rfftfreq(FFT_SIZE, 1 / rate)


def band_index(rate):
    """[bins] half-ERB band id for bins >= 4 kHz (else -1); shared 44.1/48 kHz ids."""
    hz = bin_hz(rate)
    first = int(np.floor(erb_number(LOW_HZ) / ERB_STEP))
    ids = np.floor(erb_number(hz) / ERB_STEP).astype(int) - first
    ids[hz < LOW_HZ] = -1
    return ids


BANDS = int(np.floor(erb_number(MAX_HZ) / ERB_STEP) - np.floor(erb_number(LOW_HZ) / ERB_STEP)) + 1
HIGH_BINS = int((bin_hz(48000) >= LOW_HZ).sum())  # Per-bin arm width; 44.1 kHz pads.


def band_matrix(rate):
    """[bins, BANDS] 0/1 aggregation from bin power to band power."""
    ids = band_index(rate)
    m = np.zeros((FFT_SIZE // 2 + 1, BANDS), np.float32)
    for k, b in enumerate(ids):
        if b >= 0:
            m[k, b] = 1
    return m


def high_bins(rate):
    """Indices of bins >= 4 kHz; the per-bin arm predicts them left-aligned."""
    return np.nonzero(bin_hz(rate) >= LOW_HZ)[0]


def stft_power(pcm):
    """[frames, bins] calibrated power of a mono signal on the Rust grid.

    Frame t covers samples [t*HOP - (FFT_SIZE - HOP), t*HOP + HOP): the causal
    frame completed at hop t, as in the Rust stream (zero history before t=0).
    """
    pcm = np.asarray(pcm, np.float64)
    count = -(-len(pcm) // HOP)  # Final partial hop is zero-padded, as a flushed stream.
    padded = np.concatenate([np.zeros(FFT_SIZE - HOP), pcm, np.zeros(count * HOP - len(pcm))])
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::HOP][:count]
    spec = np.fft.rfft(frames * np.hanning(FFT_SIZE + 1)[:-1], axis=-1)
    return np.square(np.abs(spec)) * CALIBRATION, spec


def features(power, rate):
    """[frames, 131]: per-bin log10 power minus a sub-4 kHz loudness reference,
    the reference itself, and a sample-rate flag. Uses only the current frame."""
    logp = np.log10(power + LOG_FLOOR)
    low = bin_hz(rate) < LOW_HZ
    ref = np.log10(power[:, low].mean(-1) + LOG_FLOOR)
    flag = np.full(len(power), 1.0 if rate == 48000 else 0.0)
    return np.concatenate([logp - ref[:, None], ref[:, None] / 10, flag[:, None]], -1), ref


def targets(power, rate, arm):
    """Reference log10 power per output unit (band or bin) and its activity mask."""
    if arm == "band":
        e = power @ band_matrix(rate)
        valid = band_matrix(rate).sum(0) > 0
    else:
        idx = high_bins(rate)
        e = np.zeros((len(power), HIGH_BINS))
        e[:, : len(idx)] = power[:, idx]
        valid = np.arange(HIGH_BINS) < len(idx)
    active = (e > ACTIVE) & valid[None]
    return np.log10(e + LOG_FLOOR), active


class EnvelopePredictor(nn.Module):
    """Causal GRU: one frame in, one frame out, explicit recurrent state."""

    def __init__(self, arm, hidden=128, layers=2):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.arm, self.hidden, self.layers = arm, hidden, layers
        self.inp = nn.Sequential(nn.Linear(FFT_SIZE // 2 + 3, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, BANDS if arm == "band" else HIGH_BINS)

    def forward(self, x, state=None):
        """x [batch, frames, 131] -> log10 power relative to the loudness reference."""
        y, state = self.gru(self.inp(x), state)
        return self.out(y), state


def envelope_loss(pred, target, active, ref, warmup, diff_weight=0.5):
    """L1 in dB on active units after warmup, plus L1 dB on adjacent-frame changes.

    `pred` is relative to the per-frame loudness reference `ref` (log10 units);
    targets are absolute log10 power. Warmup frames never enter supervision.
    Returns (total, level_db, change_db); zero at the correct reference.
    """
    absolute = pred + ref[..., None]
    err = 10 * (absolute - target)
    mask = active.clone()
    mask[:, :warmup] = False
    m = mask.float()
    level = (err.abs() * m).sum() / m.sum().clamp_min(1)
    pair = (mask[:, 1:] & mask[:, :-1]).float()
    change = ((err[:, 1:] - err[:, :-1]).abs() * pair).sum() / pair.sum().clamp_min(1)
    return level + diff_weight * change, level.detach(), change.detach()


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def as_tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)
