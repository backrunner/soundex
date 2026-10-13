"""Complex-coordinate form of the existing causal OLA and aligned analysis."""

import torch
from torch.nn import functional as F


def complex_features(features):
    magnitude = torch.pow(10.0, features[:, 0].clamp(-200, 100) / 20)
    return torch.complex(magnitude * features[:, 1].cos(), magnitude * features[:, 1].sin())


def synthesize(spectrum):
    window = torch.hann_window(256, dtype=spectrum.real.dtype, device=spectrum.device)
    frames = torch.fft.irfft(spectrum, n=256) * window
    count = spectrum.shape[1]
    length = (count + 1) * 128
    wave = F.fold(frames.transpose(1, 2), (1, length), (1, 256), stride=(1, 128))[:, 0, 0]
    weights = F.fold(
        window.square()[None, :, None].expand(1, -1, count), (1, length), (1, 256), stride=(1, 128)
    )[0, 0, 0]
    return torch.where(weights > 1e-12, wave / weights.clamp_min(1e-12), 0)[:, : count * 128]


def analyze(wave):
    window = torch.hann_window(256, dtype=wave.dtype, device=wave.device)
    return torch.fft.rfft(wave[:, :-128].unfold(-1, 256, 128) * window)
