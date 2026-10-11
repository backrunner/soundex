"""Globally connected spectral correction with optional finite causal context.

This is an original small-model experiment, not an implementation of Apollo or
UniverSR. It keeps the existing polar deployment protocol and exposes either
scaled polar features or magnitude-weighted circular features inside the graph.
"""

import math

import torch
from torch import nn


class SpectralRefiner(nn.Module):
    """Mix every frequency bin, optionally using eight past frames and no lookahead."""

    def __init__(
        self, *, bins: int, width: int, representation: str, history_frames: int = 0
    ) -> None:
        super().__init__()
        if type(bins) is not int or bins < 2:
            raise ValueError("spectral_refiner.bins must be an integer >=2")
        if type(width) is not int or not 1 <= width <= 512:
            raise ValueError("spectral_refiner.width must be an integer in 1..512")
        if representation not in {"polar", "gain_shape"}:
            raise ValueError("spectral_refiner.representation must be polar or gain_shape")
        self.bins = bins
        self.representation = representation
        if type(history_frames) is not int or history_frames not in (0, 8):
            raise ValueError("spectral_refiner.history_frames must be 0 or 8")
        self.history_frames = history_frames
        self.encoder = nn.Sequential(
            nn.Linear(3 * bins, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.head = nn.Linear(width, 2 * bins)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if history_frames:
            # Finite causal TCN: encode nine observed frames, emit only the last.
            # Zero mixing preserves the parent's mapping up to FP32 GEMM rounding.
            with torch.random.fork_rng(devices=[]):
                self.temporal_filter = nn.Conv1d(width, width, history_frames + 1, groups=width)
                self.temporal_mix = nn.Linear(width, width)
            nn.init.zeros_(self.temporal_mix.weight)
            nn.init.zeros_(self.temporal_mix.bias)

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        magnitude, phase = features[:, 0], features[:, 1]
        magnitude = magnitude.clamp(-120.0, 60.0)
        # Square-root amplitude, relative to the strongest bin in this frame.
        # No batch/time normalization: quiet bins cannot carry full phase weight.
        relative = torch.exp(
            (magnitude - magnitude.amax(dim=-1, keepdim=True)) * (math.log(10) / 40)
        )
        gain = (magnitude + 60.0) / 60.0
        if self.representation == "gain_shape":
            encoded = torch.stack((gain, relative * phase.cos(), relative * phase.sin()), dim=-1)
        else:
            encoded = torch.stack((gain, phase / math.pi, relative), dim=-1)
        return encoded.flatten(-2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.bins:
            raise ValueError("spectral_refiner.bins differs from the input FFT geometry")
        encoded = self.encoder(self.encode_features(features))
        if self.history_frames:
            if features.shape[2] != self.history_frames + 1:
                raise ValueError("temporal refiner requires exactly nine oldest-to-current frames")
            temporal = self.temporal_filter(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = encoded[:, -1:] + self.temporal_mix(torch.nn.functional.gelu(temporal))
        correction = self.head(encoded)
        correction = correction.reshape(features.shape[0], encoded.shape[1], 2, self.bins)
        correction = correction.permute(0, 2, 1, 3).tanh()
        # Finite, interpretable ranges; DSP still owns the retained-band mask.
        return torch.cat((12.0 * correction[:, :1], 0.5 * correction[:, 1:]), dim=1)
