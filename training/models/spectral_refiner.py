"""Frame-local, globally connected spectral correction with bounded outputs.

This is an original small-model experiment, not an implementation of Apollo or
UniverSR. It keeps the existing polar deployment protocol and exposes either
scaled polar features or magnitude-weighted circular features inside the graph.
"""

import math

import torch
from torch import nn


class SpectralRefiner(nn.Module):
    """Mix every frequency bin without adding temporal buffering or lookahead."""

    def __init__(self, *, bins: int, width: int, representation: str) -> None:
        super().__init__()
        if type(bins) is not int or bins < 2:
            raise ValueError("spectral_refiner.bins must be an integer >=2")
        if type(width) is not int or not 1 <= width <= 512:
            raise ValueError("spectral_refiner.width must be an integer in 1..512")
        if representation not in {"polar", "gain_shape"}:
            raise ValueError("spectral_refiner.representation must be polar or gain_shape")
        self.bins = bins
        self.representation = representation
        self.encoder = nn.Sequential(
            nn.Linear(3 * bins, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.head = nn.Linear(width, 2 * bins)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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
        correction = self.head(self.encoder(self.encode_features(features)))
        correction = correction.reshape(features.shape[0], features.shape[2], 2, self.bins)
        correction = correction.permute(0, 2, 1, 3).tanh()
        # Finite, interpretable ranges; DSP still owns the retained-band mask.
        return torch.cat((12.0 * correction[:, :1], 0.5 * correction[:, 1:]), dim=1)
