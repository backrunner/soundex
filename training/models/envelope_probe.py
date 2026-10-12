"""Experimental amplitude predictor; not a deployable SoundEx audio model.

Feature clock: frame t ends at sample (t + 1) * 128. Long FFTs update on
t % 4 == 0 and are held between updates. No lookahead or centered padding.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.causal_restorer import TemporalBlock

ARMS = ("short", "tiled", "past")
CONTEXT_BINS = 513 + 1025


def context_features(pcm: torch.Tensor, arm: str) -> torch.Tensor:
    """[B,N] -> [B,N/128,1538], with independent zero-initialized channels."""
    if arm not in ARMS or pcm.ndim != 2 or pcm.shape[-1] % 128:
        raise ValueError("expected a known arm and whole 128-sample hops")
    frames = pcm.shape[-1] // 128
    if arm == "short":
        return pcm.new_zeros(pcm.shape[0], frames, CONTEXT_BINS)
    short = F.pad(pcm, (128, 0)).unfold(-1, 256, 128)[:, ::4]
    values = []
    for size in (1024, 2048):
        wave = (
            F.pad(pcm, (size - 128, 0)).unfold(-1, size, 128)[:, ::4]
            if arm == "past"
            else short.repeat(1, 1, size // 256)
        )
        window = torch.hann_window(size, device=pcm.device, dtype=pcm.dtype)
        magnitude = torch.fft.rfft(wave * window).abs() * (256 / size)
        values.append(magnitude.repeat_interleave(4, dim=1)[:, :frames])
    return torch.cat(values, dim=-1)


class StreamingContext:
    """CPU diagnostic frontend. Caller owns one instance per physical channel batch."""

    def __init__(self, batch: int, arm: str):
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.arm = arm
        self.history = torch.zeros(batch, 2048)
        self.held = torch.zeros(batch, CONTEXT_BINS)
        self.hop = 0
        self.windows = {n: torch.hann_window(n) for n in (1024, 2048)}

    def step(self, pcm: torch.Tensor) -> torch.Tensor:
        if pcm.shape != (self.history.shape[0], 128):
            raise ValueError("step requires exactly one hop per channel")
        self.history = torch.cat((self.history[:, 128:], pcm), dim=-1)
        if self.hop % 4 == 0 and self.arm != "short":
            values = []
            for size, window in self.windows.items():
                wave = (
                    self.history[:, -size:]
                    if self.arm == "past"
                    else self.history[:, -256:].repeat(1, size // 256)
                )
                values.append(torch.fft.rfft(wave * window).abs() * (256 / size))
            self.held = torch.cat(values, dim=-1)
        self.hop += 1
        return self.held[:, None]


class EnvelopeProbe(nn.Module):
    def __init__(self, width: int = 144):
        super().__init__()
        self.width = width
        self.dilations = (1, 2, 4, 8, 16, 32)
        self.state_size = width * sum(self.dilations)
        self.encoder = nn.Sequential(
            nn.Linear(129 + CONTEXT_BINS + 4, width), nn.LayerNorm(width), nn.GELU()
        )
        self.temporal = nn.ModuleList([TemporalBlock(width, d) for d in self.dilations])
        self.decoder = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 129))
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def initial_state(self, batch, *, device=None):
        return torch.zeros(batch, self.state_size, device=device)

    def forward(self, magnitude, context, condition, state):
        # condition = [sample_rate / 48000, cutoff / Nyquist, active].
        scale = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
        inputs = torch.cat(
            (
                torch.log1p(magnitude / scale),
                torch.log1p(context / scale),
                scale.log10() / 10,
                condition,
            ),
            dim=-1,
        )
        encoded = self.encoder(inputs)
        temporal, states, offset = encoded, [], 0
        for block, dilation in zip(self.temporal, self.dilations, strict=True):
            count = dilation * self.width
            temporal, updated = block(
                temporal, state[:, offset : offset + count].reshape(-1, dilation, self.width)
            )
            states.append(updated.reshape(-1, count))
            offset += count
        delta = 2 * torch.tanh(self.decoder(torch.cat((encoded, temporal), dim=-1)))
        return (magnitude + scale * delta).abs(), torch.cat(states, dim=1)


def probe_inputs(batch, context):
    magnitude = torch.pow(10.0, batch["x"][:, 0] / 20)
    rate = batch["rate"].expand(-1, magnitude.shape[1])
    condition = torch.stack(
        (rate / 48000, batch["cutoff"] / (rate / 2), batch["active"].float()), dim=-1
    )
    return magnitude, context, condition


def predict_probe(model, batch, context, burn=64):
    magnitude, context, condition = probe_inputs(batch, context)
    state = model.initial_state(len(magnitude), device=magnitude.device)
    with torch.no_grad():
        _, state = model(magnitude[:, :burn], context[:, :burn], condition[:, :burn], state)
    prediction, _ = model(
        magnitude[:, burn:], context[:, burn:], condition[:, burn:], state.detach()
    )
    return torch.cat((magnitude[:, :burn], prediction), dim=1)
