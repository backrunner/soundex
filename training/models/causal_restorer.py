"""Original joint-complex causal backbones with explicit, per-channel state.

No external model code or weights. The public step uses the existing dB/phase
feature units, but all learned encoding and residual prediction are Cartesian.
Sequence and one-hop calls share weights and have no future-dependent norms.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 2, dilation=dilation, groups=width)
        self.expand = nn.Linear(width, width * 2)
        self.project = nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.norm(x)
        history = torch.cat((state, current), dim=1)
        temporal = self.depthwise(history.transpose(1, 2)).transpose(1, 2)
        return x + 0.25 * self.project(F.gelu(self.expand(temporal))), history[:, -self.dilation :]


class CausalRestorer(nn.Module):
    """C1 step: [B,2,1,129], [B,S], [B,1] Hz -> features, next state.

    GRU has two width-192 layers; TCN defaults to width 144, six dilations,
    receptive field 64 hops. Training also accepts T>1, without changing semantics.
    """

    def __init__(self, kind: str = "gru", width: int | None = None) -> None:
        super().__init__()
        if kind not in {"gru", "tcn"}:
            raise ValueError("kind must be gru or tcn")
        self.kind = kind
        self.width = width if width is not None else (192 if kind == "gru" else 144)
        if type(self.width) is not int or not 8 <= self.width <= 256:
            raise ValueError("width must be an integer in 8..256")
        self.encoder = nn.Sequential(
            nn.Linear(3 * 129 + 1, self.width), nn.LayerNorm(self.width), nn.GELU()
        )
        self.dilations = (1, 2, 4, 8, 16, 32)
        if kind == "gru":
            self.temporal = nn.GRU(self.width, self.width, num_layers=2, batch_first=True)
            self.state_size = 2 * self.width
        else:
            self.temporal = nn.ModuleList([TemporalBlock(self.width, d) for d in self.dilations])
            self.state_size = sum(self.dilations) * self.width
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.width, self.width), nn.GELU(), nn.Linear(self.width, 2 * 129)
        )
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)
        valid_imag = torch.ones(129)
        valid_imag[0] = valid_imag[-1] = 0
        self.register_buffer("valid_imag", valid_imag)

    def initial_state(self, batch: int, *, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.state_size, device=device)

    def forward_complex(
        self, features: torch.Tensor, state: torch.Tensor, sample_rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.pow(10.0, features[:, 0].clamp(-200, 100) / 20.0)
        phase = features[:, 1]
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase) * self.valid_imag
        scale = magnitude.square().mean(dim=-1, keepdim=True).clamp_min(1e-20).sqrt()
        compressed = (magnitude / scale).clamp_min(1e-10).pow(0.3)
        encoded_features = torch.cat(
            (
                features[:, 0].clamp(-120, 60) / 60.0,
                compressed * torch.cos(phase),
                compressed * torch.sin(phase) * self.valid_imag,
                (sample_rate / 48000.0).unsqueeze(1).expand(-1, features.shape[2], -1),
            ),
            dim=-1,
        )
        encoded = self.encoder(encoded_features)
        if self.kind == "gru":
            hidden = state.reshape(-1, 2, self.width).transpose(0, 1).contiguous()
            temporal, hidden = self.temporal(encoded, hidden)
            next_state = hidden.transpose(0, 1).reshape(-1, self.state_size)
        else:
            temporal = encoded
            next_states = []
            offset = 0
            for block, dilation in zip(self.temporal, self.dilations, strict=True):
                count = dilation * self.width
                history = state[:, offset : offset + count].reshape(-1, dilation, self.width)
                temporal, updated = block(temporal, history)
                next_states.append(updated.reshape(-1, count))
                offset += count
            next_state = torch.cat(next_states, dim=1)
        residual = 2 * torch.tanh(self.decoder(torch.cat((encoded, temporal), dim=-1)))
        residual_real, residual_imag = residual.split(129, dim=-1)
        predicted = torch.stack(
            (real + scale * residual_real, (imag + scale * residual_imag) * self.valid_imag), dim=1
        )
        return predicted, next_state

    def forward(
        self, features: torch.Tensor, state: torch.Tensor, sample_rate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cartesian, next_state = self.forward_complex(features, state, sample_rate)
        real, imag = cartesian[:, 0], cartesian[:, 1]
        power = real.square() + imag.square()
        magnitude_db = 10 * power.clamp_min(1e-20).log10()
        # Phase of a zero vector is undefined. Keep its derivative finite too.
        safe_real = torch.where(power > 1e-20, real, torch.ones_like(real) * 1e-10)
        safe_imag = torch.where(power > 1e-20, imag, torch.zeros_like(imag))
        phase = torch.atan2(safe_imag, safe_real)
        return torch.stack((magnitude_db, phase), dim=1), next_state
