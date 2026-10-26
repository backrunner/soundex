"""P62 frequency-local causal representation prototype; no quality claim.

Independent implementation. Keep P57 observations, residual coordinates and
heads; delay global compression until after band-wise temporal processing.
SPDX-License-Identifier: Apache-2.0
"""

from itertools import pairwise

import torch
from torch import nn

from models.adapter_trainability import TrainableAdapter
from models.causal_restorer import TemporalBlock

BASE_CONFIG = {
    "arm": "complex",
    "width": 96,
    "residual_slope": 1 / 64,
    "generation_budget": True,
    "budget_scope": "residual",
}
# Half-open short-FFT slices; the final band includes Nyquist. Context slices
# use the same normalized frequencies, with their own Nyquist endpoint.
BAND_EDGES = (0, 8, 16, 24, 40, 56, 80, 104, 129)


def band_slices(bins):
    if bins not in (129, 513, 1025):
        raise ValueError("expected one of the registered FFT geometries")
    factor = (bins - 1) // 128
    edges = [edge * factor for edge in BAND_EDGES[:-1]] + [bins]
    return tuple(pairwise(edges))


class BandBackbone(nn.Module):
    def __init__(self, decoder, width=96, token_width=24):
        super().__init__()
        if type(token_width) is not int or token_width < 4:
            raise ValueError("token width must be an integer >= 4")
        self.width, self.token_width = width, token_width
        self.bands = len(BAND_EDGES) - 1
        self.slices = tuple(band_slices(n) for n in (129, 513, 1025))
        self.dilations = (1, 2, 4, 8, 16, 32)
        self.state_size = self.bands * token_width * sum(self.dilations)
        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(sum(s[i][1] - s[i][0] for s in self.slices) + 4, token_width),
                nn.LayerNorm(token_width),
                nn.GELU(),
            )
            for i in range(self.bands)
        )
        # Mix only the current frame's band axis; temporal history remains explicit.
        self.mix_norms = nn.ModuleList(nn.LayerNorm(token_width) for _ in self.dilations)
        self.mixers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.bands, 2 * self.bands),
                nn.GELU(),
                nn.Linear(2 * self.bands, self.bands),
            )
            for _ in self.dilations
        )
        self.temporal = nn.ModuleList(TemporalBlock(token_width, d) for d in self.dilations)
        # Shared readout keeps the existing 192-dimensional head interface. This
        # prototype retains local state, but does not yet use a band-local decoder.
        self.readout = nn.Linear(self.bands * token_width, width)
        self.decoder = decoder

    def initial_state(self, batch, *, device=None):
        return self.readout.weight.new_zeros(batch, self.state_size, device=device)

    def encode(self, magnitude, context, condition, state):
        scale = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
        observations = (
            torch.log1p(magnitude / scale),
            torch.log1p(context[..., :513] / scale),
            torch.log1p(context[..., 513:] / scale),
        )
        tokens = []
        for index, encoder in enumerate(self.encoders):
            local = [
                values[..., slices[index][0] : slices[index][1]]
                for values, slices in zip(observations, self.slices, strict=True)
            ]
            tokens.append(encoder(torch.cat((*local, scale.log10() / 10, condition), -1)))
        encoded = torch.stack(tokens, dim=2)  # [physical channel, time, band, feature]
        temporal, states, offset = encoded, [], 0
        batch, frames = magnitude.shape[:2]
        for norm, mixer, block, dilation in zip(
            self.mix_norms, self.mixers, self.temporal, self.dilations, strict=True
        ):
            temporal = temporal + 0.25 * mixer(norm(temporal).transpose(-1, -2)).transpose(-1, -2)
            folded = temporal.permute(0, 2, 1, 3).reshape(-1, frames, self.token_width)
            count = self.bands * dilation * self.token_width
            history = state[:, offset : offset + count].reshape(-1, dilation, self.token_width)
            folded, updated = block(folded, history)
            temporal = folded.reshape(batch, self.bands, frames, self.token_width).permute(
                0, 2, 1, 3
            )
            states.append(updated.reshape(batch, count))
            offset += count
        current = self.readout(encoded.flatten(2))
        past = self.readout(temporal.flatten(2))
        return torch.cat((current, past), -1), scale, torch.cat(states, 1)


class BandTCN(TrainableAdapter):
    """Experimental four-input graph; production artifact contract is unchanged."""

    def __init__(self, token_width=24):
        super().__init__(BASE_CONFIG, "high", "full")
        self.backbone = BandBackbone(self.backbone.decoder, self.width, token_width)
        self.state_size = self.backbone.state_size
        self.token_width = token_width


def make_prototype(kind, *, seed=62, nonzero=False):
    """Fresh control/candidate, never silently load historical pretraining.

    Nonzero deterministic heads exercise deployment paths before training; this
    stress fixture is not a restored-audio checkpoint or an optimization result.
    """
    if kind not in {"global", "band"}:
        raise ValueError("unknown prototype kind")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BandTCN() if kind == "band" else TrainableAdapter(BASE_CONFIG, "high", "full")
        if nonzero:
            torch.manual_seed(seed + 1)
            with torch.no_grad():
                for layer in (model.backbone.decoder[-1], model.adapter, model.budget_head):
                    layer.weight.normal_(0, 0.03)
    return model.eval()
