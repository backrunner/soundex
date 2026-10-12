"""P9 band-relative shape updates with the P8 input-relative energy head."""

import torch
from torch import nn

from models.conditioned_energy import ConditionedEnergyEnvelope, make_conditioned_model
from models.envelope_objective import BANDS
from models.factorized_envelope import RELATIVE_FLOOR

ARMS = ("energy_conditioned", "band_conditioned", "anchored_band")


class BandConditionedEnvelope(ConditionedEnergyEnvelope):
    anchor_initial = False

    def forward(self, magnitude, context, condition, state):
        latent, scale, next_state = self.encode(magnitude, context, condition, state)
        delta = 2 * torch.tanh(self.decoder(latent))
        energy_delta = 4 * self.energy_head(latent)
        frequency = self.bin_index[None, None, :] * condition[:, :1, :1] * (48000 / 256)
        prediction = torch.zeros_like(magnitude)
        for index, (low, high) in enumerate(BANDS):
            mask = ((frequency >= low) & (frequency < high)).to(magnitude.dtype)
            count = mask.sum(-1, keepdim=True).clamp_min(1)
            input_rms = (
                (magnitude.square() * mask).sum(-1, keepdim=True) / count + (scale * 1e-8).square()
            ).sqrt()
            # One input-conditioned unit per band, not per bin: empty bins can
            # acquire structure. Keep the old positive seed and output RMS bound.
            units = (input_rms.square() + (scale * RELATIVE_FLOOR).square()).sqrt()
            seed = (
                (magnitude + units * delta + scale * 1e-8).square() + (scale * 1e-8).square()
            ).sqrt()
            seed_rms = (
                ((seed.square() * mask).sum(-1, keepdim=True) / count).clamp_min(1e-36).sqrt()
            )
            energy = self.band_energy(input_rms, scale, energy_delta[..., index : index + 1])
            restored = seed / seed_rms * energy
            if self.anchor_initial:
                # Subtract the same zero-head mapping before adding the input.
                # Exact identity at initialization, without eliminating the
                # positive seed or its missing-bin generation gradients.
                zero_seed = ((magnitude + scale * 1e-8).square() + (scale * 1e-8).square()).sqrt()
                zero_rms = (
                    ((zero_seed.square() * mask).sum(-1, keepdim=True) / count).clamp_min(1e-36)
                ).sqrt()
                zero_energy = self.band_energy(input_rms, scale, torch.zeros_like(input_rms))
                restored = (magnitude + (restored - zero_seed / zero_rms * zero_energy)).clamp_min(
                    0
                )
            prediction = prediction + mask * restored
        return prediction, next_state


class AnchoredBandEnvelope(BandConditionedEnvelope):
    anchor_initial = True


def make_shape_model(arm, width=144):
    if arm == "energy_conditioned":
        return make_conditioned_model(width)
    if arm not in ("band_conditioned", "anchored_band"):
        raise ValueError("unknown shape study arm")
    model = (AnchoredBandEnvelope if arm == "anchored_band" else BandConditionedEnvelope)(width)
    nn.init.zeros_(model.decoder[-1].weight)
    nn.init.zeros_(model.decoder[-1].bias)
    return model
