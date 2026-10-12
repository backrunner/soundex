"""P7: change only the band-energy coordinates, not features, capacity or loss."""

import math

from torch import nn

from models.factorized_envelope import FactorizedEnvelope, make_factor_model

ARMS = ("factorized", "rms_residual")


class ResidualEnergyEnvelope(FactorizedEnvelope):
    def band_energy(self, input_rms, scale, delta):
        # Add energy in units of the observed full-spectrum RMS. A missing
        # band's near-zero input RMS no longer multiplies the learned update.
        # Smooth absolute value permits suppression and generation; the same
        # upper energy bound as P6 prevents changing the output amplitude budget.
        signed = input_rms / scale + delta
        return scale * (signed.square() + 1e-16).sqrt().clamp(max=math.exp(2))


def make_energy_model(arm, width=144):
    if arm == "factorized":
        return make_factor_model(arm, width)
    if arm != "rms_residual":
        raise ValueError("unknown energy study arm")
    model = ResidualEnergyEnvelope(width)
    nn.init.zeros_(model.decoder[-1].weight)
    nn.init.zeros_(model.decoder[-1].bias)
    return model
