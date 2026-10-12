"""Diagnostic P8 intervention: input-relative energy update, unchanged capacity."""

import math

from torch import nn

from models.factorized_envelope import RELATIVE_FLOOR
from models.residual_energy import ResidualEnergyEnvelope


class ConditionedEnergyEnvelope(ResidualEnergyEnvelope):
    def band_energy(self, input_rms, scale, delta):
        ratio = input_rms / scale
        # Input-only units: weak bands no longer receive a full-spectrum-sized
        # update. Existing 1e-4 floor permits generation from a missing band.
        # No clean reference, source ID or validation-dependent threshold.
        units = (ratio.square() + RELATIVE_FLOOR**2).sqrt()
        signed = ratio + units * delta
        return scale * (signed.square() + 1e-16).sqrt().clamp(max=math.exp(2))


def make_conditioned_model(width=144):
    model = ConditionedEnergyEnvelope(width)
    nn.init.zeros_(model.decoder[-1].weight)
    nn.init.zeros_(model.decoder[-1].bias)
    return model
