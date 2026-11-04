"""P71 same-depth band capacity fixtures, with explicit full training scope.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.band_phase import BandPhaseCorrection
from models.band_tcn import BandTCN

TOKEN_WIDTHS = {"small": 24, "double": 67}


def capacity_model(kind, seed=71, *, nonzero=False):
    if kind not in TOKEN_WIDTHS:
        raise ValueError("unknown registered capacity")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BandPhaseCorrection(
            BandTCN(TOKEN_WIDTHS[kind]),
            seed=seed + 1,
            phase_support="native",
            phase_observation="adjacent",
            train_base=True,
        ).eval()
        model.study = "p71-joint-band-capacity-feasibility"
        if nonzero:
            torch.manual_seed(seed + 2)
            with torch.no_grad():
                for layer in (
                    model.base.backbone.decoder[-1],
                    model.base.adapter,
                    model.base.budget_head,
                    model.phase_head[-1],
                ):
                    layer.weight.normal_(0, 0.03)
    return model
