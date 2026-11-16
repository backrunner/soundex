"""P83 fixed circular relation arms, exposure order and protocol-v2 scoring.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from evaluation.restoration_evidence import extended_panel
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import balanced_risk, fused_spectrum_loss, phase_objective
from models.multires_spectrum import multires_spectrum_loss
from models.realized_phase_relations import auxiliary as relation_auxiliary
from models.realized_phase_relations import domain_spectra, prepare_relations, relations
from scripts.audit_phase_routing import scores

ARMS = ("old", "direct", "raw", "fused")
STEPS = 16


def auxiliary(prediction, batch, arm):
    if arm not in ARMS[1:]:
        raise ValueError("explicit registered relation arm required")
    return relation_auxiliary(prediction, batch, arm + "/circular")


def draws_for_step(fit, step):
    if len(fit) != 8 or len(set(fit)) != 8 or not 0 <= step < STEPS:
        raise ValueError("eight fixed fit draws and step in the registered16-step budget required")
    start = (step % 2) * 4
    return fit[start : start + 4]


@torch.no_grad()
def scores_v2(wave, reference, rates):
    historical = scores(wave, reference, rates)
    for row, a, b, rate in zip(
        historical, wave.double(), reference.double(), rates.reshape(-1), strict=True
    ):
        extended = extended_panel(a[:, None].numpy(), b[:, None].numpy(), int(rate))["metrics"]
        for key in row.keys() & extended.keys():
            if row[key] != extended[key]:
                raise ValueError("extended metrics changed a historical score")
        row.update(extended)
    return historical


@torch.no_grad()
def fit_scores(prediction, batch):
    """Same FP64 ruler on lifted actual FP32 predictions for every frozen arm."""
    prediction = prediction.double()
    batch = {
        k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in batch.items()
    }
    plan = prepare_relations(batch)
    spectra = domain_spectra(prediction, batch, plan)
    values = {}
    for domain in ARMS[1:]:
        result = relations(spectra[domain], plan, "circular")
        values[domain + "/risk"] = result["per_example"]
        values[domain + "/weighted_auxiliary"] = 0.1 * result["per_example"].log1p()
        values.update({domain + "/" + k: v for k, v in result["components"].items()})
    values["common_risk"] = sum(
        loss(prediction, batch)["per_example"].log1p()
        for loss in (direct_spectrum_loss, multires_spectrum_loss, fused_spectrum_loss)
    )
    torch.testing.assert_close(
        balanced_risk(values["common_risk"], batch["identity"]),
        phase_objective(prediction, batch, include_fused=True),
        rtol=1e-12,
        atol=1e-14,
    )
    return [{k: float(v[i]) for k, v in values.items()} for i in range(len(prediction))]
