"""P34 training-only smooth risk compression and explicit identity weighting.

The original direct/MR losses and evaluation metrics are unchanged. log1p is a
monotone concave transform of each dimensionless per-example objective, with
derivative 1/(1+risk). It does not discard examples, change targets, or imply an
audibility threshold. It reduces gradients on large errors, including difficult
restoration examples; this tradeoff must be evaluated with uncompressed metrics.
"""

from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss


def reconstruction_risk(direct, multires, identity, *, compress=False, identity_weight=1.0):
    if direct.ndim != 1 or direct.shape != multires.shape or identity.shape != direct.shape:
        raise ValueError("aligned per-example risks and roles required")
    identity = identity.bool()
    if not identity.any() or identity.all():
        raise ValueError("paired and identity roles required")
    if not 0 < identity_weight <= 1:
        raise ValueError("identity weight must be in (0, 1]")
    values = direct.log1p() + multires.log1p() if compress else direct + multires
    # Do not renormalize by 1+weight: the paired coefficient remains exactly .5.
    return 0.5 * (values[~identity].mean() + identity_weight * values[identity].mean())


def training_terms(prediction, budget, objective, weight):
    del weight  # Legacy runner argument; no deployed/consistency objective here.
    batch = objective.batch
    direct = direct_spectrum_loss(prediction, batch)
    multires = multires_spectrum_loss(prediction, batch)
    args = (direct["per_example"], multires["per_example"], batch["identity"])
    auxiliary = 0.1 * budget_risk(budget, batch)
    return {
        "control": reconstruction_risk(*args) + auxiliary,
        "compressed": reconstruction_risk(*args, compress=True) + auxiliary,
        "compressed_identity": reconstruction_risk(*args, compress=True, identity_weight=0.1)
        + auxiliary,
        "raw": direct["total"],
        "multires": multires["total"],
        "budget": auxiliary,
    }
