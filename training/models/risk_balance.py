"""P34 training-only smooth risk compression and explicit identity weighting.

The original direct/MR losses and evaluation metrics are unchanged. log1p is a
monotone concave transform of each dimensionless per-example objective, with
derivative 1/(1+risk). It does not discard examples, change targets, or imply an
audibility threshold. It reduces gradients on large errors, including difficult
restoration examples; this tradeoff must be evaluated with uncompressed metrics.
"""

import torch

from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss, supervised_waves


def deficit_overshoot_risk(wave, original, reference, rates):
    """One-sided MR guard on bins deficient in the original input.

    Below-reference predictions get no auxiliary gradient. The original losses
    still supervise missing energy, excess-input bins, silence and dynamics.
    Reference/input masks are detached, with no candidate-dependent thresholds.
    """
    if wave.ndim != 2 or wave.shape != original.shape or wave.shape != reference.shape:
        raise ValueError("aligned PCM required")
    reference, original = reference.detach(), original.detach()
    values = wave.new_zeros(len(wave))
    for size in (256, 1024, 2048):
        window = torch.hann_window(size, dtype=wave.dtype, device=wave.device)
        sides = wave.new_full((size // 2 + 1,), 2)
        sides[0] = sides[-1] = 1
        scale = (sides / (size * window.square().sum())).sqrt()

        def amplitude(x, size=size, window=window, scale=scale):
            return (torch.fft.rfft(x.unfold(-1, size, size // 4) * window) * scale).abs()

        predicted = amplitude(wave)
        with torch.no_grad():
            target, observed = amplitude(reference), amplitude(original)
            deficit = target > observed
            floor = target.square().sum(-1).mean(-1) * 1e-4 + 1e-12
        positive_error = (predicted - target).clamp_min(0).square() * deficit
        hz = torch.arange(size // 2 + 1, device=wave.device)[None] * rates.reshape(-1, 1) / size
        for low, high in ((4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001)):
            mask = ((hz >= low) & (hz < high))[:, None]
            denominator = (target.square() * mask).sum(-1).mean(-1) + floor
            power = (positive_error * mask).sum(-1).mean(-1)
            values = values + ((power / denominator + 1e-6).sqrt() - 1e-3) / 12
    return values


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
    wave, target = supervised_waves(prediction, batch)
    original = batch["dry"][:, objective.region]
    overshoot = (
        0.05
        * deficit_overshoot_risk(wave, original, target, batch["rate"])[
            ~batch["identity"].bool()
        ].mean()
    )
    return {
        "control": reconstruction_risk(*args) + auxiliary,
        "compressed": reconstruction_risk(*args, compress=True) + auxiliary,
        "compressed_identity": reconstruction_risk(*args, compress=True, identity_weight=0.1)
        + auxiliary,
        "compressed_identity_mid": reconstruction_risk(*args, compress=True, identity_weight=0.3)
        + auxiliary,
        "raw": direct["total"],
        "compressed_identity_guard": reconstruction_risk(*args, compress=True, identity_weight=0.3)
        + auxiliary
        + overshoot,
        "overshoot": overshoot,
        "multires": multires["total"],
        "budget": auxiliary,
    }
