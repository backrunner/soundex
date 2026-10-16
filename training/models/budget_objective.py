"""P24 input-only generation budget and globally stabilized reconstruction risk.

References enter supervision only. Historical relative metrics remain unchanged.
No per-recording thresholds, clipping of errors, or inaudibility claims.
"""

import torch
from torch.nn import functional as F

from models.complex_synthesis import complex_features
from models.envelope_objective import BANDS
from models.fit_diagnostic import BURN
from models.rendered_objective import reanalyze, waveform_norm
from models.spectral_consistency import reference_features


def balanced_spectrum(prediction, target, rate, burn=89):
    p, t = prediction[:, burn:], target[:, burn:]
    if p.shape != t.shape or p.shape[1] < 2:
        raise ValueError("aligned spectra with two supervised frames required")
    global_power = t.square().mean((1, 2))
    hz = torch.arange(129, device=p.device)[None, :] * rate / 256
    total = p.new_zeros(len(p))
    for low, high in BANDS:
        mask = ((hz >= low) & (hz < high))[:, None, :]
        count = mask.sum((1, 2)).clamp_min(1)
        ref_power = (t.square() * mask).sum((1, 2)) / (count * t.shape[1])
        # One full-spectrum reference power; no fitted dB threshold. A weak
        # band's score stays tied to the signal, while every HF error is retained.
        denominator = ref_power + global_power + 1e-14
        error = p - t
        power = (error.square() * mask).sum((1, 2)) / (count * t.shape[1])
        dynamics = (error.diff(dim=1).square() * mask).sum((1, 2)) / (count * (t.shape[1] - 1))
        weight = (0.25 if low == 0 else 1.0) / 4.25
        total = total + weight * (
            (power / denominator + 1e-6).sqrt()
            - 1e-3
            + 0.25 * ((dynamics / denominator + 1e-6).sqrt() - 1e-3)
        )
    return total


def balanced_risk(objective, prediction):
    output = objective.render(prediction)
    amplitude = reanalyze(output)
    b, region = objective.batch, objective.region
    target = torch.where(
        b["identity"][:, None, None].bool(),
        objective.baseline_amplitude,
        objective.target[:, : amplitude.shape[1]],
    )
    reference = torch.where(b["identity"][:, None].bool(), objective.baseline, b["clean_wave"])
    spectral = balanced_spectrum(amplitude, target, b["rate"])
    wave = waveform_norm(output[:, region], reference[:, region])
    # Full-scale PCM error supplements signal-relative fidelity. Smooth at zero.
    absolute = ((output[:, region] - reference[:, region]).square().mean(1) + 1e-14).sqrt() - 1e-7
    per_example = spectral + 0.25 * wave + 0.25 * absolute
    identity = b["identity"].bool()
    if not identity.any() or identity.all():
        raise ValueError("both paired and identity examples required")
    return 0.5 * (per_example[~identity].mean() + per_example[identity].mean())


def budget_target(batch):
    """Per-frame clean complex residual energy / observed global RMS, training only."""
    with torch.no_grad():
        original = complex_features(batch["x"])[:, BURN:]
        target = complex_features(reference_features(batch, phase_reference="clean"))[:, BURN:]
        scale = original.abs().square().mean(-1).clamp_min(1e-20)
        hz = torch.arange(129, device=original.device)[None, :] * batch["rate"] / 256
        values = []
        for low, high in BANDS:
            mask = ((hz >= low) & (hz < high))[:, None, :]
            power = ((target - original).abs().square() * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            values.append((power / scale).sqrt())
        return torch.stack(values, dim=-1)


def budget_risk(budget, batch):
    # The first nine post-warmup frames are also excluded from supervision.
    error = F.smooth_l1_loss(
        budget[:, 9:].log1p(), budget_target(batch)[:, 9:].log1p(), reduction="none"
    ).mean((1, 2))
    identity = batch["identity"].bool()
    return 0.5 * (error[~identity].mean() + error[identity].mean())
