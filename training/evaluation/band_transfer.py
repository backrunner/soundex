"""Frozen frequency intervention and linear OLA residual accounting (no DSP changes)."""

import torch

from models.complex_synthesis import analyze, complex_features, synthesize
from models.envelope_objective import BANDS
from models.losses import blend_deployment_features


def band_masks(rate, *, dtype, device):
    frequency = torch.arange(129, device=device)[None, :] * rate.reshape(-1, 1) / 256
    return torch.stack(
        [((frequency >= low) & (frequency < high)).to(dtype) for low, high in BANDS], dim=1
    )


def intervene(prediction, initial, masks, source, *, remove=False):
    mask = masks[:, source, None, :].bool()
    if remove:
        mask = ~mask
    return torch.where(mask[:, None], prediction, initial)


def transfer(prediction, initial, batch, burn=89):
    """[batch, source band, destination band] RMS of projected complex residual.

    Individual powers do not sum to total power because components interfere.
    No limiter/gate in this linear accounting; nonlinear interventions are scored separately.
    """
    masks = band_masks(batch["rate"], dtype=prediction.dtype, device=prediction.device)
    effective = batch["mask"] * batch["active"][:, None, :, None]
    blended = blend_deployment_features(batch["x"], prediction, effective)
    baseline = blend_deployment_features(batch["x"], initial, effective)
    delta = complex_features(blended) - complex_features(baseline)
    wave = synthesize(delta)
    projected = analyze(wave)
    parts, waves = [], []
    for source in range(len(BANDS)):
        source_wave = synthesize(delta * masks[:, source, None, :])
        waves.append(source_wave)
        parts.append(analyze(source_wave))
    waves = torch.stack(waves, 1)
    parts = torch.stack(parts, 1)
    counts = masks.sum(-1).clamp_min(1)
    matrix = (
        (parts[:, :, burn:].abs().square()[:, :, None] * masks[:, None, :, None, :]).sum((-1, -2))
        / (counts[:, None] * parts[:, :, burn:].shape[2])
    ).sqrt()
    full = (
        (projected[:, burn:].abs().square()[:, None] * masks[:, :, None, :]).sum((-1, -2))
        / (counts * projected[:, burn:].shape[1])
    ).sqrt()
    residual = projected - delta[:, : projected.shape[1]]
    consistency = (
        (residual[:, burn:].abs().square()[:, None] * masks[:, :, None, :]).sum((-1, -2))
        / (counts * residual[:, burn:].shape[1])
    ).sqrt()
    return {
        "masks": masks,
        "matrix": matrix,
        "full_residual_rms": full,
        "inconsistency_rms": consistency,
        "wave_superposition_max_error": (waves.sum(1) - wave).abs().amax(1),
        "spectrum_superposition_max_error": (parts.sum(1) - projected).abs().amax((1, 2)),
    }
