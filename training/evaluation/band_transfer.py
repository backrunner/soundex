"""Frozen frequency intervention and linear OLA residual accounting (no DSP changes)."""

import torch
from torch.nn import functional as F

from models.envelope_objective import BANDS
from models.losses import blend_deployment_features


def complex_features(features):
    magnitude = torch.pow(10.0, features[:, 0].clamp(-200, 100) / 20)
    return torch.complex(magnitude * features[:, 1].cos(), magnitude * features[:, 1].sin())


def synthesize(spectrum):
    window = torch.hann_window(256, dtype=spectrum.real.dtype, device=spectrum.device)
    frames = torch.fft.irfft(spectrum, n=256) * window
    count = spectrum.shape[1]
    length = (count + 1) * 128
    wave = F.fold(frames.transpose(1, 2), (1, length), (1, 256), stride=(1, 128))[:, 0, 0]
    weights = F.fold(
        window.square()[None, :, None].expand(1, -1, count), (1, length), (1, 256), stride=(1, 128)
    )[0, 0, 0]
    return torch.where(weights > 1e-12, wave / weights.clamp_min(1e-12), 0)[:, : count * 128]


def analyze(wave):
    window = torch.hann_window(256, dtype=wave.dtype, device=wave.device)
    return torch.fft.rfft(wave[:, :-128].unfold(-1, 256, 128) * window)


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
