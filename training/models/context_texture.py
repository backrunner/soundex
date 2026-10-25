"""P60 training-only observed-band context and structured critic negatives."""

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.texture_calibration import controls
from models.spectral_texture import SpectralTextureCritic, masked_mean, spectrum_pair


def observed_context(observed, rates, size):
    """Eight physical 3 kHz log-magnitude summaries, preserving frame time.

    This deliberately compresses frequency detail; it is not a full-resolution
    conditioning path. Only detached observed PCM determines values and scale.
    """
    observed = observed.detach()
    window = torch.hann_window(size, device=observed.device, dtype=observed.dtype)
    sides = observed.new_full((size // 2 + 1,), 2)
    sides[0] = sides[-1] = 1
    magnitude = torch.fft.rfft(observed.unfold(-1, size, size // 4) * window).abs()
    magnitude = magnitude * (sides / (size * window.square().sum())).sqrt()
    unit = (observed.square().mean(-1, keepdim=True) / len(sides) + 1e-10).sqrt()
    value = torch.log1p(magnitude / unit[:, :, None])
    hz = torch.arange(len(sides), device=observed.device)[None] * rates[:, None] / size
    bands = []
    for low in range(0, 24000, 3000):
        mask = ((hz >= low) & (hz < low + 3000)).to(value.dtype)
        bands.append((value * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1))
    return torch.stack(bands, dim=1)[:, :, None]


class ContextTextureCritic(SpectralTextureCritic):
    """Matched allocated capacity, with initially neutral optional context."""

    def __init__(self, context=False):
        super().__init__()
        self.context = context
        for layers in self.blocks:
            old = layers[0]
            expanded = nn.Conv2d(11, 8, 3, padding=1)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :3].copy_(old.weight)
                expanded.bias.copy_(old.bias)
            layers[0] = expanded

    def features(self, wave, observed, rates):
        outputs = []
        for size, layers in zip(self.fft_sizes, self.blocks, strict=True):
            value, mask = spectrum_pair(wave, observed, rates, size)
            context = observed_context(observed, rates.reshape(-1), size)
            context = context.expand(-1, -1, value.shape[2], -1)
            if not self.context:
                context = torch.zeros_like(context)
            value = torch.cat((value, context), dim=1)
            maps = []
            for layer in layers[:-1]:
                value = F.softplus(layer(value))
                maps.append(value)
            outputs.append((maps, layers[-1](value), mask))
        return outputs


def shifted_high(reference, rates, fraction):
    """Circular diagnostic HF shift with unchanged DC/Nyquist and low band."""
    length = reference.shape[-1]
    spectrum = torch.fft.rfft(reference.double())
    index = torch.arange(spectrum.shape[-1], device=reference.device, dtype=torch.float64)
    hz = index[None] * rates.reshape(-1, 1) / length
    mask = hz >= 8000
    mask[:, 0] = False
    if length % 2 == 0:
        mask[:, -1] = False
    phase = torch.exp(-2j * torch.pi * index * int(length * fraction) / length)
    return torch.fft.irfft(torch.where(mask, spectrum * phase, spectrum), n=length).to(
        reference.dtype
    )


def structured_loss(critic, predicted, reference, observed, rates, *, structured, seed, step):
    """One real and unit total fake weight; three fake forwards for every arm.

    Structured fake weights: prediction 1/2, phase 1/4, time 1/4. A transform
    indistinguishable from a numerical reference roundtrip is omitted per example
    and weights renormalized. No track-specific or audibility threshold is used.
    """
    predicted, reference, observed = (v.detach() for v in (predicted, reference, observed))
    variations = controls(predicted, reference, observed, rates, seed)
    shifted = shifted_high(reference, rates, (1 / 8, 1 / 6, 1 / 3)[step % 3])
    candidates = (predicted, variations["phase_scrambled"], shifted)
    weights = reference.new_tensor((0.5, 0.25, 0.25)).expand(len(reference), -1).clone()
    rounding = (variations["roundtrip"].double() - reference.double()).square().mean(-1)
    for i in (1, 2):
        difference = (candidates[i].double() - reference.double()).square().mean(-1)
        weights[:, i] *= difference > rounding
    weights /= weights.sum(-1, keepdim=True)
    if not structured:
        candidates = (predicted, predicted, predicted)
    real = critic.features(reference, observed, rates)
    real_loss = torch.stack([masked_mean(F.softplus(-r[1]), r[2]) for r in real]).mean(0)
    fake_losses = []
    for candidate in candidates:
        fake = critic.features(candidate, observed, rates)
        fake_losses.append(torch.stack([masked_mean(F.softplus(f[1]), f[2]) for f in fake]).mean(0))
    return real_loss + (torch.stack(fake_losses, dim=1) * weights).sum(-1)
