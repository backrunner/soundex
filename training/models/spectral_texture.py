"""Experimental training-only conditional spectral texture features.

Original SoundEx implementation. No external model, weights or code is imported.
Not connected to the production generator, optimizer, exporter or DSP.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def spectrum_pair(wave, observed, rates, fft_size):
    """Input-scaled PCM spectra on supervised support only; no reference scale.

    Noncentered analysis windows cannot cross the supplied supervised boundary.
    The critic is training-only: its symmetric convolutions are never deployed.
    Physical channels are separate examples; this is not a stereo objective.
    """
    if wave.ndim != 2 or wave.shape != observed.shape or rates.numel() != len(wave):
        raise ValueError("aligned mono views and one physical rate per example required")
    if fft_size not in (512, 1024) or wave.shape[-1] < fft_size + fft_size // 4:
        raise ValueError("two complete windows at a supported resolution required")
    if not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("unsupported physical rate")
    observed = observed.detach()
    window = torch.hann_window(fft_size, device=wave.device, dtype=wave.dtype)
    sides = wave.new_full((fft_size // 2 + 1,), 2)
    sides[0] = sides[-1] = 1
    scale = (sides / (fft_size * window.square().sum())).sqrt()
    unit = (observed.square().mean(-1, keepdim=True) / len(sides) + 1e-10).sqrt()

    def image(value):
        magnitude = (
            torch.fft.rfft(value.unfold(-1, fft_size, fft_size // 4) * window) * scale
        ).abs()
        return torch.log1p(magnitude / unit[:, :, None]).transpose(1, 2)

    candidate, condition = image(wave), image(observed)
    hz = (
        torch.arange(len(sides), device=wave.device, dtype=wave.dtype)[None, :]
        * rates.reshape(-1, 1)
        / fft_size
    )
    mask = (hz >= 8000).to(wave.dtype)[:, :, None].expand_as(candidate)
    frequency = (hz / 24000)[:, :, None].expand_as(candidate)
    return torch.stack((condition * mask, candidate * mask, frequency), dim=1), mask[:, None]


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


def masked_mean(value, mask):
    return (value * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * value.shape[1])


class SpectralTextureCritic(nn.Module):
    """Small conditional patch critic, without batch statistics or dropout."""

    def __init__(self, fft_sizes=(512, 1024)):
        super().__init__()
        if not fft_sizes or any(n not in (512, 1024) for n in fft_sizes):
            raise ValueError("unsupported texture resolutions")
        self.fft_sizes = tuple(fft_sizes)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(3, 8, 3, padding=1),
                        nn.Conv2d(8, 16, 3, padding=1),
                        nn.Conv2d(16, 1, 1),
                    ]
                )
                for _ in fft_sizes
            ]
        )

    def features(self, wave, observed, rates):
        outputs = []
        for size, layers in zip(self.fft_sizes, self.blocks, strict=True):
            value, mask = spectrum_pair(wave, observed, rates, size)
            maps = []
            for layer in layers[:-1]:
                value = F.softplus(layer(value))
                maps.append(value)
            logits = layers[-1](value)
            outputs.append((maps, logits, mask))
        return outputs

    def discriminator_loss(self, predicted, reference, observed, rates):
        """Detached audio; caller must fit on paired TRAIN examples only."""
        real = self.features(reference.detach(), observed, rates)
        fake = self.features(predicted.detach(), observed, rates)
        return torch.stack(
            [
                masked_mean(F.softplus(-r[1]) + F.softplus(f[1]), r[2])
                for r, f in zip(real, fake, strict=True)
            ]
        ).mean(0)

    def reference_loss(self, predicted, reference, observed, rates):
        """Paired learned-feature error, zero/stationary at the clean reference.

        Deliberately does not reward a discriminator realism score: that score
        generally has a nonzero generator gradient even at the exact reference.
        Reference and critic are frozen; gradients flow only to predicted PCM.
        Keep original direct/MR/complex/identity losses in any later experiment.
        """
        with frozen(self):
            with torch.no_grad():
                real = self.features(reference.detach(), observed, rates)
            fake = self.features(predicted, observed, rates)
            losses = [
                masked_mean((a - b).square(), r[2])
                for r, f in zip(real, fake, strict=True)
                for a, b in zip(f[0], r[0], strict=True)
            ]
        return torch.stack(losses).mean(0)
