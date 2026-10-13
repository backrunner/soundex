"""P11 fixed-example fitting diagnostic, never a deployable model or quality gate."""

import copy

import torch
from torch import nn

from models.causal_objective import render
from models.linear_objective import linear_objective
from models.rendered_objective import amplitude_features, reanalyze, waveform_norm

BURN = 80


class FreeSpectrum(nn.Module):
    """Independent per-example coefficients: target-fitted diagnostic, not inference."""

    def __init__(self, original, mode):
        super().__init__()
        if mode not in ("amplitude", "complex"):
            raise ValueError("unknown spectrum coordinates")
        self.mode = mode
        self.register_buffer("original", original.detach().clone())
        magnitude = torch.pow(10.0, original[:, 0] / 20)
        self.register_buffer("magnitude", magnitude)
        self.register_buffer(
            "scale", magnitude.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-10)
        )
        self.register_buffer("base_features", amplitude_features(magnitude, original))
        shape = magnitude[:, BURN:].shape
        self.delta = nn.Parameter(original.new_zeros((*shape, 2) if mode == "complex" else shape))

    @staticmethod
    def polar(real, imag):
        # Avoid atan2(0,0), including its undefined backward branch.
        tiny = (real == 0) & (imag == 0)
        real = torch.where(tiny, real.new_full((), 1e-10), real)
        imag = torch.where(tiny, torch.zeros_like(imag), imag)
        return (real.square() + imag.square()).sqrt().clamp_min(1e-10), torch.atan2(imag, real)

    def forward(self):
        magnitude, phase = self.magnitude[:, BURN:], self.original[:, 1, BURN:]
        scale = self.scale[:, BURN:]
        if self.mode == "amplitude":
            amplitude = (magnitude + scale * self.delta).clamp_min(1e-10)
            zero_amplitude = magnitude
        else:
            real, imag = magnitude * phase.cos(), magnitude * phase.sin()
            zero_amplitude, zero_phase = self.polar(real, imag)
            amplitude, angle = self.polar(
                real + scale * self.delta[..., 0], imag + scale * self.delta[..., 1]
            )
            difference = angle - zero_phase
            phase = phase + torch.atan2(difference.sin(), difference.cos())
        # Remove zero-coordinate round-trip error. Tiny float noise must not
        # launch Adam updates on independent identity coefficients.
        db = self.base_features[:, 0, BURN:] + 20 * (amplitude.log10() - zero_amplitude.log10())
        tail = torch.stack((db, phase), dim=1)
        return torch.cat((self.original[:, :, :BURN], tail), dim=2)


class FixedFitObjective:
    """P10 per-example losses, equal paired/identity risk and equal clip weight.

    All reference caches are constant. No target-derived value enters predictions.
    Corpus weighting differs from the full-library P10 score; totals are not comparable.
    """

    def __init__(self, batch):
        self.batch = batch
        self.target = torch.pow(10.0, batch["y"][:, 0] / 20)
        with torch.no_grad():
            mag = torch.pow(10.0, batch["x"][:, 0] / 20)
            self.baseline = self.render(amplitude_features(mag, batch["x"]))
            self.baseline_amplitude = reanalyze(self.baseline)
        self.region = slice((BURN + 9) * 128, self.baseline.shape[-1] - 128)

    def select(self, indices):
        """Slice immutable reference caches for a paired/identity mini-batch."""
        selected = copy.copy(self)
        selected.batch = {key: value[indices] for key, value in self.batch.items()}
        for key in ("target", "baseline", "baseline_amplitude"):
            setattr(selected, key, getattr(self, key)[indices])
        return selected

    def render(self, features):
        b = self.batch
        return render(
            features,
            b["x"],
            b["mask"],
            b["active"],
            b["gate"],
            b["dry"],
            limiter_mode="smooth-knee",
        )[0]

    def __call__(self, features):
        output = self.render(features)
        amplitude = reanalyze(output)
        reconstruction, preservation = [], []
        for i in range(len(output)):
            rate = self.batch["rate"][i : i + 1]
            reconstruction.append(
                linear_objective(
                    amplitude[i : i + 1],
                    self.target[i : i + 1, : amplitude.shape[1]],
                    rate,
                    burn=BURN + 9,
                )["total"]
            )
            preservation.append(
                linear_objective(
                    amplitude[i : i + 1], self.baseline_amplitude[i : i + 1], rate, burn=BURN + 9
                )["total"]
            )
        reconstruction, preservation = torch.stack(reconstruction), torch.stack(preservation)
        wave = waveform_norm(output[:, self.region], self.batch["clean_wave"][:, self.region])
        extra_wave = waveform_norm(output[:, self.region], self.baseline[:, self.region])
        identity = self.batch["identity"].bool()
        restoration = reconstruction + 0.25 * wave
        extra = preservation + 0.25 * extra_wave
        if not identity.any() or identity.all():
            raise ValueError("both paired and identity examples required")
        return {
            "total": 0.5 * (restoration[~identity].mean() + extra[identity].mean()),
            "restoration": restoration,
            "preservation": extra,
            "spectral": reconstruction,
            "waveform": wave,
        }
