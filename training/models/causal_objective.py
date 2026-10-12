"""P2 objective v1: raw inpainting plus actual Model-gain synthesis/gate/limiter.

All arms use the same weights. Input detector decisions are collected from Rust.
Waveform supervision excludes 64 real burn-in hops and uses fully supported OLA.
No learned or reference-conditioned runtime DSP is introduced.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.losses import blend_deployment_features, build_missing_band_mask

BURN = 64
WEIGHTS = {
    "raw_high_spectral": 1.0,
    "raw_high_complex": 0.2,
    "raw_high_energy": 0.1,
    "raw_high_temporal": 0.25,
    "raw_low_complex": 0.1,
    "output_waveform": 1.0,
    "output_high_spectral": 1.0,
    "output_high_temporal": 0.5,
    "output_high_energy": 0.1,
    "output_low_complex": 1.0,
    "prelimit_excess": 0.1,
}


def overlap_add(features):
    """Vectorized periodic-Hann causal OLA, same axis/normalization as Rust."""
    magnitude = torch.pow(10.0, features[:, 0].clamp(-200, 100) / 20)
    spectrum = torch.complex(magnitude * features[:, 1].cos(), magnitude * features[:, 1].sin())
    window = torch.hann_window(256, device=features.device, dtype=features.dtype)
    frames = torch.fft.irfft(spectrum, n=256) * window
    count = features.shape[2]
    length = (count + 1) * 128
    wave = F.fold(frames.transpose(1, 2), (1, length), (1, 256), stride=(1, 128))[:, 0, 0]
    weights = F.fold(
        window.square()[None, :, None].expand(1, -1, count), (1, length), (1, 256), stride=(1, 128)
    )[0, 0, 0]
    return torch.where(weights > 1e-12, wave / weights.clamp_min(1e-12), 0.0)


def limit(wave, mode="legacy"):
    if mode == "smooth-knee":
        shaped = 0.95 * (0.9 + 0.1 * torch.tanh((wave.abs() / 0.95 - 0.9) * 10))
        return torch.where(wave.abs() <= 0.855, wave, wave.sign() * shaped.clamp(max=0.95))
    if mode != "legacy":
        raise ValueError("unknown limiter mode")
    knee, width = 0.855, 0.095
    compressed = knee + width * (1 - torch.exp(-(wave.abs() - knee).clamp_min(0) / (width * 10)))
    return torch.where(wave.abs() <= knee, wave, wave.sign() * compressed.clamp(0, 0.95))


def gate_envelope(active: torch.Tensor) -> torch.Tensor:
    """Input-only mono gates; independent role clips are never fake stereo pairs."""
    result = []
    for sequence in active.cpu().bool().tolist():
        current = start = target = 0.0
        position, remaining = 128, 0
        frames = []
        for request in sequence:
            if request:
                remaining = 128
            values = []
            for _ in range(128):
                wet = request or remaining > 0
                if not request and remaining > 0:
                    remaining -= 1
                new_target = float(wet)
                if target != new_target:
                    target, start, position = new_target, current, 0
                if position < 128:
                    position += 1
                    # Host double here; deployment FP32 difference is measured.
                    import math

                    current = start + (target - start) * (
                        0.5 - 0.5 * math.cos(math.pi * position / 128)
                    )
                else:
                    current = target
                values.append(current)
            frames.extend(values)
        result.append(frames)
    return torch.tensor(result, dtype=torch.float32, device=active.device)


def render(predicted, degraded, mask, active, gate, dry, *, limiter_mode="legacy"):
    effective = mask * active[:, None, :, None]
    blended = blend_deployment_features(degraded, predicted, effective)
    length = predicted.shape[2] * 128
    prelimit = overlap_add(blended)[:, :length]
    return (limit(dry, limiter_mode) * (1 - gate) + limit(prelimit, limiter_mode) * gate).clamp(
        -0.95, 0.95
    ), prelimit


def relative(delta, reference, weight):
    weight = weight.expand_as(reference.real)
    # Finite gradients at exact identity/zero, reference-only normalizers.
    error = (delta.abs().square() * weight).flatten(1).sum(1).clamp_min(1e-20).sqrt()
    band = (reference.abs().square() * weight).flatten(1).sum(1).clamp_min(1e-20).sqrt()
    full = reference.abs().flatten(1).norm(dim=1)
    return (error / band.maximum(full * 0.01).clamp_min(1e-4).detach()).mean()


def energy_error(predicted, target, mask):
    denom = mask.sum(-1).clamp_min(1)
    floor = (target.square().mean(-1) * 1e-4).clamp_min(1e-8).detach()
    powers = [(a.square() * mask).sum(-1) / denom + floor for a in (predicted, target)]
    return (10 * (powers[0].log10() - powers[1].log10()).abs() * (mask.sum(-1) > 0)).mean()


def objective(predicted, batch):
    x, y, mask = batch["x"], batch["y"], batch["mask"]
    output, prelimit = render(predicted, x, mask, batch["active"], batch["gate"], batch["dry"])
    target_wave = batch["clean_wave"]
    dry_wave = batch["dry"]
    spectra = []
    for features in (predicted, y, x):
        mag = torch.pow(10.0, features[:, 0, BURN:].clamp(-200, 100) / 20)
        spectra.append(
            torch.complex(mag * features[:, 1, BURN:].cos(), mag * features[:, 1, BURN:].sin())
        )
    p, t, d = spectra
    m = mask[:, 0, BURN:]
    pm, tm = p.abs(), t.abs()
    terms = {
        "raw_high_spectral": relative(pm - tm, tm, m),
        "raw_high_complex": relative(p - t, t, m),
        "raw_high_energy": energy_error(pm, tm, m),
        "raw_high_temporal": relative(
            pm[:, 1:] - pm[:, :-1] - (tm[:, 1:] - tm[:, :-1]),
            tm[:, 1:],
            torch.minimum(m[:, 1:], m[:, :-1]),
        ),
        "raw_low_complex": relative(p - d, d, 1 - m),
    }
    # Skip burn-in plus one extra hop for common full overlap support.
    waves = [w[:, (BURN + 1) * 128 :] for w in (output, target_wave, dry_wave)]
    scale = (
        (waves[2] - waves[1])
        .abs()
        .mean(-1)
        .maximum(waves[1].square().mean(-1).sqrt() * 0.01)
        .clamp_min(1e-4)
        .detach()
    )
    terms["output_waveform"] = ((waves[0] - waves[1]).abs().mean(-1) / scale).mean()
    terms["prelimit_excess"] = (
        (prelimit[:, (BURN + 1) * 128 :].abs() - 0.855).clamp_min(0).square().mean(-1)
        / (waves[1].square().mean(-1) + 1e-4).detach()
    ).mean()
    multi = {
        k: []
        for k in (
            "output_high_spectral",
            "output_high_temporal",
            "output_high_energy",
            "output_low_complex",
        )
    }
    # Weight MR analysis by mean input-detected cutoff in the supervised region.
    # Rendering itself uses every exact per-hop cutoff/decision.
    cutoff = batch["cutoff"][:, BURN:].mean(1)
    for size in (256, 512, 2048):
        window = torch.hann_window(size, device=p.device)
        specs = [torch.fft.rfft(w.unfold(-1, size, size // 4) * window) for w in waves]
        amps = [s.abs() for s in specs]
        weight = build_missing_band_mask(
            cutoff, batch["rate"][:, 0], fft_size=size, crossover_width_hz=1000
        )[:, 0]
        multi["output_high_spectral"].append(relative(amps[0] - amps[1], amps[1], weight))
        multi["output_high_temporal"].append(
            relative(
                amps[0][:, 1:] - amps[0][:, :-1] - (amps[1][:, 1:] - amps[1][:, :-1]),
                amps[1][:, 1:],
                weight,
            )
        )
        multi["output_high_energy"].append(energy_error(amps[0], amps[1], weight))
        multi["output_low_complex"].append(relative(specs[0] - specs[2], specs[2], 1 - weight))
    terms.update({k: torch.stack(v).mean() for k, v in multi.items()})
    terms["total"] = sum(terms[k] * v for k, v in WEIGHTS.items())
    return terms
