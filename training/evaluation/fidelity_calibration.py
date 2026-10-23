"""Absolute PCM fidelity rulers; these are not audibility or release thresholds.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.multires_spectrum import spectral_loss, supervised_waves
from models.restoration_head import predict_budget

BITS = (16, 20, 24)
BANDS = {"full": (0, float("inf")), "low8k": (0, 8000), "hf8k": (8000, float("inf"))}


def lattice_round(wave: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed PCM step size, no saturation or dither; retain the original target.

    A rounding-only precision ruler, including samples outside [-1, 1]. Its
    pointwise mathematical bound is half a step = 2**(-bits). It is not a codec.
    """
    if bits not in BITS or not torch.isfinite(wave).all():
        raise ValueError("registered precision and finite PCM required")
    scale = 2 ** (bits - 1)
    return (wave.double() * scale).round() / scale


def pcm_statistics(error: torch.Tensor, rates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-example full/low/high RMS, peaks and true error time differences.

    FFT powers are Parseval-calibrated PCM units, with DC/Nyquist counted once.
    Supervised support begins on a physical hop boundary. The boundary statistic
    measures error differences, not signal jumps, and is not a playback xrun test.
    """
    if error.ndim != 2 or error.shape[-1] < 256 or len(rates) != len(error):
        raise ValueError("aligned batched PCM and rates required")
    if not torch.isfinite(error).all() or not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("finite PCM and physical sample rates required")
    error = error.double()
    n = error.shape[-1]
    spectrum = torch.fft.rfft(error) / n
    sides = torch.full_like(spectrum.real[0], 2)
    sides[0] = 1
    if n % 2 == 0:
        sides[-1] = 1
    hz = torch.arange(len(sides), device=error.device)[None] * rates.reshape(-1, 1) / n
    result = {
        band + "_rms": (spectrum.abs().square() * sides * ((hz >= low) & (hz < high)))
        .sum(-1)
        .sqrt()
        for band, (low, high) in BANDS.items()
    }
    difference = error.diff(dim=-1)
    result.update(
        peak=error.abs().amax(-1),
        difference_rms=difference.square().mean(-1).sqrt(),
        hop_boundary_difference_rms=difference[:, 127::128].square().mean(-1).sqrt(),
    )
    return result


def assess_wave(wave: torch.Tensor, reference: torch.Tensor, rates: torch.Tensor) -> dict:
    """Original-reference error in absolute units and unchanged MR metrics."""
    absolute = pcm_statistics(wave.double() - reference.double(), rates)
    spectrum = spectral_loss(wave, reference, rates)
    return {
        **{k: v.tolist() for k, v in absolute.items()},
        "mr": spectrum["per_example"].tolist(),
        "mr_dynamics": spectrum["components"]["dynamics"].tolist(),
    }


@torch.no_grad()
def calibrate(model: torch.nn.Module, item: tuple) -> dict:
    """Frozen prediction, feature roundtrips and fixed precision interventions."""
    objective, context = item
    batch = objective.batch
    rates = batch["rate"]
    predicted, _ = predict_budget(model, batch, context)
    wave, reference = supervised_waves(predicted, batch)
    passthrough, _ = supervised_waves(batch["x"], batch)
    clean32, _ = supervised_waves(batch["y"], batch)
    # Promote stored features only. This isolates synthesis precision, not the
    # earlier float32 feature analysis. Original PCM remains the target.
    clean64, _ = supervised_waves(batch["y"].double(), batch)
    results = {
        "frozen_model": assess_wave(wave, reference, rates),
        "compressed_passthrough": assess_wave(passthrough, reference, rates),
        "clean_features_float32": assess_wave(clean32, reference, rates),
        "clean_features_float64_synthesis": assess_wave(clean64, reference.double(), rates),
    }
    for bits in BITS:
        rounded = lattice_round(reference, bits)
        results[f"precision_{bits}"] = assess_wave(rounded, reference.double(), rates)
        if max(results[f"precision_{bits}"]["peak"]) > 2.0**-bits:
            raise ValueError("precision ruler exceeds its mathematical half-step bound")
    return {
        "identity": batch["identity"].bool().tolist(),
        "reference": {k: v.tolist() for k, v in pcm_statistics(reference, rates).items()},
        "model_minus_passthrough": {
            k: v.tolist()
            for k, v in pcm_statistics(wave.double() - passthrough.double(), rates).items()
        },
        "paths": results,
    }


def anchored_limits(
    current: torch.Tensor, anchor: torch.Tensor, reference: torch.Tensor, *, half_step: float
) -> dict[str, torch.Tensor]:
    """Fixed-start total-error and displacement bounds, never rolling allowances.

    A diagnostic engineering budget in PCM units. No spectral progress, phase,
    stereo or perceptual acceptance follows from passing these bounds. Calling
    code must retain the same anchor across a trajectory.
    """
    if not 0 < half_step <= 2.0**-16:
        raise ValueError("unregistered absolute budget")
    if current.shape != anchor.shape or current.shape != reference.shape or current.ndim != 2:
        raise ValueError("aligned batched PCM required")
    if not all(torch.isfinite(v).all() for v in (current, anchor, reference)):
        raise ValueError("finite anchor, reference and current PCM required")
    current, anchor, reference = (v.double() for v in (current, anchor, reference))
    before = (anchor - reference).square().mean(-1).sqrt()
    after = (current - reference).square().mean(-1).sqrt()
    displacement = (current - anchor).square().mean(-1).sqrt()
    return {
        "total_error_rms": after,
        "error_rms_increase": after - before,
        "displacement_rms": displacement,
        "within_budget": (after <= before + half_step) & (displacement <= half_step),
    }
