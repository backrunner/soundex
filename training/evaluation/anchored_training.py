"""P45 spectrum-first method pilot with fixed-start PCM preservation budgets.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from evaluation.batch_update_probe import PRIMARY, average, gradients_for
from evaluation.fidelity_calibration import anchored_limits
from evaluation.spectral_priority_probe import spectral_directions
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import restore

RADII = (0.001, 0.0003)
RELATIVE = 0.001


@torch.no_grad()
def observations(model: torch.nn.Module, items: list[tuple]) -> tuple[dict, list[torch.Tensor]]:
    values, waves = [], []
    for objective, context in items:
        values.append(measured(model, (objective, context)))
        prediction, _ = predict_budget(model, objective.batch, context)
        waves.append(supervised_waves(prediction, objective.batch)[0])
    return {"fit": average(values[:4]), "guard": values[4]}, waves


def low_band(wave: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """Offline diagnostic projection below 4 kHz; no deployment lookahead claim."""
    n = wave.shape[-1]
    frequencies = torch.arange(n // 2 + 1, device=wave.device)[None] * rates.reshape(-1, 1) / n
    return torch.fft.irfft(torch.fft.rfft(wave.double()) * (frequencies < 4000), n=n)


def pcm_guard(current: list, anchor: list, items: list, bits: int) -> dict:
    if bits not in (16, 20, 24):
        raise ValueError("registered precision ruler required")
    half = 2.0**-bits
    violations, maxima = (
        [],
        {"identity_displacement_rms": 0.0, "paired_low4k_displacement_rms": 0.0},
    )
    for index, (wave, start, item) in enumerate(zip(current, anchor, items, strict=True)):
        batch = item[0].batch
        reference = supervised_waves(batch["y"], batch)[1]
        identity = batch["identity"].bool()
        for role, a, b, target in (
            ("identity", wave[identity], start[identity], reference[identity]),
            (
                "paired_low4k",
                low_band(wave[~identity], batch["rate"][~identity]),
                low_band(start[~identity], batch["rate"][~identity]),
                low_band(reference[~identity], batch["rate"][~identity]),
            ),
        ):
            assessment = anchored_limits(a, b, target, half_step=half)
            peak = (a.double() - b.double()).abs().amax(-1)
            # Pointwise lattice bound as well as RMS: spikes cannot hide in a mean.
            passed = assessment["within_budget"] & (peak <= half)
            maxima[role + "_displacement_rms"] = max(
                maxima[role + "_displacement_rms"], float(assessment["displacement_rms"].max())
            )
            for example in (~passed).nonzero().flatten().tolist():
                violations.append(
                    {
                        "batch": index,
                        "role": role,
                        "example": example,
                        "displacement_rms": float(assessment["displacement_rms"][example]),
                        "displacement_peak": float(peak[example]),
                        "error_rms_increase": float(assessment["error_rms_increase"][example]),
                    }
                )
    return {"violations": violations, "maxima": maxima, "half_step": half}


def metric_guard(base: dict, current: dict, anchor: dict) -> dict:
    def allowance(value: float) -> float:
        return 8 * torch.finfo(torch.float32).eps * max(abs(value), 1e-12)

    progress = all(current["fit"][k] < base["fit"][k] - allowance(base["fit"][k]) for k in PRIMARY)
    violations = []
    for role in ("fit", "guard"):
        keys = ("paired_direct_complex", "paired_mr_dynamics", "budget")
        if role == "guard":
            keys = (*keys, *PRIMARY)
        for key in keys:
            initial, value = anchor[role][key], current[role][key]
            if not math.isfinite(value) or value > initial + RELATIVE * abs(initial) + allowance(
                initial
            ):
                violations.append(
                    {"role": role, "metric": key, "change_from_start": value - initial}
                )
    return {"spectral_progress": progress, "violations": violations}


def attempt(
    model: torch.nn.Module,
    items: list,
    anchor: tuple,
    bits: int | None,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    """One matched four-batch attempt. Guard data never enters gradient estimates."""
    base, _ = observations(model, items)
    gradients = average([gradients_for(model, item)[0] for item in items[:4]])
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    if bits is None:
        if optimizer is None:
            raise ValueError("Adam control requires its stored optimizer")
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for p in params:
            p.grad = gradients["total"][offset : offset + p.numel()].reshape_as(p).clone()
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
        optimizer.step()
        current, waves = observations(model, items)
        return {
            "accepted": True,
            "gradient_norm": float(norm),
            "base": base,
            "current": current,
            "metric_guard": metric_guard(base, current, anchor[0]),
            "pcm_rulers": {str(b): pcm_guard(waves, anchor[1], items, b) for b in (16, 20, 24)},
        }
    directions, certificate = spectral_directions(gradients)
    direction = directions["pooled_spectrum_guarded"]
    points, accepted = [], False
    try:
        for radius in RADII:
            restore(params, original, radius * direction)
            current, waves = observations(model, items)
            spectral = metric_guard(base, current, anchor[0])
            pcm = pcm_guard(waves, anchor[1], items, bits)
            accepted = (
                spectral["spectral_progress"]
                and not spectral["violations"]
                and not pcm["violations"]
            )
            points.append(
                {
                    "radius": radius,
                    "current": current,
                    "metric_guard": spectral,
                    "pcm_guard": pcm,
                    "accepted": accepted,
                }
            )
            if accepted:
                break
    except BaseException:
        restore(params, original)
        raise
    if not accepted:
        restore(params, original)
        if not all(torch.equal(p, start) for p, start in zip(params, original, strict=True)):
            raise ValueError("no-op changed model")
    return {
        "accepted": accepted,
        "base": base,
        "certificate": certificate,
        "points": points,
        "selected_radius": points[-1]["radius"] if accepted else None,
    }
