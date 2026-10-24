"""P47 frozen-output attribution; no trainable intervention or DSP policy.

SPDX-License-Identifier: Apache-2.0
"""

from collections import Counter

import torch

from evaluation.anchored_training import low_band
from evaluation.fidelity_calibration import assess_wave
from evaluation.reference_preservation import error_statistics, pcm_guard
from models.complex_synthesis import complex_features, synthesize
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget

PARTS = {"low4k": (0, 4000), "mid4to8k": (4000, 8000), "high8k": (8000, float("inf"))}
PATHS = ("full", "without_low4k_update", "only_high8k_update")


@torch.no_grad()
def budget_coordinate_control(model: torch.nn.Module, item: tuple) -> tuple:
    """Double budget, halve bounded decoder coordinates; unchanged real residual.

    This is an invertible output-coordinate control, not a fitted parameter model.
    Forward hooks also cover warmup, are always removed, and never edit weights.
    """
    if model.budget_scope != "residual" or model.arm != "complex":
        raise ValueError("residual-scope complex head required")

    def budget_hook(_module, _inputs, logits):
        twice = 2 * torch.nn.functional.softplus(logits)
        return twice + torch.log(-torch.expm1(-twice))

    def coordinate_hook(_module, _inputs, logits):
        return torch.atanh(0.5 * torch.tanh(model.residual_slope * logits)) / model.residual_slope

    handles = [
        model.budget_head.register_forward_hook(budget_hook),
        model.backbone.decoder[-1].register_forward_hook(coordinate_hook),
    ]
    try:
        return predict_budget(model, item[0].batch, item[1])
    finally:
        for handle in handles:
            handle.remove()


def masks(rates: torch.Tensor) -> dict:
    if not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical sample rates required")
    hz = torch.arange(129, device=rates.device)[None, None] * rates.reshape(-1, 1, 1) / 256
    return {name: (hz >= lo) & (hz < hi) for name, (lo, hi) in PARTS.items()}


def feature_paths(start: torch.Tensor, current: torch.Tensor, rates: torch.Tensor) -> dict:
    """Hard masks attribute existing changes, not a proposed deployed crossover.

    Frozen-start coefficients replace selected candidate bins. No target, role
    label, track identity or future frame enters the transformation.
    """
    if start.shape != current.shape or start.ndim != 4 or start.shape[1] != 2:
        raise ValueError("aligned magnitude/phase features required")
    bands = masks(rates)
    return {
        "full": current,
        "without_low4k_update": torch.where(bands["low4k"][:, None], start, current),
        "only_high8k_update": torch.where(bands["high8k"][:, None], current, start),
    }


def decompose(start: torch.Tensor, current: torch.Tensor, batch: dict) -> dict:
    """Linear OLA attribution in float64, preserving the exact supervised region.

    Long-FFT low-band projection is offline measurement. Per-part squared energy
    need not add: cross terms and cancellation are explicitly retained.
    """
    delta = complex_features(current).to(torch.complex128) - complex_features(start).to(
        torch.complex128
    )
    region = slice(89 * 128, delta.shape[1] * 128 - 128)
    parts = {
        name: synthesize(delta * mask)[:, region] for name, mask in masks(batch["rate"]).items()
    }
    total = synthesize(delta)[:, region]
    start_wave, _ = supervised_waves(start, batch)
    current_wave, _ = supervised_waves(current, batch)
    low = {name: low_band(wave, batch["rate"]) for name, wave in parts.items()}
    low_total = low_band(total, batch["rate"])
    energy = low_total.square().sum(-1)
    contributions = {
        name: {
            "full": {k: v.tolist() for k, v in error_statistics(wave).items()},
            "low4k": {k: v.tolist() for k, v in error_statistics(low[name]).items()},
            "low4k_projection_share": torch.where(
                energy > 0, (low[name] * low_total).sum(-1) / energy.clamp_min(1e-300), 0
            ).tolist(),
        }
        for name, wave in parts.items()
    }
    return {
        "parts": contributions,
        "total_low4k": {k: v.tolist() for k, v in error_statistics(low_total).items()},
        "additivity_max_error": float((sum(parts.values()) - total).abs().max()),
        "float32_render_difference_max": float(
            (current_wave.double() - start_wave.double() - total).abs().max()
        ),
        "low4k_cross_energy": (
            low_total.square().mean(-1) - sum(v.square().mean(-1) for v in low.values())
        ).tolist(),
    }


@torch.no_grad()
def compare_outputs(start: torch.Tensor, current: torch.Tensor, item: tuple) -> dict:
    batch = item[0].batch
    anchor, reference = supervised_waves(start, batch)
    result = {}
    for name, prediction in feature_paths(start, current, batch["rate"]).items():
        wave, _ = supervised_waves(prediction, batch)
        direct = direct_spectrum_loss(prediction, batch)
        rulers = {str(b): pcm_guard([wave], [anchor], [item], b) for b in (16, 20, 24)}
        result[name] = {
            "metrics": assess_wave(wave, reference, batch["rate"]),
            "direct": {k: v.tolist() for k, v in direct["components"].items()},
            "direct_bands": [
                {"hz": band["hz"], **{k: band[k].tolist() for k in ("amplitude", "complex")}}
                for band in direct["bands"]
            ],
            "reference_measurements": rulers["16"]["measurements"],
            "violations": {bits: v["violations"] for bits, v in rulers.items()},
        }
    return {"paths": result, "decomposition": decompose(start, current, batch)}


def budget_ablation(history: list) -> dict:
    """Replay recorded acceptance only; never imply a counterfactual trajectory.

    Test every recorded radius. An earlier failed radius can pass this ablation;
    later unrecorded radii are unknown, not assumed failures or successes.
    """
    counts = Counter()
    cases = []
    for attempt in history:
        accepted_without = False
        for point in attempt["points"]:
            metric = point["metric_guard"]
            violations = metric["violations"]
            budget = any(v["metric"] == "budget" for v in violations)
            other = any(v["metric"] != "budget" for v in violations)
            pcm = bool(point["pcm_guard"]["violations"])
            passed = metric["spectral_progress"] and not other and not pcm
            accepted_without |= passed
            counts["proposals"] += 1
            counts["budget_failing_proposals"] += budget
            counts["budget_and_pcm_failing_proposals"] += budget and pcm
            if passed and budget:
                cases.append(
                    {
                        "step": attempt["step"],
                        "radius": point["radius"],
                        "actual_attempt_accepted": attempt["accepted"],
                        "base": attempt["base"],
                        "current": point["current"],
                        "violations": violations,
                        "pcm_measurements": point["pcm_guard"]["measurements"],
                    }
                )
        counts["attempts"] += 1
        counts["actual_accepted"] += attempt["accepted"]
        counts["would_accept_recorded_proposal_without_budget"] += accepted_without
        counts["additional_accepted_attempts_without_budget"] += (
            accepted_without and not attempt["accepted"]
        )
    return {"counts": dict(counts), "budget_only_cases": cases}
