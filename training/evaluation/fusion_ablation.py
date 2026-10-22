"""Frozen-model DSP interventions, never a training loss or deployment policy."""

import torch

from models.causal_objective import gate_envelope, limit, overlap_add
from models.losses import blend_deployment_features

FUSIONS = ("current", "magnitude-only", "phase-only", "unfused", "safety-only")
CONDITIONS = ("native", "fixed-cutoff", "active-one", "fixed-detector")
# Change one factor at a time. Raw output is also scored for every condition arm.
CASES = tuple((fusion, "native") for fusion in FUSIONS) + tuple(
    ("current", condition) for condition in CONDITIONS[1:]
)


def intervene_condition(condition, mode):
    if mode not in CONDITIONS:
        raise ValueError("unknown condition intervention")
    output = condition.clone()
    if mode in {"fixed-cutoff", "fixed-detector"}:
        output[..., 1] = 0.5
    if mode in {"active-one", "fixed-detector"}:
        output[..., 2] = 1.0
    return output


def render_ablation(prediction, batch, mode):
    """Mono proxy; native audit supplies actual Rust observed spectra.

    Output warmup is 80 hops; the condition's 64-hop warmup is independent.
    Safety-only removes content gate AFTER warmup, preserving startup ramp.
    The other arms preserve input detector decisions and the original gate.
    """
    if mode not in FUSIONS:
        raise ValueError("unknown fusion intervention")
    if prediction.shape[0] != 1:
        raise ValueError("mono proxy only; stereo requires actual shared Rust gate")
    active = batch["active"].clone().bool()
    active[:, :80] = False
    gate = batch["gate"]
    if mode == "safety-only":
        active[:, 80:] = True
        gate = gate_envelope(active).to(prediction.device)
    fused = blend_deployment_features(
        batch["x"], prediction, batch["mask"] * active[:, None, :, None]
    )
    if mode in {"phase-only", "unfused", "safety-only"}:
        amplitude = torch.where(active[:, :, None], prediction[:, 0], batch["x"][:, 0])
        fused = torch.stack((amplitude, fused[:, 1]), dim=1)
    if mode in {"magnitude-only", "unfused", "safety-only"}:
        phase = torch.where(active[:, :, None], prediction[:, 1], batch["x"][:, 1])
        fused = torch.stack((fused[:, 0], phase), dim=1)
    wave = overlap_add(fused)[:, : prediction.shape[2] * 128]
    output = limit(batch["dry"], "smooth-knee") * (1 - gate)
    output = output + limit(wave, "smooth-knee") * gate
    return output.clamp(-0.95, 0.95), wave
