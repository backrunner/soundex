"""P80 diagnostic original-reference phase relations, not a default training loss.

Circular kernels reuse the legacy phase-gradient definition; wrapped L1 follows
the mathematical distance discussed in AP-BWE. Support/normalization are explicit
SoundEx choices, not a reproduction of that speech model or its training scales.
SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from models.fused_spectrum import BURN, balanced_risk

VERSION = "soundex-phase-relations-p80-v1"
KINDS = ("circular", "wrapped")


def distance(delta, kind):
    """Bound both distances by one; wrapped L1 has cusps at zero and pi.

    PyTorch chooses a zero subgradient at zero and the round-to-even branch at
    exact antipodes. The latter is not a unique physical phase direction.
    """
    if kind == "circular":
        return (1 - delta.cos()) / 2
    if kind == "wrapped":
        return (delta - 2 * math.pi * torch.round(delta / (2 * math.pi))).abs() / math.pi
    raise ValueError("registered phase distance required")


def relation_inputs(prediction, batch, burn=BURN):
    if prediction.shape != batch["x"].shape or prediction.shape != batch["y"].shape:
        raise ValueError("aligned phase/input/target features required")
    if burn < BURN or prediction.shape[2] - burn < 2 or prediction.shape[-1] != 129:
        raise ValueError("two supervised FFT256 frames after model warmup required")
    with torch.no_grad():
        reference = torch.where(
            batch["identity"].bool()[:, None, None, None], batch["x"], batch["y"]
        )[:, :, burn:]
        magnitude = torch.pow(10.0, reference[:, 0].clamp(-200, 100) / 20)
        relative = magnitude / magnitude.amax(-1, keepdim=True).clamp_min(1e-10)
        meaningful = (relative >= 1e-3) & (reference[:, 0] >= -120)
        hz = torch.arange(129, device=prediction.device)[None, :] * batch["rate"] / 256
        interior = torch.arange(129, device=prediction.device)
        available = (
            (batch["mask"][:, 0, burn:] > 0)
            & batch["active"][:, burn:, None].bool()
            & (hz[:, None, :] >= 8000)
            & ((interior > 0) & (interior < 128))[None, None, :]
        )
        # Binary native support: do not apply a blend mask twice. All weights
        # depend on the original reference and observed support, never prediction.
        weights = relative * meaningful * available
    delta = prediction[:, 1, burn:] - reference[:, 1]
    return {
        "instantaneous": (delta, weights),
        "frequency": (
            delta[..., 1:] - delta[..., :-1],
            torch.minimum(weights[..., 1:], weights[..., :-1]),
        ),
        "time": (delta[:, 1:] - delta[:, :-1], torch.minimum(weights[:, 1:], weights[:, :-1])),
    }


def phase_relations(prediction, batch, kind, burn=BURN):
    terms, support = {}, {}
    for name, (delta, weights) in relation_inputs(prediction, batch, burn).items():
        mass = weights.sum((1, 2))
        terms[name] = (distance(delta, kind) * weights).sum((1, 2)) / mass.clamp_min(1e-12)
        support[name] = mass
    return {"per_example": sum(terms.values()) / 3, "components": terms, "support_mass": support}


def auxiliary(prediction, batch, arm):
    if arm not in ("circular_relation", "wrapped_relation"):
        raise ValueError("registered relation auxiliary required")
    value = phase_relations(prediction, batch, arm.removesuffix("_relation"))["per_example"]
    return balanced_risk(0.1 * value.log1p(), batch["identity"])
