"""P82 common-reference phase relations before/after causal synthesis.

Diagnostic only. Original lossless PCM, input-only availability and P80 kernels;
no inference operation, predictor-dependent support or default objective change.
SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from models.complex_synthesis import analyze, complex_features, synthesize
from models.fused_spectrum import BURN, balanced_risk, fused_stages, prepare_fusion
from models.phase_relations import distance

VERSION = "soundex-realized-phase-relations-p82-v1"
DOMAINS = ("direct", "raw", "fused")
COMPONENTS = ("instantaneous", "frequency", "time")


@torch.no_grad()
def prepare_relations(batch, burn=BURN):
    """One original-PCM analysis and identical detached support for every domain.

    Analyze uses complete Hann256/hop128 windows wholly inside the existing
    supervised PCM region [burn*128, N-128). Frames are burn through T-3 inclusive.
    The original P80 cached feature reference and last two frames remain historical
    controls; this common PCM reference is never reconstructed from those features.
    """
    target = analyze(batch["clean_wave"].detach())
    frames = batch["x"].shape[2]
    if burn < BURN or target.shape != (len(batch["x"]), frames - 2, 129) or frames - 2 - burn < 2:
        raise ValueError(
            "aligned original PCM and at least two supported post-burn frames required"
        )
    target = target[:, burn:]
    magnitude = target.abs()
    relative = magnitude / magnitude.amax(-1, keepdim=True).clamp_min(1e-10)
    # P80 -120 dB feature amplitude equals 1e-6 in unnormalized FFT units.
    meaningful = (relative >= 1e-3) & (magnitude >= 1e-6)
    bins = torch.arange(129, device=target.device)
    hz = bins[None, :] * batch["rate"] / 256
    available = (
        (batch["mask"][:, 0, burn : frames - 2] > 0)
        & batch["active"][:, burn : frames - 2, None].bool()
        & (hz[:, None, :] >= 8000)
        & ((bins > 0) & (bins < 128))[None, None, :]
    )
    weight = relative * meaningful * available
    return {"reference": target, "weight": weight, "burn": burn, "end": frames - 2}


def domain_spectra(prediction, batch, plan):
    """Analysis windows are offline loss support, not model lookahead."""
    if prediction.shape != batch["x"].shape:
        raise ValueError("aligned prediction required")
    direct = complex_features(prediction)
    raw = synthesize(direct)
    fused = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
    burn, end = plan["burn"], plan["end"]
    return {
        "direct": direct[:, burn:end],
        "raw": analyze(raw)[:, burn:],
        "fused": analyze(fused)[:, burn:],
    }


def edges(delta, weight):
    return {
        "instantaneous": (delta, weight),
        "frequency": (
            delta[..., 1:] - delta[..., :-1],
            torch.minimum(weight[..., 1:], weight[..., :-1]),
        ),
        "time": (delta[:, 1:] - delta[:, :-1], torch.minimum(weight[:, 1:], weight[:, :-1])),
    }


def relations(spectrum, plan, kind):
    target, weight = plan["reference"], plan["weight"]
    if spectrum.shape != target.shape:
        raise ValueError("common original-reference phase geometry required")
    # Unsupported bins are constants before angle(), so irrelevant tiny/zero
    # predictions cannot produce 0*nonfinite backward values. Supported prediction
    # zeros retain torch.angle(0)=0 and its zero derivative; magnitude loss remains.
    observed = torch.where(weight > 0, spectrum, torch.ones_like(spectrum))
    reference = torch.where(weight > 0, target, torch.ones_like(target))
    delta = observed.angle() - reference.angle()
    components, support = {}, {}
    for name, (error, w) in edges(delta, weight).items():
        mass = w.sum((1, 2))
        components[name] = (distance(error, kind) * w).sum((1, 2)) / mass.clamp_min(1e-12)
        support[name] = mass
    return {
        "per_example": sum(components.values()) / 3,
        "components": components,
        "support_mass": support,
    }


def auxiliary(prediction, batch, arm):
    domain, kind = arm.split("/")
    if domain not in DOMAINS:
        raise ValueError("registered phase-relation domain required")
    plan = prepare_relations(batch)
    spectrum = domain_spectra(prediction, batch, plan)[domain]
    return balanced_risk(
        0.1 * relations(spectrum, plan, kind)["per_example"].log1p(), batch["identity"]
    )


@torch.no_grad()
def correspondence(a, b, plan, kind):
    """Measured phase-risk differences and a per-case analytic PCM-analysis bound.

    Circular and wrapped kernels are geodesically Lipschitz with constants 1/2
    and 1/pi. Phase separation <= pi*min(1, |a-b|/min(|a|,|b|)); zero bins use pi.
    Edge separations add. This bounds same-target risk discrepancy for these two
    spectra, not renderer repeatability, derivative parity or model quality.
    """
    if kind not in ("circular", "wrapped"):
        raise ValueError("registered kernel required")
    ra, rb = relations(a, plan, kind), relations(b, plan, kind)
    minimum = torch.minimum(a.abs(), b.abs())
    angular = math.pi * ((a - b).abs() / minimum.clamp_min(torch.finfo(minimum.dtype).tiny)).clamp(
        max=1
    )
    angular = torch.where(minimum > 0, angular, torch.full_like(angular, math.pi))
    weights = plan["weight"]
    limits = {
        "instantaneous": (angular, weights),
        "frequency": (
            angular[..., 1:] + angular[..., :-1],
            torch.minimum(weights[..., 1:], weights[..., :-1]),
        ),
        "time": (angular[:, 1:] + angular[:, :-1], torch.minimum(weights[:, 1:], weights[:, :-1])),
    }
    lipschitz = 0.5 if kind == "circular" else 1 / math.pi
    bounds = {
        k: ((v * lipschitz).clamp(max=1) * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-12)
        for k, (v, w) in limits.items()
    }
    differences = {k: (ra["components"][k] - rb["components"][k]).abs() for k in COMPONENTS}
    return {
        "component_difference": differences,
        "component_bound": bounds,
        "risk_difference": (ra["per_example"] - rb["per_example"]).abs(),
        "risk_bound": sum(bounds.values()) / 3,
        "supported_prediction_zero_count": ((minimum == 0) & (weights > 0)).sum((1, 2)),
    }
