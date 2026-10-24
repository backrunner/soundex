"""P52 original-reference high-band preservation alongside unchanged spectra.

SPDX-License-Identifier: Apache-2.0
"""

import math

import torch

from evaluation.adapter_scale import FIXED_SCALE, place
from evaluation.batch_update_probe import average
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_guarded_transfer import measured
from scripts.audit_update_directions import gradient, metrics

ARMS = ("control", "logit", "physical")
GRADIENT_FRACTION = 0.1
BANDS = ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24001))


def band_errors(wave, reference, rates):
    """Per-example original-reference error power in calibrated PCM squared units.

    One-sided Parseval weights include DC and Nyquist exactly once. Analysis is
    offline over the already supervised region, not a deployable filter or extra
    lookahead. Summed disjoint bands equal waveform MSE. No signal normalization,
    reference limiting, RMS square-root singularity or clean-output teacher.
    """
    rates = rates.reshape(-1)
    if wave.ndim != 2 or wave.shape != reference.shape or len(rates) != len(wave):
        raise ValueError("aligned batched PCM required")
    if wave.shape[-1] < 256 or not torch.all((rates == 44100) | (rates == 48000)):
        raise ValueError("physical rates and complete PCM support required")
    error = wave.double() - reference.detach().double()
    size = error.shape[-1]
    spectrum = torch.fft.rfft(error)
    sides = error.new_full((spectrum.shape[-1],), 2)
    sides[0] = 1
    if size % 2 == 0:
        sides[-1] = 1
    power = (spectrum.real.square() + spectrum.imag.square()) * sides / size**2
    hz = torch.arange(len(sides), device=wave.device)[None] * rates[:, None] / size
    return torch.stack([(power * ((hz >= low) & (hz < high))).sum(-1) for low, high in BANDS], -1)


def penalties(model, item):
    """Actual identity reconstruction error versus the original clean waveform.

    The logit control observes the same clean clips but precedes input-dependent
    units, tanh, phase and OLA. It is deliberately not called a PCM objective.
    """
    batch = item[0].batch
    identity = batch["identity"].bool()
    if identity.ndim != 1 or not identity.any() or identity.all():
        raise ValueError("paired and identity roles required")
    captured = []
    handle = model.adapter.register_forward_hook(lambda _m, _args, value: captured.append(value))
    try:
        prediction, _ = predict_budget(model, batch, item[1])
    finally:
        handle.remove()
    if len(captured) != 2 or captured[0].shape[1] != 80:
        raise ValueError("separate 80-frame causal warmup required")
    logits = torch.cat(captured, 1)[:, 89:]
    if logits.shape[1] < 1:
        raise ValueError("supervised frames required")
    bands = band_errors(*supervised_waves(prediction, batch), batch["rate"])
    return {
        "logit": logits[identity].double().square().mean(),
        "physical": bands[identity, 1:].sum(-1).mean(),
    }, bands


def calibrate(model, items):
    """Freeze two weights from fit-only pooled initial gradients, never scores.

    Each added gradient has 10% of the common objective's pooled gradient norm
    at this one calibration point. This is not equal fidelity or equal gradients
    throughout training. Zero gradients are reported as inactive, not divided by.
    """
    params = list(model.adapter.parameters())
    pooled, values = {}, []
    for item in items:
        common = metrics(model, *item)[0]["total"]
        terms, _ = penalties(model, item)
        losses = {"common": common, **terms}
        values.append({k: float(v.detach()) for k, v in losses.items()})
        for key, value in losses.items():
            g = gradient(value, params).detach().double() / len(items)
            pooled[key] = pooled.get(key, 0) + g
    norms = {k: float(g.norm()) for k, g in pooled.items()}
    if not all(math.isfinite(v) for v in norms.values()) or norms["common"] <= 0:
        raise ValueError("finite nonzero common calibration gradient required")
    weights = {
        k: GRADIENT_FRACTION * norms["common"] / norms[k] if norms[k] else 0.0
        for k in ("logit", "physical")
    }
    return {
        "weights": weights,
        "gradient_norms": norms,
        "gradient_fraction": GRADIENT_FRACTION,
        "values": average(values),
        "cosine_with_common": {
            k: float(pooled[k].dot(pooled["common"]) / (norms[k] * norms["common"]))
            if norms[k]
            else None
            for k in weights
        },
        "batches": len(items),
    }


def update(model, items, optimizer, arm, calibration):
    """Identical P50 Adam displacement; one fixed auxiliary, no acceptance gate."""
    if arm not in ARMS or len(items) != 4:
        raise ValueError("registered arm and four fit batches required")
    params = list(model.adapter.parameters())
    original = [p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    values, auxiliary = [], []
    for item in items:
        scores = metrics(model, *item)[0]
        values.append({k: float(v.detach()) for k, v in scores.items()})
        loss = scores["total"]
        if arm != "control":
            terms, _ = penalties(model, item)
            term = calibration["weights"][arm] * terms[arm]
            loss = loss + term
            auxiliary.append(float(term.detach()))
        (loss / 4).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(params, original, strict=True)]
    if not all(torch.isfinite(d).all() for d in delta):
        place(params, original, delta, 0)
        raise ValueError("nonfinite proposal; optimizer state must not be reused")
    place(params, original, delta, FIXED_SCALE)
    optimizer.zero_grad(set_to_none=True)
    return {
        "fit_before": average(values),
        "fit_after": average([measured(model, item) for item in items]),
        "weighted_auxiliary_before": sum(auxiliary) / 4,
        "gradient_norm": float(norm),
        "changed_parameters": sum(
            int(torch.count_nonzero(p.detach() != old))
            for p, old in zip(params, original, strict=True)
        ),
        "applied_displacement_l2": float(
            sum(
                (p.detach() - old).square().sum() for p, old in zip(params, original, strict=True)
            ).sqrt()
        ),
    }
