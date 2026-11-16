"""P76 finite Adam diagnostic: auxiliary derivatives only in the phase head.

The backbone keeps every derivative from the unchanged P67 joint objective.
Routing applies before the common global clip and persistent Adam update. It is
not a scalar full-model loss, a detached-magnitude surrogate or a frozen base.
SPDX-License-Identifier: Apache-2.0
"""

import torch

from evaluation.realized_complex import realized_hf_terms
from evaluation.spectral_training import adam_update
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import BURN, balanced_risk, fused_stages, phase_objective, prepare_fusion
from models.restoration_head import predict_budget

ARMS = ("old", "realized_phase", "direct_phase")
COEFFICIENT = 0.1


def auxiliary(prediction, batch, arm):
    if arm == "realized_phase":
        final = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
        region = slice(BURN * 128, final.shape[-1] - 128)
        terms = realized_hf_terms(final[:, region], batch["clean_wave"][:, region], batch["rate"])
        values = COEFFICIENT * terms["complex"].log1p()
    elif arm == "direct_phase":
        direct = direct_spectrum_loss(prediction, batch)
        total = direct["per_example"]
        # Exact change from direct complex coefficient .1 to .2 inside log1p.
        # Scales/support differ from realized_phase; this is a mechanism control,
        # not a norm-matched claim isolating loss location alone.
        values = (total + COEFFICIENT * direct["components"]["complex"]).log1p() - total.log1p()
    else:
        raise ValueError("an explicit phase auxiliary is required")
    return balanced_risk(values, batch["identity"])


def backward(prediction, batch, model, arm, divisor, *, auxiliary_fn=None):
    if (arm not in ARMS and auxiliary_fn is None) or divisor < 1:
        raise ValueError("registered arm and positive batch divisor required")
    old = phase_objective(prediction, batch, include_fused=True)
    extra, gradients = None, None
    head = [p for name, p in model.named_parameters() if name.startswith("phase_head.")]
    if not head or not all(p.requires_grad for p in model.parameters()):
        raise ValueError("joint trainable base and explicit phase head required")
    if arm != "old":
        extra = (auxiliary if auxiliary_fn is None else auxiliary_fn)(prediction, batch, arm)
        gradients = torch.autograd.grad(extra / divisor, head, retain_graph=True)
    (old / divisor).backward()
    if gradients is not None:
        for parameter, gradient in zip(head, gradients, strict=True):
            if parameter.grad is None or not torch.isfinite(gradient).all():
                raise ValueError("missing old head derivative or nonfinite auxiliary")
            parameter.grad.add_(gradient)
    return float(old.detach()), float(extra.detach()) if extra is not None else 0.0, gradients


def update(model, items, optimizer, arm, *, adam_step=adam_update, auxiliary_fn=None):
    if len(items) != 4:
        raise ValueError("four matched paired/identity batches required")
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    auxiliary_values = []
    head = [(k, p) for k, p in model.named_parameters() if k.startswith("phase_head.")]
    extra_sum = [torch.zeros_like(p) for _, p in head]

    def closure():
        values = []
        for objective, context in items:
            prediction, _ = predict_budget(model, objective.batch, context)
            value, extra, gradients = backward(
                prediction, objective.batch, model, arm, len(items), auxiliary_fn=auxiliary_fn
            )
            values.append(value)
            auxiliary_values.append(extra)
            if gradients is not None:
                for total, gradient in zip(extra_sum, gradients, strict=True):
                    total.add_(gradient.detach())
        return sum(values) / len(values)

    result = adam_step(model.parameters(), optimizer, closure)
    derivative = float(
        sum(
            g.double().mul(p.detach().double() - before[k].double()).sum()
            for (k, p), g in zip(head, extra_sum, strict=True)
        )
    )
    result["auxiliary_directional_derivative"] = derivative
    result["old_directional_derivative"] = result["actual_directional_derivative"] - derivative
    result["auxiliary_gradient_norm"] = float(
        sum(g.double().square().sum() for g in extra_sum).sqrt()
    )
    result["weighted_auxiliary_before"] = sum(auxiliary_values) / len(auxiliary_values)
    result["displacement_by_group"] = {
        group: float(
            sum(
                (p.detach().double() - before[k].double()).square().sum()
                for k, p in model.named_parameters()
                if k.startswith(group + ".")
            ).sqrt()
        )
        for group in ("base", "phase_head")
    }
    result["scope"] = (
        "objective_before is unchanged P67 only; routed derivatives are not a full-model scalar objective"
    )
    return result
