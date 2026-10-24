"""P54 explicit objectives and actual Adam updates, without post-step scaling.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from models.auraloss_spectrum import spectral_loss as auraloss_spectrum
from models.direct_spectrum_objective import direct_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss, supervised_waves
from models.restoration_head import predict_budget

OBJECTIVES = ("common", "auraloss")


def objective_value(model, item, objective):
    """Change only the rendered spectral term; preserve direct HF/fidelity terms.

    The frozen budget auxiliary has zero adapter gradient and is not optimized.
    Historical common scores (including that constant) remain separately reported.
    Equal paired/identity means and the input schedule retain the 80:20 corpus mix.
    """
    if objective not in OBJECTIVES:
        raise ValueError("unknown spectral objective")
    batch = item[0].batch
    prediction, _ = predict_budget(model, batch, item[1])
    direct = direct_spectrum_loss(prediction, batch)["per_example"]
    if objective == "common":
        rendered = multires_spectrum_loss(prediction, batch)["per_example"]
    else:
        rendered = auraloss_spectrum(*supervised_waves(prediction, batch), batch["rate"])[
            "per_example"
        ]
    risk = direct.log1p() + rendered.log1p()
    identity = batch["identity"].bool()
    return 0.5 * (risk[~identity].mean() + risk[identity].mean())


def adam_update(parameters, optimizer, closure):
    """One genuine optimizer step with measured float32 parameter displacement.

    The closure owns averaging/backward and returns a detached scalar. No line
    search, hidden displacement multiplier, accept/reject gate or moment reset.
    The caller must stop on nonfinite failure; do not reuse contaminated state.
    """
    parameters = list(parameters)
    before = [p.detach().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    value = float(closure())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("nonfinite objective")
    if any(p.grad is None for p in parameters):
        raise ValueError("missing trainable parameter gradient")
    gradients = [p.grad.detach().clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(parameters, before, strict=True)]
    if not all(torch.isfinite(d).all() for d in delta):
        raise ValueError("nonfinite Adam displacement; stop and retain failure")
    result = {
        "objective_before": value,
        "gradient_norm": float(norm),
        "actual_displacement_l2": float(sum(d.double().square().sum() for d in delta).sqrt()),
        "actual_directional_derivative": float(
            sum((g.double() * d.double()).sum() for g, d in zip(gradients, delta, strict=True))
        ),
        "changed_parameters": sum(int(torch.count_nonzero(d)) for d in delta),
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
    }
    optimizer.zero_grad(set_to_none=True)
    return result


def update(model, items, optimizer, objective):
    if len(items) != 4:
        raise ValueError("four fit batches required")

    def closure():
        values = []
        for item in items:
            value = objective_value(model, item, objective)
            (value / len(items)).backward()
            values.append(float(value.detach()))
        return sum(values) / len(values)

    return adam_update(model.adapter.parameters(), optimizer, closure)
