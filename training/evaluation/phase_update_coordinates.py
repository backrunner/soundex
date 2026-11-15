"""P77 phase readout units with a function/Adam-equivalent control.

Canonical W=64U. Gradients, moments, epsilon, learning rate AND global clipping
must be transformed. The optional gain changes the physical last-layer step,
not inference, loss weights, momentum history or a post-Adam displacement.
SPDX-License-Identifier: Apache-2.0
"""

import copy
from functools import partial

import torch
from torch import nn

from evaluation.spectral_training import adam_update
from scripts.run_phase_temporal_training import load

SCALE = 64.0
NAMES = ("phase_head.2.weight", "phase_head.2.bias")


class UnitPhaseLinear(nn.Linear):
    """Training coordinates only; export converts back to the original layer."""

    def forward(self, value):
        return super().forward(value) * SCALE


def continued(saved, gain):
    if gain not in (1, 64):
        raise ValueError("registered physical readout gains are 1 and 64")
    model = load(saved).eval()
    previous = model.phase_head[-1]
    layer = UnitPhaseLinear(previous.in_features, previous.out_features).to(previous.weight)
    with torch.no_grad():
        layer.weight.copy_(previous.weight / SCALE)
        layer.bias.copy_(previous.bias / SCALE)
    model.phase_head[-1] = layer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    if len(optimizer.param_groups) != 1:
        raise ValueError("one canonical Adam group required")
    group = optimizer.param_groups[0]
    if group["weight_decay"] != 0:
        raise ValueError("weight decay requires a separately derived coordinate rule")
    head = [layer.weight, layer.bias]
    ids = {id(p) for p in head}
    group["params"] = [p for p in group["params"] if id(p) not in ids]
    optimizer.add_param_group(
        {
            **{k: v for k, v in group.items() if k != "params"},
            "params": head,
            "lr": group["lr"] / SCALE * gain,
            "eps": group["eps"] * SCALE,
        }
    )
    for p in head:
        state = optimizer.state[p]
        state["exp_avg"].mul_(SCALE)
        state["exp_avg_sq"].mul_(SCALE**2)
        if "max_exp_avg_sq" in state:
            state["max_exp_avg_sq"].mul_(SCALE**2)
    return model, optimizer


def canonical_model_state(model):
    return {
        k: v.detach().clone() * (SCALE if k in NAMES else 1) for k, v in model.state_dict().items()
    }


def canonical_optimizer_state(model, optimizer, original):
    """Map moments to the original IDs/group; only the gain-one control is equivalent.

    Gain64 checkpoints retain their actual coordinate optimizer separately.
    This conversion is for comparing state tensors, not for resuming gain64 with
    the historical one-group learning rate.
    """
    result = copy.deepcopy(original)
    identifiers = result["param_groups"][0]["params"]
    named = list(model.named_parameters())
    if len(identifiers) != len(named):
        raise ValueError("canonical parameter IDs changed")
    for identifier, (name, parameter) in zip(identifiers, named, strict=True):
        state = copy.deepcopy(optimizer.state[parameter])
        if name in NAMES:
            state["exp_avg"].div_(SCALE)
            state["exp_avg_sq"].div_(SCALE**2)
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].div_(SCALE**2)
        result["state"][identifier] = state
    return result


def original_clip(model):
    """Preserve the canonical global norm; never clip the mixed coordinate norm."""
    head = [p for name, p in model.named_parameters() if name in NAMES]

    def clip(parameters, max_norm, *, error_if_nonfinite):
        for p in head:
            p.grad.div_(SCALE)
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=error_if_nonfinite
            )
        finally:
            for p in head:
                p.grad.mul_(SCALE)
        return norm

    return clip


def stepper(model):
    return partial(adam_update, clip_gradients=original_clip(model))
