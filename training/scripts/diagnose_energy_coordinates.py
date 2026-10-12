#!/usr/bin/env python3
"""Deterministic energy-coordinate probe, not a music generalization score.

Only one band-energy bias learns; the target shape is uniform and known. This
isolates optimizer conditioning from backbone capacity and shape prediction.
"""

import argparse
from pathlib import Path

import torch

from evaluation.reporting import write_json_report
from models.factorized_envelope import factor_objective
from models.residual_energy import ARMS, make_energy_model


def diagnose(steps=128):
    torch.set_num_threads(1)
    records = []
    for level in (0.0, 1e-10, 1e-5, 1e-3):
        source = torch.full((1, 8, 129), 0.1, dtype=torch.float64)
        source[:, :, 86:] = level
        target = source.clone()
        target[:, :, 86:] = 0.01
        scale = source.square().mean(-1, keepdim=True).sqrt()
        input_rms = (
            source[:, :, 86:].square().mean(-1, keepdim=True) + (scale * 1e-8).square()
        ).sqrt()
        for arm in ARMS:
            model = make_energy_model(arm, width=16).double()
            bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.AdamW([bias], lr=1e-4, weight_decay=1e-4)
            trace = []
            for step in range(steps + 1):
                energy = model.band_energy(input_rms, scale, 4 * bias)
                prediction = torch.cat((source[:, :, :86], energy.expand(-1, -1, 43)), -1)
                terms = factor_objective(
                    prediction, target, torch.tensor([[48000.0]], dtype=torch.float64), burn=0
                )
                if step in (0, 1, 32, steps):
                    gradients = {
                        key: float(torch.autograd.grad(terms[key], bias, retain_graph=True)[0])
                        for key in ("total", "band_envelope", "band_shape")
                    }
                    trace.append(
                        {
                            "step": step,
                            "rms": float(energy.detach().mean()),
                            "bias": float(bias.detach()),
                            "loss": float(terms["total"].detach()),
                            "gradients": gradients,
                        }
                    )
                if step == steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_([bias], 5, error_if_nonfinite=True)
                optimizer.step()
            records.append(
                {"arm": arm, "input_band_rms": level, "target_band_rms": 0.01, "trace": trace}
            )
    return {
        "scope": "synthetic fixed-shape single-bias optimization; not convergence or generalization evidence",
        "steps": steps,
        "learning_rate": 1e-4,
        "rows": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_json_report(diagnose(), args.output)
