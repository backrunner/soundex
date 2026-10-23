"""P46 original-reference PCM constraints, with output change as telemetry only.

SPDX-License-Identifier: Apache-2.0
"""

import torch

from evaluation.anchored_training import low_band
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_update_directions import gradient


def error_statistics(error: torch.Tensor) -> dict[str, torch.Tensor]:
    """Calibrated per-example errors, preserving reference transients in time."""
    if error.ndim != 2 or error.shape[-1] < 256 or not torch.isfinite(error).all():
        raise ValueError("finite batched PCM with complete hop support required")
    error = error.double()
    difference = error.diff(dim=-1)
    return {
        "rms": error.square().mean(-1).sqrt(),
        "peak": error.abs().amax(-1),
        "difference_rms": difference.square().mean(-1).sqrt(),
        "boundary_difference_rms": difference[:, 127::128].square().mean(-1).sqrt(),
    }


def reference_limits(
    wave: torch.Tensor, anchor: torch.Tensor, reference: torch.Tensor, bits: int
) -> dict:
    """Bound growth of original-target errors, never rolling output displacement.

    q=2^-bits is retained from P45 as an engineering sensitivity ruler, not an
    audibility limit. The first-difference allowance is 2q by the triangle
    inequality for two samples with error perturbation bounded by q. This does
    not mean every candidate perturbation itself has that pointwise bound.
    """
    if bits not in (16, 20, 24) or wave.shape != anchor.shape or wave.shape != reference.shape:
        raise ValueError("registered ruler and aligned PCM required")
    wave, anchor, reference = (v.double() for v in (wave, anchor, reference))
    before, after = error_statistics(anchor - reference), error_statistics(wave - reference)
    displacement = error_statistics(wave - anchor)
    allowance = {k: 2.0**-bits * (2 if "difference" in k else 1) for k in before}
    changes = {k: after[k] - before[k] for k in before}
    passed = torch.stack([changes[k] <= allowance[k] for k in before]).all(0)
    return {
        "passed": passed,
        "before": before,
        "after": after,
        "changes": changes,
        "allowance": allowance,
        "displacement": displacement,
    }


def reference_groups(wave: torch.Tensor, reference: torch.Tensor, batch: dict) -> dict:
    """Identity is full-band; paired low-band projection is offline analysis."""
    mask = batch["identity"].bool()
    if not mask.any() or mask.all() or mask.ndim != 1:
        raise ValueError("both paired and identity examples required")
    if wave.ndim != 2 or wave.shape != reference.shape or len(wave) != len(mask):
        raise ValueError("aligned batched waveforms required")
    if not torch.all((batch["rate"] == 44100) | (batch["rate"] == 48000)):
        raise ValueError("physical sample rates required")
    return {
        "identity": wave[mask].double() - reference[mask].detach().double(),
        "paired_low4k": low_band(wave[~mask], batch["rate"][~mask])
        - low_band(reference[~mask].detach(), batch["rate"][~mask]),
    }


def reference_mse(wave: torch.Tensor, reference: torch.Tensor, batch: dict) -> dict:
    """Absolute PCM squared errors, with finite zero gradients at exact recovery.

    Group means supply first-order direction constraints, not per-example or
    peak certificates. Finite-step checks independently protect every example.
    """
    errors = reference_groups(wave, reference, batch)
    return {key: value.square().mean() for key, value in errors.items()}


def reference_gradients(model: torch.nn.Module, item: tuple) -> dict:
    objective, context = item
    prediction, _ = predict_budget(model, objective.batch, context)
    wave, reference = supervised_waves(prediction, objective.batch)
    return {
        key: gradient(value, list(model.parameters())).detach()
        for key, value in reference_mse(wave, reference, objective.batch).items()
    }


def pcm_guard(current: list, anchors: list, items: list, bits: int) -> dict:
    violations, measurements = [], []
    for index, (wave, anchor, item) in enumerate(zip(current, anchors, items, strict=True)):
        batch = item[0].batch
        reference = supervised_waves(batch["y"], batch)[1]
        before, after = (
            reference_groups(anchor, reference, batch),
            reference_groups(wave, reference, batch),
        )
        # The groups already contain errors against original PCM. Comparing them
        # to zero retains exactly that original reference, not a surrogate target.
        for role in before:
            limits = reference_limits(
                after[role], before[role], torch.zeros_like(before[role]), bits
            )
            measurements.append(
                {
                    "batch": index,
                    "role": role,
                    "allowance": limits["allowance"],
                    **{
                        key: {name: values.tolist() for name, values in limits[key].items()}
                        for key in ("before", "after", "changes", "displacement")
                    },
                }
            )
            for example in (~limits["passed"]).nonzero().flatten().tolist():
                violations.append(
                    {
                        "batch": index,
                        "role": role,
                        "example": example,
                        "failed_metrics": [
                            k
                            for k, v in limits["changes"].items()
                            if v[example] > limits["allowance"][k]
                        ],
                    }
                )
    return {"violations": violations, "measurements": measurements, "bits": bits}
