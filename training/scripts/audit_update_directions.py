#!/usr/bin/env python3
"""P40 reversible local update probes, never a saved training trajectory."""

import argparse
import copy
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import budget_risk
from models.direct_spectrum_objective import direct_spectrum_loss
from models.envelope_probe import probe_inputs
from models.fit_diagnostic import BURN
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import load_restoration_head, predict_budget
from scripts.run_consistency_followup import read, registered


def flatten(values: list, params: list) -> torch.Tensor:
    return torch.cat(
        [
            (v if v is not None else torch.zeros_like(p)).flatten()
            for v, p in zip(values, params, strict=True)
        ]
    )


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-30)


def gradient(value: torch.Tensor, params: list) -> torch.Tensor:
    return flatten(torch.autograd.grad(value, params, retain_graph=True, allow_unused=True), params)


def metrics(
    model: torch.nn.Module,
    objective: object,
    context: object,
    fixed_state: torch.Tensor | None = None,
) -> tuple:
    batch = objective.batch
    if fixed_state is None:
        prediction, budget = predict_budget(model, batch, context)
    else:
        _, _, condition = probe_inputs(batch, context)
        tail, _, budget = model.forward_details(
            batch["x"][:, :, BURN:], context[:, BURN:], condition[:, BURN:], fixed_state
        )
        prediction = torch.cat((batch["x"][:, :, :BURN], tail), dim=2)
    direct = direct_spectrum_loss(prediction, batch)
    mr = multires_spectrum_loss(prediction, batch)
    identity = batch["identity"].bool()
    pair = ~identity
    risk = direct["per_example"].log1p() + mr["per_example"].log1p()
    auxiliary = 0.1 * budget_risk(budget, batch)
    values = {
        "total": 0.5 * (risk[pair].mean() + risk[identity].mean()) + auxiliary,
        "paired_direct_amplitude": direct["components"]["amplitude"][pair].mean(),
        "paired_direct_complex": direct["components"]["complex"][pair].mean(),
        "paired_mr": mr["per_example"][pair].mean(),
        "paired_mr_dynamics": mr["components"]["dynamics"][pair].mean(),
        "identity_mr": mr["per_example"][identity].mean(),
        "identity_risk": 0.5 * risk[identity].mean(),
        "budget": auxiliary,
    }
    return values, direct, pair


def restore(params: list, original: list, delta: torch.Tensor | None = None) -> None:
    offset = 0
    with torch.no_grad():
        for p, base in zip(params, original, strict=True):
            p.copy_(base)
            if delta is not None:
                p.add_(delta[offset : offset + p.numel()].reshape_as(p))
            offset += p.numel()


def probe(
    model: torch.nn.Module,
    checkpoint: dict,
    objective: object,
    context: object,
    fixed_history: bool = False,
) -> dict:
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    encoder = torch.cat(
        [
            torch.full_like(p, n.startswith("backbone.encoder."), dtype=torch.bool).flatten()
            for n, p in model.named_parameters()
        ]
    )
    fixed_state = None
    if fixed_history:
        _, _, condition = probe_inputs(objective.batch, context)
        with torch.no_grad():
            _, fixed_state = model(
                objective.batch["x"][:, :, :BURN],
                context[:, :BURN],
                condition[:, :BURN],
                model.initial_state(len(context), device=context.device),
            )
        fixed_state = fixed_state.detach().clone()
    values, direct, pair = metrics(model, objective, context, fixed_state)
    base = {k: float(v.detach()) for k, v in values.items()}
    gradients = {k: gradient(v, params) for k, v in values.items()}
    denominator = 1 + direct["per_example"].detach()
    amp = gradient(0.5 * (direct["components"]["amplitude"] / denominator)[pair].mean(), params)
    comp = gradient(0.05 * (direct["components"]["complex"] / denominator)[pair].mean(), params)
    examples = []
    band = direct["bands"][-1]
    for row in pair.nonzero().flatten().tolist():
        a = gradient(band["amplitude"][row] / denominator[row], params)[encoder]
        c = gradient(0.1 * band["complex"][row] / denominator[row], params)[encoder]
        examples.append(
            {
                "batch_row": row,
                "encoder_hf_cosine": float(unit(a).dot(unit(c))),
                "amplitude_norm": float(a.norm()),
                "complex_norm": float(c.norm()),
            }
        )
    optimizer = torch.optim.Adam(params)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    offset = 0
    for p in params:
        p.grad = gradients["total"][offset : offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    unclipped_norm = float(torch.nn.utils.clip_grad_norm_(params, 5.0))
    optimizer.step()
    adam_delta = torch.cat(
        [(p.detach() - b).flatten() for p, b in zip(params, original, strict=True)]
    )
    restore(params, original)
    for p in params:
        p.grad = None
    directions = {
        "sgd_total": -unit(gradients["total"]),
        "stored_adam": unit(adam_delta),
        "without_encoder": -unit(gradients["total"] * ~encoder),
        "encoder_bisector": -unit(unit(amp * encoder) + unit(comp * encoder)),
        "full_bisector": -unit(unit(amp) + unit(comp)),
    }
    results = {}
    try:
        for name, direction in directions.items():
            derivatives = {k: float(v.dot(direction)) for k, v in gradients.items()}
            points = []
            # Fixed L2 radii; no selection or line search. Negative radius supplies
            # a central finite-difference check at the smaller radius.
            radii = [-0.001, 0.001, 0.01]
            if name == "stored_adam":
                radii.append(float(adam_delta.norm()))
            for radius in radii:
                restore(params, original, direction * radius)
                with torch.no_grad():
                    measured, _, _ = metrics(model, objective, context, fixed_state)
                measured = {k: float(v) for k, v in measured.items()}
                if not all(torch.isfinite(torch.tensor(v)) for v in measured.values()):
                    raise ValueError("nonfinite perturbed objective")
                points.append(
                    {
                        "radius": radius,
                        "values": measured,
                        "changes": {k: measured[k] - base[k] for k in base},
                    }
                )
            finite_difference = {
                k: (points[1]["values"][k] - points[0]["values"][k]) / 0.002 for k in base
            }
            results[name] = {
                "direction_norm": float(direction.norm()),
                "derivatives": derivatives,
                "finite_difference": finite_difference,
                "points": points,
            }
    finally:
        restore(params, original)
    for p, b in zip(params, original, strict=True):
        if not torch.equal(p, b):
            raise ValueError("probe did not restore frozen parameters")
    return {
        "base": base,
        "directions": results,
        "individual_hf": examples,
        "adam_step_norm": float(adam_delta.norm()),
        "unclipped_gradient_norm": unclipped_norm,
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    study, output = args.study.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("fresh output directory required")
    config, endpoints = read(study / "registration.json"), read(study / "review-registration.json")
    if (
        config["format"] != "soundex-detector-condition-v1"
        or read(study / "progress.json")["stage"] != "complete"
    ):
        raise ValueError("completed P38 controls required")
    indices = [int(i * (config["steps"] - 1) / 15) for i in range(16)]
    hashes = dict(config["input_hashes"])
    source = Path(__file__).resolve().parents[1]
    for name, digest in config["source_hashes"].items():
        if name.startswith(("models/", "data/")):
            hashes[str(source / name)] = digest
    for p in [
        Path(__file__),
        study / "registration.json",
        study / "review-registration.json",
        *(Path(p) for p in config["schedule_files"].values()),
    ]:
        hashes[str(p.resolve())] = sha256_file(p)
    selected = {k: v for k, v in endpoints.items() if k.startswith("control-")}
    for v in selected.values():
        hashes[v["path"]] = v["sha256"]
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift: {name}")
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    registered(
        {
            "scope": "P40 reversible single-update diagnostics on P39 training draws; no checkpoint saves, new training, selection, native quality or generalization claim",
            "draw_indices": indices,
            "hashes": hashes,
            "endpoints": selected,
            "directions": [
                "sgd_total",
                "stored_adam",
                "without_encoder",
                "encoder_bisector",
                "full_bisector",
            ],
            "radii_l2": [-0.001, 0.001, 0.01],
            "fixed_history": args.fixed_history,
            "adam": "load recorded optimizer independently for each draw; actual current gradient clipped at 5; probe actual Adam norm as an extra point; no persistent optimizer update",
            "comparison": "all metrics retain original raw scales; total uses exact compressed P38 objective; bisectors use P39 local direct-component gradients, not a new loss",
            "finite_difference": "central 1e-3 radius, report all discrepancies rather than relabel tiny float32 differences as a pass",
        },
        output / "registration.json",
    )
    paths = [Path(p) for p in config["input_hashes"]]
    rows, _ = load_sources([p for p in paths if p.name == "manifest.jsonl"])
    clips = BoundedEnvelopeClips(
        rows,
        next(p for p in paths if p.name == "soundex-gain-data"),
        output / "feature-cache",
        capacity=320,
    )
    reports = {}
    for seed in config["seeds"]:
        schedule = read(Path(config["schedule_files"][str(seed)]))
        pool = LazyPool(schedule["specs"], clips)
        label = f"control-{config['steps']}-{seed}"
        checkpoint = torch.load(selected[label]["path"], map_location="cpu", weights_only=False)
        model = load_restoration_head(checkpoint).eval()
        cases = []
        for index in indices:
            objective, context = pool.select(schedule["draws"][index])
            case = probe(model, checkpoint, objective, context, args.fixed_history)
            cases.append(
                {
                    "draw_index": index,
                    "sample_specs": [schedule["specs"][j] for j in schedule["draws"][index]],
                    **case,
                }
            )
            write_json_report(cases, output / f"{label}.json")
            print(label, index, "complete", flush=True)
        reports[label] = cases
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"input/source drift after probe: {name}")
    write_json_report(
        {
            "status": "complete",
            "cases": reports,
            "registration_sha256": sha256_file(output / "registration.json"),
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fixed-history", action="store_true")
    run(parser.parse_args())
