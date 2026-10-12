#!/usr/bin/env python3
"""P8 optimization audit: matched early trajectories and fixed-batch fitting.

No candidate selection, test scoring or production export. Read P8 draws exactly;
compare learning rates and an optional input-relative energy coordinate.
Repeated-batch fits diagnose learnability, not quality.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from data.weak_band_study import identity_specs
from evaluation.reporting import sha256_file, write_json_report
from models.conditioned_energy import make_conditioned_model
from models.envelope_objective import BANDS
from models.envelope_probe import context_features, predict_probe
from models.factorized_envelope import BURN
from models.linear_objective import linear_objective, make_linear_model
from scripts.run_envelope_study import validate


def parameter_group(name):
    if name.startswith("energy_head."):
        return "energy"
    if name.startswith("decoder."):
        return "shape"
    return "backbone"


def gradient_summary(model):
    result = {}
    for group in ("energy", "shape", "backbone"):
        grads = [
            p.grad.detach().flatten()
            for n, p in model.named_parameters()
            if parameter_group(n) == group and p.grad is not None
        ]
        v = torch.cat(grads)
        result[group] = {"norm": float(v.norm()), "max": float(v.abs().max())}
    return result


def prepared(clips, specs, device):
    batch = clips.batch(specs, device)
    return batch, context_features(batch["pcm"], "past").to(device)


def forward(model, pair):
    batch, context = pair
    p = predict_probe(model, batch, context, burn=BURN)
    t = torch.pow(10.0, batch["y"][:, 0] / 20)
    return p, t, linear_objective(p, t, batch["rate"])


@torch.no_grad()
def measure(model, pair):
    p, t, loss = forward(model, pair)
    batch = pair[0]
    rows = []
    for i in range(len(p)):
        values = linear_objective(p[i : i + 1], t[i : i + 1], batch["rate"][i : i + 1])
        frequency = torch.arange(129, device=p.device) * batch["rate"][i, 0] / 256
        pp, tt = p[i, BURN:], t[i, BURN:]
        scale = tt.square().mean().clamp_min(1e-30)
        bands = []
        for low, high in BANDS:
            a, b = (
                pp[:, (frequency >= low) & (frequency < high)],
                tt[:, (frequency >= low) & (frequency < high)],
            )
            bands.append(
                {
                    "low": low,
                    "reference_relative_power_db": float(
                        10 * (b.square().mean().clamp_min(1e-30) / scale).log10()
                    ),
                    "prediction_relative_power_db": float(
                        10 * (a.square().mean().clamp_min(1e-30) / scale).log10()
                    ),
                    "sc": float((a - b).norm() / b.norm().clamp_min(1e-10)),
                }
            )
        rows.append({"loss": {k: float(v) for k, v in values.items()}, "bands": bands})
    return {"loss": {k: float(v) for k, v in loss.items()}, "rows": rows}


def initial_model(seed, device, coordinate="full_spectrum"):
    torch.manual_seed(seed)
    model = (
        make_conditioned_model()
        if coordinate == "input_relative"
        else make_linear_model("linear_reconstruction")
    )
    return model.to(device)


def first_step(seed, pair, coordinate="full_spectrum"):
    model = initial_model(seed, pair[0]["x"].device, coordinate)
    before = copy.deepcopy(model.state_dict())
    initial = measure(model, pair)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    _, _, terms = forward(model, pair)
    parameters = list(model.parameters())
    a = torch.autograd.grad(terms["linear_reconstruction"], parameters, retain_graph=True)
    b = torch.autograd.grad(terms["linear_dynamics"], parameters, retain_graph=True)
    aa, bb = torch.cat([v.flatten() for v in a]), torch.cat([v.flatten() for v in b])
    cosine = float((aa @ bb) / (aa.norm() * bb.norm()).clamp_min(1e-30))
    terms["total"].backward()
    gradients = gradient_summary(model)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True))
    optimizer.step()
    after = copy.deepcopy(model.state_dict())
    scans = []
    # Interpolate the SAME Adam update; isolate which output branch causes damage.
    for group in ("all", "energy", "shape"):
        for fraction in (1.0, 0.1, 0.01, 0.001):
            state = {
                n: v + fraction * (after[n] - v)
                if group == "all" or parameter_group(n) == group
                else v
                for n, v in before.items()
            }
            model.load_state_dict(state)
            scans.append({"group": group, "fraction": fraction, **measure(model, pair)})
    return {
        "seed": seed,
        "initial": initial,
        "gradients": gradients,
        "preclip_norm": norm,
        "amplitude_dynamics_gradient_cosine": cosine,
        "scans": scans,
    }


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root, output = args.study.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((root / "config.json").read_text())
    rows, manifests = load_sources([Path(p) for p in config["manifests"]])
    if manifests != config["manifests"]:
        raise ValueError("source manifests changed")
    detector = root / "binaries/soundex-gain-data"
    if sha256_file(detector) != config["detector_sha256"]:
        raise ValueError("detector changed")
    clips = EnvelopeClips(rows, detector, config["cache_path"])
    # Content-independent choice from the already registered training probes.
    music = [
        s for s in config["training_probe_specs"] if rows[s["row_id"]]["corpus"] == "music_library"
    ]
    speech = [
        s for s in config["training_probe_specs"] if rows[s["row_id"]]["corpus"] == "speech_library"
    ]
    fit_specs = music[:2] + speech[:1] + identity_specs(music[:1])
    for spec in fit_specs:
        row = clips.rows[spec["row_id"]]
        if row["split"] != "train" or row["held_out"]:
            raise ValueError("fit must use training records only")
    protocol = {
        "parent_config_sha256": sha256_file(root / "config.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "rates": args.rates,
        "coordinate": args.coordinate,
        "source_files": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
            for package in ("models", "scripts", "data", "evaluation")
            for p in sorted((Path(__file__).resolve().parents[1] / package).rglob("*.py"))
        },
        "seeds": config["seeds"],
        "early_steps": args.steps,
        "fit_steps": args.fit_steps,
        "fit_specs": fit_specs,
        "device": args.device,
        "scope": "post-hoc development diagnosis; no independent test or promotion",
        "schedule": "original P8 cosine horizon 2048 for early; constant rate for fixed-batch fit",
        "draw_sha256": {str(s): sha256_file(root / f"draws-{s}.json") for s in config["seeds"]},
    }
    write_json_report(protocol, output / "protocol.json")
    fit_pair = prepared(clips, fit_specs, args.device)
    results = []
    for seed in config["seeds"]:
        draws = json.loads((root / f"draws-{seed}.json").read_text())["draws"]
        first = prepared(clips, draws[:4], args.device)
        write_json_report(
            first_step(seed, first, args.coordinate), output / f"first-step-{seed}.json"
        )
        for mode in ("early", "fit"):
            steps = args.steps if mode == "early" else args.fit_steps
            for lr in protocol["rates"]:
                model = initial_model(seed, args.device, args.coordinate)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                trace = []
                initial = measure(model, fit_pair)
                for step in range(1, steps + 1):
                    pair = (
                        prepared(clips, draws[(step - 1) * 4 : step * 4], args.device)
                        if mode == "early"
                        else fit_pair
                    )
                    optimizer.param_groups[0]["lr"] = (
                        lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / config["steps"])))
                        if mode == "early"
                        else lr
                    )
                    optimizer.zero_grad(set_to_none=True)
                    _, _, loss = forward(model, pair)
                    loss["total"].backward()
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 5, error_if_nonfinite=True
                        )
                    )
                    optimizer.step()
                    if step in {1, 2, 4, 8, 16, 32, 64, 128, 256, steps}:
                        record = {
                            "step": step,
                            "pre_update_loss": float(loss["total"].detach()),
                            "preclip_norm": norm,
                            "fixed_probe": measure(model, fit_pair),
                        }
                        if mode == "early" and step in {1, 8, 32, steps}:
                            record["development"] = validate(
                                model,
                                clips,
                                config["validation_specs"],
                                "past",
                                args.device,
                                objective_fn=linear_objective,
                                burn=BURN,
                            )
                        trace.append(record)
                        print(
                            json.dumps(
                                {
                                    "mode": mode,
                                    "seed": seed,
                                    "lr": lr,
                                    "step": step,
                                    "fixed_loss": record["fixed_probe"]["loss"]["total"],
                                    "development": record.get("development", {}).get("score"),
                                }
                            ),
                            flush=True,
                        )
                result = {"seed": seed, "mode": mode, "lr": lr, "initial": initial, "trace": trace}
                path = output / f"{mode}-{seed}-{lr}.json"
                write_json_report(result, path)
                results.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "lr": lr,
                        "initial_fixed_loss": initial["loss"]["total"],
                        "final_fixed_loss": trace[-1]["fixed_probe"]["loss"]["total"],
                        "final_development_score": trace[-1].get("development", {}).get("score"),
                        "report_sha256": sha256_file(path),
                    }
                )
                write_json_report(
                    {"status": "running", "completed": results}, output / "summary.json"
                )
    write_json_report({"status": "complete", "completed": results}, output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coordinate", choices=("full_spectrum", "input_relative"), default="full_spectrum"
    )
    parser.add_argument("--rates", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fit-steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.fit_steps < 1
        or any(not math.isfinite(r) or r <= 0 for r in args.rates)
    ):
        parser.error("positive steps required")
    run(args)
