"""P78 reproduce P77 endpoints exactly and attribute their first Adam update.

No changed candidate, extra training budget, moment reset or output scoring.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adam_attribution import components, summarize
from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import (
    NAMES,
    SCALE,
    canonical_model_state,
    continued,
    stepper,
)
from evaluation.reporting import sha256_file, write_json_report
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_routing import exact
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh audit required")
    output.mkdir(parents=True)
    snapshot(output)
    checked_receipt(prior / "complete.json")
    p77 = read(prior / "registration.json")
    hashes = dict(p77["hashes"])
    hashes.update(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "format": "soundex-adam-contributions-p78-v1",
            "prior": str(prior),
            "hashes": hashes,
            "budget": "Replay all16 P77 two-update trajectories exactly (32 already-studied updates); every final coordinate model and optimizer must be bitwise identical. Capture first-update gradients and physical displacement only; no new candidate checkpoints, held-out scoring or audio rendering.",
            "question": "Separate inherited first moment, current old gradient, auxiliary numerator and changed second-moment denominator in the actual first physical phase update. P77's larger readout step did not jointly improve spectral rulers; cumulative endpoint differences cannot identify optimizer contribution.",
            "analysis": "Canonical FP64 Adam algebra at common initial state, groups base/phase hidden/phase readout. Effective extra gradient is the captured routed-minus-old difference, including FP32 accumulation. Require inactive clipping to avoid a hidden coupling interpretation; retain norms and all groups/origins. Validate actual FP32 parameter changes against analytic predictions within two local FP32 ULP plus16*epsilon*abs(step). Exact algebra is checked independently in FP64 tests. This bound concerns numerical reconstruction, not audio acceptance.",
            "scope": "Fit-only first-step local attribution on P77's fixed eight TRAIN draws. No reset-Adam treatment, coefficient selection, convergence/architecture ceiling, native improvement or device qualification. Existing P77 failures remain.",
        },
        output / "registration.json",
    )


def run(output):
    registration = read(output / "registration.json")
    verify(registration["hashes"])
    if (output / "first-step.pth").exists():
        raise ValueError("existing diagnostic retained")
    prior = Path(registration["prior"])
    config = read(prior / "registration.json")
    frozen = read(prior / "frozen.json")["endpoints"]
    p74 = Path(config["p74"])
    old = read(p74 / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        config["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), p74 / "cache", capacity=100),
    )
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"]}
    captured, checks, analyses = {}, {}, {}
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for gain in (1, 64):
            for arm in ("old", "realized_phase"):
                model, optimizer = continued(saved, gain)
                named = list(model.named_parameters())
                before = canonical_model_state(model)
                first = {}

                def observed(parameters, opt, closure, *, model=model, first=first, named=named):
                    def capture():
                        value = closure()
                        first["gradients"] = {
                            name: p.grad.detach().clone() / (SCALE if name in NAMES else 1)
                            for name, p in named
                        }
                        return value

                    return stepper(model)(parameters, opt, capture)

                selected = [items[i] for i in config["fit"][:4]]
                measurement = update(model, selected, optimizer, arm, adam_step=observed)
                if measurement["gradient_norm"] >= 5:
                    raise ValueError(
                        "active global clipping needs a different attribution; retain audit"
                    )
                first["norm"] = measurement["gradient_norm"]
                first["delta"] = {
                    name: value.double() - before[name].double()
                    for name, value in canonical_model_state(model).items()
                }
                first["before"] = before
                update(
                    model,
                    [items[i] for i in config["fit"][4:]],
                    optimizer,
                    arm,
                    adam_step=stepper(model),
                )
                label = origin + "/" + ("unit_" if gain == 1 else "gain64_") + arm
                target = torch.load(frozen[label]["path"], map_location="cpu", weights_only=False)
                exact(model.state_dict(), target["coordinate_model"])
                exact(optimizer.state_dict(), target["coordinate_optimizer"])
                exact(canonical_model_state(model), target["model"])
                checks[label] = True
                captured[label] = first
                torch.save(captured, output / "first-step.pth")
                write_json_report(
                    {"exact_endpoint_checks": checks, "analyses": analyses},
                    output / "progress.json",
                )
                print("exact replay", label, flush=True)
            old_first = captured[origin + "/" + ("unit_" if gain == 1 else "gain64_") + "old"]
            routed_first = captured[
                origin + "/" + ("unit_" if gain == 1 else "gain64_") + "realized_phase"
            ]
            group = saved["optimizer"]["param_groups"][0]
            if group["weight_decay"] or group["amsgrad"]:
                raise ValueError("only registered unregularized Adam is analyzed")
            by_group = {key: [] for key in ("base.", "phase_head.0.", "phase_head.2.")}
            numerical = []
            for identifier, (name, _) in zip(group["params"], named, strict=True):
                state = saved["optimizer"]["state"][identifier]
                g0, g1 = old_first["gradients"][name], routed_first["gradients"][name]
                vectors = components(
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    g0,
                    g1,
                    lr=group["lr"] * (gain if name in NAMES else 1),
                    betas=group["betas"],
                    eps=group["eps"],
                    step=int(state["step"]) + 1,
                )
                actual = routed_first["delta"][name]
                magnitude = torch.maximum(
                    routed_first["before"][name].abs(),
                    (routed_first["before"][name].double() + actual).float().abs(),
                )
                ulp = (
                    torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude
                ).double()
                error = (actual - vectors["routed_step"]).abs()
                bound = 2 * ulp + 16 * torch.finfo(torch.float32).eps * vectors["routed_step"].abs()
                if not (error <= bound).all():
                    raise ValueError(
                        "Adam algebra does not reconstruct FP32 writes within registered rounding bound"
                    )
                numerical.append(float(error.max()))
                prefix = next(p for p in by_group if name.startswith(p))
                by_group[prefix].append((vectors, g0.double(), g1.double() - g0.double(), actual))
            result = {}
            for prefix, pieces in by_group.items():
                merged = {
                    key: torch.cat([p[0][key].flatten() for p in pieces]) for key in pieces[0][0]
                }
                g0 = torch.cat([p[1].flatten() for p in pieces])
                extra = torch.cat([p[2].flatten() for p in pieces])
                actual = torch.cat([p[3].flatten() for p in pieces])
                result[prefix] = {
                    **summarize(merged, g0, extra),
                    "old_gradient_l2": float(g0.norm()),
                    "effective_extra_gradient_l2": float(extra.norm()),
                    "actual_step_l2": float(actual.norm()),
                }
            analyses[origin + "/gain" + str(gain)] = {
                "groups": result,
                "max_algebra_vs_actual_write_error": max(numerical),
            }
    torch.save(captured, output / "first-step.pth")
    registered(
        {
            "exact_endpoint_checks": checks,
            "analyses": analyses,
            "scope": registration["scope"],
            "qualified": False,
        },
        output / "assessment.json",
    )
    verify(registration["hashes"])
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.log" and "__pycache__" not in p.parts
            }
        },
        output / "complete.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        failure = args.output / "failure.json"
        if args.output.exists() and not failure.exists() and (args.mode == "run" or fresh):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
