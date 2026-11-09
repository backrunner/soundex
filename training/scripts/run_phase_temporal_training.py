"""P74 fresh matched small temporal-observation training with unchanged P67 loss.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import time
import traceback
from pathlib import Path

import torch

from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.trainability_audit import assert_reserved_isolation
from models.phase_temporal import OBSERVATIONS, phase_temporal_model
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.run_band_capacity_training import STEPS
from scripts.run_band_capacity_training import pack as capacity_pack
from scripts.run_band_capacity_training import run as run_matched
from scripts.run_band_training import jsonl
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def fresh(kind, seed):
    return phase_temporal_model(kind, seed)


def pack(model, kind, seed, optimizer, history, signature):
    return {
        **capacity_pack(model, kind, seed, optimizer, history, signature),
        "format": "soundex-phase-temporal-training-v1",
    }


def load(saved, _scope=None):
    if saved["format"] != "soundex-phase-temporal-training-v1":
        raise ValueError("explicit P74 checkpoint required")
    model = fresh(saved["kind"], saved["seed"])
    model.load_state_dict(saved["model"], strict=True)
    return model


def inspect(model, items, changed):
    if changed is not None and not any(
        count for k, count in changed.items() if k.startswith("base.relation_projections.")
    ):
        raise ValueError("new observation projections did not update")
    records = []
    for index, (objective, context) in enumerate(items):
        batch = objective.batch
        x = batch["x"].detach().clone().requires_grad_()
        prediction, _ = predict_budget(model, {**batch, "x": x}, context)
        gradient = torch.autograd.grad(prediction[:, 0, -1, 80:120].mean(), x)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite TRAIN phase-history derivative")
        history_gradient = float(gradient[:, 1, 89:-1].abs().sum())
        with torch.no_grad():
            # A feature intervention only: keep magnitudes/context/targets fixed,
            # repeat the final observed phase over past frames, then inspect only
            # the final output whose current input phase is still unchanged.
            repeated = batch["x"].clone()
            repeated[:, 1] = repeated[:, 1, -1:, :].clone().expand_as(repeated[:, 1])
            ablated, _ = predict_budget(model, {**batch, "x": repeated}, context)
            magnitude_change = float(
                (prediction[:, 0, -1] - ablated[:, 0, -1]).square().mean().sqrt()
            )
        if model.temporal_observation == "magnitude" and (history_gradient or magnitude_change):
            raise ValueError("magnitude control consumed phase history")
        records.append(
            {
                "draw": index,
                "phase_history_gradient_l1": history_gradient,
                "final_magnitude_db_rms_change_repeated_phase": magnitude_change,
            }
        )
    if model.temporal_observation == "phase" and not any(
        r["phase_history_gradient_l1"] > 0 for r in records
    ):
        raise ValueError("no phase-history path on registered TRAIN probes")
    return {
        "records": records,
        "scope": "First eight registered TRAIN draws, original versus repeated-final-phase feature history with common magnitude/context/warmup/targets. Final-frame magnitude only; appended rotation cannot change it. Derivatives exclude detached89-hop warmup. Learning/history connectivity, not restored-audio quality or physical-waveform intervention.",
    }


def prepare(prior, feasibility, audit, output):
    if output.exists():
        raise ValueError("fresh registration required")
    for p in (prior / "p72-complete.json", feasibility / "complete.json", audit / "complete.json"):
        checked_receipt(p)
    screen = read(feasibility / "assessment.json")
    if not all(
        screen[k]
        for k in ("all_compute_under_5ms", "all_service_under_hop", "all_paced_completions_on_time")
    ):
        raise ValueError("P74 feasibility prerequisite failed; retain timing failures")
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    if len(schedule["draws"]) != 4 * STEPS:
        raise ValueError("incomplete matched exposure")
    selected = {schedule["specs"][i]["row_id"] for d in schedule["draws"] for i in d}
    review = read(Path(old["review_schedule"]))
    reserved = [rows[review["specs"][i]["row_id"]] for d in review["review"] for i in d]
    repo = Path(old["repository"])
    catalogs = sorted((repo / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repo / "data/catalogs/current-approved.jsonl"
    assert_reserved_isolation(
        [rows[k] for k in selected],
        list(rows.values()) + jsonl(catalog),
        [r for p in catalogs for r in jsonl(p)] + reserved,
    )
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(old["hashes"])
    hashes.update(read(feasibility / "complete.json")["files"])
    hashes.update(read(audit / "complete.json")["files"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    for p in (
        prior / "registration.json",
        prior / "p72-complete.json",
        feasibility / "complete.json",
        audit / "complete.json",
        catalog,
        *catalogs,
    ):
        hashes[str(p)] = sha256_file(p)
    for key in ("schedule", "review_schedule"):
        if hashes.get(old[key]) != sha256_file(old[key]):
            raise ValueError("inherited schedule drift")
    config = {
        k: old[k]
        for k in (
            "repository",
            "manifests",
            "detector",
            "schedule",
            "review_schedule",
            "review_cases",
            "p55",
            "controls",
            "packages",
            "coverage",
            "seeds",
            "panel",
            "objective",
        )
    }
    config.update(
        {
            "format": "soundex-phase-temporal-p74-v1",
            "prior": str(prior),
            "feasibility": str(feasibility),
            "audit": str(audit),
            "parents": {},
            "assets": str(feasibility / "assets"),
            "binary": str(feasibility / "assets/soundex-context-replay"),
            "arms": list(OBSERVATIONS),
            "steps": STEPS,
            "hashes": hashes,
            "evaluation_catalogs": list(map(str, catalogs)),
            "question": "Does adding phase-relation information BEFORE the small temporal band backbone improve original-reference restoration versus equal-parameter magnitude-only observations? P68/P69 modified only the rotation head; P72 widened magnitude tokens. Neither answers this observation-location hypothesis.",
            "budget_rationale": "Four fresh trajectories, each16384 updates x4 matched batches, complete inherited65536-draw exposure:327680 paired crops,182 recordings,27627.408 unique supervised seconds.80:20 music/speech. Same adequate fresh-model budget asP72, not unchanged continuation or convergence proof. Finish all endpoints before scoring.",
            "optimizer": "Fresh Adam .9/.999 eps1e-8, all216802 parameters, global clip5. Same256-update1e-5->1e-4 warmup then cosine->1e-5. No intermediate selection, step veto or shortened control.",
            "preflight": "Two actual TRAIN updates per seed/arm, finite full gradients, changed relation projection, phase-history derivative and repeated-final-phase magnitude intervention; both-rate/channel native correspondence. Discard diagnostic states, start all formal trajectories fresh.",
            "selection": "Same P71 spectrum-first ruler and named controls: phase versus matched magnitude must lower source-equal native paired-music MR and independent HFSC in both seeds on the fixed P55 recheck. Retain recording bootstrap intervals and all shape/energy/dynamics/complex/absolute/stereo/identity tradeoffs. Other native music panel must also be reported; no favorable-panel selection or automatic promotion.",
            "review": "Freeze all four endpoints first. Common fit loss/history probes, source-isolated raw/native review, complete160-case P55 recheck with P71 panel, all codecs/roles/adverse sources, trained native runtime and blinded listening materials. Legacy/untreated/identityDSP separately named.",
            "scope": "Fresh matched representation experiment with UNCHANGED P67 loss. Realized-complex supervision is a separate future control, not silently introduced. Previously exposed development panels are rechecks; no independent-test, listening, convergence or device qualification.",
        }
    )
    registered(config, output / "registration.json")
    registered(
        {"sha256": sha256_file(output / "registration.json")}, output / "registration.receipt.json"
    )
    write_json_report(
        {"stage": "registered", "trajectories": 4, "steps": STEPS}, output / "progress.json"
    )


def run(output):
    if read(output / "registration.json")["format"] != "soundex-phase-temporal-p74-v1":
        raise ValueError("P74 registration required")
    run_matched(
        output,
        make_model=fresh,
        load_model=load,
        pack_model=pack,
        inspect=inspect,
        study="p74",
        candidate="phase",
        control="magnitude",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    for name in ("prior", "feasibility", "audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.feasibility.resolve(),
                args.audit.resolve(),
                args.output.resolve(),
            )
        else:
            run(args.output.resolve())
    except Exception:
        if args.output.exists():
            write_json_report(
                {"traceback": traceback.format_exc()},
                args.output / f"failure-{time.time_ns()}.json",
            )
            write_json_report(
                {"stage": "failed", "qualified": False}, args.output / "progress.json"
            )
        raise
