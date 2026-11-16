"""P83 matched finite Adam trajectories for direct/raw/fused circular relations.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export
from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import canonical_model_state, continued, stepper
from evaluation.realized_relation_updates import (
    ARMS,
    STEPS,
    auxiliary,
    draws_for_step,
    fit_scores,
    scores_v2,
)
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_evidence import VERSION as PANEL_VERSION
from evaluation.restoration_evidence import describe_comparison, strict_paired_recordings
from evaluation.rust_bridge import read_sxa
from models.fused_spectrum import BURN, fused_stages, prepare_fusion
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_routing import add_records, exact, native
from scripts.audit_realized_relations import native_check
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import fresh

VERSION = "soundex-realized-relation-updates-p83-v1"
TRACE_ROLES = (0, 4, 5, 9)  # paired/identity first music and speech roles


def prepare(prior, balance, controls, output):
    if output.exists():
        raise ValueError("fresh experiment directory required")
    for root in (prior, balance, controls):
        checked_receipt(root / "complete.json")
    p82 = read(prior / "registration.json")
    p80 = Path(p82["prior"])
    hashes = dict(p82["hashes"])
    for root in (prior, balance, controls):
        hashes.update(read(root / "complete.json")["files"])
        hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    # Literal historical paths are receipt identities, including any symlinks.
    native_controls = {}
    for label in ["identity_dsp", *(origin + "/parent" for origin in p82["endpoints"])]:
        native_controls[label] = {}
        for draw in p82["native_draws"]:
            paths = []
            for role in range(10):
                suffix = f"/native/{label}/{draw}/{role}/output.sxa"
                matches = [name for name in hashes if name.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f"unique frozen native control required: {suffix}")
                if str(Path(matches[0]).with_name("input.sxa")) not in hashes:
                    raise ValueError("frozen control input must be hash-bound")
                paths.append(matches[0])
            native_controls[label][str(draw)] = paths
    output.mkdir(parents=True)
    snapshot(output)
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    protocol = Path(__file__).resolve().parents[2] / "docs/evaluation-protocol.md"
    hashes[str(protocol)] = sha256_file(protocol)
    registered(
        {
            **{
                k: p82[k]
                for k in (
                    "p74",
                    "endpoints",
                    "fit",
                    "transfer",
                    "native_draws",
                    "coverage",
                    "evaluation_catalogs",
                )
            },
            "format": VERSION,
            "panel": PANEL_VERSION,
            "prior": str(prior),
            "p80": str(p80),
            "hashes": hashes,
            "native_controls": native_controls,
            "arms": list(ARMS),
            "steps": STEPS,
            "gain": 64,
            "trace_roles": list(TRACE_ROLES),
            "changed_hypothesis": "P82 establishes distinct original-reference relational gradients after reconstruction and remaining mixed-objective conflicts. P83 tests actual persistent Adam and output transfer for common-target direct/raw/fused circular losses, not another unchanged P80 recipe or a capacity comparison.",
            "loss": "Old direct+raw+fused spectral objective for every parameter; only phase_head additionally receives fixed0.1*log1p(mean instantaneous/frequency/time circular relations). P82 common original-PCM Hann256/hop128 target, frames89..T-3, support and floors unchanged. No target limiting, magnitude detachment, loss-weight selection or inferred phase-sign quality veto.",
            "optimizer": "All4 frozen P74 origins, copied persistent FP32 Adam, original base/hidden LR1e-5 and betas(.9,.999), canonical global clip5. Same P77 gain64 readout step and equivalent W=64U coordinates in every arm. No new multiplier, reset, line search or early stopping. Old arm at step2 must exactly reproduce P80 old canonical model, coordinate model and optimizer; that checkpoint is not a quality-selection endpoint.",
            "budget_rationale": "16 updates/trajectory x4 arms x4 origins=256 updates, four draws/update. Alternate the two fixed four-draw groups for8 complete passes of the same40 recording bank. In Adam's first-moment recursion the original m0 coefficient is .9^16=0.185302 versus .9^2=.81; this tests accumulated new-direction contributions beyond the inherited-first-moment-dominated initial pair of steps. Second-moment history remains .999^16=0.984119. This is not a convergence or sufficient full-training budget, and adds no unique source duration.",
            "freeze": "Finish all16 trajectories and freeze all16 endpoint hashes and all16 ONNX graph hashes before fit/output/transfer scoring. No intermediate quality scores, best-checkpoint selection, discarded endpoint or shortened control. Keep every update record and all old step2 exact-control results.",
            "assessment": "Protocol-v2 strict schemas and recording-cluster paired bootstrap2000 seed83. Fixed main comparison raw-vs-old native paired transfer at all4 origins; also direct/fused, raw-vs-direct, fused-vs-direct, each-vs-parent, untreated and identityDSP. MR/HFSC primary; squared shape, envelope, within-band and envelope dynamics, coherent/absolute/preserved-band errors separate. Music/speech and paired/identity never merged away. Inspect both signs and intervals without epsilon/phase veto; no automatic promotion or full-training approval.",
            "native": "800 fresh native roles:16 endpoints x5 draws x10 roles. Trace roles0,4,5,9 for paired/identity music/speech:320 PCM and analytic phase-risk correspondence checks under unchanged P67/P82 bounds. Reuse250 hash-bound parent/identityDSP outputs after exact input/rate/timeline checks and rescore in v2; do not retry historical timing. Original native-v1 scores must reproduce exactly. Inference graph structure and fixed buffering unchanged; exports are experimental, no timing/device qualification.",
            "fit_review": "Score all20 frozen parent/endpoint models on all8 fit draws using the same FP64 common objective and all3 relation rulers applied to lifted FP32 predictions. Compare common-risk and each own-risk separately; never compare differently weighted totals. Raw and proxy wave scores use actual FP32 reconstruction lifted to FP64. All scores target original lossless PCM.",
            "scope": "40 fit recordings:32music8speech,18.724499s unique MR support.8 passes do not multiply unique duration.20 previously exposed transfer recordings:16music4speech,9.283955s. Native fit only draw2/five recordings; raw/proxy fit all8/forty. Source/work split retained within this diagnostic, but every source was historically eligible for training. Not unseen generalization, physical stereo, full-library training, convergence, blind listening or default-weight qualification. Historical legacy is not rescored.",
        },
        output / "registration.json",
    )
    print("registered16 trajectories x16 updates,800 native roles", flush=True)


def compare(records, config, *, seed=83):
    comparisons, interpretations = {}, {}
    pairs = (
        [(a, "old") for a in ARMS[1:]]
        + [("raw", "direct"), ("fused", "direct")]
        + [(a, "parent") for a in ARMS]
    )
    for population, role, corpus, route, origin in itertools.product(
        ("fit", "transfer"),
        ("paired", "identity"),
        ("all", "music_library", "speech_library"),
        ("raw", "proxy", "native"),
        config["endpoints"],
    ):
        selected = [
            r
            for r in records
            if r["population"] == population
            and r["role"] == role
            and (corpus == "all" or r["corpus"] == corpus)
        ]
        paths = [
            (f"{route}/{origin}/{a}", f"{route}/{origin}/{b}", f"{a}-vs-{b}") for a, b in pairs
        ]
        if route == "native":
            # Untreated is available on all draws, but native fit covers only one.
            selected = [r for r in selected if r["draw"] in config["native_draws"]]
            paths += [
                (f"native/{origin}/{a}", b, f"{a}-vs-{b}")
                for a, b in itertools.product(ARMS, ("untreated", "native/identity_dsp"))
            ]
        for candidate, control, label in paths:
            key = f"{population}/{role}/{corpus}/{route}/{origin}/{label}"
            values = strict_paired_recordings(selected, candidate, control, seed=seed)
            comparisons[key], interpretations[key] = values, describe_comparison(values)
    return {
        "comparisons": comparisons,
        "interpretations": interpretations,
        "qualified": False,
        "scope": config["scope"],
    }


def compare_fit(records, config, *, seed=83):
    comparisons = {}
    pairs = (
        [(a, "old") for a in ARMS[1:]]
        + [("raw", "direct"), ("fused", "direct")]
        + [(a, "parent") for a in ARMS]
    )
    for origin, role, corpus in itertools.product(
        config["endpoints"], ("paired", "identity"), ("all", "music_library", "speech_library")
    ):
        selected = [
            r for r in records if r["role"] == role and (corpus == "all" or r["corpus"] == corpus)
        ]
        for a, b in pairs:
            comparisons[f"{origin}/{role}/{corpus}/{a}-vs-{b}"] = strict_paired_recordings(
                selected, f"{origin}/{a}", f"{origin}/{b}", seed=seed
            )
    return {
        "comparisons": comparisons,
        "scope": "Frozen endpoint fixed-fit risks under identical separate rulers; not generalization.",
    }


def frozen_native(paths, batch):
    waves = []
    if len(paths) != len(batch["pcm"]):
        raise ValueError("native control role coverage changed")
    for i, path in enumerate(map(Path, paths)):
        original, rate = read_sxa(path.with_name("input.sxa"))
        wave, output_rate = read_sxa(path)
        if (
            rate != int(batch["rate"][i, 0])
            or output_rate != rate
            or original.shape != wave.shape
            or original.shape != (len(batch["pcm"][i]), 1)
        ):
            raise ValueError("reused native control timeline mismatch")
        if not torch.equal(torch.from_numpy(original[:, 0]), batch["pcm"][i]):
            raise ValueError("reused native control input differs")
        waves.append(torch.from_numpy(wave[:, 0]))
    return torch.stack(waves)


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "updates").exists() or (output / "frozen.json").exists():
        raise ValueError("existing execution retained; do not silently restart")
    p74, p80 = Path(config["p74"]), Path(config["p80"])
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
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"] + config["transfer"]}
    previous = read(p80 / "frozen.json")["endpoints"]
    endpoints, exact_controls = {}, {}
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for arm in ARMS:
            model, optimizer = continued(saved, config["gain"])
            history = []
            for step in range(config["steps"]):
                draws = draws_for_step(config["fit"], step)
                result = update(
                    model,
                    [items[d] for d in draws],
                    optimizer,
                    arm,
                    adam_step=stepper(model),
                    auxiliary_fn=auxiliary if arm != "old" else None,
                )
                history.append({"step": step + 1, "draws": draws, **result})
                registered(history[-1], output / "updates" / origin / arm / f"{step + 1:02d}.json")
                if arm == "old" and step == 1:
                    control = torch.load(
                        previous[origin + "/old"]["path"], map_location="cpu", weights_only=False
                    )
                    exact(canonical_model_state(model), control["model"])
                    exact(model.state_dict(), control["coordinate_model"])
                    exact(optimizer.state_dict(), control["coordinate_optimizer"])
                    exact_controls[origin] = True
                print("update", origin, arm, step + 1, flush=True)
            name = origin + "/" + arm
            path = output / "proposals" / name / "endpoint.pth"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    "format": VERSION,
                    "kind": saved["kind"],
                    "seed": saved["seed"],
                    "model": canonical_model_state(model),
                    "coordinate_model": model.state_dict(),
                    "coordinate_optimizer": optimizer.state_dict(),
                    "updates": history,
                    "parent": entry,
                    "steps": config["steps"],
                    "gain": config["gain"],
                    "arm": arm,
                },
                path,
            )
            endpoints[name] = {"path": str(path), "sha256": sha256_file(path)}
    registered(
        {"endpoints": endpoints, "step2_old_controls_exact": exact_controls}, output / "frozen.json"
    )
    graphs = {}
    with torch.no_grad():
        for name, entry in endpoints.items():
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = fresh(saved["kind"], saved["seed"]).eval()
            model.load_state_dict(saved["model"], strict=True)
            export(model, output / "exports" / name)
            graphs[name] = output / "exports" / name / "diagnostic.onnx"
        registered({str(p): sha256_file(p) for p in graphs.values()}, output / "native-graphs.json")
        print("all16 endpoints and graphs frozen before scoring", flush=True)
        records, fit_records, parity, reused = [], [], {}, []
        all_models = {**endpoints, **{k + "/parent": v for k, v in config["endpoints"].items()}}
        historical = {(r["path"], r["case"]): r["scores"] for r in read(p80 / "records.json")}
        for name, entry in all_models.items():
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = fresh(saved["kind"], saved["seed"]).eval()
            model.load_state_dict(saved["model"], strict=True)
            for draw, (objective, context) in items.items():
                batch = objective.batch
                prediction, _ = predict_budget(model, batch, context)
                raw, reference = supervised_waves(prediction, batch)
                final = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
                region = slice(BURN * 128, final.shape[-1] - 128)
                for route, wave in (("raw", raw), ("proxy", final[:, region])):
                    add_records(
                        records,
                        config,
                        schedule,
                        rows,
                        draw,
                        route + "/" + name,
                        scores_v2(wave, reference, batch["rate"]),
                    )
                if draw in config["fit"]:
                    add_records(
                        fit_records,
                        config,
                        schedule,
                        rows,
                        draw,
                        name,
                        fit_scores(prediction, batch),
                    )
            for draw in config["native_draws"]:
                batch = items[draw][0].batch
                if name.endswith("/parent"):
                    wave = frozen_native(config["native_controls"][name][str(draw)], batch)
                else:
                    wave, check = native(
                        Path(old["binary"]),
                        graphs[name],
                        batch,
                        output / "native" / name / str(draw),
                        True,
                        trace_roles=TRACE_ROLES,
                    )
                    parity[f"{name}/{draw}"] = check
                    for role in TRACE_ROLES:
                        current = native_check(
                            output / "native" / name / str(draw) / str(role),
                            {
                                k: v[role : role + 1]
                                for k, v in batch.items()
                                if isinstance(v, torch.Tensor)
                            },
                            check["roles"][str(role)]["native"][0],
                        )
                        registered(
                            current,
                            output / "relation-correspondence" / name / str(draw) / f"{role}.json",
                        )
                    if not check["passed"]:
                        raise ValueError("native PCM correspondence failed; retain every case")
                region = slice(BURN * 128, wave.shape[-1] - 128)
                values = scores_v2(wave[:, region], batch["clean_wave"][:, region], batch["rate"])
                if name.endswith("/parent"):
                    for role, value in enumerate(values):
                        reference = historical[("native/" + name, f"{draw}/{role}")]
                        if any(value[k] != v for k, v in reference.items()):
                            raise ValueError("parent native historical metric changed")
                    reused.append(name + "/" + str(draw))
                add_records(records, config, schedule, rows, draw, "native/" + name, values)
            write_json_report(records, output / "records.json")
            write_json_report(fit_records, output / "fit-records.json")
            write_json_report(parity, output / "correspondence.json")
            print("scored all paths", name, flush=True)
        for draw, (objective, _) in items.items():
            batch = objective.batch
            region = slice(BURN * 128, batch["dry"].shape[-1] - 128)
            add_records(
                records,
                config,
                schedule,
                rows,
                draw,
                "untreated",
                scores_v2(batch["dry"][:, region], batch["clean_wave"][:, region], batch["rate"]),
            )
            if draw in config["native_draws"]:
                wave = frozen_native(config["native_controls"]["identity_dsp"][str(draw)], batch)
                values = scores_v2(wave[:, region], batch["clean_wave"][:, region], batch["rate"])
                for role, value in enumerate(values):
                    reference = historical[("native/identity_dsp", f"{draw}/{role}")]
                    if any(value[k] != v for k, v in reference.items()):
                        raise ValueError("identity DSP historical metric changed")
                reused.append("identity_dsp/" + str(draw))
                add_records(records, config, schedule, rows, draw, "native/identity_dsp", values)
    write_json_report(records, output / "records.json")
    write_json_report(fit_records, output / "fit-records.json")
    registered(
        {
            "reused_controls_exact": reused,
            "native_batches": len(parity),
            "native_role_checks": sum(len(v["roles"]) for v in parity.values()),
        },
        output / "control-assessment.json",
    )
    registered(compare(records, config), output / "assessment.json")
    registered(compare_fit(fit_records, config), output / "fit-assessment.json")
    verify(config["hashes"])
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
    verify(read(output / "native-graphs.json"))
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
    print("complete:256 updates,800 new native roles,320 traced correspondence checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--balance", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh_directory = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.balance.resolve(),
                args.controls.resolve(),
                args.output.resolve(),
            )
        else:
            run(args.output.resolve())
    except BaseException:
        if (
            args.output.exists()
            and (fresh_directory or args.mode == "run")
            and not (args.output / "failure.json").exists()
        ):
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
