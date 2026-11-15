"""P77 fixed physical phase-readout step with equivalent persistent-Adam controls.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export
from evaluation.phase_routed_update import update
from evaluation.phase_update_coordinates import (
    canonical_model_state,
    canonical_optimizer_state,
    continued,
    stepper,
)
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings
from models.fused_spectrum import BURN, fused_stages, prepare_fusion
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_routing import add_records, exact, native, scores
from scripts.audit_phase_routing import continued as original
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import fresh

OBJECTIVES = ("old", "realized_phase")


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh directory required")
    output.mkdir(parents=True)
    snapshot(output)
    checked_receipt(prior / "complete.json")
    old = read(prior / "registration.json")
    hashes = dict(old["hashes"])
    hashes.update(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    config = {
        k: old[k]
        for k in ("endpoints", "fit", "transfer", "native_draws", "coverage", "evaluation_catalogs")
    }
    config.update(
        {
            "format": "soundex-phase-step-p77-v1",
            "prior": str(prior),
            "p74": old["prior"],
            "hashes": hashes,
            "objectives": list(OBJECTIVES),
            "gains": [1, 64],
            "steps": 2,
            "question": "Does a physically larger last-layer phase update make the realized auxiliary observable/useful, after proving coordinate and persistent-Adam equivalence? P76 increments were near native FP32 discrepancy; P49 instead reduced oversized frozen-adapter steps.",
            "coordinates": "Canonical W=64U in only phase_head.2 weight/bias. mU=64mW, vU=4096vW, epsU=64epsW; lrU=lrW/64 for unit gain. Clip the full canonical gradient norm, then restore coordinate gradients. All other parameters and moments unchanged. Gain64 changes only that group's learning rate, with the structural64 factor fixed before scoring; not a post-step multiplier or parameter expansion.",
            "equivalence": "Both objectives, all4 origins: exact starting forward state, exact canonical model AND Adam tensors after each of two steps versus an original-coordinate duplicate, and exact final equality with P76 saved endpoints. Reuse only those proven-equivalent P76 outputs; no native control rerun. Preserve any failure, no tolerance loosening.",
            "budget": "16 coordinate proposals x2 updates plus8 original-coordinate controls x2 =48 diagnostic updates. Freeze every proposal before new output scoring. Eight gain64 endpoints receive12 raw/proxy draws and5 native draws (400 mono roles); reuse hash-bound P76 unit controls/parents/identityDSP/untreated. Same TRAIN source/work partition, original references, loss, crops, codecs,80:20 mix and warmup.",
            "selection": "No automatic training/promotion. Compare gain64 realized versus gain64 old, and both against their same-objective unit controls; retain interaction, raw/proxy/native, music/speech, identity, all recording counterexamples. Positive advancement requires native paired-transfer MR and independentHFSC decreases greater than8*FP32epsilon*abs(control_mean), and realized complex decrease, at every origin, without undefined scores. This prospective numerical ruler comes from P49, not an audibility threshold or retroactive P76 reclassification. No coefficient selection on transfer.",
            "scope": "Two-step conditioning diagnostic, not adequate training, convergence, architecture ceiling, independent generalization, listening or device qualification. Original P76 counterexamples remain; no native timing retry. Canonical exports use unchanged forward/runtime ABI.",
        }
    )
    registered(config, output / "registration.json")
    print("registered P77 gains1/64, two objectives, four origins", flush=True)


def assess(records, config):
    comparisons = {}
    pairs = (
        ("gain64_realized_phase", "gain64_old"),
        ("gain64_old", "unit_old"),
        ("gain64_realized_phase", "unit_realized_phase"),
        ("unit_realized_phase", "unit_old"),
    )
    for population in ("fit", "transfer"):
        for role in ("paired", "identity"):
            for corpus in ("all", "music_library", "speech_library"):
                selected = [
                    r
                    for r in records
                    if r["population"] == population
                    and r["role"] == role
                    and (corpus == "all" or r["corpus"] == corpus)
                ]
                for endpoint in config["endpoints"]:
                    for route in ("raw", "proxy", "native"):
                        for candidate, control in pairs:
                            key = f"{population}/{role}/{corpus}/{route}/{endpoint}/{candidate}-vs-{control}"
                            comparisons[key] = paired_recordings(
                                selected,
                                f"{route}/{endpoint}/{candidate}",
                                f"{route}/{endpoint}/{control}",
                            )
    gates = {}
    for endpoint in config["endpoints"]:
        v = comparisons[
            f"transfer/paired/all/native/{endpoint}/gain64_realized_phase-vs-gain64_old"
        ]
        gates[endpoint] = (
            all(
                v[k].get("mean_delta", float("inf"))
                < -8 * torch.finfo(torch.float32).eps * abs(v[k].get("control_mean", 0))
                and not v[k]["undefined_sources"]
                for k in ("mr", "hf_sc")
            )
            and v["realized_complex"].get("mean_delta", float("inf")) < 0
            and not v["realized_complex"]["undefined_sources"]
        )
    return {
        "comparisons": comparisons,
        "joint_transfer_gate": gates,
        "qualified": False,
        "scope": config["scope"],
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "proposals").exists():
        raise ValueError("existing proposals retained")
    prior, p74 = Path(config["prior"]), Path(config["p74"])
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
    previous = read(prior / "frozen.json")["endpoints"]
    endpoints, checks = {}, {}
    for label, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        for objective in OBJECTIVES:
            for gain in config["gains"]:
                model, optimizer = continued(saved, gain)
                if gain == 1:
                    control, control_optimizer = original(saved)
                history = []
                for step in range(2):
                    selected = [items[i] for i in config["fit"][4 * step : 4 * step + 4]]
                    if gain == 1:
                        with torch.no_grad():
                            for item in selected:
                                exact(
                                    predict_budget(model, item[0].batch, item[1])[0],
                                    predict_budget(control, item[0].batch, item[1])[0],
                                )
                    history.append(
                        update(model, selected, optimizer, objective, adam_step=stepper(model))
                    )
                    if gain == 1:
                        update(control, selected, control_optimizer, objective)
                        exact(canonical_model_state(model), control.state_dict())
                        exact(
                            canonical_optimizer_state(model, optimizer, saved["optimizer"]),
                            control_optimizer.state_dict(),
                        )
                name = label + "/" + ("unit_" if gain == 1 else "gain64_") + objective
                if gain == 1:
                    prior_endpoint = torch.load(
                        previous[label + "/" + objective]["path"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    exact(canonical_model_state(model), prior_endpoint["model"])
                    exact(
                        canonical_optimizer_state(model, optimizer, saved["optimizer"]),
                        prior_endpoint["optimizer"],
                    )
                    checks[name] = {
                        "canonical_steps_bitwise_equal": 2,
                        "p76_endpoint_bitwise_equal": True,
                    }
                path = output / "proposals" / name / "endpoint.pth"
                path.parent.mkdir(parents=True)
                torch.save(
                    {
                        "format": "soundex-phase-step-p77-v1",
                        "kind": saved["kind"],
                        "seed": saved["seed"],
                        "model": canonical_model_state(model),
                        "coordinate_model": model.state_dict(),
                        "coordinate_optimizer": optimizer.state_dict(),
                        "parent": entry,
                        "gain": gain,
                        "objective": objective,
                        "updates": history,
                        "steps": 2,
                    },
                    path,
                )
                endpoints[name] = {"path": str(path), "sha256": sha256_file(path), "gain": gain}
                print("frozen", name, flush=True)
    registered({"endpoints": endpoints, "equivalence": checks}, output / "frozen.json")
    records = []
    for row in read(prior / "records.json"):
        path = row["path"]
        if path.endswith("/direct_phase"):
            continue
        for objective in OBJECTIVES:
            if path.endswith("/" + objective):
                path = path[: -(len(objective))] + "unit_" + objective
                break
        records.append({**row, "path": path})
    graphs, parity = {}, {}
    with torch.no_grad():
        for label, entry in endpoints.items():
            if entry["gain"] == 1:
                continue
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = fresh(saved["kind"], saved["seed"])
            model.load_state_dict(saved["model"], strict=True)
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
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
                        route + "/" + label,
                        scores(wave, reference, batch["rate"]),
                    )
            print("proxy scored", label, flush=True)
        registered({str(p): sha256_file(p) for p in graphs.values()}, output / "native-graphs.json")
        for label, graph in graphs.items():
            for draw in config["native_draws"]:
                batch = items[draw][0].batch
                wave, correspondence = native(
                    Path(old["binary"]),
                    graph,
                    batch,
                    output / "native" / label / str(draw),
                    draw == config["fit"][0],
                )
                region = slice(BURN * 128, wave.shape[-1] - 128)
                add_records(
                    records,
                    config,
                    schedule,
                    rows,
                    draw,
                    "native/" + label,
                    scores(wave[:, region], batch["clean_wave"][:, region], batch["rate"]),
                )
                if correspondence is not None:
                    parity[label] = correspondence
            write_json_report(records, output / "records.json")
            write_json_report(parity, output / "correspondence.json")
            print("native scored", label, flush=True)
    result = assess(records, config)
    result["native_correspondence_pass"] = all(p["passed"] for p in parity.values())
    registered(result, output / "assessment.json")
    if not result["native_correspondence_pass"]:
        raise ValueError("native correspondence failure retained")
    verify(config["hashes"])
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
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
    print("complete", result["joint_transfer_gate"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh_directory = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        failure = args.output / "failure.json"
        if (
            args.output.exists()
            and not failure.exists()
            and (args.mode != "prepare" or fresh_directory)
        ):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
