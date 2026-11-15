"""P79 complete block-factorial output attribution on frozen P77 states.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export
from evaluation.parameter_factorial import GROUPS, hybrid, shapley
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings
from models.fused_spectrum import BURN, fused_stages, prepare_fusion
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_routing import add_records, exact, native, scores
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import fresh


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh factorial audit required")
    output.mkdir(parents=True)
    snapshot(output)
    checked_receipt(prior / "complete.json")
    p77 = read(prior / "registration.json")
    records = read(prior / "records.json")
    native_draws = sorted(
        {
            next(
                d
                for d in p77["fit"]
                if any(
                    r["draw"] == d and r["sample"]["row"]["sample_rate"] == rate for r in records
                )
            )
            for rate in (44100, 48000)
        }
    )
    hashes = dict(p77["hashes"])
    hashes.update(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "format": "soundex-parameter-output-p79-v1",
            "prior": str(prior),
            "fit": p77["fit"],
            "native_draws": native_draws,
            "hashes": hashes,
            "question": "Which finite base/hidden/readout changes explain P77's final spectral tradeoff? P78 first-step Adam attribution cannot identify output-level group interactions.",
            "factors": list(GROUPS),
            "budget": "All4 P74 origins; frozen P77 gain64_old versus gain64_realized_phase. All8 exact parameter-block choices, freeze32 states/graphs before scoring. No optimizer or candidate search. Eight historical fit draws for raw/proxy; first fit draw covering each native rate for actual Rust. Reuse only exact existing anchor records. Expected640 native roles, including80 retained anchor roles; identityDSP adds20 roles with10 retained. All8 draws have untreated control.",
            "assessment": "Unchanged FP64 P76 spectral/fidelity scores against original lossless targets, recording aggregation before bootstrap. Retain paired/identity and music/speech, all32 interventions, undefined scores and adverse sources. Complete three-factor Shapley marginals and nonadditivity; contributions sum to mask7-minus-mask0. Show each component against8*FP32epsilon*abs(control_mean) as a numerical scale, not audibility or new acceptance. Native/proxy correspondence retains max2^-14/RMS2e-5 bounds.",
            "scope": "Reused fit-only diagnostic; no unseen generalization, hybrid selection, full training, architecture ceiling, listening, default promotion or timing qualification. Frozen original P77 decisions remain unchanged.",
        },
        output / "registration.json",
    )


def attributed(records, prefix):
    masks = {mask: {} for mask in range(8)}
    for row in records:
        for mask in range(8):
            if row["path"] == prefix + str(mask):
                key = (row["source"], row["case"])
                if key in masks[mask]:
                    raise ValueError("duplicate intervention case")
                masks[mask][key] = row
    if not masks[0] or any(m.keys() != masks[0].keys() for m in masks.values()):
        raise ValueError("complete matched eight-way coverage required")
    synthetic = []
    for key, row in masks[0].items():
        values = {group: {} for group in (*GROUPS, "nonadditivity")}
        for metric in row["scores"]:
            contribution = shapley([masks[m][key]["scores"][metric] for m in range(8)])
            for group, value in contribution.items():
                values[group][metric] = value
        for group, metrics in values.items():
            synthetic.append({**row, "path": group, "scores": metrics})
        synthetic.append({**row, "path": "zero", "scores": {k: 0.0 for k in row["scores"]}})
    return {
        group: paired_recordings(synthetic, group, "zero", seed=79)
        for group in (*GROUPS, "nonadditivity")
    }


def assess(records, origins):
    result = {}
    for origin in origins:
        for route in ("raw", "proxy", "native"):
            for role in ("paired", "identity"):
                for corpus in ("all", "music_library", "speech_library"):
                    selected = [
                        r
                        for r in records
                        if r["role"] == role and (corpus == "all" or r["corpus"] == corpus)
                    ]
                    prefix = route + "/" + origin + "/mask"
                    anchor = paired_recordings(selected, prefix + "7", prefix + "0", seed=79)
                    for v in anchor.values():
                        v["p77_numerical_scale"] = (
                            8 * torch.finfo(torch.float32).eps * abs(v["control_mean"])
                            if "control_mean" in v
                            else None
                        )
                    value = {"routed_vs_old": anchor, "contributions": attributed(selected, prefix)}
                    for baseline in (
                        ("untreated", "native/identity_dsp")
                        if route == "native"
                        else ("untreated",)
                    ):
                        draws = {r["draw"] for r in selected if r["path"] == prefix + "0"}
                        common = [r for r in selected if r["draw"] in draws]
                        value[baseline] = {
                            "old": paired_recordings(common, prefix + "0", baseline, seed=79),
                            "routed": paired_recordings(common, prefix + "7", baseline, seed=79),
                        }
                    result[f"{origin}/{route}/{role}/{corpus}"] = value
    return result


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "interventions").exists():
        raise ValueError("existing output audit retained")
    prior = Path(config["prior"])
    p77 = read(prior / "registration.json")
    frozen = read(prior / "frozen.json")["endpoints"]
    p74 = Path(p77["p74"])
    old = read(p74 / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        p77["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), p74 / "cache", capacity=100),
    )
    items = {d: pool.select(schedule["draws"][d]) for d in config["fit"]}
    graphs, endpoints = {}, {}
    for origin in p77["endpoints"]:
        anchors = [
            torch.load(
                frozen[origin + "/gain64_" + arm]["path"], map_location="cpu", weights_only=False
            )
            for arm in ("old", "realized_phase")
        ]
        for mask in range(8):
            label = origin + "/mask" + str(mask)
            state = hybrid(anchors[0]["model"], anchors[1]["model"], mask)
            if mask in (0, 7):
                exact(state, anchors[int(mask == 7)]["model"])
            model = fresh(anchors[0]["kind"], anchors[0]["seed"]).eval()
            model.load_state_dict(state, strict=True)
            path = output / "interventions" / label / "state.pth"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    "format": "soundex-parameter-output-p79-intervention",
                    "model": state,
                    "kind": anchors[0]["kind"],
                    "seed": anchors[0]["seed"],
                    "mask": mask,
                },
                path,
            )
            endpoints[label] = {"path": str(path), "sha256": sha256_file(path)}
            if mask in (0, 7):
                graph = (
                    prior
                    / "exports"
                    / origin
                    / ("gain64_old" if mask == 0 else "gain64_realized_phase")
                    / "diagnostic.onnx"
                )
            else:
                export(model, output / "exports" / label)
                graph = output / "exports" / label / "diagnostic.onnx"
            graphs[label] = graph
            print("frozen intervention", label, flush=True)
    graphs["identity_dsp"] = Path(old["controls"]["identity_dsp"])
    registered(
        {"endpoints": endpoints, "graphs": {str(p): sha256_file(p) for p in graphs.values()}},
        output / "frozen.json",
    )
    records = []
    for row in read(prior / "records.json"):
        if row["draw"] not in config["fit"]:
            continue
        path = row["path"]
        if path in ("untreated", "native/identity_dsp"):
            if path == "untreated" or row["draw"] in config["native_draws"]:
                records.append(row)
        elif any(path.endswith("/gain64_" + arm) for arm in ("old", "realized_phase")) and (
            not path.startswith("native/") or row["draw"] in config["native_draws"]
        ):
            records.append(
                {
                    **row,
                    "path": path.rsplit("/", 1)[0]
                    + ("/mask0" if path.endswith("/gain64_old") else "/mask7"),
                }
            )
    correspondence = {}
    with torch.no_grad():
        for label, entry in endpoints.items():
            state = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = fresh(state["kind"], state["seed"]).eval()
            model.load_state_dict(state["model"], strict=True)
            for draw, (objective, context) in items.items():
                batch = objective.batch
                if any(r["path"] == "raw/" + label and r["draw"] == draw for r in records):
                    continue
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
            write_json_report(records, output / "records.json")
            print("raw/proxy", label, flush=True)
        for label, graph in graphs.items():
            for draw in config["native_draws"]:
                if any(r["path"] == "native/" + label and r["draw"] == draw for r in records):
                    continue
                batch = items[draw][0].batch
                wave, parity = native(
                    Path(old["binary"]), graph, batch, output / "native" / label / str(draw), True
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
                correspondence[label + "/" + str(draw)] = parity
                write_json_report(records, output / "records.json")
                write_json_report(correspondence, output / "correspondence.json")
            print("native", label, flush=True)
        for draw, (objective, _) in items.items():
            if any(r["path"] == "untreated" and r["draw"] == draw for r in records):
                continue
            batch = objective.batch
            region = slice(BURN * 128, batch["dry"].shape[-1] - 128)
            add_records(
                records,
                config,
                schedule,
                rows,
                draw,
                "untreated",
                scores(batch["dry"][:, region], batch["clean_wave"][:, region], batch["rate"]),
            )
    write_json_report(records, output / "records.json")
    result = {
        "attribution": assess(records, p77["endpoints"]),
        "native_correspondence_pass": all(p["passed"] for p in correspondence.values()),
        "qualified": False,
        "scope": config["scope"],
    }
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
            and (args.mode == "run" or fresh_directory)
        ):
            write_json_report({"traceback": traceback.format_exc()}, failure)
        raise
