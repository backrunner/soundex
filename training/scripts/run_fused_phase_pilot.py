"""P67 matched output-support x fused-MR calibration on frozen small backbones.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import fcntl
import importlib.metadata
import time
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export, native_review
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_training import adam_update
from evaluation.trainability_audit import assert_reserved_isolation
from models.band_phase import BandPhaseCorrection
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import (
    balanced_risk,
    fused_spectrum_loss,
    phase_objective,
    prepare_fusion,
)
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import predict_budget
from scripts.assess_band_phase import compare, records
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_band_phase_pilot import SEEDS, canonical_runtime_alias, model_for, raw_review
from scripts.run_band_training import assert_catalogs_unchanged, jsonl
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt

ARMS = ("broad-common", "native-common", "broad-fused", "native-fused")


def make_model(config, seed, arm):
    broad = model_for(config["parents"][str(seed)], seed)
    mode = arm.split("-")[0]
    if mode == "broad":
        model = broad
    else:
        model = BandPhaseCorrection(broad.base, seed=seed + 65, phase_support="native").eval()
    model.study = "p67-phase-support-fusion-factorial"
    return model


def prepare(prior, audit, output):
    if output.exists():
        raise ValueError("fresh directory required; retain all attempts")
    checked_receipt(prior / "complete.json")
    checked_receipt(audit / "complete.json")
    old = read(prior / "registration.json")
    if not read(audit / "assessment.json")["passed"]:
        raise ValueError("native fusion correspondence required before calibration")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    selected = {schedule["specs"][i]["row_id"] for draw in schedule["draws"] for i in draw}
    repository = Path(old["repository"])
    catalogs = sorted((repository / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repository / "data/catalogs/current-approved.jsonl"
    assert_reserved_isolation(
        [rows[k] for k in selected],
        list(rows.values()) + jsonl(catalog),
        [r for p in catalogs for r in jsonl(p)],
    )
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(old["hashes"])
    aliases = canonical_runtime_alias(hashes, old["repository"], old["binary"])
    hashes.update(read(audit / "complete.json")["files"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    for path in (
        prior / "registration.json",
        prior / "complete.json",
        audit / "complete.json",
        *catalogs,
        catalog,
    ):
        hashes[str(path)] = sha256_file(path)
    registered({"optimization": schedule}, output / "schedule.json")
    hashes[str(output / "schedule.json")] = sha256_file(output / "schedule.json")
    p63 = read(Path(old["prior"]) / "registration.json")
    controls = {k: p63["controls"][k] for k in ("legacy", "identity_dsp")}
    for path in controls.values():
        if sha256_file(path) != hashes[path]:
            raise ValueError("frozen control drift")
    config = {
        "format": "soundex-fused-phase-p67-v1",
        "prior": str(prior),
        "audit": str(audit),
        "repository": str(repository),
        "parents": old["parents"],
        "seeds": SEEDS,
        "arms": ARMS,
        "steps": 512,
        "learning_rate": 1e-4,
        "schedule": str(output / "schedule.json"),
        "manifests": old["manifests"],
        "detector": old["detector"],
        "binary": old["binary"],
        "controls": controls,
        "packages": old["packages"],
        "evaluation_catalogs": list(map(str, catalogs)),
        "exposure": old["exposure"],
        "paired_crops_per_trajectory": old["paired_crops_per_seed"],
        "question": "P66 identified discarded phase output and fusion-dependent gains. Isolate output availability and a verified fused-MR auxiliary with a 2x2 design, rather than extend unchanged P65 or vary capacity.",
        "objective": "common = log1p(original-reference direct) + log1p(raw MR); fused adds 1.0*log1p(final-output MR). Equal paired/identity means, unchanged 80:20 music/speech. Original PCM targets for every term, including identity. No waveform-distance term or DSP teacher.",
        "optimizer": "Fresh Adam per trajectory, lr 1e-4, default betas/eps, clip 5, four batches/update; phase head only. Equal initialization within origin; no scaled displacement, early winner or adaptive weight.",
        "barrier": "Finish all eight 512-update endpoints and freeze hashes before any post-training raw/native comparison. Fixed endpoints only; no held-out scoring, no timing during training. Incomplete trajectories cannot support comparison.",
        "review": "First eight inherited TRAIN draws: 32 music/eight speech paired+identity raw/common/fused scores; same two native batches: eight music/two speech, both roles, all eight endpoints, zero controls, legacy and identity DSP. Record all per-source counterexamples; no promotion.",
        "scope": "Early optimization calibration of one 196162-parameter architecture with frozen magnitude backbones, not full training, convergence, capacity ceiling, independent generalization or release qualification.",
        "hashes": hashes,
        "inherited_runtime_aliases": aliases,
    }
    registered(config, output / "registration.json")
    print("registered eight matched 512-update phase trajectories", flush=True)


def update(model, items, optimizer, include_fused):
    def closure():
        values = []
        for objective, context in items:
            batch = objective.batch
            prediction, _ = predict_budget(model, batch, context)
            value = phase_objective(prediction, batch, include_fused=include_fused)
            (value / len(items)).backward()
            values.append(float(value.detach()))
        return sum(values) / len(values)

    if len(items) != 4:
        raise ValueError("four batches required")
    return adam_update(model.phase_head.parameters(), optimizer, closure)


@torch.no_grad()
def loss_review(model, items):
    result = []
    for objective, context in items:
        batch = objective.batch
        prediction, _ = predict_budget(model, batch, context)
        values = {
            "direct": direct_spectrum_loss(prediction, batch)["per_example"],
            "raw_mr": multires_spectrum_loss(prediction, batch)["per_example"],
            "fused_mr": fused_spectrum_loss(prediction, batch, prepare_fusion(batch))[
                "per_example"
            ],
        }
        common = values["direct"].log1p() + values["raw_mr"].log1p()
        values.update(common=common, augmented=common + values["fused_mr"].log1p())
        result.append(
            {
                "per_example": {k: v.tolist() for k, v in values.items()},
                "balanced": {
                    k: float(balanced_risk(v, batch["identity"])) for k, v in values.items()
                },
            }
        )
    return result


def run(output):
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "training").exists() or (output / "preflight").exists():
            raise ValueError("previous execution retained; no silent restart")
        config = read(output / "registration.json")
        write_json_report(
            {"stage": "verifying-inputs", "files": len(config["hashes"])}, output / "progress.json"
        )
        if any(importlib.metadata.version(k) != v for k, v in config["packages"].items()):
            raise ValueError("registered packages changed")
        catalogs = list(
            (Path(config["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")
        )
        assert_catalogs_unchanged(catalogs, config["evaluation_catalogs"])
        verify(config["hashes"])
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        specs, draws = schedule["specs"], schedule["draws"]
        if len(draws) != 4 * config["steps"]:
            raise ValueError("unequal exposure budget")
        pool = LazyPool(
            specs,
            BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=100),
        )
        items = [pool.select(d) for d in draws[:8]]
        descriptions = [
            {
                "index": i,
                "samples": {
                    "calibration": [{"row": rows[specs[j]["row_id"]], "spec": specs[j]} for j in d]
                },
            }
            for i, d in enumerate(draws[:8])
        ]
        graphs, preflight = {}, {}
        for seed in SEEDS:
            for mode in ("broad", "native"):
                model = make_model(config, seed, mode + "-common")
                stress = copy.deepcopy(model)
                with torch.no_grad():
                    stress.phase_head[-1].bias.fill_(2.0)
                root = output / "preflight" / f"{seed}-{mode}"
                export(stress, root / "export")
                preflight[f"{seed}-{mode}"] = [
                    native_parity(
                        stress,
                        root / "export/diagnostic.onnx",
                        Path(config["binary"]),
                        root / str(rate),
                        rate,
                    )
                    for rate in (44100, 48000)
                ]
            model = make_model(config, seed, "broad-common")
            label = f"{seed}-zero"
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
            write_json_report(
                raw_review(model, items, descriptions, rows, specs, draws),
                output / "raw-review" / f"{label}.json",
            )
            write_json_report(loss_review(model, items), output / "loss-review" / f"{label}.json")
        write_json_report(preflight, output / "preflight.json")
        endpoints = {}
        for seed in SEEDS:
            for arm in config["arms"]:
                label = f"{seed}-{arm}"
                model = make_model(config, seed, arm)
                frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
                optimizer = torch.optim.Adam(
                    model.phase_head.parameters(), lr=config["learning_rate"]
                )
                history, directory = [], output / "training" / label
                directory.mkdir(parents=True)
                for index in range(config["steps"]):
                    begin = time.monotonic()
                    value = update(
                        model,
                        [pool.select(d) for d in draws[4 * index : 4 * index + 4]],
                        optimizer,
                        arm.endswith("-fused"),
                    )
                    history.append(
                        {"step": index + 1, **value, "seconds": time.monotonic() - begin}
                    )
                    if index == 0 or (index + 1) % 32 == 0:
                        if any(
                            not torch.equal(v, frozen[k])
                            for k, v in model.base.state_dict().items()
                        ):
                            raise ValueError("frozen backbone changed")
                        saved = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "history": history,
                            "step": index + 1,
                            "seed": seed,
                            "arm": arm,
                        }
                        atomic_save(saved, directory / "latest.pth")
                        write_json_report(
                            {"sha256": sha256_file(directory / "latest.pth")},
                            directory / "latest.json",
                        )
                        progress = {
                            "stage": "training",
                            "trajectory": label,
                            "step": index + 1,
                            "steps": config["steps"],
                            "completed_trajectories": len(endpoints),
                            "total_trajectories": 8,
                            "seconds_per_step": history[-1]["seconds"],
                        }
                        write_json_report(progress, output / "progress.json")
                        print(progress, flush=True)
                endpoint = directory / "endpoint.pth"
                atomic_save(saved, endpoint)
                endpoints[label] = {"path": str(endpoint), "sha256": sha256_file(endpoint)}
        registered(endpoints, output / "endpoints.json")
        # All trajectories complete before any endpoint is reviewed/exported.
        for label, entry in endpoints.items():
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
            if saved["step"] != config["steps"] or sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError("endpoint barrier failed")
            model = make_model(config, saved["seed"], saved["arm"])
            model.load_state_dict(saved["model"])
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
            write_json_report(
                raw_review(model, items, descriptions, rows, specs, draws),
                output / "raw-review" / f"{label}.json",
            )
            write_json_report(loss_review(model, items), output / "loss-review" / f"{label}.json")
        graphs.update({k: Path(v) for k, v in config["controls"].items()})
        write_json_report(
            {"stage": "native-review", "completed_trajectories": 8}, output / "progress.json"
        )
        native_review(Path(config["binary"]), graphs, items, descriptions, output / "native-review")
        comparisons = {}
        for seed in SEEDS:
            for after, before in (
                ("native-common", "broad-common"),
                ("broad-fused", "broad-common"),
                ("native-fused", "native-common"),
                ("native-fused", "broad-fused"),
                *((arm, "zero") for arm in ARMS),
            ):
                name = f"{seed}-{after}-vs-{before}"
                comparisons[name] = {}
                for path, key in (("raw-review", "scores"), ("native-review", "native")):

                    def file_for(arm, path=path, seed=seed):
                        return (
                            output
                            / path
                            / (
                                f"{seed}-{arm}.json"
                                if path == "raw-review"
                                else f"{seed}-{arm}/report.json"
                            )
                        )

                    comparisons[name][path] = compare(
                        records(read(file_for(before)), key), records(read(file_for(after)), key)
                    )
        registered(
            {"comparisons": comparisons, "scope": config["scope"]}, output / "assessment.json"
        )
        verify(config["hashes"])
        assert_catalogs_unchanged(
            list((Path(config["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
            config["evaluation_catalogs"],
        )
        registered(
            {
                "files": {
                    str(p): sha256_file(p)
                    for p in output.rglob("*")
                    if p.is_file()
                    and p.suffix in {".json", ".pth", ".onnx"}
                    and "cache" not in p.parts
                    and p.name != "progress.json"
                }
            },
            output / "complete.json",
        )
        write_json_report(
            {"stage": "complete", "qualified": False, "scope": config["scope"]},
            output / "progress.json",
        )
        print(
            "P67 calibration and fixed review complete; not promotion or runtime qualification",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.audit.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report(
                {"traceback": traceback.format_exc()},
                args.output / f"failure-{time.time_ns()}.json",
            )
            write_json_report(
                {"stage": "failed", "qualified": False}, args.output / "progress.json"
            )
        raise
