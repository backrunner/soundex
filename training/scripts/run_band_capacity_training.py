"""P72 fresh joint small/double band training under frozen P71 evaluation.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import importlib.metadata
import time
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_runtime import export
from evaluation.band_training import freeze_endpoints, learning_rate
from evaluation.longform_review import aligned_output
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_panel import paired_recordings, panel
from evaluation.rust_bridge import read_sxa
from evaluation.spectral_training import adam_update
from evaluation.trainability_audit import assert_reserved_isolation
from models.band_capacity import TOKEN_WIDTHS, capacity_model
from models.fused_spectrum import phase_objective
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged, jsonl
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_cross_phase_training import export_model
from scripts.run_fused_phase_pilot import loss_review
from scripts.run_joint_training import finalize
from scripts.run_longform_review import checked_receipt

STEPS = 16384


def fresh(kind, seed):
    model = capacity_model(kind, seed)
    model.study = "p72-fresh-joint-band-capacity-training"
    return model


def load(saved, _scope=None):
    if saved["format"] != "soundex-band-capacity-training-v1":
        raise ValueError("explicit P72 checkpoint required")
    model = fresh(saved["kind"], saved["seed"])
    model.load_state_dict(saved["model"], strict=True)
    return model


def pack(model, kind, seed, optimizer, history, signature):
    return {
        "format": "soundex-band-capacity-training-v1",
        "kind": kind,
        "seed": seed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "step": len(history),
        "signature": signature,
    }


def update(model, items, optimizer):
    if len(items) != 4:
        raise ValueError("four matched paired/identity batches required")

    def closure():
        values = []
        for objective, context in items:
            prediction, _ = predict_budget(model, objective.batch, context)
            value = phase_objective(prediction, objective.batch, include_fused=True)
            (value / len(items)).backward()
            values.append(float(value.detach()))
        return sum(values) / len(values)

    # The P69 helper clips/checks only phase_head: full training must use all tensors.
    return adam_update(model.parameters(), optimizer, closure)


def prepare(prior, feasibility, panel_root, output):
    if output.exists():
        raise ValueError("fresh registration required")
    for p in (
        prior / "p69-complete.json",
        feasibility / "complete.json",
        panel_root / "complete.json",
    ):
        checked_receipt(p)
    screen = read(feasibility / "assessment.json")
    if not all(
        screen[k]
        for k in ("all_compute_under_5ms", "all_service_under_hop", "all_paced_completions_on_time")
    ):
        raise ValueError("capacity runtime prerequisite failed; retain all timings")
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    if len(schedule["draws"]) != 4 * STEPS:
        raise ValueError("full matched exposure required")
    selected = {schedule["specs"][i]["row_id"] for d in schedule["draws"] for i in d}
    review = read(Path(old["review_schedule"]))
    reserved_review = [rows[review["specs"][i]["row_id"]] for d in review["review"] for i in d]
    repo = Path(old["repository"])
    catalogs = sorted((repo / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repo / "data/catalogs/current-approved.jsonl"
    assert_reserved_isolation(
        [rows[k] for k in selected],
        list(rows.values()) + jsonl(catalog),
        [r for p in catalogs for r in jsonl(p)] + reserved_review,
    )
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(old["hashes"])
    hashes.update(read(feasibility / "complete.json")["files"])
    hashes.update(read(panel_root / "complete.json")["files"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    for p in (prior / "registration.json", prior / "p69-complete.json", catalog, *catalogs):
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
        )
    }
    config.update(
        {
            "format": "soundex-band-capacity-p72-v1",
            "prior": str(prior),
            "feasibility": str(feasibility),
            "panel": str(panel_root),
            "parents": {},
            "assets": str(feasibility / "assets"),
            "binary": str(feasibility / "assets/soundex-context-replay"),
            "arms": list(TOKEN_WIDTHS),
            "token_widths": TOKEN_WIDTHS,
            "steps": STEPS,
            "hashes": hashes,
            "evaluation_catalogs": list(map(str, catalogs)),
            "question": "Does approximately twice the band representation capacity improve original-reference restoration under fresh full magnitude/phase training, equal exposure and objective? P23/P26 used a different global representation; P69 froze magnitudes. Small and double are both fresh; no architecture inference from historical-pretraining comparisons.",
            "budget_rationale": "16384 updates x4 matched batches matches established P63 fresh-model scale and consumes the complete 65536-draw schedule. 327680 paired crops, 182 recordings, 27627.408 unique supervised seconds per trajectory; 80:20 music/speech. New capacity/fully connected output hypothesis, not a longer repeat of failed phase-only training. Completion is not convergence; retain full history and final trend.",
            "objective": "Unchanged P67: balanced log1p(direct original-reference HF spectra) + log1p(raw reconstructed MR) + log1p(fused original-reference MR). Equal paired/identity roles, all full-model tensors optimized. No new loss weights, perceptual loss or DSP teacher.",
            "optimizer": "Fresh Adam .9/.999 eps1e-8 all model parameters, global clip5; four batches/update. Same256-update1e-5->1e-4 warmup/cosine->1e-5. No intermediate selection or shortened controls. Joint small204418/double408023, fixed phase hidden32 and six dilations.",
            "preflight": "Two real TRAIN updates for all four fresh seed/capacity configurations, full-parameter finite gradients/displacement, native parity both rates and channels. Diagnostic states discarded before full training. No loss-based selection.",
            "selection": read(panel_root / "registration.json")["future_capacity_rule"],
            "review": "All four endpoints frozen first; fixed fit components, raw/native source-isolated review, full160-case20-source P55 longform recheck, all raw/native P71 panel metrics with paired recording bootstrap, separate codec and role strata, all counterexamples. Actual trained native runtime grid, blinded materials, no automatic promotion.",
            "scope": "Matched fresh capacity experiment; repeated development sources are rechecks. No independent-test, listening, convergence or device qualification until separately established.",
        }
    )
    registered(config, output / "registration.json")
    registered(
        {"sha256": sha256_file(output / "registration.json")}, output / "registration.receipt.json"
    )
    write_json_report(
        {"stage": "registered", "trajectories": 4, "steps": STEPS}, output / "progress.json"
    )


def preflight(config, pool, draws, output):
    receipt = output / "preflight-complete.json"
    if receipt.exists():
        checked_receipt(receipt)
        return
    root = output / "preflight"
    if root.exists():
        raise ValueError("incomplete preflight retained")
    root.mkdir()
    items = [pool.select(d) for d in draws[:8]]
    for seed in config["seeds"]:
        for kind in config["arms"]:
            model = fresh(kind, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            updates = [update(model, items[i : i + 4], optimizer) for i in (0, 4)]
            changed = {
                k: int(torch.count_nonzero(v != before[k])) for k, v in model.state_dict().items()
            }
            if not any(v for k, v in changed.items() if k.startswith("base.backbone.encoders.")):
                raise ValueError("full magnitude encoder did not update")
            directory = root / f"{seed}-{kind}"
            export(model, directory / "export")
            parity = [
                native_parity(
                    model,
                    directory / "export/diagnostic.onnx",
                    Path(config["binary"]),
                    directory / str(rate),
                    rate,
                )
                for rate in (44100, 48000)
            ]
            atomic_save(
                pack(model, kind, seed, optimizer, updates, "preflight-only"),
                directory / "diagnostic.pth",
            )
            registered(
                {"updates": updates, "changed_tensors": changed, "native": parity},
                directory / "result.json",
            )
            print("joint preflight", seed, kind, flush=True)
    registered({"files": {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}}, receipt)


def trajectory(config, pool, draws, kind, seed, output):
    directory = output / "training" / f"{seed}-{kind}"
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest({"config": config, "kind": kind, "seed": seed})
    endpoint, latest, done = [
        directory / n for n in ("endpoint.pth", "latest.pth", "complete.json")
    ]
    if done.exists():
        info = read(done)
        if info["signature"] != signature or sha256_file(endpoint) != info["sha256"]:
            raise ValueError("completed trajectory drift")
        return {"path": str(endpoint), "sha256": info["sha256"], "scope": kind}
    model = fresh(kind, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    history = []
    if latest.exists():
        if sha256_file(latest) != read(directory / "latest.json")["sha256"]:
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            saved["signature"] != signature
            or saved["step"] != len(saved["history"])
            or not 0 <= saved["step"] <= config["steps"]
        ):
            raise ValueError("resume history/signature drift")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
    for index in range(len(history), config["steps"]):
        begin = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(index, config["steps"])
        result = update(
            model, [pool.select(d) for d in draws[4 * index : 4 * index + 4]], optimizer
        )
        history.append({"step": index + 1, **result, "seconds": time.monotonic() - begin})
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == config["steps"]:
            atomic_save(pack(model, kind, seed, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, directory / "latest.json")
            progress = {
                "stage": "training",
                "trajectory": f"{seed}-{kind}",
                "step": index + 1,
                "steps": config["steps"],
                "total_trajectories": 4,
                "seconds_per_step": history[-1]["seconds"],
            }
            write_json_report(progress, output / "progress.json")
            print(progress, flush=True)
    atomic_save(pack(model, kind, seed, optimizer, history, signature), endpoint)
    registered({"signature": signature, "sha256": sha256_file(endpoint)}, done)
    return {"path": str(endpoint), "sha256": sha256_file(endpoint), "scope": kind}


def panel_review(config, output):
    root = output / "longform-recheck"
    checked_receipt(root / "complete.json")
    saved = read(root / "summary.json")["records"]
    rows = []
    for case in read(Path(config["p55"]) / "cases.json"):
        target, rate = read_sxa(Path(case["reference"]))
        start, n = case["score_start"], case["score_length"]
        reference = target[start : start + n]
        records = []
        for row in (r for r in saved if r["id"] == case["id"]):
            if row["path"] == "untreated":
                wave, actual = read_sxa(Path(case["input"]))
                wave = wave[start : start + n]
            else:
                mode, label = row["path"].split("/", 1)
                wave, actual = read_sxa(root / "renders" / label / case["id"] / (mode + ".sxa"))
                wave = aligned_output(wave, start, n)
            if actual != rate:
                raise ValueError("panel rate mismatch")
            scored = panel(wave, reference, rate)
            records.append(
                {
                    "source": case["source"],
                    "case": case["id"],
                    "path": row["path"],
                    "codec": case["codec"],
                    "role": case["role"],
                    "scores": {**row["scores"], **scored["metrics"]},
                    "resolutions": scored["resolutions"],
                }
            )
        registered(records, output / "panel-records" / (case["id"] + ".json"))
        rows.extend(records)
    comparisons = panel_comparisons(rows, config["seeds"])
    registered(
        {"comparisons": comparisons, "scope": config["scope"], "selection": config["selection"]},
        output / "panel-assessment.json",
    )


def panel_comparisons(rows, seeds):
    comparisons = {}
    for role in ("paired", "identity"):
        for codec in ("all", *sorted({r["codec"] for r in rows})):
            subset = [
                r for r in rows if r["role"] == role and (codec == "all" or r["codec"] == codec)
            ]
            if not subset:
                continue
            for seed in seeds:
                for mode in ("raw", "native"):
                    a = f"{mode}/{seed}-double"
                    for b in (
                        f"{mode}/{seed}-small",
                        f"{mode}/legacy",
                        "untreated",
                        f"{mode}/identity_dsp",
                    ):
                        comparisons[f"{role}/{codec}/{a}-vs-{b}"] = paired_recordings(subset, a, b)
    return comparisons


def run(output):
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            sha256_file(output / "registration.json")
            != read(output / "registration.receipt.json")["sha256"]
        ):
            raise ValueError("registration drift")
        config = read(output / "registration.json")
        if any(importlib.metadata.version(k) != v for k, v in config["packages"].items()):
            raise ValueError("registered packages changed")

        def check_inputs():
            assert_catalogs_unchanged(
                list((Path(config["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
                config["evaluation_catalogs"],
            )
            verify(config["hashes"])

        write_json_report({"stage": "verifying-inputs"}, output / "progress.json")
        check_inputs()
        if (output / "p72-complete.json").exists():
            checked_receipt(output / "p72-complete.json")
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        if len(schedule["draws"]) != config["steps"] * 4:
            raise ValueError("schedule length mismatch")
        pool = LazyPool(
            schedule["specs"],
            BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=100),
        )
        write_json_report({"stage": "preflight"}, output / "progress.json")
        preflight(config, pool, schedule["draws"], output)
        entries = {
            f"{seed}-{kind}": trajectory(config, pool, schedule["draws"], kind, seed, output)
            for seed in config["seeds"]
            for kind in config["arms"]
        }
        freeze_endpoints(config, entries, output)
        check_inputs()
        items = [pool.select(d) for d in schedule["draws"][:8]]
        for label, entry in entries.items():
            model = load(torch.load(entry["path"], map_location="cpu", weights_only=False))
            registered(loss_review(model, items), output / "fit-loss-review" / (label + ".json"))
        finalize(config, entries, output, load_model=load, export_model=export_model, study="p72")
        panel_review(config, output)
        check_inputs()
        registered(
            {
                "files": {
                    str(p): sha256_file(p)
                    for p in (
                        output / "complete.json",
                        output / "panel-assessment.json",
                        output / "endpoints.json",
                        output / "preflight-complete.json",
                    )
                }
            },
            output / "p72-complete.json",
        )
        write_json_report({"stage": "complete", "qualified": False}, output / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--feasibility", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.feasibility.resolve(),
                args.panel.resolve(),
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
