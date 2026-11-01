"""P69 matched cross-frequency phase training with complete-endpoint review.

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
from data.causal_study import digest, load_sources
from evaluation.adapter_runtime import export
from evaluation.band_training import freeze_endpoints, fresh, learning_rate
from evaluation.band_training import load as load_base
from evaluation.reporting import sha256_file, write_json_report
from evaluation.trainability_audit import assert_reserved_isolation
from models.band_phase import BandPhaseCorrection
from scripts.assess_band_phase import compare, records
from scripts.audit_fused_spectrum import snapshot
from scripts.bench_band_tcn import native_parity
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged, jsonl
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_fused_phase_pilot import loss_review, update
from scripts.run_joint_training import finalize
from scripts.run_longform_review import checked_receipt

ARMS = ("temporal_repeat", "adjacent")
STEPS = 16384


def make_model(config, seed, kind):
    base = load_base(
        torch.load(config["parents"][str(seed)]["path"], map_location="cpu", weights_only=False)
    )
    return wrap(base, seed, kind)


def wrap(base, seed, kind):
    if kind not in ARMS:
        raise ValueError("unregistered phase observation")
    model = BandPhaseCorrection(
        base, seed=seed + 68, phase_support="native", phase_observation=kind
    ).eval()
    model.study = "p69-matched-cross-frequency-phase-training"
    return model


def load(saved, _scope=None):
    if saved["format"] == "soundex-band-training-v1":
        return wrap(load_base(saved), saved["seed"], "temporal_repeat")
    if saved["format"] != "soundex-cross-phase-training-v1":
        raise ValueError("unknown phase checkpoint")
    model = wrap(fresh("band", saved["seed"]), saved["seed"], saved["kind"])
    model.load_state_dict(saved["model"], strict=True)
    return model


def pack(model, kind, seed, optimizer, history, signature):
    return {
        "format": "soundex-cross-phase-training-v1",
        "kind": kind,
        "seed": seed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "step": len(history),
        "signature": signature,
    }


def prepare(prior, feasibility, output):
    if output.exists():
        raise ValueError("fresh registration directory required")
    checked_receipt(prior / "complete.json")
    checked_receipt(feasibility / "complete.json")
    screen = read(feasibility / "assessment.json")
    if not all(
        screen[k]
        for k in ("all_compute_under_5ms", "all_service_under_hop", "all_paced_completions_on_time")
    ):
        raise ValueError("P68 feasibility prerequisite incomplete")
    old = read(prior / "registration.json")
    p63 = Path(read(Path(old["prior"]) / "registration.json")["prior"])
    base = read(p63 / "registration.json")
    checked_receipt(p63 / "complete.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(base["schedule"]))["optimization"]
    if len(schedule["draws"]) != 4 * STEPS:
        raise ValueError("full registered crop exposure required")
    selected = {schedule["specs"][i]["row_id"] for d in schedule["draws"] for i in d}
    review = read(Path(base["review_schedule"]))
    review_rows = [rows[review["specs"][i]["row_id"]] for d in review["review"] for i in d]
    repository = Path(old["repository"])
    catalogs = sorted((repository / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = repository / "data/catalogs/current-approved.jsonl"
    assert_reserved_isolation(
        [rows[k] for k in selected],
        list(rows.values()) + jsonl(catalog),
        [r for p in catalogs for r in jsonl(p)] + review_rows,
    )
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(old["hashes"])
    hashes.update(read(feasibility / "complete.json")["files"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    for p in (
        prior / "complete.json",
        prior / "registration.json",
        p63 / "registration.json",
        p63 / "complete.json",
        feasibility / "complete.json",
        catalog,
        *catalogs,
    ):
        hashes[str(p)] = sha256_file(p)
    # The full P63 schedule and source bytes were already frozen by P67. Never
    # rebind a changed consumed file to its current hash during registration.
    for path in (base["schedule"], base["review_schedule"]):
        if hashes.get(path) != sha256_file(path):
            raise ValueError("inherited full schedule drift")
    config = {
        "format": "soundex-cross-phase-training-p69-v1",
        **{
            k: old[k]
            for k in (
                "parents",
                "seeds",
                "manifests",
                "detector",
                "repository",
                "packages",
                "controls",
            )
        },
        **{k: base[k] for k in ("schedule", "review_schedule", "review_cases", "p55")},
        "prior": str(prior),
        "feasibility": str(feasibility),
        "binary": str(feasibility / "assets/soundex-context-replay"),
        "assets": str(feasibility / "assets"),
        "hashes": hashes,
        "evaluation_catalogs": list(map(str, catalogs)),
        "arms": list(ARMS),
        "steps": STEPS,
        "coverage": base["coverage"],
        "objective": old["objective"],
        "optimized_risk": "P67 fused objective only: balanced log1p(direct original-reference) + log1p(raw MR) + log1p(final-DSP MR). Same weights for both observations; no DSP teacher or added waveform loss.",
        "optimizer": "Fresh phase-head-only Adam .9/.999 eps1e-8, clip5, four paired/identity batches/update. P63 indexed schedule: 256-update 1e-5 to 1e-4 warmup, cosine to 1e-5 at 16384. No post-step scaling or early selection. Frozen base tensors checked at every checkpoint.",
        "question": "Does observing adjacent-bin phase relationships improve restoration over equal-width repeated temporal observations? P68 establishes missing information, nonsaturated gradients and native feasibility, not learned quality.",
        "budget_rationale": "P67 phase heads saw only the first 2048 draws. Use the complete P63 65536-draw exposure, 327680 paired crops and 27627.408 unique supervised seconds per trajectory instead of recycling that small bank. 16384 matches the established fresh-training scale; both arms reset and finish equally. This is a substantial fixed comparison, not proof of convergence; report full update history and final-window trend.",
        "preflight": "Two real Adam updates for each arm/origin on first eight TRAIN draws, export and native parity at both rates/physical channels. Keep diagnostic states, discard them for full training; only finite/contract gates, no score-based choice.",
        "barrier": "All four exact 16384-update endpoints with optimizer/history/signatures freeze before any endpoint comparison. Exact persistent Adam resume, no shortened controls or intermediate winners.",
        "selection": "No automatic promotion. A useful cross-frequency direction requires lower source-equal native paired-music MR versus repeated-input control and zero parent in BOTH seeds on the reserved-source review, plus reported shape/band/energy/dynamics, identity, absolute PCM and every counterexample. Small waveform regression alone is not rejection. Reused P55 long-form results are rechecks, not independent generalization. No claim from tiny TRAIN means.",
        "review": "First eight TRAIN draws for fixed-objective components; P63 source-isolated 32-batch raw and 8-batch native panel, then P55 20-source160-case physical-stereo raw/native recheck. Four endpoints, two zero parents, legacy, identity DSP. Same original references and all per-source results. Full four-endpoint 52000-hop cost grid only after training; no device/listening qualification.",
        "scope": "Matched phase-observation training on frozen P63 magnitudes, no whole-model architecture ceiling or qualified weight claim. 80:20 music/speech, mono/left/right/mid/side views, no added lookahead or buffering.",
    }
    registered(config, output / "registration.json")
    registered(
        {"sha256": sha256_file(output / "registration.json")}, output / "registration.receipt.json"
    )
    write_json_report(
        {"stage": "registered", "trajectories": 4, "steps": STEPS}, output / "progress.json"
    )
    print("registered four matched 16384-update phase trajectories", flush=True)


def preflight(config, pool, draws, output):
    receipt = output / "preflight-complete.json"
    if receipt.exists():
        checked_receipt(receipt)
        return
    root = output / "preflight"
    if root.exists():
        raise ValueError("incomplete preflight retained; no silent retry")
    root.mkdir()
    items = [pool.select(d) for d in draws[:8]]
    for seed in config["seeds"]:
        for kind in config["arms"]:
            model = make_model(config, seed, kind)
            optimizer = torch.optim.Adam(model.phase_head.parameters(), lr=1e-5)
            updates = [update(model, items[i : i + 4], optimizer, True) for i in (0, 4)]
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
            write_json_report(
                {"updates": updates, "native": parity, "qualification": False},
                directory / "result.json",
            )
            print("preflight complete", seed, kind, flush=True)
    registered({"files": {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}}, receipt)


def trajectory(config, pool, draws, kind, seed, output):
    directory = output / "training" / f"{seed}-{kind}"
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest({"config": config, "kind": kind, "seed": seed})
    endpoint, latest = directory / "endpoint.pth", directory / "latest.pth"
    done = directory / "complete.json"
    if done.exists():
        info = read(done)
        if info["signature"] != signature or sha256_file(endpoint) != info["sha256"]:
            raise ValueError("completed trajectory drift")
        return {"path": str(endpoint), "sha256": info["sha256"], "scope": kind}
    model = make_model(config, seed, kind)
    frozen = copy.deepcopy(model.base.state_dict())
    optimizer = torch.optim.Adam(model.phase_head.parameters(), lr=1e-5)
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
            model, [pool.select(d) for d in draws[4 * index : 4 * index + 4]], optimizer, True
        )
        history.append({"step": index + 1, **result, "seconds": time.monotonic() - begin})
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == config["steps"]:
            if any(not torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items()):
                raise ValueError("frozen magnitude backbone changed")
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


def export_model(model, directory, _kind):
    export(model, directory)
    return directory / "diagnostic.onnx"


def assess(config, output):
    comparisons = {}
    for seed in config["seeds"]:
        for after, before in (
            (f"{seed}-adjacent", f"{seed}-temporal_repeat"),
            *((f"{seed}-{kind}", f"{seed}-parent") for kind in ARMS),
        ):
            result = {}
            for folder, key in (("raw-review", "scores"), ("native-review", "native")):

                def rows(label, folder=folder, key=key):
                    path = (
                        output
                        / folder
                        / (label + ".json" if folder == "raw-review" else label + "/report.json")
                    )
                    return records(read(path), key)

                result[folder] = compare(rows(before), rows(after))
            comparisons[after + "-vs-" + before] = result
    registered(
        {"comparisons": comparisons, "scope": config["scope"], "selection": config["selection"]},
        output / "assessment.json",
    )


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

        write_json_report(
            {"stage": "verifying-inputs", "files": len(config["hashes"])}, output / "progress.json"
        )
        check_inputs()
        if (output / "p69-complete.json").exists():
            checked_receipt(output / "p69-complete.json")
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        if len(schedule["draws"]) != config["steps"] * 4:
            raise ValueError("registered schedule length mismatch")
        pool = LazyPool(
            schedule["specs"],
            BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=100),
        )
        write_json_report(
            {"stage": "preflight", "input_hashes_verified": len(config["hashes"])},
            output / "progress.json",
        )
        preflight(config, pool, schedule["draws"], output)
        entries = {}
        for seed in config["seeds"]:
            for kind in config["arms"]:
                entries[f"{seed}-{kind}"] = trajectory(
                    config, pool, schedule["draws"], kind, seed, output
                )
        freeze_endpoints(config, entries, output)
        check_inputs()
        items = [pool.select(d) for d in schedule["draws"][:8]]
        for label, entry in {
            **{k + "-parent": v for k, v in config["parents"].items()},
            **entries,
        }.items():
            model = load(torch.load(entry["path"], map_location="cpu", weights_only=False))
            write_json_report(
                loss_review(model, items), output / "fit-loss-review" / (label + ".json")
            )
        write_json_report(
            {"stage": "reviewing-frozen-endpoints", "trajectories": 4}, output / "progress.json"
        )
        finalize(config, entries, output, load_model=load, export_model=export_model, study="p69")
        assess(config, output)
        check_inputs()
        registered(
            {
                "files": {
                    str(p): sha256_file(p)
                    for p in (
                        output / "complete.json",
                        output / "assessment.json",
                        output / "endpoints.json",
                        output / "preflight-complete.json",
                    )
                }
            },
            output / "p69-complete.json",
        )
        write_json_report({"stage": "complete", "qualified": False}, output / "progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--feasibility", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.feasibility.resolve(), args.output.resolve())
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
