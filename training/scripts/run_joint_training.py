#!/usr/bin/env python3
"""P57 matched small-model adapter versus full reconstruction training.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import onnx
import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_runtime import cost, export, native_review
from evaluation.reporting import sha256_file, write_json_report
from evaluation.spectral_training import adam_update, objective_value
from evaluation.trainability_audit import assert_reserved_isolation
from models.adapter_trainability import load_trainable
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.compare_historical_spectra import aggregate, scores
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_longform_review import review as longform_review

STEPS = 4096
RATE = 1e-7
SCOPES = ("adapter", "full")


def step(model, items, optimizer):
    if len(items) != 4:
        raise ValueError("four paired/identity batches required")

    def closure():
        values = []
        for item in items:
            value = objective_value(model, item, "common")
            (value / len(items)).backward()
            values.append(float(value.detach()))
        return sum(values) / len(items)

    return adam_update((p for p in model.parameters() if p.requires_grad), optimizer, closure)


def pack(model, start, optimizer, history, signature):
    # Runtime-compatible forward weights; training scope and history are explicit.
    return {
        "format": "soundex-spectral-adapter-v1",
        "study": "p57-joint-small-training",
        "training_scope": model.scope,
        "base_config": start["base_config"],
        "support": start["support"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "step": len(history),
        "signature": signature,
    }


def assert_frozen(model, start):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if any(
        not torch.equal(v, start["model"][n])
        for n, v in model.state_dict().items()
        if n not in trainable
    ):
        raise ValueError("frozen parameter/statistic changed")


def prepare(args):
    output, study, audit, p55 = (
        getattr(args, k).resolve() for k in ("output", "study", "audit", "review")
    )
    if (output / "registration.json").exists():
        config = read(output / "registration.json")
        if any(config[k] != str(v) for k, v in (("study", study), ("audit", audit), ("p55", p55))):
            raise ValueError("resume arguments changed")
        verify(config["hashes"])
        return config
    old = read(study / "registration.json")
    checked_receipt(audit / "complete.json")
    checked_receipt(p55 / "complete.json")
    if read(study / "progress.json").get("stage") != "complete":
        raise ValueError("completed P54 required")
    endpoints = {
        k: v for k, v in read(study / "review-registration.json").items() if k.endswith("-common")
    }
    if len(endpoints) != 2:
        raise ValueError("both predetermined common-loss seeds required")
    rows, hashes = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))
    specs = schedule["optimization"]["specs"]
    selected = {s["row_id"]: rows[s["row_id"]] for s in specs}
    root = Path(__file__).resolve().parents[1]
    catalogs = sorted((root.parent / "data/runs").glob("*/evaluation-catalog.jsonl"))
    catalog = root.parent / "data/catalogs/current-approved.jsonl"
    reserved = [json.loads(s) for p in catalogs for s in p.read_text().splitlines() if s]
    universe = list(rows.values()) + [json.loads(s) for s in catalog.read_text().splitlines() if s]
    assert_reserved_isolation(list(selected.values()), universe, reserved)
    for row in selected.values():
        for kind in ("clean", "degraded"):
            hashes[str(Path(row["_root"]) / row[kind + "_path"])] = row[kind + "_checksum"]
    previous = read(Path(old["prior"]) / "registration.json")
    assets = Path(previous["assets"])
    old_review = read(p55 / "registration.json")
    controls = {k: old_review["models"][k] for k in ("identity_dsp", "legacy")}
    for path in [
        study / "registration.json",
        study / "review-registration.json",
        study / "complete.json",
        study / "runtime-complete.json",
        audit / "registration.json",
        audit / "summary.json",
        audit / "complete.json",
        p55 / "registration.json",
        p55 / "cases.json",
        p55 / "prepared.json",
        p55 / "complete.json",
        Path(old["schedule"]),
        Path(old["review_schedule"]),
        Path(old["binary"]),
        Path(old["detector"]),
        *map(Path, controls.values()),
        *catalogs,
        catalog,
        *[assets / f"input-{r}-{c}.sxa" for r in (44100, 48000) for c in (1, 2)],
        root / "requirements.txt",
        root / "requirements-evaluation.txt",
        root / "constraints-cpu-py312.txt",
        root / "evaluation/listening.html",
    ]:
        hashes[str(path)] = sha256_file(path)
    hashes.update({v["path"]: v["sha256"] for v in endpoints.values()})
    output.mkdir(parents=True, exist_ok=True)
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(p in {".venv", "__pycache__", "outputs"} for p in relative.parts):
            continue
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path)] = sha256_file(path)
        hashes[str(target)] = sha256_file(target)
    config = {
        "format": "soundex-joint-small-p57-v1",
        "study": str(study),
        "audit": str(audit),
        "p55": str(p55),
        "parents": endpoints,
        "hashes": hashes,
        "manifests": old["manifests"],
        "schedule": old["schedule"],
        "review_schedule": old["review_schedule"],
        "review_cases": old["review_cases"],
        "coverage": old["coverage"]["optimization"],
        "detector": old["detector"],
        "binary": old["binary"],
        "controls": controls,
        "assets": str(assets),
        "steps": STEPS,
        "rate": RATE,
        "scopes": SCOPES,
        "question": "Does jointly adapting existing features and reconstruction heads improve on adapter-only continuation under identical original-reference loss, clean protection, start, rate, source order and 4096 actual updates? No parameter increase; not repeated P54 loss comparison.",
        "optimizer": "Fresh Adam for BOTH scopes; persistent moments within each run, lr 1e-7, betas .9/.999, eps 1e-8, clip 5, four batches per update. No rate search, identity reduction, veto or post-step scaling.",
        "decision_basis": "P56 full-scope directions reduce pooled TRAIN transfer MR at all three fixed rates in both seeds, but effects are tiny and individual batches regress. Retain P54's existing 1e-7 rate and all original fidelity terms; larger rates lose mixed-risk transfer. One step cannot establish capacity or quality; this matched budget tests cumulative transfer.",
        "exposure": "Same complete P54 optimization schedule reused by both scopes, after excluding all reserved source/work aliases. 81920 paired crops per trajectory; original 4:1 music/speech and clean identities. Repeated source/crops are not additional independent material.",
        "barrier": "Finish all four 4096-update endpoints, then freeze hashes before any comparative raw/native review. No early winner or budget reduction. Loss telemetry only during training. Preserve failures and exact resume moments.",
        "review": "Historical TRAIN raw/native music+speech recheck, then all-endpoint 52000-hop timing, then P55 20-source/160-case stereo recheck with all four endpoints, both parents, legacy and identity DSP. These exposed sources are NOT a new independent test. Freeze all graph hashes before any P55 recheck. Blinded materials without invented ratings; no automatic promotion.",
        "runtime": "Same existing 480809-parameter causal graph, no added history/lookahead/buffer. Measure every trained endpoint's model+DSP latency, CPU/RSS and tails. Fixed 256 frames excluded from compute; no device qualification.",
    }
    verify(hashes)
    registered(config, output / "registration.json")
    return config


def trajectory(config, pool, seed, scope, output):
    label = seed + "-" + scope
    directory = output / "training" / label
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest({"config": config, "seed": seed, "scope": scope})
    endpoint, latest = directory / "endpoint.pth", directory / "latest.pth"
    if (directory / "complete.json").exists():
        done = read(directory / "complete.json")
        if done["signature"] != signature or done["sha256"] != sha256_file(endpoint):
            raise ValueError("completed trajectory drift")
        return {"path": str(endpoint), "sha256": done["sha256"], "scope": scope}
    start = torch.load(config["parents"][seed]["path"], map_location="cpu", weights_only=False)
    saved = None
    if latest.exists():
        if sha256_file(latest) != read(directory / "latest.json")["sha256"]:
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            saved["signature"] != signature
            or saved["training_scope"] != scope
            or saved["step"] != len(saved["history"])
        ):
            raise ValueError("resume contract drift")
    model = load_trainable(saved or start, scope)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config["rate"]
    )
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
    draws = read(Path(config["schedule"]))["optimization"]["draws"]
    for index in range(len(history), config["steps"]):
        begin = time.monotonic()
        result = step(model, [pool.select(d) for d in draws[4 * index : 4 * index + 4]], optimizer)
        history.append({"step": index + 1, **result, "seconds": time.monotonic() - begin})
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == config["steps"]:
            assert_frozen(model, start)
            atomic_save(pack(model, start, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, directory / "latest.json")
            write_json_report(history, directory / "updates.json")
            progress = {
                "stage": "training",
                "trajectory": label,
                "step": index + 1,
                "steps": config["steps"],
                "seconds_per_step": history[-1]["seconds"],
            }
            write_json_report(progress, output / "progress.json")
            print(progress, flush=True)
    atomic_save(pack(model, start, optimizer, history, signature), endpoint)
    done = {"signature": signature, "sha256": sha256_file(endpoint)}
    registered(done, directory / "complete.json")
    return {"path": str(endpoint), "sha256": done["sha256"], "scope": scope}


def export_joint(model, path, scope):
    report = export(model, path)
    graph_path = path / "diagnostic.onnx"
    graph = onnx.load(graph_path)
    properties = {p.key: p.value for p in graph.metadata_props}
    properties.update(
        {"soundex.study": "p57-joint-small-training", "soundex.training_scope": scope}
    )
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, graph_path)
    report.update(
        {
            "onnx_sha256": sha256_file(graph_path),
            "study": "p57-joint-small-training",
            "training_scope": scope,
        }
    )
    write_json_report(report, path / "report.json")
    return graph_path


@torch.no_grad()
def finalize(config, entries, output):
    registered(entries, output / "endpoints.json")
    verify({e["path"]: e["sha256"] for e in entries.values()})
    rows, _ = load_sources(list(map(Path, config["manifests"])))
    schedule = read(Path(config["review_schedule"]))
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(config["detector"]), output / "review-cache", capacity=320),
    )
    items = [pool.select(d) for d in schedule["review"]]
    all_entries = {**{k + "-parent": v for k, v in config["parents"].items()}, **entries}
    graphs, summary = {}, {}
    for label, entry in all_entries.items():
        model = load_trainable(
            torch.load(entry["path"], map_location="cpu", weights_only=False),
            entry.get("scope", "adapter"),
        )
        graphs[label] = export_joint(
            model, output / "exports" / label, entry.get("scope", "p54-parent")
        )
        cases = []
        for index, item in enumerate(items):
            batch = item[0].batch
            wave, target = supervised_waves(predict_budget(model, batch, item[1])[0], batch)
            cases.append(
                {
                    "index": index,
                    "samples": config["review_cases"][index]["samples"],
                    "corpora": [
                        rows[schedule["specs"][j]["row_id"]]["corpus"]
                        for j in schedule["review"][index]
                    ],
                    "identity": batch["identity"].bool().tolist(),
                    "scores": scores(wave, target, batch["rate"]),
                }
            )
        write_json_report(
            {"checkpoint": entry, "cases": cases}, output / "raw-review" / (label + ".json")
        )
        summary[label] = aggregate(cases)
    write_json_report(summary, output / "raw-summary.json")
    graphs.update({k: Path(v) for k, v in config["controls"].items()})
    native_review(
        Path(config["binary"]), graphs, items, config["review_cases"], output / "native-review"
    )
    costs = {}
    for label in entries:
        for rate, channels, paced in (
            (44100, 1, False),
            (44100, 2, False),
            (48000, 1, False),
            (48000, 2, False),
            (48000, 2, True),
        ):
            name = f"{label}/{'paced-' if paced else ''}{rate}-{channels}"
            costs[name] = cost(
                Path(config["binary"]),
                graphs[label],
                Path(config["assets"]) / f"input-{rate}-{channels}.sxa",
                output / "native-cost" / name,
                paced=paced,
            )
    write_json_report(costs, output / "cost-summary.json")
    recheck = output / "longform-recheck"
    recheck.mkdir(exist_ok=True)
    source = Path(config["p55"])
    review_config = {
        "format": "soundex-p57-p55-recheck-v1",
        "scope": "Previously exposed P55 sources: recheck, NOT independent test or qualification",
        "models": {k: str(v) for k, v in graphs.items()},
        "sources": read(source / "registration.json")["sources"],
        "binary": config["binary"],
    }
    registered(review_config, recheck / "registration.json")
    registered(read(source / "prepared.json"), recheck / "prepared.json")
    checked_receipt(recheck / "prepared.json")
    registered(
        {
            "models": review_config["models"],
            "files": {str(p): sha256_file(p) for p in graphs.values()},
        },
        recheck / "endpoints.json",
    )
    if not (recheck / "complete.json").exists():
        longform_review(review_config, read(source / "cases.json"), recheck)
    checked_receipt(recheck / "complete.json")
    # The reused renderer keeps its P55 protocol; override only the run status's
    # interpretation, never frozen scores or the prior P55 registration.
    write_json_report(
        {
            "stage": "complete",
            "qualified": False,
            "scope": review_config["scope"],
            "listening_ratings": "pending",
        },
        recheck / "progress.json",
    )
    verify(config["hashes"])
    paths = [
        output / "registration.json",
        output / "endpoints.json",
        output / "raw-summary.json",
        output / "cost-summary.json",
        recheck / "complete.json",
    ]
    registered({"files": {str(p): sha256_file(p) for p in paths}}, output / "complete.json")
    write_json_report(
        {
            "stage": "complete",
            "qualified": False,
            "scope": "All matched training, raw/native rechecks and synthetic timing complete; no new independent test or human/device qualification.",
        },
        output / "progress.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args)
        if (output / "complete.json").exists():
            checked_receipt(output / "complete.json")
            return
        if args.prepare_only:
            print("registered", config["coverage"], flush=True)
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        pool = LazyPool(
            schedule["specs"],
            BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=100),
        )
        entries = {}
        for seed in config["parents"]:
            for scope in SCOPES:
                entries[seed + "-" + scope] = trajectory(config, pool, seed, scope, output)
        finalize(config, entries, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
