#!/usr/bin/env python3
"""P56 frozen-endpoint trainability, identity risk and Adam-history attribution.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from evaluation.trainability_audit import assert_reserved_isolation, gradients, measure
from models.adapter_trainability import SCOPES, load_trainable
from models.restoration_head import predict_budget
from models.spectral_adapter import load_adapter
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_guarded_transfer import coverage

RATES = (1e-7, 1e-6, 1e-5)
IDENTITY_WEIGHTS = (1.0, 0.0)


def save_proposal(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(value, path)
    return {"path": str(path), "sha256": sha256_file(path)}


def prepare(study, output):
    if output.exists():
        raise ValueError("fresh output required; retain failed or completed diagnostics")
    config = read(study / "registration.json")
    if read(study / "progress.json").get("stage") != "complete":
        raise ValueError("all P54 training/review/timing must finish")
    endpoints = {
        k: v for k, v in read(study / "review-registration.json").items() if k.endswith("-common")
    }
    if len(endpoints) != 2:
        raise ValueError("both common-objective seeds required")
    verify({e["path"]: e["sha256"] for e in endpoints.values()})
    rows, hashes = load_sources(list(map(Path, config["manifests"])))
    schedule = read(Path(config["schedule"]))
    selected, roles = [], {}
    for role in ("optimization", "calibration"):
        source = schedule[role]
        indices = [i * (len(source["draws"]) - 1) // 3 for i in range(4)]
        specs, draws = [], []
        for index in indices:
            draw = []
            for j in source["draws"][index]:
                draw.append(len(specs))
                specs.append(source["specs"][j])
            draws.append(draw)
        samples = [{"spec": s, "row": rows[s["row_id"]]} for s in specs]
        roles[role] = {
            "indices": indices,
            "specs": specs,
            "draws": draws,
            "samples": samples,
            "coverage": coverage([{"samples": {"registered": samples}}], "registered"),
        }
        selected.extend(s["row"] for s in samples)
    for field in ("track_id", "source_checksum"):
        sets = [{s["row"][field] for s in r["samples"]} for r in roles.values()]
        if sets[0] & sets[1]:
            raise ValueError("optimization/calibration recording overlap")
    root = Path(__file__).resolve().parents[1]
    catalogs = sorted((root.parent / "data/runs").glob("*/evaluation-catalog.jsonl"))
    if not catalogs:
        raise ValueError("registered evaluation reservation catalogs required")
    reserved = [json.loads(line) for p in catalogs for line in p.read_text().splitlines() if line]
    catalog = root.parent / "data/catalogs/current-approved.jsonl"
    universe = list(rows.values()) + [json.loads(s) for s in catalog.read_text().splitlines() if s]
    assert_reserved_isolation(selected, universe, reserved)
    assert_reserved_isolation(
        [s["row"] for s in roles["optimization"]["samples"]],
        universe,
        [s["row"] for s in roles["calibration"]["samples"]],
    )
    for row in selected:
        for kind in ("clean", "degraded"):
            path = str(Path(row["_root"]) / row[kind + "_path"])
            hashes[path] = row[kind + "_checksum"]
    for path in [
        study / "registration.json",
        study / "review-registration.json",
        study / "complete.json",
        study / "runtime-complete.json",
        Path(config["schedule"]),
        Path(config["detector"]),
        *catalogs,
        catalog,
        root / "requirements.txt",
        root / "constraints-cpu-py312.txt",
    ]:
        hashes[str(path)] = sha256_file(path)
    hashes.update({e["path"]: e["sha256"] for e in endpoints.values()})
    output.mkdir(parents=True)
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(p in {".venv", "__pycache__", "outputs"} for p in relative.parts):
            continue
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path)] = sha256_file(path)
        hashes[str(target)] = sha256_file(target)
    registration = {
        "format": "soundex-trainability-audit-p56-v1",
        "study": str(study),
        "endpoints": endpoints,
        "roles": roles,
        "hashes": hashes,
        "detector": config["detector"],
        "rates": RATES,
        "identity_weights": IDENTITY_WEIGHTS,
        "scopes": SCOPES,
        "optimizer": "Fresh Adam in all three scopes; additional adapter-only control resumes exact P54 moments. Betas .9/.999, eps 1e-8, clip 5. One actual step averaged across four TRAIN batches per proposal; no post-step scaling or veto.",
        "objective": "Unchanged common direct and raw MR losses. Keep paired coefficient .5; identity coefficient .5 or zero. Zero is a causal diagnostic, never a production policy. Full scope trains the budget via reconstruction only, no budget teacher.",
        "history": "Real warmup recomputed for each measurement, detached in gradients; no frozen-statistic updates. Adapter branch no longer detaches features in full scope; forward function/state remain exact.",
        "barrier": "Save/hash all 48 one-step proposals before any calibration measurements. Retain every seed/scope/rate/identity/optimizer-history result; no automatic selection, continuation or promotion.",
        "scope": "Post-P54 local optimization diagnosis on reused TRAIN recordings. Not a capacity ceiling, converged training, new generalization test, native DSP or listening result. Reserved P55 aliases excluded. Old P39-P53 probes did not compare current trained adapter/decoder/full scopes with matched Adam and identity ablation.",
    }
    verify(hashes)
    registered(registration, output / "registration.json")
    return registration, rows


def proposal(start, scope, parts, weight, rate, history):
    model = load_trainable(start, scope)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    before = torch.cat([p.detach().flatten() for p in parameters])
    optimizer = torch.optim.Adam(parameters, lr=rate, betas=(0.9, 0.999), eps=1e-8)
    if history == "continued":
        if scope != "adapter":
            raise ValueError("only adapter has matched historical moments")
        optimizer.load_state_dict(copy.deepcopy(start["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = rate
    gradient = 0.5 * (
        parts["paired_direct"]
        + parts["paired_rendered"]
        + weight * (parts["identity_direct"] + parts["identity_rendered"])
    )
    offset = 0
    for p in parameters:
        p.grad = gradient[offset : offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
    optimizer.step()
    after = torch.cat([p.detach().flatten() for p in parameters])
    delta = (after - before).double()
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite actual update")
    trainable = {n for n, _ in named}
    if any(
        not torch.equal(value, start["model"][name])
        for name, value in model.state_dict().items()
        if name not in trainable
    ):
        raise ValueError("frozen tensor changed")
    result = {
        "scope": scope,
        "identity_weight": weight,
        "rate": rate,
        "optimizer_history": history,
        "trainable_parameters": len(before),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "gradient_norm": float(norm),
        "displacement_l2": float(delta.norm()),
        "changed_parameters": int(torch.count_nonzero(delta)),
        "component_directional_derivatives": {
            k: float(v.double().dot(delta)) for k, v in parts.items()
        },
    }
    return model, optimizer, result


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    config, rows = prepare(args.study.resolve(), output)
    clips = BoundedEnvelopeClips(rows, Path(config["detector"]), output / "cache", capacity=80)
    items = {}
    for role, entry in config["roles"].items():
        pool = LazyPool(entry["specs"], clips)
        items[role] = [pool.select(d) for d in entry["draws"]]
    entries, attribution = {}, {}
    # Calibration PCM is loaded but never scored or backpropagated before freeze.
    for seed, entry in config["endpoints"].items():
        start = torch.load(entry["path"], map_location="cpu", weights_only=False)
        old = load_adapter(start).eval()
        expected = [
            predict_budget(old, item[0].batch, item[1])[0].detach()
            for item in items["optimization"]
        ]
        attribution[seed] = {}
        for scope in SCOPES:
            model = load_trainable(start, scope)
            with torch.no_grad():
                for item, reference in zip(items["optimization"], expected, strict=True):
                    if not torch.equal(predict_budget(model, item[0].batch, item[1])[0], reference):
                        raise ValueError("trainability control changes starting function")
            parts, report, named = gradients(model, items["optimization"])
            offset, groups = 0, {}
            for name, p in named:
                group = (
                    "adapter"
                    if name.startswith("adapter.")
                    else (
                        "decoder"
                        if name.startswith("backbone.decoder.")
                        else ("budget" if name.startswith("budget_head.") else "features")
                    )
                )
                groups.setdefault(group, []).extend(range(offset, offset + p.numel()))
                offset += p.numel()
            from evaluation.trainability_audit import geometry

            report["parameter_groups"] = {
                group: geometry({k: v[index] for k, v in parts.items()})
                for group, index in groups.items()
            }
            attribution[seed][scope] = report
            for history in ("fresh", "continued") if scope == "adapter" else ("fresh",):
                for weight in IDENTITY_WEIGHTS:
                    for rate in RATES:
                        label = f"{seed}-{scope}-{history}-identity{weight:g}-lr{rate:g}"
                        candidate, optimizer, result = proposal(
                            start, scope, parts, weight, rate, history
                        )
                        path = output / "proposals" / (label + ".pth")
                        saved_entry = save_proposal(
                            {
                                **start,
                                "model": candidate.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "p56": result,
                            },
                            path,
                        )
                        entries[label] = {
                            **saved_entry,
                            "seed": seed,
                            **result,
                        }
            write_json_report(
                {"stage": "proposals", "seed": seed, "scope": scope, "saved": len(entries)},
                output / "progress.json",
            )
            print("proposals", seed, scope, len(entries), flush=True)
    registered(entries, output / "endpoints.json")
    registered(attribution, output / "attribution.json")
    verify({e["path"]: e["sha256"] for e in entries.values()})
    results = {}
    for label, entry in {**config["endpoints"], **entries}.items():
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = load_trainable(checkpoint, entry.get("scope", "adapter"))
        results[label] = {role: measure(model, values) for role, values in items.items()}
        write_json_report(results[label], output / "measurements" / (label + ".json"))
        write_json_report(
            {"stage": "measurements", "completed": len(results), "total": len(entries) + 2},
            output / "progress.json",
        )
        print("measured", len(results), "of", len(entries) + 2, label, flush=True)
    verify(config["hashes"])
    verify({e["path"]: e["sha256"] for e in entries.values()})
    registered(
        {"results": results, "qualified": False, "scope": config["scope"]}, output / "summary.json"
    )
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in [
                    output / "registration.json",
                    output / "endpoints.json",
                    output / "attribution.json",
                    output / "summary.json",
                ]
            }
        },
        output / "complete.json",
    )
    write_json_report(
        {"stage": "complete", "proposals": len(entries), "qualified": False},
        output / "progress.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
