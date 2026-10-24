#!/usr/bin/env python3
"""P51 fit-only frozen-feature response separation versus matched scalar suppression.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import fcntl
import shutil
from collections import Counter
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import assert_frozen, load_endpoint
from evaluation.adapter_response import (
    coefficients,
    features,
    grams,
    install,
    paired_geometry,
    response_bands,
    separate,
)
from evaluation.adapter_runtime import cost, export, native_review
from evaluation.anchored_review import compare, review_case
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_guarded_transfer import measured
from scripts.run_anchored_pilot import verify
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.summarize_adapter_scale import changes


def prepare(prior, output):
    path = output / "registration.json"
    if path.exists():
        config = read(path)
        if config["prior"] != str(prior):
            raise ValueError("prior changed")
        verify(config["hashes"])
        return config
    old = read(prior / "registration.json")
    verify(old["hashes"])
    verify(read(prior / "complete.json")["files"])
    parents = read(prior / "review-registration.json")
    verify({entry["path"]: entry["sha256"] for entry in parents.values()})
    hashes = {
        **old["hashes"],
        **read(prior / "complete.json")["files"],
        **{entry["path"]: entry["sha256"] for entry in parents.values()},
    }
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(
            part in {".venv", "__pycache__", "outputs"} for part in p.relative_to(root).parts
        )
    ] + list((root / "evaluation").glob("*.json"))
    for source in sources:
        hashes[str(source)] = sha256_file(source)
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (prior / "registration.json", prior / "complete.json"):
        hashes[str(source)] = sha256_file(source)
    config = {
        "format": "soundex-adapter-response-p51-v1",
        "prior": str(prior),
        "hashes": hashes,
        "parents": parents,
        **{
            key: old[key]
            for key in (
                "starts",
                "schedule",
                "cases",
                "coverage",
                "binary",
                "assets",
                "detector",
                "manifests",
                "objective",
            )
        },
        "fit": "All 512 P50 fit batches, frames 89..255 only after real 80-frame warmup. Float64 augmented-feature Grams, equal clip/frame weights; two frozen encoders, shared by persistent/reset endpoints. No new source material or target-spectrum fitting.",
        "intervention": "For each of four P50 heads, minimize paired logit distortion plus lambda=1 clean logit power plus 1e-6 normalized ridge toward original coefficients. Save float32 weights. Scalar control matches resulting fit clean logit power. No validation-selected penalty, PCA rank, scalar or checkpoint.",
        "barrier": "Freeze all eight derived checkpoint hashes and fit statistics before reading review feature statistics or scoring any new candidate. Review same 32 fit and 32 globally source-disjoint reused-TRAIN batches. All original controls retained unchanged from verified P50 reports.",
        "scope": "Affine frozen-feature mechanism probe, not original-reference training, classifier impossibility, capacity limit, new independent generalization or release qualification. Feature/logit energy is not calibrated PCM. Exact original-reference spectral/PCM and native scores judge whether suppression preserves real restoration.",
        "native": "Export all eight transformed heads, same P48 ABI and unchanged Rust DSP. Each: four 3000-hop rate/channel runs and one 1000-hop paced 48k stereo run; eight evenly spaced P50 review batches (640 new native roles). Reuse hash-verified P50 six start/parent controls and identity DSP on identical clips. No simultaneous timing and training/tests; retain every miss.",
    }
    verify(hashes)
    registered(config, path)
    return config


def fit_features(config, pool, schedule, output):
    result = {}
    for seed in config["starts"]:
        path = output / "features" / f"{seed}.pth"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            if read(receipt)["sha256"] != sha256_file(path):
                raise ValueError("fit feature drift")
            result[seed] = torch.load(path, weights_only=False)
            if result[seed]["signature"] != digest(config):
                raise ValueError("fit feature registration drift")
            continue
        if path.exists():
            raise ValueError("incomplete feature artifact retained")
        model = load_endpoint(config["parents"][seed + "-persistent"])
        accum, geometry = {}, []
        for index, draw in enumerate(schedule["fit"]):
            z = features(model, pool.select(draw))
            for key, value in grams(z).items():
                accum[key] = accum.get(key, 0) + value / len(schedule["fit"])
            geometry.append({"index": index, **paired_geometry(z)})
            if (index + 1) % 64 == 0:
                write_json_report(
                    {
                        "stage": "fit_features",
                        "seed": seed,
                        "batches": index + 1,
                        "total": len(schedule["fit"]),
                    },
                    output / "progress.json",
                )
                print("features", seed, index + 1, flush=True)
        saved = {"grams": accum, "geometry": geometry, "signature": digest(config)}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(saved, path)
        registered({"sha256": sha256_file(path)}, receipt)
        result[seed] = saved
    return result


def candidates(config, statistics, output):
    entries = {}
    for parent, entry in config["parents"].items():
        if parent in config["starts"]:
            continue
        seed = next(s for s in config["starts"] if parent.startswith(s + "-"))
        original = load_endpoint(entry)
        projected, scalar, report = separate(coefficients(original), **statistics[seed]["grams"])
        for arm, weights in (("separated", projected), ("scalar", scalar)):
            label = parent + "-" + arm
            path = output / "candidates" / label / "endpoint.pth"
            receipt = path.with_suffix(".json")
            signature = digest({"config": config, "label": label})
            if receipt.exists():
                saved = read(receipt)
                if saved["signature"] != signature or saved["sha256"] != sha256_file(path):
                    raise ValueError("candidate drift")
            else:
                if path.exists():
                    raise ValueError("incomplete candidate retained")
                model = copy.deepcopy(original)
                install(model, weights)
                initial = torch.load(config["starts"][seed]["path"], weights_only=False)
                assert_frozen(model, initial)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_save(
                    {
                        "format": "soundex-spectral-adapter-v1",
                        "model": model.state_dict(),
                        "base_config": model.base_config,
                        "support": model.support,
                        "signature": signature,
                        "parent_sha256": entry["sha256"],
                        "operation": arm,
                        "fit": report,
                    },
                    path,
                )
                registered(
                    {"signature": signature, "sha256": sha256_file(path), "fit": report}, receipt
                )
            entries[label] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "parent": parent,
                "seed": seed,
                "arm": arm,
            }
    registered(entries, output / "candidate-registration.json")
    return entries


@torch.no_grad()
def review(config, entries, pool, schedule, output):
    if read(output / "candidate-registration.json") != entries:
        raise ValueError("complete candidate barrier required")
    verify({e["path"]: e["sha256"] for e in entries.values()})
    result, geometry, parent_responses = {}, {}, {}
    prior = Path(config["prior"])
    for role in ("fit", "review"):
        indices = (
            list(range(0, len(schedule[role]), 16))
            if role == "fit"
            else list(range(len(schedule[role])))
        )
        items = [pool.select(schedule[role][i]) for i in indices]
        anchors, originals, reports, feature_bank = {}, {}, {}, {}
        for seed, entry in config["starts"].items():
            model = load_endpoint(entry)
            anchors[seed] = [
                supervised_waves(predict_budget(model, item[0].batch, item[1])[0], item[0].batch)[0]
                for item in items
            ]
            feature_model = load_endpoint(config["parents"][seed + "-persistent"])
            feature_bank[seed] = [features(feature_model, item) for item in items]
            geometry[role + "/" + seed] = [
                {"index": indices[i], **paired_geometry(z)}
                for i, z in enumerate(feature_bank[seed])
            ]
        for parent in config["parents"]:
            originals[parent] = read(prior / "review" / role / f"{parent}.json")["cases"]
            if parent not in config["starts"]:
                seed = next(s for s in config["starts"] if parent.startswith(s + "-"))
                model = load_endpoint(config["parents"][parent])
                parent_responses[role + "/" + parent] = [
                    {
                        "index": indices[j],
                        "response": response_bands(model, z, item[0].batch["rate"]),
                    }
                    for j, (z, item) in enumerate(zip(feature_bank[seed], items, strict=True))
                ]
        for label, entry in entries.items():
            path = output / "review" / role / f"{label}.json"
            receipt = path.with_suffix(".receipt.json")
            if receipt.exists():
                saved = read(path)
                if (
                    read(receipt)["sha256"] != sha256_file(path)
                    or saved["checkpoint_sha256"] != entry["sha256"]
                ):
                    raise ValueError("review drift")
                reports[label] = saved["cases"]
                continue
            if path.exists():
                raise ValueError("incomplete review retained")
            model = load_endpoint(entry)
            cases = []
            for j, item in enumerate(items):
                wave = supervised_waves(
                    predict_budget(model, item[0].batch, item[1])[0], item[0].batch
                )[0]
                cases.append(
                    {
                        **config["cases"][role][indices[j]],
                        "metrics": measured(model, item),
                        **review_case(model, item),
                        "adapter_response": response_bands(
                            model, feature_bank[entry["seed"]][j], item[0].batch["rate"]
                        ),
                        "reference_rulers": {
                            str(bits): pcm_guard([wave], [anchors[entry["seed"]][j]], [item], bits)
                            for bits in (16, 20, 24)
                        },
                    }
                )
            write_json_report({"checkpoint_sha256": entry["sha256"], "cases": cases}, path)
            registered({"sha256": sha256_file(path)}, receipt)
            reports[label] = cases
            print("raw", role, label, "complete", flush=True)
        result[role] = {}
        for label, entry in entries.items():
            cases = reports[label]
            references = {
                "start": originals[entry["seed"]],
                "parent": originals[entry["parent"]],
                "scalar": reports[entry["parent"] + "-scalar"],
            }
            result[role][label] = {
                "comparisons": {
                    key: {
                        "raw": compare(cases, ref),
                        "metrics": changes(
                            [c["metrics"] for c in cases], [c["metrics"] for c in ref]
                        ),
                    }
                    for key, ref in references.items()
                },
                "ruler_violations": {
                    str(bits): dict(
                        Counter(
                            v["role"]
                            for c in cases
                            for v in c["reference_rulers"][str(bits)]["violations"]
                        )
                    )
                    for bits in (16, 20, 24)
                },
            }
    write_json_report(result, output / "assessment.json")
    write_json_report(geometry, output / "review-feature-geometry.json")
    write_json_report(parent_responses, output / "parent-responses.json")


def native(config, entries, pool, schedule, output):
    models, costs = {}, []
    for label, entry in entries.items():
        directory = output / "exports" / label
        export(load_endpoint(entry), directory)
        models[label] = directory / "diagnostic.onnx"
        for rate in (44100, 48000):
            for channels in (1, 2):
                row = cost(
                    Path(config["binary"]),
                    models[label],
                    Path(config["assets"]) / f"input-{rate}-{channels}.sxa",
                    output / "native-cost" / label / f"{rate}-{channels}",
                )
                costs.append({"label": label, "rate": rate, "channels": channels, **row})
        row = cost(
            Path(config["binary"]),
            models[label],
            Path(config["assets"]) / "input-48000-2.sxa",
            output / "native-cost" / label / "48000-2-paced",
            paced=True,
        )
        costs.append({"label": label, "rate": 48000, "channels": 2, **row})
        write_json_report(costs, output / "cost-summary.json")
        print("native cost", label, "complete", flush=True)
    items = [pool.select(draw) for draw in schedule["review"]]
    native_review(
        Path(config["binary"]), models, items, config["cases"]["review"], output / "native-review"
    )
    controls = read(Path(config["prior"]) / "native-review/summary.json")
    for entry in controls.values():
        if sha256_file(Path(entry["report"])) != entry["sha256"]:
            raise ValueError("prior native report drift")
        verify(read(Path(entry["report"]))["artifacts"])
    registered(controls, output / "native-controls.json")


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), output)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            print("P51 complete", flush=True)
            return
        if args.prepare_only:
            print("P51 registered", flush=True)
            return
        rows, _ = load_sources([Path(p) for p in config["manifests"]])
        clips = BoundedEnvelopeClips(
            rows, Path(config["detector"]), output / "feature-cache", capacity=320
        )
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(schedule["specs"], clips)
        statistics = fit_features(config, pool, schedule, output)
        entries = candidates(config, statistics, output)
        verify({e["path"]: e["sha256"] for e in entries.values()})
        write_json_report(
            {"stage": "frozen_review", "candidates": len(entries)}, output / "progress.json"
        )
        review(config, entries, pool, schedule, output)
        native(config, entries, pool, schedule, output)
        verify(config["hashes"])
        files = [
            output / "assessment.json",
            output / "review-feature-geometry.json",
            output / "parent-responses.json",
            output / "candidate-registration.json",
            output / "native-controls.json",
            output / "cost-summary.json",
            output / "native-review/summary.json",
            *(output / "features").glob("*"),
            *(output / "candidates").rglob("*.*"),
            *(output / "review").rglob("*.json"),
            *(output / "exports").rglob("report.json"),
        ]
        for receipt in (output / "native-review/summary.json", output / "native-controls.json"):
            for entry in read(receipt).values():
                path = Path(entry["report"])
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError("native report drift")
                verify(read(path)["artifacts"])
                files.extend([path, *(Path(p) for p in read(path)["artifacts"])])
        registered(
            {
                "registration_sha256": sha256_file(output / "registration.json"),
                "files": {str(p): sha256_file(p) for p in files},
            },
            output / "complete.json",
        )
        write_json_report(
            {"stage": "complete", "derived_checkpoints": len(entries), "native_roles": 640},
            output / "progress.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
