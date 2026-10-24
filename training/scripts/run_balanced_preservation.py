#!/usr/bin/env python3
"""P53 corpus-balanced preservation with fixed batch-gradient RMS calibration.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import fcntl
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import assert_frozen, load_endpoint
from evaluation.anchored_review import compare, review_case
from evaluation.balanced_preservation import ARMS, CorpusPool, calibrate, update
from evaluation.physical_preservation import penalties
from evaluation.reference_preservation import pcm_guard
from evaluation.reporting import sha256_file, write_json_report
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from models.spectral_adapter import load_adapter
from scripts.audit_adapter_response import native
from scripts.audit_guarded_transfer import measured
from scripts.run_adapter_pilot import checkpoint, optimizer_for
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
    if old["format"] != "soundex-physical-preservation-p52-v1":
        raise ValueError("completed P52 required")
    verify(old["hashes"])
    complete = read(prior / "complete.json")
    verify(complete["files"])
    parents = old["parents"]
    hashes = {**old["hashes"], **complete["files"]}
    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for p in root.rglob("*.py")
        if not any(x in {".venv", "__pycache__", "outputs"} for x in p.relative_to(root).parts)
    ] + list((root / "evaluation").glob("*.json"))
    for source in sources:
        hashes[str(source)] = sha256_file(source)
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (prior / "registration.json", prior / "complete.json"):
        hashes[str(source)] = sha256_file(source)
    config = {
        "format": "soundex-balanced-preservation-p53-v1",
        "prior": str(prior),
        "p50": old["p50"],
        "p51": old["prior"],
        "parents": parents,
        "hashes": hashes,
        **{
            k: old[k]
            for k in (
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
        "steps": 128,
        "arms": ARMS,
        "calibration_indices": list(range(0, 512, 16)),
        "optimizer": "Fresh Adam per trajectory, lr=1e-4 betas=.9/.999 eps=1e-8 clip=5; retain moments across 128 steps. Apply fixed .001 of proposal; four equal batches per step, same 512 P50 fit draws. No veto or early stop.",
        "hypothesis": "P52 physical preservation was corpus-imbalanced and pooled gradients hid batch variation. Keep the P50 parents, spectral objective, 128-update schedule and DSP. Compare global RMS physical, corpus-balanced RMS physical, and corpus-balanced RMS logit; reuse frozen P52 control/logit/physical endpoints with identical starts and budgets.",
        "penalty": "Identity original-reference raw OLA error power >=4 kHz in PCM squared, or affine clean logits squared. Corpus means use metadata, never row position. No division by signal level, teacher DSP or review-selected floor.",
        "calibration": "32 original evenly spaced fit batches per seed before training. Global physical: 10% common batch-gradient RMS. Balanced: normalize corpus-mean gradients to 80:20 RMS shares, then one common factor sets their actual combined RMS, including cross terms, to 10% common RMS. Freeze coefficients. Report component shares and per-batch ratios; no per-batch/future upper bound. Zero or cancelled gradients stop preflight, not silently drop a corpus.",
        "barrier": "Finish all six new trajectories and freeze hashes before endpoint scoring. Reuse verified P52 controls at unchanged 128-update budgets, P50 parents, P51 controls and P38 starts. Same original-reference scoring and 16/20/24-bit rulers. Historical TRAIN recheck, not independent generalization or convergence.",
        "native": "Six unchanged ABI exports, 480 new mono roles on same eight review batches; reuse verified P50/P51/P52 controls including identity DSP. Each new head: 4 x 3000 compute plus 1000 paced hops. No training/tests during timing. Preserve all misses and separate fixed 256-frame buffer.",
        "decision": "Bounded mechanism pilot, not full-library/converged training, architecture capacity, listening, stereo/device qualification or promotion. Original spectral supervision stays primary; report common totals separately from augmented training totals. No automatic longer rerun.",
    }
    verify(hashes)
    registered(config, path)
    return config


def calibration(config, pool, schedule, output):
    path = output / "calibration.json"
    receipt = output / "calibration.receipt.json"
    if receipt.exists():
        saved = read(path)
        if read(receipt)["sha256"] != sha256_file(path) or saved["signature"] != digest(config):
            raise ValueError("calibration drift")
        return saved["seeds"]
    if path.exists():
        raise ValueError("incomplete calibration retained")
    items = [pool.select(schedule["fit"][i]) for i in config["calibration_indices"]]
    result = {}
    for seed, entry in config["parents"].items():
        result[seed] = calibrate(load_endpoint(entry), items)
        print("calibration", seed, result[seed], flush=True)
    registered({"signature": digest(config), "seeds": result}, path)
    registered({"sha256": sha256_file(path)}, receipt)
    return result


def train(seed, arm, config, scales, pool, schedule, output):
    directory = output / "training" / f"{seed}-{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    latest, final = directory / "latest.pth", directory / "endpoint.pth"
    signature = digest({"config": config, "calibration": scales, "seed": seed, "arm": arm})
    receipt = directory / "complete.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["signature"] != signature or saved["sha256"] != sha256_file(final):
            raise ValueError("completed training drift")
        return final
    initial = torch.load(config["starts"][seed]["path"], map_location="cpu", weights_only=False)
    saved = None
    if latest.exists():
        if read(directory / "latest.json")["sha256"] != sha256_file(latest):
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or saved["step"] != len(saved["history"]):
            raise ValueError("resume registration/history drift")
    model = (load_adapter(saved) if saved else load_endpoint(config["parents"][seed])).eval()
    optimizer = optimizer_for(model)
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
    for index in range(len(history), config["steps"]):
        started = time.monotonic()
        items = [pool.select(d) for d in schedule["fit"][4 * index : 4 * index + 4]]
        result = update(model, items, optimizer, arm, scales[seed])
        history.append({"step": index + 1, "seconds": time.monotonic() - started, **result})
        if index == 0 or (index + 1) % 8 == 0:
            assert_frozen(model, initial)
            atomic_save(checkpoint(model, initial, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, directory / "latest.json")
            write_json_report(history, directory / "updates.json")
            info = {
                "stage": "training",
                "seed": seed,
                "arm": arm,
                "step": index + 1,
                "steps": config["steps"],
                "seconds_per_step": history[-1]["seconds"],
            }
            write_json_report(info, output / "progress.json")
            print(info, flush=True)
    assert_frozen(model, initial)
    atomic_save(checkpoint(model, initial, optimizer, history, signature), final)
    registered({"signature": signature, "sha256": sha256_file(final)}, receipt)
    return final


@torch.no_grad()
def review(config, entries, pool, schedule, output):
    if read(output / "review-registration.json") != entries:
        raise ValueError("complete endpoint barrier required")
    verify({e["path"]: e["sha256"] for e in entries.values()})
    result = {}
    for role in ("fit", "review"):
        indices = (
            list(range(0, len(schedule[role]), 16))
            if role == "fit"
            else list(range(len(schedule[role])))
        )
        items = [pool.select(schedule[role][i]) for i in indices]
        anchors, reports = {}, {}
        for seed, entry in config["starts"].items():
            model = load_endpoint(entry)
            anchors[seed] = [
                supervised_waves(predict_budget(model, item[0].batch, item[1])[0], item[0].batch)[0]
                for item in items
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
                terms, bands = penalties(model, item)
                cases.append(
                    {
                        **config["cases"][role][indices[j]],
                        "metrics": measured(model, item),
                        **review_case(model, item),
                        "preservation": {
                            "terms": {k: float(v) for k, v in terms.items()},
                            "band_mse": bands.tolist(),
                        },
                        "reference_rulers": {
                            str(b): pcm_guard([wave], [anchors[entry["seed"]][j]], [item], b)
                            for b in (16, 20, 24)
                        },
                    }
                )
            write_json_report({"checkpoint_sha256": entry["sha256"], "cases": cases}, path)
            registered({"sha256": sha256_file(path)}, receipt)
            reports[label] = cases
            print("raw", role, label, "complete", flush=True)
        result[role] = {}
        for label, entry in entries.items():
            seed = entry["seed"]
            references = {
                "start": read(Path(config["p50"]) / "review" / role / f"{seed}.json")["cases"],
                "parent": read(Path(config["p50"]) / "review" / role / f"{seed}-persistent.json")[
                    "cases"
                ],
                **{arm: reports[seed + "-" + arm] for arm in ARMS},
                **{
                    f"p52_{arm}": read(
                        Path(config["prior"]) / "review" / role / f"{seed}-{arm}.json"
                    )["cases"]
                    for arm in ("control", "logit", "physical")
                },
                **{
                    f"p51_{arm}": read(
                        Path(config["p51"]) / "review" / role / f"{seed}-persistent-{arm}.json"
                    )["cases"]
                    for arm in ("scalar", "separated")
                },
            }
            cases = reports[label]
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


def finalize(config, entries, pool, schedule, output):
    registered(entries, output / "review-registration.json")
    review(config, entries, pool, schedule, output)
    write_json_report({"stage": "native", "models": len(entries)}, output / "progress.json")
    native({**config, "prior": config["p50"]}, entries, pool, schedule, output)
    p51_controls = read(Path(config["p51"]) / "native-review/summary.json")
    registered(p51_controls, output / "p51-native-controls.json")
    registered(
        read(Path(config["prior"]) / "native-review/summary.json"),
        output / "p52-native-controls.json",
    )
    verify(config["hashes"])
    files = [
        output / name
        for name in (
            "registration.json",
            "calibration.json",
            "calibration.receipt.json",
            "assessment.json",
            "review-registration.json",
            "cost-summary.json",
            "native-review/summary.json",
            "native-controls.json",
            "p51-native-controls.json",
            "p52-native-controls.json",
        )
    ]
    for directory in ("training", "review", "exports", "native-cost"):
        files.extend(p for p in (output / directory).rglob("*") if p.is_file())
    for receipt in (
        "native-review/summary.json",
        "native-controls.json",
        "p51-native-controls.json",
        "p52-native-controls.json",
    ):
        for entry in read(output / receipt).values():
            path = Path(entry["report"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("native report drift")
            verify(read(path)["artifacts"])
            files.extend([path, *(Path(p) for p in read(path)["artifacts"])])
    registered({"files": {str(p): sha256_file(p) for p in files}}, output / "complete.json")
    write_json_report(
        {
            "stage": "complete",
            "endpoints": len(entries),
            "steps_each": config["steps"],
            "native_roles": 480,
        },
        output / "progress.json",
    )


def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), output)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            print("P53 complete", flush=True)
            return
        if args.prepare_only:
            print("P53 registered", flush=True)
            return
        rows, _ = load_sources([Path(p) for p in config["manifests"]])
        clips = BoundedEnvelopeClips(
            rows, Path(config["detector"]), output / "feature-cache", capacity=320
        )
        schedule = read(Path(config["schedule"]))
        pool = CorpusPool(LazyPool(schedule["specs"], clips))
        scales = calibration(config, pool, schedule, output)
        entries = {}
        for seed in config["starts"]:
            for arm in ARMS:
                path = train(seed, arm, config, scales, pool, schedule, output)
                entries[seed + "-" + arm] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "seed": seed,
                    "arm": arm,
                }
        finalize(config, entries, pool, schedule, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
