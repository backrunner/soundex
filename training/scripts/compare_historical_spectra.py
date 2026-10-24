#!/usr/bin/env python3
"""P54 same-material historical/current raw and actual Rust DSP comparison."""

import argparse
import fcntl
import shutil
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import digest, load_sources
from evaluation.adapter_pilot import load_endpoint
from evaluation.adapter_runtime import native_review
from evaluation.fidelity_calibration import pcm_statistics
from evaluation.head_export import session_for
from evaluation.legacy_context import wrap_legacy
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from models.auraloss_spectrum import spectral_loss as auraloss_spectrum
from models.multires_spectrum import spectral_loss, supervised_waves
from models.restoration_head import predict_budget
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered


def scores(wave, reference, rates):
    spectrum = spectral_loss(wave, reference, rates)
    return {
        "mr": spectrum["per_example"].tolist(),
        **{k: v.tolist() for k, v in spectrum["components"].items()},
        "auraloss": auraloss_spectrum(wave, reference, rates)["per_example"].tolist(),
        **{k: v.tolist() for k, v in pcm_statistics(wave - reference, rates).items()},
    }


def prepare(prior, legacy, output):
    path = output / "registration.json"
    if path.exists():
        config = read(path)
        verify(config["hashes"])
        if config["prior"] != str(prior) or config["legacy"] != str(legacy):
            raise ValueError("comparison inputs changed")
        return config
    old = read(prior / "registration.json")
    verify(read(prior / "complete.json")["files"])
    p52 = Path(old["prior"])
    p52_entries = read(p52 / "review-registration.json")
    p53_entries = read(prior / "review-registration.json")
    models = {}
    for seed, start in old["starts"].items():
        for name, entry, graph in (
            ("p38", start, Path(old["p50"]) / "exports" / seed / "diagnostic.onnx"),
            (
                "p52-control",
                p52_entries[seed + "-control"],
                p52 / "exports" / (seed + "-control") / "diagnostic.onnx",
            ),
            (
                "p53-logit",
                p53_entries[seed + "-balanced-logit"],
                prior / "exports" / (seed + "-balanced-logit") / "diagnostic.onnx",
            ),
        ):
            models[name + "-" + seed[-8:]] = {**entry, "onnx": str(graph)}
    binary = output / "binaries" / "soundex-context-replay"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old["binary"], binary)
    wrapper = output / "legacy-context.onnx"
    parity = wrap_legacy(legacy, wrapper)
    native = {k: v["onnx"] for k, v in models.items()}
    identity_report = read(prior / "native-controls.json")["identity_dsp"]["report"]
    # Its graph is preserved beside the P50 exports, not inferred from scores.
    identity = Path(old["p50"]) / "exports" / "identity_dsp" / "diagnostic.onnx"
    if not identity.exists():
        raise ValueError(f"missing frozen identity graph: {identity}; report {identity_report}")
    native.update(legacy=str(wrapper), identity_dsp=str(identity))
    paths = [
        legacy,
        wrapper,
        binary,
        Path(old["schedule"]),
        prior / "registration.json",
        *map(Path, old["manifests"]),
        Path(old["detector"]),
        *(Path(v["path"]) for v in models.values()),
        *map(Path, native.values()),
    ]
    root = Path(__file__).resolve().parents[1]
    for source in root.rglob("*.py"):
        if any(p in {".venv", "__pycache__", "outputs"} for p in source.relative_to(root).parts):
            continue
        target = output / "source" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        paths.append(target)
    config = {
        "format": "soundex-common-comparison-p54-v1",
        "prior": str(prior),
        "legacy": str(legacy),
        "legacy_wrapper_parity": parity,
        "schedule": old["schedule"],
        "manifests": old["manifests"],
        "detector": old["detector"],
        "cases": old["cases"]["review"],
        "review_indices": list(range(0, 32, 4)),
        "models": models,
        "native": native,
        "binary": str(binary),
        "scope": "Fixed historical TRAIN recheck, eight batches/32 music+8 speech mono views plus identities. Not independent generalization; uneven inherited budgets. Current ModelGain/SmoothKnee DSP for every model; legacy wrapper is diagnostic only. All candidates fixed before scores, no winner promotion.",
        "hashes": {str(p): sha256_file(p) for p in paths},
    }
    registered(config, path)
    return config


def aggregate(cases):
    result = {}
    for corpus in ("music_library", "speech_library"):
        for identity in (False, True):
            rows = []
            for case in cases:
                for i, c in enumerate(case["corpora"] * 2):
                    if c == corpus and case["identity"][i] == identity:
                        rows.append({k: v[i] for k, v in case["scores"].items()})
            result[corpus + ("/identity" if identity else "/paired")] = {
                "clips": len(rows),
                **{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
            }
    return result


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args.prior.resolve(), args.legacy.resolve(), output)
        if (output / "complete.json").exists():
            verify(read(output / "complete.json")["files"])
            return
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))
        pool = LazyPool(
            schedule["specs"],
            BoundedEnvelopeClips(
                rows, Path(config["detector"]), output / "feature-cache", capacity=80
            ),
        )
        items = [pool.select(draw) for draw in schedule["review"]]
        native_review(
            Path(config["binary"]),
            {k: Path(v) for k, v in config["native"].items()},
            items,
            config["cases"],
            output / "native",
        )
        results = {}
        for label in ["untreated", "legacy", *config["models"]]:
            path = output / "raw" / f"{label}.json"
            model = load_endpoint(config["models"][label]) if label in config["models"] else None
            legacy = session_for(config["legacy"]) if label == "legacy" else None
            cases = []
            for index in config["review_indices"]:
                item = items[index]
                batch, region = item[0].batch, item[0].region
                if label == "untreated":
                    wave = batch["dry"][:, region]
                elif legacy:
                    features = batch["x"].permute(0, 2, 1, 3)
                    flat = features.reshape(-1, 2, 1, 129).numpy()
                    predicted = torch.from_numpy(
                        legacy.run(None, {legacy.get_inputs()[0].name: flat})[0]
                    )
                    predicted = predicted.reshape(features.shape).permute(0, 2, 1, 3)
                    wave = supervised_waves(predicted, batch)[0]
                else:
                    wave = supervised_waves(predict_budget(model, batch, item[1])[0], batch)[0]
                cases.append(
                    {
                        "index": index,
                        "samples": config["cases"][index]["samples"],
                        "corpora": [
                            rows[schedule["specs"][i]["row_id"]]["corpus"]
                            for i in schedule["review"][index]
                        ],
                        "identity": batch["identity"].bool().tolist(),
                        "scores": scores(wave, batch["clean_wave"][:, region], batch["rate"]),
                    }
                )
            write_json_report({"cases": cases}, path)
            results["raw/" + label] = aggregate(cases)
            print("raw", label, results["raw/" + label], flush=True)
        for label in config["native"]:
            cases = []
            for index in config["review_indices"]:
                batch, region = items[index][0].batch, items[index][0].region
                waves = [
                    torch.from_numpy(
                        read_sxa(output / "native" / label / f"{index:02d}-{i:02d}" / "output.sxa")[
                            0
                        ][:, 0]
                    )
                    for i in range(len(batch["pcm"]))
                ]
                cases.append(
                    {
                        "index": index,
                        "corpora": [
                            rows[schedule["specs"][i]["row_id"]]["corpus"]
                            for i in schedule["review"][index]
                        ],
                        "identity": batch["identity"].bool().tolist(),
                        "scores": scores(
                            torch.stack(waves)[:, region],
                            batch["clean_wave"][:, region],
                            batch["rate"],
                        ),
                    }
                )
            write_json_report({"cases": cases}, output / "scored-native" / f"{label}.json")
            results["native/" + label] = aggregate(cases)
        write_json_report({"scope": config["scope"], "results": results}, output / "summary.json")
        registered(
            {
                "registration": digest(config),
                "files": {
                    str(p): sha256_file(p)
                    for p in output.rglob("*")
                    if p.is_file() and p.name not in {"complete.json", "run.lock"}
                },
            },
            output / "complete.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "legacy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
