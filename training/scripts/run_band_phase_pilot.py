"""P65 fixed-budget TRAIN-role phase-only calibration, not a capacity verdict.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import fcntl
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.adapter_runtime import export, native_review
from evaluation.band_training import load
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from models.band_phase import BandPhaseCorrection
from models.multires_spectrum import supervised_waves
from models.restoration_head import predict_budget
from scripts.bench_band_tcn import native_parity
from scripts.compare_historical_spectra import aggregate, scores
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_joint_training import step
from scripts.run_longform_review import checked_receipt
from scripts.summarize_guarded_transfer import coverage

STEPS = 512
SEEDS = (20261012, 20261013)


def canonical_runtime_alias(hashes, repository, binary):
    """Bind an inherited build alias to identical frozen bytes, never live target/.

    Native calls consume the explicit binary argument. Only this known duplicate
    can be removed from the inherited verification set; retain its provenance.
    """
    alias = str(Path(repository) / "target/release/soundex-context-replay")
    binary = str(Path(binary))
    if alias == binary or alias not in hashes:
        return []
    expected = hashes[alias]
    if hashes.get(binary) != expected or sha256_file(binary) != expected:
        raise ValueError("non-equivalent inherited runtime alias")
    del hashes[alias]
    return [{"historical_build_path": alias, "consumed_frozen_path": binary, "sha256": expected}]


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve every attempt")
    checked_receipt(prior / "complete.json")
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    draws = schedule["draws"][: 4 * STEPS]
    used = sorted({i for d in draws for i in d})
    samples = [
        {"spec": schedule["specs"][i], "row": rows[schedule["specs"][i]["row_id"]]} for i in used
    ]
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for directory in ("models", "data", "evaluation", "scripts"):
        for path in (source / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".toml", ".html"}:
                destination = output / "source/training" / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".toml", ".html"}:
            shutil.copy2(path, output / "source/training" / path.name)
    parents = {str(seed): read(prior / "endpoints.json")[f"{seed}-band"] for seed in SEEDS}
    hashes = dict(old["hashes"])
    aliases = canonical_runtime_alias(hashes, old["repository"], old["binary"])
    for e in parents.values():
        hashes[e["path"]] = e["sha256"]
    hashes.update(
        {str(p): sha256_file(p) for p in (output / "source/training").rglob("*") if p.is_file()}
    )
    registered(
        {"optimization": {"specs": schedule["specs"], "draws": draws}}, output / "schedule.json"
    )
    hashes[str(output / "schedule.json")] = sha256_file(output / "schedule.json")
    config = {
        "format": "soundex-band-phase-p65-v1",
        "prior": str(prior),
        "parents": parents,
        "steps": STEPS,
        "learning_rate": 1e-4,
        "objective": "unchanged P63 common original-reference direct plus raw-OLA MR",
        "schedule": str(output / "schedule.json"),
        "manifests": old["manifests"],
        "detector": old["detector"],
        "binary": old["binary"],
        "repository": old["repository"],
        "evaluation_catalogs": old["evaluation_catalogs"],
        "packages": old["packages"],
        "exposure": coverage([{"samples": {"registered": samples}}], "registered"),
        "paired_crops_per_seed": 20 * STEPS,
        "hashes": hashes,
        "inherited_runtime_aliases": aliases,
        "scope": "512-step TRAIN-only learnability calibration, not full training, convergence, architecture ceiling, held-out score or promotion. P63 magnitude branch and all base state tensors frozen. First eight TRAIN draws used for shared before/after fit diagnostics. Same 80-hop history and frame-89 supervision; no DSP teacher.",
        "barrier": "Both origins finish the fixed budget and freeze endpoint hashes before all post-training native/raw fit reviews. No intermediate winner or budget adjustment.",
        "native_preflight": "zero-correction export parity plus deterministic nonzero phase stress on both rates/physical channels, before optimizer updates; retain raw versus actual Rust DSP differences",
        "next_gate": "Any fit improvement is only local trainability; retain all spectral, phase, identity and raw/native regressions before considering a new adequately budgeted source-isolated study.",
    }
    registered(config, output / "registration.json")
    print("registered", output, config["exposure"], flush=True)


def model_for(entry, seed):
    base = load(torch.load(entry["path"], map_location="cpu", weights_only=False))
    return BandPhaseCorrection(base, seed=seed + 65).eval()


def raw_review(model, items, descriptions, rows, specs, draws):
    cases = []
    with torch.no_grad():
        for i, item in enumerate(items):
            batch = item[0].batch
            prediction, _ = predict_budget(model, batch, item[1])
            wave, target = supervised_waves(prediction, batch)
            cases.append(
                {
                    "index": i,
                    "samples": descriptions[i]["samples"],
                    "corpora": [rows[specs[j]["row_id"]]["corpus"] for j in draws[i]],
                    "identity": batch["identity"].bool().tolist(),
                    "scores": scores(wave, target, batch["rate"]),
                }
            )
    return {"cases": cases, "summary": aggregate(cases)}


def run(output):
    torch.set_num_threads(1)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "training").exists():
            raise ValueError("existing calibration retained; inspect failure, no silent restart")
        config = read(output / "registration.json")
        import importlib.metadata

        if any(importlib.metadata.version(k) != v for k, v in config["packages"].items()):
            raise ValueError("registered packages changed")
        catalogs = sorted(
            str(p)
            for p in (Path(config["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")
        )
        assert_catalogs_unchanged(catalogs, config["evaluation_catalogs"])
        write_json_report(
            {"stage": "verifying-inputs", "files": len(config["hashes"])}, output / "progress.json"
        )
        verify(config["hashes"])
        rows, _ = load_sources(list(map(Path, config["manifests"])))
        schedule = read(Path(config["schedule"]))["optimization"]
        specs, draws = schedule["specs"], schedule["draws"]
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
        # Preflight uses no held-out cases, no quality-based candidate selection.
        for seed in SEEDS:
            model = model_for(config["parents"][str(seed)], seed)
            label = f"{seed}-zero"
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
            write_json_report(
                raw_review(model, items, descriptions, rows, specs, draws),
                output / "raw-review" / f"{label}.json",
            )
            stress = copy.deepcopy(model)
            with torch.no_grad():
                stress.phase_head[-1].bias.fill_(
                    2.0
                )  # fixed ~0.098-radian support-weighted rotation
            export(stress, output / "preflight" / str(seed) / "export")
            graph = output / "preflight" / str(seed) / "export/diagnostic.onnx"
            preflight[str(seed)] = []
            for rate in (44100, 48000):
                root = output / "preflight" / str(seed) / str(rate)
                parity = {}
                for name, candidate, candidate_graph in (
                    ("zero", model, graphs[label]),
                    ("stress", stress, graph),
                ):
                    parity[name] = native_parity(
                        candidate, candidate_graph, Path(config["binary"]), root / name, rate
                    )
                dry, _ = read_sxa(root / "zero/mono.output.sxa")
                rotated, _ = read_sxa(root / "stress/mono.output.sxa")
                difference = rotated.astype(np.float64) - dry.astype(np.float64)
                response = float(np.sqrt(np.mean(difference**2)))
                if not np.isfinite(response) or response == 0:
                    raise ValueError("nonzero rotation has no finite native PCM response")
                parity["native_pcm_response_rms"] = response
                preflight[str(seed)].append(parity)
        native_review(Path(config["binary"]), graphs, items, descriptions, output / "native-before")
        write_json_report(preflight, output / "preflight.json")
        endpoints = {}
        for seed in SEEDS:
            model = model_for(config["parents"][str(seed)], seed)
            frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
            optimizer = torch.optim.Adam(model.phase_head.parameters(), lr=config["learning_rate"])
            history = []
            directory = output / "training" / str(seed)
            directory.mkdir(parents=True)
            for index in range(config["steps"]):
                started = time.monotonic()
                result = step(
                    model, [pool.select(d) for d in draws[4 * index : 4 * index + 4]], optimizer
                )
                history.append({"step": index + 1, **result, "seconds": time.monotonic() - started})
                if (index + 1) % 32 == 0 or index == 0:
                    if any(
                        not torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items()
                    ):
                        raise ValueError("frozen magnitude model drift")
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "history": history,
                            "seed": seed,
                            "step": index + 1,
                        },
                        directory / "latest.pth",
                    )
                    write_json_report(
                        {
                            "stage": "training",
                            "seed": seed,
                            "step": index + 1,
                            "steps": config["steps"],
                        },
                        output / "progress.json",
                    )
                    print(seed, index + 1, result["objective_before"], flush=True)
            path = directory / "endpoint.pth"
            atomic_save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "step": len(history),
                    "history": history,
                },
                path,
            )
            endpoints[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
        registered(endpoints, output / "endpoints.json")
        for seed in SEEDS:
            entry = endpoints[str(seed)]
            if sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError("endpoint drift")
            model = model_for(config["parents"][str(seed)], seed)
            saved = torch.load(entry["path"], weights_only=False, map_location="cpu")
            if saved["step"] != config["steps"]:
                raise ValueError("incomplete endpoint barrier")
            model.load_state_dict(saved["model"])
            label = f"{seed}-phase"
            export(model, output / "exports" / label)
            graphs[label] = output / "exports" / label / "diagnostic.onnx"
            write_json_report(
                raw_review(model, items, descriptions, rows, specs, draws),
                output / "raw-review" / f"{label}.json",
            )
        native_review(Path(config["binary"]), graphs, items, descriptions, output / "native-after")
        verify(config["hashes"])
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
        print("P65 calibration complete; no held-out or timing qualification", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), output)
        else:
            run(output)
    except BaseException:
        if output.exists():
            write_json_report(
                {"traceback": traceback.format_exc()}, output / f"failure-{time.time_ns()}.json"
            )
        raise
