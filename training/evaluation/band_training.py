"""P63 fresh architecture controls and resumable, fixed-budget optimization.

SPDX-License-Identifier: Apache-2.0
"""

import math
import time
from pathlib import Path

import onnx
import torch

from data.causal_study import digest
from evaluation.adapter_runtime import export
from evaluation.reporting import sha256_file, write_json_report
from models.adapter_trainability import TrainableAdapter
from models.band_tcn import BASE_CONFIG, make_prototype
from scripts.run_causal_study import atomic_save
from scripts.run_consistency_followup import read, registered
from scripts.run_joint_training import step

ARMS = ("global", "global_matched", "band")
SEEDS = (20261012, 20261013)
STEPS = 16384


def fresh(kind, seed):
    if kind == "global_matched":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return TrainableAdapter({**BASE_CONFIG, "width": 46}, "high", "full").eval()
    return make_prototype(kind, seed=seed, nonzero=False)


def pack(model, kind, seed, optimizer, history, signature):
    return {
        "format": "soundex-band-training-v1",
        "study": "p63-fresh-band-training",
        "kind": kind,
        "seed": seed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "step": len(history),
        "signature": signature,
    }


def load(saved, _scope=None):
    if saved["format"] != "soundex-band-training-v1" or saved["kind"] not in ARMS:
        raise ValueError("explicit fresh architecture checkpoint required")
    model = fresh(saved["kind"], saved["seed"])
    model.load_state_dict(saved["model"], strict=True)
    return model


def learning_rate(index, steps):
    """Same deterministic update-index schedule for every fresh trajectory."""
    if not 0 <= index < steps or steps <= 257:
        raise ValueError("invalid registered optimization index/budget")
    if index < 256:
        return 1e-4 * (0.1 + 0.9 * index / 255)
    progress = (index - 256) / (steps - 257)
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))


def export_model(model, directory, kind):
    report = export(model, directory)
    path = directory / "diagnostic.onnx"
    graph = onnx.load(path)
    properties = {p.key: p.value for p in graph.metadata_props}
    properties.update({"soundex.study": "p63-fresh-band-training", "soundex.architecture": kind})
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, path)
    report.update(onnx_sha256=sha256_file(path), study="p63-fresh-band-training", architecture=kind)
    write_json_report(report, directory / "report.json")
    return path


def freeze_endpoints(config, entries, output):
    expected = {f"{seed}-{kind}" for seed in config["seeds"] for kind in config["arms"]}
    if set(entries) != expected:
        raise ValueError("complete all registered architecture/seed endpoints before review")
    for label, entry in entries.items():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("endpoint hash drift")
        saved = torch.load(entry["path"], weights_only=False, map_location="cpu")
        if (
            saved["step"] != config["steps"]
            or len(saved["history"]) != config["steps"]
            or label != f"{saved['seed']}-{saved['kind']}"
            or saved["signature"]
            != digest({"config": config, "kind": saved["kind"], "seed": saved["seed"]})
        ):
            raise ValueError("incomplete or mismatched registered endpoint")
    registered(entries, output / "endpoints.json")


def trajectory(config, pool, draws, kind, seed, output):
    label = f"{seed}-{kind}"
    directory = output / "training" / label
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest({"config": config, "kind": kind, "seed": seed})
    endpoint, latest = directory / "endpoint.pth", directory / "latest.pth"
    complete = directory / "complete.json"
    if complete.exists():
        done = read(complete)
        if done["signature"] != signature or done["sha256"] != sha256_file(endpoint):
            raise ValueError("completed trajectory drift")
        return {"path": str(endpoint), "sha256": done["sha256"], "scope": kind}
    saved = None
    if latest.exists():
        if sha256_file(latest) != read(directory / "latest.json")["sha256"]:
            raise ValueError("latest checkpoint drift")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or saved["step"] != len(saved["history"]):
            raise ValueError("resume registration/history drift")
    model = load(saved) if saved else fresh(kind, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    history = saved["history"] if saved else []
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
    if len(draws) != 4 * config["steps"]:
        raise ValueError("wrong common schedule length")
    for index in range(len(history), config["steps"]):
        begin = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(index, config["steps"])
        result = step(model, [pool.select(d) for d in draws[4 * index : 4 * index + 4]], optimizer)
        history.append({"step": index + 1, **result, "seconds": time.monotonic() - begin})
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == config["steps"]:
            atomic_save(pack(model, kind, seed, optimizer, history, signature), latest)
            write_json_report({"sha256": sha256_file(latest)}, directory / "latest.json")
            write_json_report(history, directory / "updates.json")
            progress = {
                "stage": "training",
                "trajectory": label,
                "step": index + 1,
                "steps": config["steps"],
                "seconds_per_step": history[-1]["seconds"],
                "total_trajectories": len(config["arms"]) * len(config["seeds"]),
                "paired_crops_completed": (index + 1) * 20,
            }
            write_json_report(progress, output / "progress.json")
            print(progress, flush=True)
    atomic_save(pack(model, kind, seed, optimizer, history, signature), endpoint)
    done = {"signature": signature, "sha256": sha256_file(endpoint)}
    registered(done, complete)
    return {"path": str(endpoint), "sha256": done["sha256"], "scope": kind}
