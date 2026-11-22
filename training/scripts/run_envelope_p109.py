"""P109: clean-input preservation for the learned envelope (2x2 factorial).

Factors: target (absolute band level as P108, or band *deficit* = max(reference
- input, 0) floored 30 dB below the reference band) x identity share (0 or 25%
of batch items use lossless input with the same lossless reference). Same P108
rows/segments, model, budget and seeds. Final endpoints only.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import envelope_fill, missing_band  # noqa: E402
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from models.envelope_predictor import (  # noqa: E402
    ACTIVE,
    BANDS,
    LOG_FLOOR,
    EnvelopePredictor,
    band_matrix,
    envelope_loss,
    features,
    stft_power,
)
from scripts import run_envelope_p108 as p108  # noqa: E402

P108 = p108.REPO / "data/runs/envelope-predictor-p108-20261122"
DEFICIT_FLOOR_REL = 1e-3  # -30 dB below the reference band: <= +0.0043 dB if added.
CONFIG = {
    **{k: v for k, v in p108.CONFIG.items() if k not in ("arms",)},
    "threads_per_run": 2,
    "targets": ["level", "deficit"],
    "identity_shares": [0.0, 0.25],
    "deficit_floor_rel": DEFICIT_FLOOR_REL,
}
RULES = (
    "Prospective, descriptive. Cells: target in {level, deficit} x identity share in "
    "{0, 0.25}; seeds 1 and 2; level/0 reproduces the P108 band design. (1) Clean "
    "preservation: for each target, identity 0.25 lowers clean-role preserved LSD versus "
    "identity 0 in both seeds (paired recording intervals below zero). (2) Registered "
    "primary candidate deficit/0.25 advances to the native Rust port only if, in both "
    "seeds, its missing-region half-ERB RMS error is below the frozen native P74 parent "
    "(interval below zero) and its clean-role preserved LSD versus the P74 parent is not "
    "an 'increase' (interval not entirely above zero). Bias, temporal correlation, LSD and "
    "paired preserved-band errors are reported beside it. Offline proxy; not a promotion."
)


def cells():
    return [
        (t, s, seed)
        for t in CONFIG["targets"]
        for s in CONFIG["identity_shares"]
        for seed in CONFIG["train_seeds"]
    ]


def cell_name(target, share, seed):
    return f"{target}-id{int(share * 100)}-{seed}"


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    reg108 = json.loads((P108 / "registration.json").read_text())
    code = [
        ROOT / "scripts/run_envelope_p109.py",
        ROOT / "scripts/run_envelope_p108.py",
        ROOT / "models/envelope_predictor.py",
        ROOT / "evaluation/envelope_fill.py",
        ROOT / "evaluation/missing_band.py",
    ]
    registration = {
        "study": "p109-envelope-clean-preservation",
        "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": CONFIG,
        "rules": RULES,
        "cells": [cell_name(*c) for c in cells()],
        "selection_from": str(P108 / "registration.json"),
        "selection_sha256": p108.sha(P108 / "registration.json"),
        "coverage": reg108["coverage"],
        "excluded_reserved_sources": reg108["excluded_reserved_sources"],
        "motivation": "P108 band endpoints beat the P74 parent on missing-region envelope error "
        "but raised clean-role preserved LSD to 5.7-6.0 dB (P74 0.70): the model never saw "
        "lossless input and the ported Rust detector is active on ~99.9% of clean frames.",
        "code_sha256": {str(p.relative_to(p108.REPO)): p108.sha(p) for p in code},
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=1))
    print(json.dumps({"cells": registration["cells"]}))


def _extract_row(task):
    row, seg_count, seg_frames, seed = task
    clean, rate = sf.read(p108.LIBRARY / row["clean_path"], dtype="float64")
    dry, _ = sf.read(p108.LIBRARY / row["degraded_path"], dtype="float64")
    pd, _ = stft_power(dry)
    pr, _ = stft_power(clean)
    xd, rd = features(pd, rate)
    xc, rc = features(pr, rate)
    m = band_matrix(rate)
    er, ed = pr @ m, pd @ m
    level = np.log10(er + LOG_FLOOR)
    deficit = np.log10(np.maximum(er - ed, DEFICIT_FLOOR_REL * er) + LOG_FLOOR)
    rng = np.random.default_rng([seed, int(row["row_id"][:8], 16)])  # Same starts as P108.
    count = len(xd) - seg_frames
    if count <= 0:
        return None
    starts = rng.choice(count, size=min(seg_count, count), replace=False)
    out = []
    for s in sorted(starts):
        sl = slice(s, s + seg_frames)
        out.append(
            (
                xd[sl, :129].astype(np.float16),
                np.stack([rd[sl], xd[sl, 130]], -1).astype(np.float32),
                xc[sl, :129].astype(np.float16),
                np.stack([rc[sl], xc[sl, 130]], -1).astype(np.float32),
                level[sl].astype(np.float16),
                deficit[sl].astype(np.float16),
            )
        )
    return out


KEYS = ("xd", "rd", "xc", "rc", "level", "deficit")


def extract(out, workers):
    reg108 = json.loads((P108 / "registration.json").read_text())
    rows = {r["row_id"]: r for r in p108.jsonl(p108.LIBRARY / "manifest.jsonl")}
    cache = out / "cache"
    cache.mkdir(exist_ok=False)
    n, f = CONFIG["segments_per_row"], CONFIG["segment_frames"]
    shapes = {
        "xd": (129, np.float16),
        "rd": (2, np.float32),
        "xc": (129, np.float16),
        "rc": (2, np.float32),
        "level": (BANDS, np.float16),
        "deficit": (BANDS, np.float16),
    }
    for split, ids in reg108["selection"].items():
        total = len(ids) * n
        arrays = {
            k: np.lib.format.open_memmap(cache / f"{split}-{k}.npy", "w+", dt, (total, f, d))
            for k, (d, dt) in shapes.items()
        }
        index = 0
        tasks = [(rows[i], n, f, CONFIG["seed"]) for i in ids]
        with ProcessPoolExecutor(workers) as pool:
            for segs in pool.map(_extract_row, tasks, chunksize=16):
                for seg in segs or []:
                    for key, value in zip(KEYS, seg, strict=True):
                        arrays[key][index] = value
                    index += 1
        for a in arrays.values():
            a.flush()
        (cache / f"{split}-count.json").write_text(json.dumps({"count": index}))
        print(json.dumps({"split": split, "segments": index}))


def _load(cache, split):
    count = json.loads((cache / f"{split}-count.json").read_text())["count"]
    return {k: np.load(cache / f"{split}-{k}.npy", mmap_mode="r")[:count] for k in KEYS}


def _batch(data, idx, target, share, gain_db, rng):
    use_clean = rng.random(len(idx)) < share
    x = np.where(use_clean[:, None, None], data["xc"][idx], data["xd"][idx]).astype(np.float32)
    r = np.where(use_clean[:, None, None], data["rc"][idx], data["rd"][idx]).astype(np.float32)
    level = torch.from_numpy(np.asarray(data["level"][idx], np.float32))
    active = level > math.log10(ACTIVE)
    if target == "level":
        t = level
    else:
        t = torch.from_numpy(np.asarray(data["deficit"][idx], np.float32))
        floor = torch.log10(10**level * DEFICIT_FLOOR_REL + LOG_FLOOR)
        t = torch.where(torch.from_numpy(use_clean)[:, None, None], floor, t)
    r = torch.from_numpy(r)
    g = torch.from_numpy(rng.uniform(-gain_db, gain_db, (len(idx), 1)).astype(np.float32)) / 10
    lv = r[..., 0] + g
    inputs = torch.cat([torch.from_numpy(x), (lv / 10)[..., None], r[..., 1:2]], -1)
    return inputs, t + g[..., None], active, lv


def _validate(model, data, target, share):
    model.eval()
    rng = np.random.default_rng(0)
    vals = []
    with torch.no_grad():
        for start in range(0, len(data["xd"]), 256):
            idx = np.arange(start, min(start + 256, len(data["xd"])))
            x, t, a, r = _batch(data, idx, target, share, 0.0, rng)
            y, _ = model(x)
            vals.append(
                float(envelope_loss(y, t, a, r, CONFIG["warmup_frames"], 0.0)[1]) * len(idx)
            )
    model.train()
    return sum(vals) / len(data["xd"])


def train(out, target, share, seed):
    torch.set_num_threads(CONFIG["threads_per_run"])
    directory = out / "training" / cell_name(target, share, seed)
    directory.mkdir(parents=True, exist_ok=False)
    data, val = _load(out / "cache", "train"), _load(out / "cache", "validation")
    torch.manual_seed(seed)
    model = EnvelopePredictor("band", CONFIG["hidden"], CONFIG["layers"])
    opt = torch.optim.AdamW(model.parameters(), CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps, rng = CONFIG["steps"], np.random.default_rng([CONFIG["seed"], seed])
    history = [
        {
            "step": 0,
            "val_degraded": _validate(model, val, target, 0.0),
            "val_identity": _validate(model, val, target, 1.0),
        }
    ]
    started = time.time()
    for step in range(1, steps + 1):
        lr = CONFIG["lr_final"] + (CONFIG["lr"] - CONFIG["lr_final"]) * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / steps)
        )
        for group in opt.param_groups:
            group["lr"] = lr
        idx = np.sort(rng.choice(len(data["xd"]), CONFIG["batch"], replace=False))
        x, t, a, r = _batch(data, idx, target, share, CONFIG["gain_augment_db"], rng)
        y, _ = model(x)
        loss, level, _change = envelope_loss(
            y, t, a, r, CONFIG["warmup_frames"], CONFIG["diff_weight"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()
        if step % 512 == 0:
            with (directory / "train.log").open("a") as log:
                log.write(
                    json.dumps(
                        {"step": step, "level_db": float(level), "seconds": time.time() - started}
                    )
                    + "\n"
                )
        if step % 4096 == 0:
            history.append(
                {
                    "step": step,
                    "val_degraded": _validate(model, val, target, 0.0),
                    "val_identity": _validate(model, val, target, 1.0),
                }
            )
    torch.save(
        {
            "model": model.state_dict(),
            "target": target,
            "share": share,
            "seed": seed,
            "config": CONFIG,
        },
        directory / "endpoint.pth",
    )
    (directory / "history.json").write_text(json.dumps(history, indent=1))
    (directory / "complete.json").write_text(
        json.dumps(
            {
                "endpoint_sha256": p108.sha(directory / "endpoint.pth"),
                "seconds": time.time() - started,
            }
        )
    )
    print(json.dumps(history[-1]))


def _score_case(task):
    case, paths = task
    torch.set_num_threads(1)
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows = []
    for label, (path, mode) in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = EnvelopePredictor("band", CONFIG["hidden"], CONFIG["layers"])
        model.load_state_dict(state["model"])
        model.eval()
        wave, _ = envelope_fill.render(dry, rate, mode, model=model)
        wave = wave[begin : begin + n]
        if not np.isfinite(wave).all():
            raise ValueError(f"nonfinite {label}")
        scores = missing_band.score(wave, target, mask_dry, rate)
        scores["peak_abs"] = float(np.abs(wave).max())
        rows.append(
            {
                "case": case["id"],
                "codec": case["codec"],
                "role": case["role"],
                "source": case["source"],
                "path": label,
                "scores": scores,
            }
        )
    return rows


def evaluate(out, workers):
    paths = {}
    for t, s, seed in cells():
        d = out / "training" / cell_name(t, s, seed)
        receipt = json.loads((d / "complete.json").read_text())
        if p108.sha(d / "endpoint.pth") != receipt["endpoint_sha256"]:
            raise ValueError("endpoint hash drift")
        paths[f"p109_{cell_name(t, s, seed)}"] = (
            str(d / "endpoint.pth"),
            "band" if t == "level" else "band_deficit",
        )
    (out / "evaluation-frozen.json").write_text(
        json.dumps({k: p108.sha(v[0]) for k, v in paths.items()}, indent=1)
    )
    cases = json.loads((p108.PANEL / "cases.json").read_text())
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for chunk in pool.map(_score_case, [(c, paths) for c in cases]):
            rows.extend(chunk)
    keep = {
        "untreated",
        "20261013-magnitude",
        "identity_dsp",
        "oracle_detector_band",
        "p108_band-1",
        "p108_band-2",
        "oracle_patch_erb",
        "legacy",
    }
    rows.extend(r for r in p108.jsonl(P108 / "records.jsonl") if r["path"] in keep)
    with (out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    headline = (
        "missing_lsd_db",
        "missing_erb_rmse_db",
        "missing_erb_bias_db",
        "missing_temporal_corr",
        "preserved_lsd_db",
        "preserved_complex_rel",
    )
    summary, contrasts = {}, {}
    for role in ("paired", "identity"):
        summary[role] = {}
        for p in sorted({r["path"] for r in rows}):
            summary[role][p] = {}
            for m in headline:
                groups = {}
                for r in rows:
                    if r["role"] == role and r["path"] == p and r["scores"][m] is not None:
                        groups.setdefault(r["source"], []).append(r["scores"][m])
                summary[role][p][m] = (
                    float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
                )
        trimmed = [
            {**r, "scores": {k: r["scores"][k] for k in headline}}
            for r in rows
            if r["role"] == role
        ]
        pairs = []
        for seed in CONFIG["train_seeds"]:
            for t in CONFIG["targets"]:
                pairs.append((f"p109_{t}-id25-{seed}", f"p109_{t}-id0-{seed}"))
                pairs.append((f"p109_{t}-id25-{seed}", "20261013-magnitude"))
            pairs.append((f"p109_deficit-id25-{seed}", f"p109_level-id25-{seed}"))
            pairs.append((f"p109_level-id0-{seed}", f"p108_band-{seed}"))
        for a, b in pairs:
            c = strict_paired_recordings(trimmed, a, b)
            contrasts[f"{role}: {a} - {b}"] = {
                k: {
                    "mean_delta": c[k].get("mean_delta"),
                    "ci95": c[k].get("paired_source_bootstrap_95"),
                    "sources": c[k].get("sources"),
                    "adverse": len(c[k].get("adverse_sources", [])),
                }
                for k in headline
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    print(json.dumps({"rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "extract", "train", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=CONFIG["targets"])
    parser.add_argument("--share", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=10)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out)
    elif a.stage == "extract":
        extract(a.out, a.workers)
    elif a.stage == "train":
        if (a.target, a.share, a.seed) not in cells():
            raise SystemExit("unregistered cell")
        train(a.out, a.target, a.share, a.seed)
    else:
        evaluate(a.out, a.workers)


if __name__ == "__main__":
    main()
