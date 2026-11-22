"""P108: learned causal half-ERB envelope versus matched per-bin control.

Stages: register (frozen pool, exclusions, budget, rules) -> extract (feature
cache) -> train (fixed budget, final endpoint only) -> evaluate (P107 rulers on
the exposed P88 panel through the offline fill proxy). No default promotion.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from evaluation import envelope_fill, missing_band  # noqa: E402
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from models.envelope_predictor import (  # noqa: E402
    ACTIVE,
    BANDS,
    HIGH_BINS,
    EnvelopePredictor,
    envelope_loss,
    features,
    parameter_count,
    stft_power,
    targets,
)

LIBRARY = REPO / "data/processed/music_library/processed/music_library-7fda9e58ab583617"
PANEL = REPO / "data/runs/raw-hf-shape-p88-20261119-v2"
P107 = REPO / "data/runs/missing-band-baseline-p107-20261122-v3"
CONFIG = {
    "seed": 108,
    "train_rows": 16000,
    "validation_rows": 1000,
    "segments_per_row": 3,
    "segment_frames": 256,
    "warmup_frames": 32,
    "batch": 64,
    "steps": 16384,
    "lr": 1e-3,
    "lr_final": 1e-5,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "gain_augment_db": 10.0,
    "diff_weight": 0.5,
    "hidden": 128,
    "layers": 2,
    "arms": ["band", "bin"],
    "train_seeds": [1, 2],
    "threads_per_run": 4,
}
RULES = (
    "Prospective, descriptive: (1) learning - band arm validation level-L1 falls below its "
    "step-0 value; (2) transfer - on the exposed P88 panel, recording-mean missing-region "
    "half-ERB RMS error of the band arm is below the frozen native P74 parent in both seeds "
    "with paired recording-bootstrap intervals below zero; (3) parameterization - band arm "
    "below the matched bin arm in both seeds. Bias, temporal correlation, LSD, preserved "
    "band and clean-role errors are reported beside it; no threshold. The result is an "
    "offline proxy: native Rust correspondence, runtime and listening remain required "
    "before any default or release decision. Final endpoints only; no intermediate selection."
)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserved_hashes():
    hashes, groups = set(), set()
    for f in glob.glob(str(REPO / "data/runs/*/evaluation-catalog.jsonl")):
        for d in jsonl(f):
            for k in ("audio_sha256", "decoded_pcm_sha256"):
                if isinstance(d.get(k), str):
                    hashes.add(d[k])
            for k in ("split_group", "work_id"):
                if isinstance(d.get(k), str):
                    groups.add(d[k])
    for s in json.loads((PANEL / "registration.json").read_text())["sources"]:
        hashes.update([s["audio_sha256"], s["decoded_pcm_sha256"]])
        groups.add(s["split_group"])
    return hashes, groups


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    rows = jsonl(LIBRARY / "manifest.jsonl")
    hashes, groups = reserved_hashes()

    def reserved(r):
        m = r["source_metadata"]
        return (
            r["source_checksum"] in hashes
            or m.get("decoded_pcm_sha256") in hashes
            or m.get("audio_sha256") in hashes
            or m.get("split_group") in groups
            or m.get("work_id") in groups
        )

    excluded_sources = sorted({r["source_id"] for r in rows if reserved(r)})
    pools = {}
    for split in ("train", "validation"):
        pools[split] = [
            r
            for r in rows
            if r["clean_path"].startswith(f"audio/{split}/")
            and not r["held_out"]
            and not reserved(r)
        ]
    rng = np.random.default_rng(CONFIG["seed"])
    chosen = {
        "train": sorted(
            rng.choice(len(pools["train"]), CONFIG["train_rows"], replace=False).tolist()
        ),
        "validation": sorted(
            rng.choice(len(pools["validation"]), CONFIG["validation_rows"], replace=False).tolist()
        ),
    }
    selection = {k: [pools[k][i]["row_id"] for i in v] for k, v in chosen.items()}
    coverage = {}
    for k, v in chosen.items():
        sel = [pools[k][i] for i in v]
        coverage[k] = {
            "rows": len(sel),
            "sources": len({r["source_id"] for r in sel}),
            "clean_crops": len({r["clean_checksum"] for r in sel}),
            "codecs": sorted({r["codec_id"] for r in sel}),
            "channel_roles": sorted({r["channel_role"] for r in sel}),
            "rates": sorted({r["sample_rate"] for r in sel}),
        }
    code = [
        ROOT / "models/envelope_predictor.py",
        ROOT / "evaluation/envelope_fill.py",
        ROOT / "evaluation/missing_band.py",
        Path(__file__).resolve(),
    ]
    registration = {
        "study": "p108-learned-envelope",
        "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": CONFIG,
        "rules": RULES,
        "library_manifest_sha256": sha(LIBRARY / "manifest.jsonl"),
        "excluded_reserved_sources": excluded_sources,
        "reserved_hash_count": len(hashes),
        "reserved_group_count": len(groups),
        "selection": selection,
        "coverage": coverage,
        "parameters": {a: parameter_count(EnvelopePredictor(a)) for a in CONFIG["arms"]},
        "p74_parent_parameters": 216802,
        "evaluation": {
            "panel": str(PANEL),
            "panel_cases_sha256": sha(PANEL / "cases.json"),
            "panel_status": "exposed development recheck; P88 works absent from the training "
            "library by file, decoded-PCM and split-group hashes",
            "comparators_from": str(P107),
            "scored_region": "P107 v3 (BURN 89, 10 s, native lag 128 for Rust renders)",
        },
        "code_sha256": {str(p.relative_to(REPO)): sha(p) for p in code},
        "proxy_note": "Offline Python fill with the ported Rust detector and Hann WOLA; "
        "no crossover, limiter or Rust stream. Native correspondence is a later gate.",
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=1))
    print(json.dumps({k: v for k, v in registration.items() if k in ("coverage", "parameters")}))


def _extract_row(task):
    row, seg_count, seg_frames, seed = task
    clean, rate = sf.read(LIBRARY / row["clean_path"], dtype="float64")
    dry, rate2 = sf.read(LIBRARY / row["degraded_path"], dtype="float64")
    if rate != rate2 or clean.shape != dry.shape or clean.ndim != 1:
        raise ValueError(f"pair geometry {row['row_id']}")
    pd, _ = stft_power(dry)
    pr, _ = stft_power(clean)
    x, ref = features(pd, rate)
    tb, _ = targets(pr, rate, "band")
    ti, _ = targets(pr, rate, "bin")
    rng = np.random.default_rng([seed, int(row["row_id"][:8], 16)])
    count = len(x) - seg_frames
    if count <= 0:
        return None
    starts = rng.choice(count, size=min(seg_count, count), replace=False)
    return [
        (
            x[s : s + seg_frames, :129].astype(np.float16),
            np.stack([ref[s : s + seg_frames], x[s : s + seg_frames, 130]], -1).astype(np.float32),
            tb[s : s + seg_frames].astype(np.float16),
            ti[s : s + seg_frames].astype(np.float16),
        )
        for s in sorted(starts)
    ]


def extract(out, workers):
    reg = json.loads((out / "registration.json").read_text())
    rows = {r["row_id"]: r for r in jsonl(LIBRARY / "manifest.jsonl")}
    cache = out / "cache"
    cache.mkdir(exist_ok=False)
    n, f = CONFIG["segments_per_row"], CONFIG["segment_frames"]
    for split, ids in reg["selection"].items():
        total = len(ids) * n
        arrays = {
            "x": np.lib.format.open_memmap(
                cache / f"{split}-x.npy", "w+", np.float16, (total, f, 129)
            ),
            "ref": np.lib.format.open_memmap(
                cache / f"{split}-ref.npy", "w+", np.float32, (total, f, 2)
            ),
            "band": np.lib.format.open_memmap(
                cache / f"{split}-band.npy", "w+", np.float16, (total, f, BANDS)
            ),
            "bin": np.lib.format.open_memmap(
                cache / f"{split}-bin.npy", "w+", np.float16, (total, f, HIGH_BINS)
            ),
        }
        index, meta = 0, []
        tasks = [(rows[i], n, f, CONFIG["seed"]) for i in ids]
        with ProcessPoolExecutor(workers) as pool:
            for row_id, segs in zip(ids, pool.map(_extract_row, tasks, chunksize=16), strict=True):
                for seg in segs or []:
                    for key, value in zip(("x", "ref", "band", "bin"), seg, strict=True):
                        arrays[key][index] = value
                    meta.append(row_id)
                    index += 1
        for a in arrays.values():
            a.flush()
        (cache / f"{split}-rows.json").write_text(json.dumps({"count": index, "rows": meta}))
        seconds = index * (f - CONFIG["warmup_frames"]) * 128 / 44100
        print(json.dumps({"split": split, "segments": index, "approx_supervised_s": seconds}))


def _load(cache, split):
    count = json.loads((cache / f"{split}-rows.json").read_text())["count"]
    return {
        k: np.load(cache / f"{split}-{k}.npy", mmap_mode="r")[:count]
        for k in ("x", "ref", "band", "bin")
    }


def _batch(data, idx, arm, gain_db, rng):
    x = torch.from_numpy(np.asarray(data["x"][idx], np.float32))
    ref = torch.from_numpy(np.asarray(data["ref"][idx], np.float32))
    target = torch.from_numpy(np.asarray(data[arm][idx], np.float32))
    active = target > math.log10(ACTIVE)
    if gain_db:
        g = torch.from_numpy(rng.uniform(-gain_db, gain_db, (len(idx), 1)).astype(np.float32)) / 10
        level, target = ref[..., 0] + g, target + g[..., None]
    else:
        level = ref[..., 0]
    inputs = torch.cat([x, (level / 10)[..., None], ref[..., 1:2]], -1)
    return inputs, target, active, level


def evaluate_loss(model, data, arm):
    model.eval()
    rng = np.random.default_rng(0)
    totals = []
    with torch.no_grad():
        for start in range(0, len(data["x"]), 256):
            idx = np.arange(start, min(start + 256, len(data["x"])))
            x, t, a, r = _batch(data, idx, arm, 0.0, rng)
            y, _ = model(x)
            totals.append(
                float(envelope_loss(y, t, a, r, CONFIG["warmup_frames"], 0.0)[1]) * len(idx)
            )
    model.train()
    return sum(totals) / len(data["x"])


def train(out, arm, seed):
    torch.set_num_threads(CONFIG["threads_per_run"])
    directory = out / "training" / f"{arm}-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    cache = out / "cache"
    data, val = _load(cache, "train"), _load(cache, "validation")
    torch.manual_seed(seed)
    model = EnvelopePredictor(arm, CONFIG["hidden"], CONFIG["layers"])
    opt = torch.optim.AdamW(model.parameters(), CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps = CONFIG["steps"]

    def lr_at(step):
        cos = 0.5 * (1 + math.cos(math.pi * step / steps))
        return CONFIG["lr_final"] + (CONFIG["lr"] - CONFIG["lr_final"]) * cos

    rng = np.random.default_rng([CONFIG["seed"], seed])
    history = [{"step": 0, "validation_level_db": evaluate_loss(model, val, arm)}]
    started = time.time()
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1)
        idx = np.sort(rng.choice(len(data["x"]), CONFIG["batch"], replace=False))
        x, t, a, r = _batch(data, idx, arm, CONFIG["gain_augment_db"], rng)
        y, _ = model(x)
        loss, level, change = envelope_loss(
            y, t, a, r, CONFIG["warmup_frames"], CONFIG["diff_weight"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()
        if step % 512 == 0:
            entry = {
                "step": step,
                "level_db": float(level),
                "change_db": float(change),
                "seconds": time.time() - started,
            }
            with (directory / "train.log").open("a") as log:
                log.write(json.dumps(entry) + "\n")
        if step % 2048 == 0 or step == steps:
            history.append({"step": step, "validation_level_db": evaluate_loss(model, val, arm)})
    torch.save(
        {"model": model.state_dict(), "arm": arm, "seed": seed, "config": CONFIG},
        directory / "endpoint.pth",
    )
    (directory / "history.json").write_text(json.dumps(history, indent=1))
    (directory / "complete.json").write_text(
        json.dumps(
            {
                "endpoint_sha256": sha(directory / "endpoint.pth"),
                "seconds": time.time() - started,
                "steps": steps,
            }
        )
    )
    print(json.dumps(history[-1]))


def _score_case(task):
    _, case, paths = task
    torch.set_num_threads(1)
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows = []
    signals = {}
    oracle, _ = envelope_fill.render(dry, rate, "oracle_band", reference=ref)
    signals["oracle_detector_band"] = oracle
    for label, path in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = EnvelopePredictor(state["arm"], CONFIG["hidden"], CONFIG["layers"])
        model.load_state_dict(state["model"])
        model.eval()
        signals[label], _ = envelope_fill.render(dry, rate, state["arm"], model=model)
    for label, wave in signals.items():
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
    for complete in sorted((out / "training").glob("*/complete.json")):
        receipt = json.loads(complete.read_text())
        endpoint = complete.parent / "endpoint.pth"
        if sha(endpoint) != receipt["endpoint_sha256"]:
            raise ValueError("endpoint hash drift")
        paths[f"p108_{complete.parent.name}"] = str(endpoint)
    if len(paths) != len(CONFIG["arms"]) * len(CONFIG["train_seeds"]):
        raise ValueError("every registered trajectory must finish before evaluation")
    (out / "evaluation-frozen.json").write_text(
        json.dumps(
            {
                "endpoints": {k: sha(v) for k, v in paths.items()},
                "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=1,
        )
    )
    cases = json.loads((PANEL / "cases.json").read_text())
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for chunk in pool.map(_score_case, [(out, c, paths) for c in cases]):
            rows.extend(chunk)
    keep = {
        "untreated",
        "20261013-magnitude",
        "20261012-magnitude",
        "legacy",
        "identity_dsp",
        "oracle_patch_erb",
        "oracle_complex",
        "spectral_dsp",
    }
    rows.extend(r for r in jsonl(P107 / "records.jsonl") if r["path"] in keep)
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
    summary = {}
    for role in ("paired", "identity"):
        table = {}
        for p in sorted({r["path"] for r in rows}):
            table[p] = {}
            for m in headline:
                g = defaultdict(list)
                for r in rows:
                    if r["role"] == role and r["path"] == p and r["scores"][m] is not None:
                        g[r["source"]].append(r["scores"][m])
                table[p][m] = float(np.mean([np.mean(v) for v in g.values()])) if g else None
        summary[role] = table
    paired = [
        {**r, "scores": {k: r["scores"][k] for k in headline}}
        for r in rows
        if r["role"] == "paired"
    ]
    contrasts = {}
    pairs = []
    for seed in CONFIG["train_seeds"]:
        pairs += [
            (f"p108_band-{seed}", "20261013-magnitude"),
            (f"p108_band-{seed}", f"p108_bin-{seed}"),
            (f"p108_bin-{seed}", "20261013-magnitude"),
            (f"p108_band-{seed}", "untreated"),
        ]
    pairs += [
        ("oracle_detector_band", "oracle_patch_erb"),
        ("oracle_detector_band", "20261013-magnitude"),
    ]
    for a, b in pairs:
        c = strict_paired_recordings(paired, a, b)
        contrasts[f"{a} - {b}"] = {
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
    parser.add_argument("--arm", choices=CONFIG["arms"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.out)
    elif args.stage == "extract":
        extract(args.out, args.workers)
    elif args.stage == "train":
        if args.arm is None or args.seed not in CONFIG["train_seeds"]:
            raise SystemExit("train requires a registered --arm and --seed")
        train(args.out, args.arm, args.seed)
    else:
        evaluate(args.out, args.workers)


if __name__ == "__main__":
    main()
