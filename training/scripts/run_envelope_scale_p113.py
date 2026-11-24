"""P113: generalization of the envelope predictor with broader exposure and budget.

Frozen P109 recipe (deficit target, 25% identity, EnvelopePredictor('band', 128, 2),
P111 gate [0, 6] dB at deployment). Changes: every eligible TRAIN row (two
segments each) instead of 16,000 rows, and 65,536 updates instead of 16,384.
Exclusions are recomputed from every evaluation catalog, including P112 and the
P113 reserved panel. Primary evaluation is native on the P113 panel; P112 is a
labelled recheck. Final endpoints only. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import missing_band  # noqa: E402
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from export_envelope import write  # noqa: E402
from models.envelope_predictor import (  # noqa: E402
    EnvelopePredictor,
    envelope_loss,
    parameter_count,
)
from scripts import run_envelope_fresh_p112 as p112  # noqa: E402
from scripts import run_envelope_gate_p111 as p111  # noqa: E402
from scripts import run_envelope_p108 as p108  # noqa: E402
from scripts import run_envelope_p109 as p109  # noqa: E402
from scripts.score_missing_band import check_lag  # noqa: E402

REPO = p108.REPO
PANEL = REPO / "data/runs/reserved-panel-p113-20261124"
P112_RUN = REPO / "data/runs/envelope-fresh-p112-20261124"
CONFIG = {
    **{
        k: p109.CONFIG[k]
        for k in (
            "segment_frames",
            "warmup_frames",
            "batch",
            "lr",
            "lr_final",
            "weight_decay",
            "grad_clip",
            "gain_augment_db",
            "diff_weight",
            "hidden",
            "layers",
            "seed",
        )
    },
    "segments_per_row": 2,
    "validation_rows": 2000,
    "steps": 65536,
    "target": "deficit",
    "identity_share": 0.25,
    "train_seeds": [1, 2],
    "threads_per_run": 8,
    "gate_db": list(p111.GATE_DB),
}
RULES = (
    "Prospective. Rationale: P108/P109 validation loss was still falling at 16,384 updates; "
    "P84 found source-time coverage helps transfer; P112 showed the development gain did "
    "not transfer. Data exposure and budget change together (confounded by design). "
    "(1) Primary, P113 reserved panel (24 non-electronic works), native SmoothKnee: in both "
    "seeds the gated P113 endpoint has missing-region half-ERB RMS below the native P74 "
    "parent (interval below zero) AND clean preserved LSD versus P74 is not an 'increase'. "
    "(2) Incremental: in both seeds, ERB RMS below the gated P111 candidate of the same seed "
    "(interval below zero). (3) Reported: P112 panel as an exposed recheck, missing LSD, "
    "bias, temporal correlation, paired preserved LSD, absolute clean HF dBFS, per genre/codec, "
    "per-genre validation loss (monitoring only). Final endpoints; no promotion."
)


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    rows = p108.jsonl(p108.LIBRARY / "manifest.jsonl")
    hashes, groups = p108.reserved_hashes()
    if not (PANEL / "evaluation-catalog.jsonl").exists():
        raise ValueError("reserve the P113 panel before registering training")

    def reserved(r):
        m = r["source_metadata"]
        return (
            r["source_checksum"] in hashes
            or m.get("decoded_pcm_sha256") in hashes
            or m.get("audio_sha256") in hashes
            or m.get("split_group") in groups
            or m.get("work_id") in groups
        )

    pools = {
        split: [
            r
            for r in rows
            if r["clean_path"].startswith(f"audio/{split}/")
            and not r["held_out"]
            and not reserved(r)
        ]
        for split in ("train", "validation")
    }
    rng = np.random.default_rng(113)
    val = sorted(
        rng.choice(len(pools["validation"]), CONFIG["validation_rows"], replace=False).tolist()
    )
    selection = {
        "train": [r["row_id"] for r in pools["train"]],
        "validation": [pools["validation"][i]["row_id"] for i in val],
    }
    genre = {
        r["row_id"]: r["source_metadata"].get("genre", "unlabeled")
        for split in pools.values()
        for r in split
    }
    code = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_envelope_p109.py",
        ROOT / "models/envelope_predictor.py",
        ROOT / "export_envelope.py",
        ROOT / "evaluation/envelope_fill.py",
        ROOT / "evaluation/missing_band.py",
    ]
    (out / "registration.json").write_text(
        json.dumps(
            {
                "study": "p113-envelope-scale",
                "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": CONFIG,
                "rules": RULES,
                "selection": selection,
                "genres": {k: genre[k] for k in selection["train"] + selection["validation"]},
                "coverage": {
                    k: {"rows": len(v), "sources": len({r["source_id"] for r in pools[k]})}
                    for k, v in selection.items()
                },
                "excluded_reserved_sources": sorted({r["source_id"] for r in rows if reserved(r)}),
                "panel_catalog_sha256": p108.sha(PANEL / "evaluation-catalog.jsonl"),
                "code_sha256": {str(p.relative_to(REPO)): p108.sha(p) for p in code},
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "train_rows": len(selection["train"]),
                "train_sources": len({r["source_id"] for r in pools["train"]}),
                "excluded_sources": len({r["source_id"] for r in rows if reserved(r)}),
            }
        )
    )


def extract(out, workers):
    reg = json.loads((out / "registration.json").read_text())
    rows = {r["row_id"]: r for r in p108.jsonl(p108.LIBRARY / "manifest.jsonl")}
    cache = out / "cache"
    cache.mkdir(exist_ok=False)
    n, f = CONFIG["segments_per_row"], CONFIG["segment_frames"]
    shapes = {
        "xd": (129, np.float16),
        "rd": (2, np.float32),
        "xc": (129, np.float16),
        "rc": (2, np.float32),
        "level": (p109.BANDS, np.float16),
        "deficit": (p109.BANDS, np.float16),
    }
    for split, ids in reg["selection"].items():
        total = len(ids) * n
        arrays = {
            k: np.lib.format.open_memmap(cache / f"{split}-{k}.npy", "w+", dt, (total, f, d))
            for k, (d, dt) in shapes.items()
        }
        index, meta = 0, []
        tasks = [(rows[i], n, f, CONFIG["seed"]) for i in ids]
        with ProcessPoolExecutor(workers) as pool:
            for row_id, segs in zip(
                ids, pool.map(p109._extract_row, tasks, chunksize=32), strict=True
            ):
                for seg in segs or []:
                    for key, value in zip(p109.KEYS, seg, strict=True):
                        arrays[key][index] = value
                    meta.append(row_id)
                    index += 1
        for arr in arrays.values():
            arr.flush()
        (cache / f"{split}-count.json").write_text(json.dumps({"count": index, "rows": meta}))
        print(json.dumps({"split": split, "segments": index}))


def _genre_validation(model, data, rows, genres):
    """Monitoring only: deficit-target validation level error per genre (degraded input)."""
    by_genre = defaultdict(list)
    for i, row_id in enumerate(rows):
        by_genre[genres.get(row_id, "unlabeled")].append(i)
    model.eval()
    out = {}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for g, idx in sorted(by_genre.items()):
            vals = []
            for start in range(0, len(idx), 256):
                part = np.asarray(idx[start : start + 256])
                x, t, a, r = p109._batch(data, part, "deficit", 0.0, 0.0, rng)
                y, _ = model(x)
                vals.append(
                    float(envelope_loss(y, t, a, r, CONFIG["warmup_frames"], 0.0)[1]) * len(part)
                )
            out[g] = sum(vals) / len(idx)
    model.train()
    return out


def train(out, seed):
    torch.set_num_threads(CONFIG["threads_per_run"])
    reg = json.loads((out / "registration.json").read_text())
    directory = out / "training" / f"deficit-id25-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    data, val = p109._load(out / "cache", "train"), p109._load(out / "cache", "validation")
    val_rows = json.loads((out / "cache/validation-count.json").read_text())["rows"]
    torch.manual_seed(seed)
    model = EnvelopePredictor("band", CONFIG["hidden"], CONFIG["layers"])
    opt = torch.optim.AdamW(model.parameters(), CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps, rng = CONFIG["steps"], np.random.default_rng([CONFIG["seed"], 113, seed])

    def snapshot(step):
        return {
            "step": step,
            "val_degraded": p109._validate(model, val, "deficit", 0.0),
            "val_identity": p109._validate(model, val, "deficit", 1.0),
            "val_by_genre": _genre_validation(model, val, val_rows, reg["genres"]),
        }

    history = [snapshot(0)]
    started = time.time()
    for step in range(1, steps + 1):
        lr = CONFIG["lr_final"] + (CONFIG["lr"] - CONFIG["lr_final"]) * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / steps)
        )
        for group in opt.param_groups:
            group["lr"] = lr
        idx = np.sort(rng.choice(len(data["xd"]), CONFIG["batch"], replace=False))
        x, t, a, r = p109._batch(
            data, idx, "deficit", CONFIG["identity_share"], CONFIG["gain_augment_db"], rng
        )
        y, _ = model(x)
        loss, level, _change = envelope_loss(
            y, t, a, r, CONFIG["warmup_frames"], CONFIG["diff_weight"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        opt.step()
        if step % 1024 == 0:
            with (directory / "train.log").open("a") as log:
                log.write(
                    json.dumps(
                        {"step": step, "level_db": float(level), "seconds": time.time() - started}
                    )
                    + "\n"
                )
        if step % 8192 == 0:
            history.append(snapshot(step))
            (directory / "history.json").write_text(json.dumps(history, indent=1))
    torch.save(
        {
            "model": model.state_dict(),
            "target": "deficit",
            "share": 0.25,
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
                "parameters": parameter_count(model),
            }
        )
    )
    print(json.dumps(history[-1]))


def export(out):
    (out / "artifacts").mkdir(exist_ok=False)
    arts = {}
    for seed in CONFIG["train_seeds"]:
        d = out / "training" / f"deficit-id25-{seed}"
        receipt = json.loads((d / "complete.json").read_text())
        if p108.sha(d / "endpoint.pth") != receipt["endpoint_sha256"]:
            raise ValueError("endpoint drift")
        state = torch.load(d / "endpoint.pth", map_location="cpu", weights_only=False)
        model = EnvelopePredictor("band", CONFIG["hidden"], CONFIG["layers"])
        model.load_state_dict(state["model"])
        path = out / "artifacts" / f"p113-deficit-id25-gate0-6-{seed}.sxenv"
        write(
            model.eval(),
            "deficit",
            path,
            warmup_frames=CONFIG["warmup_frames"],
            fill_gate_db=tuple(CONFIG["gate_db"]),
            provenance={"checkpoint_sha256": receipt["endpoint_sha256"], "study": "P113"},
        )
        arts[str(seed)] = {"path": str(path), "sha256": p108.sha(path)}
    (out / "evaluation-frozen.json").write_text(
        json.dumps({"artifacts": arts, "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1)
    )
    print(json.dumps(arts))


def _render(binary, artifact, source, work):
    work.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        SOUNDEX_EVAL_ENHANCEMENT_MODE="envelope",
        SOUNDEX_EVAL_LIMITER_MODE="smooth-knee",
    )
    subprocess.run(
        [binary, artifact, source, str(work / "output.sxa"), str(work / "report.json"), "offline"],
        env=env,
        check=True,
        capture_output=True,
    )
    wave, _ = read_sxa(work / "output.sxa")
    return wave


def _case(task):
    out, panel_name, case, binary, mine, p111_arts, prior = task
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows = []

    def add(label, wave):
        scores = missing_band.score(wave, target, mask_dry, rate)
        scores["clean_hf_error_dbfs"] = (
            p111.hf_error_dbfs(wave, mask_dry, rate) if case["role"] == "identity" else None
        )
        rows.append(
            {
                "panel": panel_name,
                "case": case["id"],
                "codec": case["codec"],
                "role": case["role"],
                "source": case["source"],
                "path": label,
                "scores": scores,
            }
        )

    for seed, art in mine.items():
        wave = _render(
            binary,
            art["path"],
            case["input"],
            out / "renders" / panel_name / case["id"] / f"p113_{seed}",
        )
        check_lag(wave, dry, rate, 0)
        add(f"p113_seed{seed}", wave[begin : begin + n])
    for label in ("untreated", "p74_parent", "gated_seed1", "gated_seed2"):
        found = [r for r in prior.get(case["id"], []) if r["path"] == label]
        if found:
            rows.append({**found[0], "panel": panel_name})
    if not prior:  # The P113 panel has no prior renders: render P74 and P111 here.
        add("untreated", mask_dry)
        work = out / "renders" / panel_name / case["id"] / "p74"
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(p112.P74_BRIDGE),
                str(p112.P74_GRAPH),
                case["input"],
                str(work / "output.sxa"),
                str(work / "runtime.json"),
                "1",
            ],
            env=p112.environment(),
            check=True,
            capture_output=True,
        )
        p74, _ = read_sxa(work / "output.sxa")
        check_lag(p74, dry, rate, 128)
        add("p74_parent", p74[begin + 128 : begin + 128 + n])
        for seed, art in p111_arts.items():
            wave = _render(
                binary,
                art["path"],
                case["input"],
                out / "renders" / panel_name / case["id"] / f"p111_{seed}",
            )
            add(f"gated_seed{seed}", wave[begin : begin + n])
    return rows


def evaluate(out, binary, workers):
    frozen = json.loads((out / "evaluation-frozen.json").read_text())
    mine = frozen["artifacts"]
    p111_arts = json.loads(
        (p111.p108.REPO / "data/runs/envelope-gate-p111-20261124/registration.json").read_text()
    )["artifacts"]
    for art in [*mine.values(), *p111_arts.values()]:
        if p108.sha(art["path"]) != art["sha256"]:
            raise ValueError("artifact drift")
    panel_dir = out / "panel"
    if not (panel_dir / "cases.json").exists():
        raise ValueError("prepare the P113 panel pairs first")
    tasks = []
    for case in json.loads((panel_dir / "cases.json").read_text()):
        tasks.append((out, "p113", case, str(binary), mine, p111_arts, {}))
    prior = defaultdict(list)
    for r in p108.jsonl(P112_RUN / "records.jsonl"):
        prior[r["case"]].append(r)
    for case in json.loads((P112_RUN / "cases.json").read_text()):
        tasks.append(
            (
                out,
                "p112_recheck",
                case,
                str(binary),
                mine,
                p111_arts,
                {case["id"]: prior[case["id"]]},
            )
        )
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_case, tasks):
            rows.extend(r)
    with (out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary, contrasts = {}, {}
    keys = p111.HEADLINE
    for panel_name in ("p113", "p112_recheck"):
        for role in ("paired", "identity"):
            sel = [r for r in rows if r["panel"] == panel_name and r["role"] == role]
            table = {}
            for p in sorted({r["path"] for r in sel}):
                table[p] = {}
                for m in keys:
                    g = defaultdict(list)
                    for r in sel:
                        if r["path"] == p and r["scores"].get(m) is not None:
                            g[r["source"]].append(r["scores"][m])
                    table[p][m] = float(np.mean([np.mean(v) for v in g.values()])) if g else None
            summary[f"{panel_name}/{role}"] = table
            use = keys if role == "identity" else keys[:-1]
            trimmed = [{**r, "scores": {k: r["scores"][k] for k in use}} for r in sel]
            for seed in CONFIG["train_seeds"]:
                for b in ("p74_parent", f"gated_seed{seed}", "untreated"):
                    c = strict_paired_recordings(trimmed, f"p113_seed{seed}", b)
                    contrasts[f"{panel_name}/{role}: p113_seed{seed} - {b}"] = {
                        k: {
                            "mean_delta": c[k].get("mean_delta"),
                            "ci95": c[k].get("paired_source_bootstrap_95"),
                            "sources": c[k].get("sources"),
                            "undefined": [u[:12] for u in c[k].get("undefined_sources", [])],
                            "adverse": len(c[k].get("adverse_sources", [])),
                        }
                        for k in use
                    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    print(json.dumps({"rows": len(rows)}))


def prepare_panel(out):
    """Freeze P113 panel pairs with the P112 (P87/P88) preparation."""
    panel_dir = out / "panel"
    panel_dir.mkdir(exist_ok=False)
    sources = p108.jsonl(PANEL / "evaluation-catalog.jsonl")
    (panel_dir / "registration.json").write_text(
        json.dumps(
            {"sources": sources, "ffmpeg": p112.P88["ffmpeg"], "profile": p112.P88["profile"]},
            indent=1,
        )
    )
    p112.prepare(panel_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("register", "extract", "train", "export", "prepare-panel", "evaluate")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out)
    elif a.stage == "extract":
        extract(a.out, a.workers)
    elif a.stage == "train":
        if a.seed not in CONFIG["train_seeds"]:
            raise SystemExit("unregistered seed")
        train(a.out, a.seed)
    elif a.stage == "export":
        export(a.out)
    elif a.stage == "prepare-panel":
        prepare_panel(a.out)
    else:
        evaluate(a.out, a.binary, a.workers)


if __name__ == "__main__":
    main()
