"""P115: more distinct, rights-screened works at a matched exposure budget, x capacity.

Training works = TRAIN-split library works with an accepted manual rights review +
rights-screen-passing (v3) expansion works with a maintainer acceptance record, isolated from every evaluation
catalog (P112-P115 panels included). Codec pairs are generated directly from the
publisher originals with the project codec/alignment code and recipe weights;
only features are stored. Total segments and updates match P113 so arm A versus
P113 isolates work diversity; arm B (hidden 192) tests capacity. Sampling is
genre-tempered (genre mass proportional to sqrt(work count)). Final endpoints
only. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audio_prep import align_codec_pair, encode_decode_ffmpeg  # noqa: E402
from data.protocol import load_recipe_from_profile  # noqa: E402
from data.rights_screen import POLICY_VERSION, screen  # noqa: E402
from evaluation.longform_review import aliases, history_files, source_candidates  # noqa: E402
from models.envelope_predictor import (  # noqa: E402
    BANDS,
    LOG_FLOOR,
    EnvelopePredictor,
    band_matrix,
    envelope_loss,
    features,
    parameter_count,
    stft_power,
)
from scripts import run_envelope_p108 as p108  # noqa: E402
from scripts import run_envelope_p109 as p109  # noqa: E402
from scripts import run_envelope_scale_p113 as p113  # noqa: E402

REPO = p108.REPO
P113_RUN = REPO / "data/runs/envelope-scale-p113-20261124"
PROFILE = REPO / "training/configs/music_library.yaml"
FFMPEG = "/opt/homebrew/Cellar/ffmpeg-full/9.0.2_1/bin/ffmpeg"
EXPANSION = [
    REPO / "data/runs/reserved-panel-p113-20261124/verified-candidates.jsonl",
    REPO / "data/runs/reserved-panel-p114-20261124/verified-candidates.jsonl",
]
CONFIG = {
    **{
        k: p113.CONFIG[k]
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
            "steps",
            "identity_share",
            "gate_db",
            "layers",
        )
    },
    "seed": 115,
    "target_segments": 132298,  # = P113 training segments.
    "crops_per_work": 8,
    "crop_seconds": [2.0, 4.0],
    "segments_per_row": 2,
    "validation_works_fraction": 0.05,
    "genre_temper": 0.5,
    "arms": {"A": 128, "B": 192},
    "train_seeds": [1, 2],
}
RULES = (
    "Prospective. Fresh panels: P115 multi-genre (24) and P114 electronic (16); primary analysis "
    "on rights-screen-passing works of each panel (all works also reported). (1) Work diversity: "
    "arm A (P113 architecture, matched segments/updates) has missing-region half-ERB RMS below the "
    "preserved P113 candidate, pooled over both panels, interval below zero in both seeds. "
    "(2) Capacity: arm B below arm A of the same seed, pooled, interval below zero in both seeds. "
    "(3) Candidate: an arm qualifies if, in both seeds and on each panel separately, ERB RMS is "
    "below the native P74 parent (interval below zero) and clean preserved LSD versus P74 is not "
    "an 'increase'. If both qualify, B is preferred only if rule 2 holds; otherwise A. "
    "(4) Runtime: arm B native worker processing max below 2.0 ms per hop at 48 kHz stereo. "
    "Hip hop remains an unresolved coverage gap. No promotion."
)


def _passes(row):
    return not screen(row, json.loads(Path(row["source_metadata_path"]).read_text()))


ACCEPTANCE = REPO / "data/catalogs/expansion-review-acceptance.jsonl"


def library_reviewed():
    """TRAIN-split library works with an accepted manual rights review (all 300 have one)."""
    seen = {}
    for r in p108.jsonl(p108.LIBRARY / "manifest.jsonl"):
        if r["clean_path"].startswith("audio/train/"):
            seen.setdefault(r["source_id"], r["source_metadata"])
    works = []
    for m in seen.values():
        review = m.get("source_review") or {}
        if (
            review.get("status") == "accepted"
            and review.get("commercial_training") is True
            and review.get("apache_weight_distribution") is True
            and not review.get("unresolved_issues")
        ):
            works.append({**m, "id": m.get("source_file"), "path": m["source_file"]})
    return works


def pool(out_dir):
    """Rights-accepted training works, isolated from every evaluation catalog.

    Library works need their accepted manual review; expansion works additionally
    need a maintainer acceptance record (the automated screen alone is not enough).
    """
    hashes, groups = p108.reserved_hashes()
    library = library_reviewed()
    accepted = set()
    if ACCEPTANCE.exists():
        accepted = {
            r["audio_sha256"] for r in p108.jsonl(ACCEPTANCE) if r.get("status") == "accepted"
        }
    verified = [r for p in EXPANSION for r in p108.jsonl(p)]
    eligible, _ = source_candidates(verified, history_files(REPO / "data"))
    expansion_pass = [r for r in eligible if _passes(r)]
    expansion_ok = [r for r in expansion_pass if r["audio_sha256"] in accepted]

    def reserved(r):
        keys = aliases(r)
        return any(k[1] in hashes for k in keys if k[0] == "hash") or any(
            k[1] in groups for k in keys if k[0] == "work"
        )

    works, seen = [], set()
    for origin, rows in (("library", library), ("expansion", expansion_ok)):
        for r in rows:
            if reserved(r) or r["audio_sha256"] in seen:
                continue
            seen.add(r["audio_sha256"])
            path = Path(r["path"]) if Path(r["path"]).is_absolute() else REPO / "data" / r["path"]
            works.append(
                {
                    "id": r["id"],
                    "audio_sha256": r["audio_sha256"],
                    "path": str(path),
                    "genre": r.get("genre", "unlabeled"),
                    "author": r.get("author"),
                    "origin": origin,
                }
            )
    report = {
        "library_reviewed_train": len(library),
        "expansion_eligible": len(eligible),
        "expansion_screen_pass": len(expansion_pass),
        "expansion_maintainer_accepted": len(expansion_ok),
    }
    return works, report


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    works, report = pool(out)
    if report["expansion_maintainer_accepted"] == 0:
        raise SystemExit(
            "no maintainer-accepted expansion works: P115 tests work diversity and "
            f"needs {ACCEPTANCE.name} (see docs/training-rights-screen.md)"
        )
    rng = np.random.default_rng(CONFIG["seed"])
    by_genre = defaultdict(list)
    for w in works:
        by_genre[w["genre"]].append(w)
    validation = []
    for _genre, items in sorted(by_genre.items()):
        k = (
            max(1, round(len(items) * CONFIG["validation_works_fraction"]))
            if len(items) >= 10
            else 0
        )
        validation += [
            items[int(i)]["audio_sha256"] for i in rng.choice(len(items), k, replace=False)
        ]
    for w in works:
        w["split"] = "validation" if w["audio_sha256"] in validation else "train"
    train = [w for w in works if w["split"] == "train"]
    per_work = CONFIG["target_segments"] / len(train)
    counts = Counter(w["genre"] for w in train)
    for w in works:
        w["sampling_weight"] = (
            counts[w["genre"]] ** (CONFIG["genre_temper"] - 1) if w["split"] == "train" else 0.0
        )
    code = [
        Path(__file__).resolve(),
        ROOT / "data/rights_screen.py",
        ROOT / "scripts/run_envelope_p109.py",
        ROOT / "models/envelope_predictor.py",
    ]
    registration = {
        "study": "p115-envelope-diverse",
        "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": CONFIG,
        "rules": RULES,
        "rights_screen": POLICY_VERSION,
        "pool_report": report,
        "works": works,
        "train_works": len(train),
        "validation_works": len(works) - len(train),
        "segments_per_train_work": per_work,
        "genres_train": dict(counts),
        "hip_hop_gap": "no rights-safe hip-hop source; see docs/training-rights-screen.md",
        "profile": str(PROFILE),
        "profile_sha256": p108.sha(PROFILE),
        "ffmpeg": FFMPEG,
        "ffmpeg_sha256": p108.sha(FFMPEG),
        "panel_catalogs": {
            name: p108.sha(path)
            for name, path in (
                ("p115", REPO / "data/runs/reserved-panel-p115-20261125/evaluation-catalog.jsonl"),
                ("p114", REPO / "data/runs/reserved-panel-p114-20261124/evaluation-catalog.jsonl"),
            )
        },
        "code_sha256": {str(p.relative_to(REPO)): p108.sha(p) for p in code},
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=1))
    print(
        json.dumps(
            {
                "works": len(works),
                "train": len(train),
                "genres": dict(counts),
                "segments_per_work": round(per_work, 1),
                **report,
            }
        )
    )


def _generate(task):
    """Codec pairs from one original; returns per-row feature segments and provenance."""
    work, quota, index, seed = task
    os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ["PATH"]
    recipe = load_recipe_from_profile(PROFILE)
    codecs = [c for c in recipe.codecs if not c.held_out]
    roles = dict(recipe.channel_roles)
    rng = np.random.default_rng([seed, int(work["audio_sha256"][:12], 16)])
    info = sf.info(work["path"])
    sr, total = info.samplerate, info.frames
    stereo = info.channels >= 2
    segs, prov = [], []
    starts = np.sort(rng.uniform(0.05, 0.95, CONFIG["crops_per_work"]))
    for c, frac in enumerate(starts):
        seconds = rng.uniform(*CONFIG["crop_seconds"]) + 0.5
        frames = int(seconds * sr)
        start = int(frac * max(total - frames, 0))
        audio, _ = sf.read(
            work["path"], start=start, frames=frames, dtype="float32", always_2d=True
        )
        if len(audio) < frames // 2:
            continue
        audio = audio[:, :2] if stereo else audio[:, :1]
        rate = (44100, 48000)[(index + c) % 2]
        g = math.gcd(sr, rate)
        clean = np.ascontiguousarray(
            resample_poly(audio, rate // g, sr // g, axis=0), dtype=np.float32
        )
        for codec in codecs:
            try:
                pair = align_codec_pair(
                    clean, encode_decode_ffmpeg(clean, rate, codec), rate, recipe
                )
            except Exception as error:  # Retained as provenance, never silently dropped.
                prov.append({"crop": c, "codec": codec.id, "error": str(error)})
                continue
            if stereo:
                names = ["left", "right", "mid", "side"]
                role = names[
                    int(
                        rng.choice(
                            4, p=np.array([roles[n] for n in names]) / sum(roles[n] for n in names)
                        )
                    )
                ]
                pick = {
                    "left": lambda x: x[:, 0],
                    "right": lambda x: x[:, 1],
                    "mid": lambda x: x.mean(1),
                    "side": lambda x: (x[:, 0] - x[:, 1]) / 2,
                }[role]
            else:
                role, pick = "mono", (lambda x: x[:, 0])
            cl, dg = pick(pair.clean).astype(np.float64), pick(pair.degraded).astype(np.float64)
            pd, _ = stft_power(dg)
            pr, _ = stft_power(cl)
            xd, rd = features(pd, rate)
            xc, rc = features(pr, rate)
            m = band_matrix(rate)
            er, ed = pr @ m, pd @ m
            level = np.log10(er + LOG_FLOOR)
            deficit = np.log10(np.maximum(er - ed, p109.DEFICIT_FLOOR_REL * er) + LOG_FLOOR)
            count = len(xd) - CONFIG["segment_frames"]
            if count <= 0:
                continue
            for s in np.sort(
                rng.choice(count, min(CONFIG["segments_per_row"], count), replace=False)
            ):
                sl = slice(s, s + CONFIG["segment_frames"])
                segs.append(
                    (
                        xd[sl, :129].astype(np.float16),
                        np.stack([rd[sl], xd[sl, 130]], -1).astype(np.float32),
                        xc[sl, :129].astype(np.float16),
                        np.stack([rc[sl], xc[sl, 130]], -1).astype(np.float32),
                        level[sl].astype(np.float16),
                        deficit[sl].astype(np.float16),
                    )
                )
            prov.append(
                {
                    "crop": c,
                    "start": start,
                    "rate": rate,
                    "codec": codec.id,
                    "role": role,
                    "offset": int(pair.offset_samples),
                }
            )
    if quota is not None and len(segs) > quota:
        keep = np.sort(rng.choice(len(segs), quota, replace=False))
        segs = [segs[i] for i in keep]
    return work["audio_sha256"], segs, prov


def generate(out, workers, limit=None):
    reg = json.loads((out / "registration.json").read_text())
    cache = out / "cache"
    cache.mkdir(exist_ok=limit is not None)
    shapes = {
        "xd": (129, np.float16),
        "rd": (2, np.float32),
        "xc": (129, np.float16),
        "rc": (2, np.float32),
        "level": (BANDS, np.float16),
        "deficit": (BANDS, np.float16),
    }
    for split in ("train", "validation"):
        works = [w for w in reg["works"] if w["split"] == split][:limit]
        quota = math.ceil(reg["segments_per_train_work"]) if split == "train" else 40
        capacity = len(works) * quota
        arrays = {
            k: np.lib.format.open_memmap(
                cache / f"{split}-{k}.npy", "w+", dt, (capacity, CONFIG["segment_frames"], d)
            )
            for k, (d, dt) in shapes.items()
        }
        meta, provenance, index = [], {}, 0
        tasks = [(w, quota, i, CONFIG["seed"]) for i, w in enumerate(works)]
        with ProcessPoolExecutor(workers) as pool:
            for work_hash, segs, prov in pool.map(_generate, tasks, chunksize=1):
                provenance[work_hash] = prov
                for seg in segs:
                    for key, value in zip(p109.KEYS, seg, strict=True):
                        arrays[key][index] = value
                    meta.append(work_hash)
                    index += 1
        for arr in arrays.values():
            arr.flush()
        (cache / f"{split}-count.json").write_text(json.dumps({"count": index, "works": meta}))
        (cache / f"{split}-provenance.json").write_text(json.dumps(provenance))
        print(json.dumps({"split": split, "works": len(works), "segments": index}))


def train(out, arm, seed):
    torch.set_num_threads(int(os.environ.get("P115_THREADS", "4")))
    reg = json.loads((out / "registration.json").read_text())
    hidden = CONFIG["arms"][arm]
    directory = out / "training" / f"{arm}-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    data, val = p109._load(out / "cache", "train"), p109._load(out / "cache", "validation")
    work_of = json.loads((out / "cache/train-count.json").read_text())["works"]
    weight = {w["audio_sha256"]: w["sampling_weight"] for w in reg["works"]}
    per_work = Counter(work_of)
    probs = np.array([weight[h] / per_work[h] for h in work_of], np.float64)
    cdf = np.cumsum(probs / probs.sum())
    torch.manual_seed(seed)
    model = EnvelopePredictor("band", hidden, CONFIG["layers"])
    opt = torch.optim.AdamW(model.parameters(), CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps, rng = CONFIG["steps"], np.random.default_rng([CONFIG["seed"], seed, hidden])

    def snapshot(step):
        return {
            "step": step,
            "val_degraded": p109._validate(model, val, "deficit", 0.0),
            "val_identity": p109._validate(model, val, "deficit", 1.0),
        }

    history, started = [snapshot(0)], time.time()
    for step in range(1, steps + 1):
        lr = CONFIG["lr_final"] + (CONFIG["lr"] - CONFIG["lr_final"]) * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / steps)
        )
        for group in opt.param_groups:
            group["lr"] = lr
        idx = np.sort(np.minimum(np.searchsorted(cdf, rng.random(CONFIG["batch"])), len(cdf) - 1))
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
        {"model": model.state_dict(), "arm": arm, "hidden": hidden, "seed": seed, "config": CONFIG},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "generate", "smoke", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(CONFIG["arms"]))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=14)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out)
    elif a.stage == "generate":
        generate(a.out, a.workers)
    elif a.stage == "smoke":
        generate(a.out, a.workers, limit=3)
    else:
        if a.arm is None or a.seed not in CONFIG["train_seeds"]:
            raise SystemExit("train requires a registered --arm and --seed")
        train(a.out, a.arm, a.seed)


if __name__ == "__main__":
    main()
