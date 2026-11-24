"""P112: first source-disjoint test of the gated envelope mode against the P74 parent.

Reserves up to 16 stereo works isolated from every processed/archived manifest
and historical run catalog by file, decoded-PCM and connected work aliases (the
P88 rule), writes an evaluation catalog so future training excludes them,
prepares MP3 64 / AAC 96 / Vorbis q3 / clean pairs exactly as P87/P88, and
renders untreated, the native P74 parent (context-replay, SmoothKnee) and both
gated envelope seeds (stream-eval, SmoothKnee). Rules and hashes are frozen at
registration, before any audio is prepared. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audio_prep import (  # noqa: E402
    align_codec_pair,
    encode_decode_ffmpeg,
    estimate_delay_samples,
)
from data.protocol import load_recipe_from_profile  # noqa: E402
from data.source_quality import inspect_master  # noqa: E402
from evaluation import missing_band  # noqa: E402
from evaluation.adapter_runtime import environment  # noqa: E402
from evaluation.longform_review import (  # noqa: E402
    aliases,
    digest,
    history_files,
    publisher_evidence,
    source_candidates,
)
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa, write_sxa  # noqa: E402
from scripts import run_envelope_gate_p111 as p111  # noqa: E402
from scripts import run_envelope_p108 as p108  # noqa: E402
from scripts.run_band_training import jsonl  # noqa: E402
from scripts.score_missing_band import check_lag  # noqa: E402

REPO = p108.REPO
BURN, HOP = 89, 128
CODECS = ("mp3-cbr-64", "aac-lc-cbr-96", "vorbis-vbr-q3")
WORKS = 16
P88 = json.loads((p108.PANEL / "registration.json").read_text())
P74_GRAPH = REPO / "data/preserved/p107-best-p74-20261013-magnitude/diagnostic.onnx"
P74_BRIDGE = REPO / "data/preserved/p107-best-p74-20261013-magnitude/soundex-context-replay"
P111_RUN = REPO / "data/runs/envelope-gate-p111-20261124"
HEADLINE = p111.HEADLINE
RULES = (
    "Prospective; first source-disjoint test of the frozen P111 gated candidate (both seeds). "
    "(1) Restoration: missing-region half-ERB RMS below the native P74 parent with a paired "
    "recording interval below zero, in both seeds. (2) Clean input: preserved LSD versus P74 "
    "not an 'increase' in both seeds; absolute clean >4 kHz error in PCM dBFS reported beside "
    "it. (3) Reported: missing LSD, ERB bias, temporal correlation, paired preserved LSD, "
    "per-codec means, every adverse recording. Coverage is limited to the works the isolation "
    "rule leaves (declared genre/author gaps); no artist-independence or listening claim; no "
    "threshold changes after scoring; no promotion."
)


def reserve():
    data = REPO / "data"
    histories = history_files(data)
    inventory = sorted(set(histories) | set((data / "catalogs").glob("*.jsonl")))
    universe = list({frozenset(aliases(r)): r for p in inventory for r in jsonl(p)}.values())
    masters = jsonl(data / "catalogs/current-approved.jsonl")
    eligible, _ = source_candidates(masters, histories)
    bridges = {frozenset(aliases(r)) for r in universe}

    def closure(keys):
        blocked, before = set(keys), -1
        while before != len(blocked):
            before = len(blocked)
            for component in bridges:
                if component & blocked:
                    blocked.update(component)
        return blocked

    used = closure({k for p in histories for r in jsonl(p) for k in aliases(r)})
    pool = [
        {**r, "evaluation_rights_evidence": publisher_evidence(r)}
        for r in eligible
        if r["source_quality"]["channels"] == 2 and not aliases(r) & used
    ]
    count = len(pool)
    chosen, authors, genres = [], Counter(), Counter()
    while pool and len(chosen) < WORKS:
        row = min(
            pool,
            key=lambda r: (
                genres[r["genre"]],
                authors[r["author"]],
                digest([112, r["audio_sha256"]]),
            ),
        )
        chosen.append(row)
        authors[row["author"]] += 1
        genres[row["genre"]] += 1
        protected = closure(aliases(row))
        pool = [r for r in pool if not aliases(r) & protected]
    return chosen, count, [str(p) for p in histories]


def register(out, binary, commit):
    out.mkdir(parents=True, exist_ok=False)
    sources, eligible, histories = reserve()
    if len(sources) < 8:
        raise ValueError("fewer than eight isolated stereo works")
    with (out / "evaluation-catalog.jsonl").open("x") as f:
        for row in sources:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    p111_reg = json.loads((P111_RUN / "registration.json").read_text())
    hashes = {
        str(p): p108.sha(p) for p in (binary, P74_GRAPH, P74_BRIDGE, P88["ffmpeg"], P88["profile"])
    }
    for art in p111_reg["artifacts"].values():
        if p108.sha(art["path"]) != art["sha256"]:
            raise ValueError("P111 artifact drift")
        hashes[art["path"]] = art["sha256"]
    code = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_envelope_gate_p111.py",
        ROOT / "evaluation/missing_band.py",
        ROOT / "evaluation/envelope_fill.py",
    ]
    (out / "registration.json").write_text(
        json.dumps(
            {
                "study": "p112-envelope-fresh-sources",
                "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "rules": RULES,
                "sources": sources,
                "eligible_isolated_stereo_works": eligible,
                "coverage": {
                    "works": len(sources),
                    "genres": dict(Counter(r["genre"] for r in sources)),
                    "authors": dict(Counter(r["author"] for r in sources)),
                },
                "history_files": histories,
                "artifacts": p111_reg["artifacts"],
                "binary": str(binary),
                "binary_commit": commit,
                "p74": {"graph": str(P74_GRAPH), "bridge": str(P74_BRIDGE)},
                "ffmpeg": P88["ffmpeg"],
                "profile": P88["profile"],
                "hashes": hashes,
                "code_sha256": {str(p.relative_to(REPO)): p108.sha(p) for p in code},
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "works": len(sources),
                "eligible": eligible,
                "genres": Counter(r["genre"] for r in sources),
            }
        )
    )


def prepare(out):
    """P87/P88 pair preparation (13 s crop, alternating 44.1/48 kHz, aligned codecs)."""
    c = json.loads((out / "registration.json").read_text())
    os.environ["PATH"] = str(Path(c["ffmpeg"]).parent) + os.pathsep + os.environ["PATH"]
    recipe = load_recipe_from_profile(Path(c["profile"]))
    codecs = [next(k for k in recipe.codecs if k.id == name) for name in CODECS]
    cases = []
    for i, s in enumerate(c["sources"]):
        directory = out / "inputs" / f"{i:02d}"
        directory.mkdir(parents=True)
        q = inspect_master(Path(s["path"]), s["audio_sha256"])
        if (
            q["review_flags"]
            or q["decoded_pcm_sha256"] != s["decoded_pcm_sha256"]
            or q["channels"] != 2
        ):
            raise ValueError("stereo source integrity changed")
        rate, sr = (44100, 48000)[i % 2], q["sample_rate"]
        start = int((q["duration_seconds"] - 13) * (0.25, 0.65)[i % 2] * sr)
        original, actual = sf.read(
            s["path"], start=start, frames=13 * sr, dtype="float32", always_2d=True
        )
        if actual != sr or original.shape != (13 * sr, 2):
            raise ValueError("incomplete simultaneous stereo source")
        g = math.gcd(sr, rate)
        clean = np.ascontiguousarray(
            resample_poly(original, rate // g, sr // g, axis=0), dtype=np.float32
        )
        origin = 2 * rate - (BURN - 1) * HOP
        frames = math.ceil((BURN * HOP + 10 * rate) / HOP) * HOP
        reference = clean[origin : origin + frames]
        write_sxa(directory / "reference.sxa", reference, rate)
        for codec in [*codecs, None]:
            label = codec.id if codec else "identity"
            if codec:
                pair = align_codec_pair(
                    clean, encode_decode_ffmpeg(clean, rate, codec), rate, recipe
                )
                lo = origin - max(0, -pair.offset_samples)
                matched, wave = pair.clean[lo : lo + frames], pair.degraded[lo : lo + frames]
            else:
                matched = wave = reference
            if not np.array_equal(matched, reference) or wave.shape != reference.shape:
                raise ValueError("codec or clean role changed original reference support")
            residual = [
                estimate_delay_samples(
                    matched[:, ch],
                    wave[:, ch],
                    rate,
                    max_offset_samples=4,
                    correlation_samples=65536,
                )
                for ch in range(2)
            ]
            if any(abs(d) > recipe.alignment.residual_tolerance_samples for d in residual):
                raise ValueError("physical-channel residual alignment")
            write_sxa(directory / f"{label}.sxa", wave, rate)
            cases.append(
                {
                    "id": f"{i:02d}-{label}",
                    "index": i,
                    "source": s["audio_sha256"],
                    "role": "paired" if codec else "identity",
                    "codec": label,
                    "rate": rate,
                    "input": str(directory / f"{label}.sxa"),
                    "ref": str(directory / "reference.sxa"),
                    "residual": list(map(int, residual)),
                }
            )
    (out / "cases.json").write_text(json.dumps(cases, indent=1))
    print(json.dumps({"cases": len(cases)}))


def _case(task):
    out, case, reg = task
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = BURN * HOP - HOP, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows = []

    def add(label, wave):
        scores = missing_band.score(wave, target, mask_dry, rate)
        scores["clean_hf_error_dbfs"] = (
            p111.hf_error_dbfs(wave, mask_dry, rate) if case["role"] == "identity" else None
        )
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

    add("untreated", mask_dry)
    work = out / "renders" / case["id"]
    (work / "p74").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            reg["p74"]["bridge"],
            reg["p74"]["graph"],
            case["input"],
            str(work / "p74/output.sxa"),
            str(work / "p74/runtime.json"),
            "1",
        ],
        env=environment(),
        check=True,
        capture_output=True,
    )
    report = json.loads((work / "p74/runtime.json").read_text())
    p74, _ = read_sxa(work / "p74/output.sxa")
    if not report["finite"] or report["dsp_delay_samples"] != HOP or p74.shape != dry.shape:
        raise ValueError("P74 native output contract")
    check_lag(p74, dry, rate, HOP)
    add("p74_parent", p74[begin + HOP : begin + HOP + n])
    for seed, art in reg["artifacts"].items():
        (work / f"gated{seed}").mkdir(exist_ok=True)
        env = dict(
            os.environ,
            SOUNDEX_EVAL_ENHANCEMENT_MODE="envelope",
            SOUNDEX_EVAL_LIMITER_MODE="smooth-knee",
        )
        subprocess.run(
            [
                reg["binary"],
                art["path"],
                case["input"],
                str(work / f"gated{seed}/output.sxa"),
                str(work / f"gated{seed}/report.json"),
                "offline",
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        wave, _ = read_sxa(work / f"gated{seed}/output.sxa")
        check_lag(wave, dry, rate, 0)
        add(f"gated_seed{seed}", wave[begin : begin + n])
    return rows


def evaluate(out, workers):
    reg = json.loads((out / "registration.json").read_text())
    for path, digest_ in reg["hashes"].items():
        if p108.sha(path) != digest_:
            raise ValueError(f"hash drift {path}")
    cases = json.loads((out / "cases.json").read_text())
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_case, [(out, c, reg) for c in cases]):
            rows.extend(r)
    with (out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary, contrasts = {}, {}
    for role in ("paired", "identity"):
        summary[role] = {}
        for codec in ["all", *sorted({r["codec"] for r in rows if r["role"] == role})]:
            table = {}
            for p in sorted({r["path"] for r in rows}):
                table[p] = {}
                for m in HEADLINE:
                    groups = {}
                    for r in rows:
                        if (
                            r["role"] == role
                            and r["path"] == p
                            and codec in ("all", r["codec"])
                            and r["scores"].get(m) is not None
                        ):
                            groups.setdefault(r["source"], []).append(r["scores"][m])
                    table[p][m] = (
                        float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
                    )
            summary[role][codec] = table
        keys = HEADLINE if role == "identity" else HEADLINE[:-1]
        trimmed = [
            {**r, "scores": {k: r["scores"][k] for k in keys}} for r in rows if r["role"] == role
        ]
        for seed in reg["artifacts"]:
            for b in ("p74_parent", "untreated"):
                c = strict_paired_recordings(trimmed, f"gated_seed{seed}", b)
                contrasts[f"{role}: gated_seed{seed} - {b}"] = {
                    k: {
                        "mean_delta": c[k].get("mean_delta"),
                        "ci95": c[k].get("paired_source_bootstrap_95"),
                        "sources": c[k].get("sources"),
                        "undefined": [u[:12] for u in c[k].get("undefined_sources", [])],
                        "adverse": [a[:12] for a in c[k].get("adverse_sources", [])],
                    }
                    for k in keys
                }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    print(json.dumps({"rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "prepare", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--workers", type=int, default=12)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out, a.binary, a.commit)
    elif a.stage == "prepare":
        prepare(a.out)
    else:
        evaluate(a.out, a.workers)


if __name__ == "__main__":
    main()
