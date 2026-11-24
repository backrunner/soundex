"""P110: native Rust correspondence, quality and timing of the P109 envelope candidate.

Exports both registered P109 deficit/identity seeds to `.sxenv`, renders every
P88 panel case through the native stream (`soundex-stream-eval`, Envelope mode,
offline delay-compensated), re-renders the Python fill proxy, and scores both
with the P107 rulers against the native P74 parent. Registration first.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import envelope_fill, missing_band  # noqa: E402
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from export_envelope import write  # noqa: E402
from models.envelope_predictor import EnvelopePredictor  # noqa: E402
from scripts import run_envelope_p108 as p108  # noqa: E402
from scripts.score_missing_band import check_lag  # noqa: E402

P109 = p108.REPO / "data/runs/envelope-clean-preservation-p109-20261122"
LIMITER = "smooth-knee"  # Matches soundex-context-replay, which rendered P74/P107 native paths.
CEILING = 0.95


def smooth_knee(x, ceiling=CEILING):
    """soundex-dsp LimiterMode::SmoothKnee, memoryless, applied to the proxy."""
    a = np.abs(x)
    shaped = ceiling * (0.9 + 0.1 * np.tanh((a / ceiling - 0.9) * 10.0))
    return np.where(a <= 0.9 * ceiling, x, np.sign(x) * np.minimum(shaped, ceiling))


SEEDS = (1, 2)
HEADLINE = (
    "missing_lsd_db",
    "missing_erb_rmse_db",
    "missing_erb_bias_db",
    "missing_temporal_corr",
    "preserved_lsd_db",
    "preserved_complex_rel",
)
RULES = (
    "Prospective. (1) Correspondence: for each seed and role, the native recording-mean "
    "missing ERB RMS and clean preserved LSD lie within 0.1 dB of the Python proxy; larger "
    "deviations are attributed (limiter, float32 detector/cutoff rounding, warmup) before any "
    "quality reading, and waveform error-to-signal ratios are reported. (2) Native quality: "
    "in both seeds, missing ERB RMS below the native P74 parent (paired recording interval "
    "below zero) and clean preserved LSD versus the P74 parent not an 'increase'. Bias, "
    "temporal correlation, LSD, paired preserved band and per-codec means are reported. "
    "(3) Runtime: realtime_bench Envelope mode at 44.1/48 kHz mono/stereo; report compute, "
    "deadline misses and the fixed buffer; no retries to pass. Exposed panel; no promotion."
)


def register(out, binary, commit):
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    artifacts = {}
    for seed in SEEDS:
        ckpt = P109 / "training" / f"deficit-id25-{seed}" / "endpoint.pth"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = EnvelopePredictor("band", state["config"]["hidden"], state["config"]["layers"])
        model.load_state_dict(state["model"])
        path = out / "artifacts" / f"deficit-id25-{seed}.sxenv"
        header = write(
            model.eval(),
            "deficit",
            path,
            warmup_frames=state["config"]["warmup_frames"],
            provenance={"checkpoint_sha256": p108.sha(ckpt), "study": "P109"},
        )
        artifacts[str(seed)] = {
            "path": str(path),
            "sha256": p108.sha(path),
            "payload_sha256": header["payload_sha256"],
            "checkpoint_sha256": p108.sha(ckpt),
        }
    code = [
        Path(__file__).resolve(),
        ROOT / "export_envelope.py",
        ROOT / "evaluation/envelope_fill.py",
        ROOT / "evaluation/missing_band.py",
        ROOT / "models/envelope_predictor.py",
    ]
    registration = {
        "study": "p110-envelope-native",
        "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rules": RULES,
        "binary": str(binary),
        "binary_sha256": p108.sha(binary),
        "binary_commit": commit,
        "artifacts": artifacts,
        "panel": str(p108.PANEL),
        "panel_cases_sha256": p108.sha(p108.PANEL / "cases.json"),
        "native_settings": {
            "SOUNDEX_EVAL_ENHANCEMENT_MODE": "envelope",
            "mode": "offline",
            "limiter": "legacy (stream-eval default)",
            "lag": 0,
        },
        "code_sha256": {str(p.relative_to(p108.REPO)): p108.sha(p) for p in code},
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=1))
    print(json.dumps({"artifacts": artifacts}))


def _case(task):
    out, case, binary, artifacts = task
    torch.set_num_threads(1)
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows, correspondence = [], {}
    for seed, art in artifacts.items():
        work = out / "renders" / case["id"] / f"seed{seed}"
        work.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ, SOUNDEX_EVAL_ENHANCEMENT_MODE="envelope", SOUNDEX_EVAL_LIMITER_MODE=LIMITER
        )
        subprocess.run(
            [
                binary,
                art["path"],
                case["input"],
                str(work / "output.sxa"),
                str(work / "report.json"),
                "offline",
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        native, _ = read_sxa(work / "output.sxa")
        if native.shape != dry.shape or not np.isfinite(native).all():
            raise ValueError("native output contract")
        check_lag(native, dry, rate, 0)  # Offline stream-eval output is delay-compensated.
        state = torch.load(
            P109 / "training" / f"deficit-id25-{seed}" / "endpoint.pth",
            map_location="cpu",
            weights_only=False,
        )
        model = EnvelopePredictor("band", state["config"]["hidden"], state["config"]["layers"])
        model.load_state_dict(state["model"])
        raw_proxy, _ = envelope_fill.render(dry, rate, "band_deficit", model=model.eval())
        proxy = smooth_knee(raw_proxy)
        a, b = native[begin : begin + n], proxy[begin : begin + n]
        diff = a - b
        correspondence[seed] = {
            "max_abs": float(np.abs(diff).max()),
            "error_to_signal_db": float(
                10 * np.log10(np.sum(diff**2) / max(np.sum(b**2), 1e-30) + 1e-30)
            ),
            "added_energy_native_vs_proxy_db": float(
                10
                * np.log10(
                    (np.sum((a - mask_dry) ** 2) + 1e-30) / (np.sum((b - mask_dry) ** 2) + 1e-30)
                )
            ),
            "native_peak": float(np.abs(a).max()),
        }
        raw = raw_proxy[begin : begin + n]
        for label, wave in (
            (f"native_seed{seed}", a),
            (f"proxy_seed{seed}", b),
            (f"proxy_unlimited_seed{seed}", raw),
        ):
            scores = missing_band.score(wave, target, mask_dry, rate)
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
    return rows, {case["id"]: correspondence}


def evaluate(out, workers):
    reg = json.loads((out / "registration.json").read_text())
    for seed, art in reg["artifacts"].items():
        if p108.sha(art["path"]) != art["sha256"]:
            raise ValueError(f"artifact drift seed {seed}")
    if p108.sha(reg["binary"]) != reg["binary_sha256"]:
        raise ValueError("binary drift")
    cases = json.loads((p108.PANEL / "cases.json").read_text())
    rows, corr = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for r, c in pool.map(_case, [(out, c, reg["binary"], reg["artifacts"]) for c in cases]):
            rows.extend(r)
            corr.update(c)
    keep = {"untreated", "20261013-magnitude", "identity_dsp", "legacy", "oracle_patch_erb"}
    rows.extend(r for r in p108.jsonl(p108.P107 / "records.jsonl") if r["path"] in keep)
    with (out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "correspondence.json").write_text(json.dumps(corr, indent=1))
    summary, contrasts = {}, {}
    for role in ("paired", "identity"):
        summary[role] = {}
        codecs = ["all", *sorted({r["codec"] for r in rows if r["role"] == role})]
        for codec in codecs:
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
                            and r["scores"][m] is not None
                        ):
                            groups.setdefault(r["source"], []).append(r["scores"][m])
                    table[p][m] = (
                        float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
                    )
            summary[role][codec] = table
        trimmed = [
            {**r, "scores": {k: r["scores"][k] for k in HEADLINE}}
            for r in rows
            if r["role"] == role
        ]
        for seed in SEEDS:
            for a, b in (
                (f"native_seed{seed}", "20261013-magnitude"),
                (f"native_seed{seed}", f"proxy_seed{seed}"),
                (f"native_seed{seed}", "untreated"),
            ):
                c = strict_paired_recordings(trimmed, a, b)
                contrasts[f"{role}: {a} - {b}"] = {
                    k: {
                        "mean_delta": c[k].get("mean_delta"),
                        "ci95": c[k].get("paired_source_bootstrap_95"),
                        "adverse": len(c[k].get("adverse_sources", [])),
                    }
                    for k in HEADLINE
                }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    print(json.dumps({"rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--workers", type=int, default=10)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out, a.binary, a.commit)
    else:
        evaluate(a.out, a.workers)


if __name__ == "__main__":
    main()
