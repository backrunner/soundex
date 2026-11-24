"""P111: soft fill gate for clean-input preservation of the native envelope mode.

Re-exports both P109 deficit/identity seeds with fill_gate_db = [0, 6] (fill
scaled linearly in dB of fill/present energy; bounds from the target definition,
fixed before scoring) and renders the P88 panel natively with the SmoothKnee
limiter used by every comparator. Compares with the native P74 parent and the
ungated P110 v2 native renders. Registration first. SPDX-License-Identifier: Apache-2.0
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
from scripts import run_envelope_native_p110 as p110  # noqa: E402
from scripts import run_envelope_p108 as p108  # noqa: E402
from scripts.score_missing_band import check_lag  # noqa: E402

GATE_DB = (0.0, 6.0)
P110 = p108.REPO / "data/runs/envelope-native-p110-20261124-v2"
SEEDS = (1, 2)
HEADLINE = (*p110.HEADLINE, "clean_hf_error_dbfs")
RULES = (
    "Prospective. Gate [0, 6] dB on 10*log10(fill/present band energy): 0 dB is where a fill "
    "would only double existing band energy; 6 dB is where present content is <= 20% of "
    "the target; the deficit target floors identity bands at -30 dB, so clean bands sit far "
    "below the gate and fully missing bands far above it. (1) Correspondence: native "
    "recording means within 0.1 dB of the gated proxy through the same SmoothKnee curve. "
    "(2) Primary: in both seeds, clean preserved LSD versus the native P74 parent is not an "
    "'increase' AND missing-region half-ERB RMS versus P74 has an interval below zero. "
    "(3) Reported: gated minus ungated P110 native, absolute clean high-band (>4 kHz) error "
    "in PCM dBFS, per-codec means. Exposed panel; no promotion."
)


def register(out, binary, commit):
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    artifacts = {}
    for seed in SEEDS:
        ckpt = p110.P109 / "training" / f"deficit-id25-{seed}" / "endpoint.pth"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = EnvelopePredictor("band", state["config"]["hidden"], state["config"]["layers"])
        model.load_state_dict(state["model"])
        path = out / "artifacts" / f"deficit-id25-gate0-6-{seed}.sxenv"
        header = write(
            model.eval(),
            "deficit",
            path,
            warmup_frames=state["config"]["warmup_frames"],
            fill_gate_db=GATE_DB,
            provenance={"checkpoint_sha256": p108.sha(ckpt), "study": "P111"},
        )
        artifacts[str(seed)] = {
            "path": str(path),
            "sha256": p108.sha(path),
            "payload_sha256": header["payload_sha256"],
        }
    code = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_envelope_native_p110.py",
        ROOT / "export_envelope.py",
        ROOT / "evaluation/envelope_fill.py",
        ROOT / "evaluation/missing_band.py",
    ]
    (out / "registration.json").write_text(
        json.dumps(
            {
                "study": "p111-envelope-fill-gate",
                "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "rules": RULES,
                "gate_db": GATE_DB,
                "binary": str(binary),
                "binary_sha256": p108.sha(binary),
                "binary_commit": commit,
                "artifacts": artifacts,
                "native_settings": {
                    "SOUNDEX_EVAL_ENHANCEMENT_MODE": "envelope",
                    "SOUNDEX_EVAL_LIMITER_MODE": p110.LIMITER,
                    "mode": "offline",
                },
                "ungated_comparator": str(P110),
                "code_sha256": {str(p.relative_to(p108.REPO)): p108.sha(p) for p in code},
            },
            indent=1,
        )
    )
    print(json.dumps(artifacts))


def hf_error_dbfs(wave, dry, rate):
    """RMS (PCM dBFS) of the >4 kHz part of wave - dry: absolute added/removed high band."""
    spec = np.fft.rfft(wave - dry, axis=0)
    spec[np.fft.rfftfreq(len(wave), 1 / rate) < 4000] = 0
    d = np.fft.irfft(spec, len(wave), axis=0)
    return float(20 * np.log10(np.sqrt(np.mean(d**2)) + 1e-12))


def _case(task):
    out, case, binary, artifacts = task
    torch.set_num_threads(1)
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows, corr = [], {}

    def add(label, wave):
        scores = missing_band.score(wave, target, mask_dry, rate)
        scores["clean_hf_error_dbfs"] = (
            hf_error_dbfs(wave, mask_dry, rate) if case["role"] == "identity" else None
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

    for seed, art in artifacts.items():
        work = out / "renders" / case["id"] / f"seed{seed}"
        work.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            SOUNDEX_EVAL_ENHANCEMENT_MODE="envelope",
            SOUNDEX_EVAL_LIMITER_MODE=p110.LIMITER,
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
        check_lag(native, dry, rate, 0)
        state = torch.load(
            p110.P109 / "training" / f"deficit-id25-{seed}" / "endpoint.pth",
            map_location="cpu",
            weights_only=False,
        )
        model = EnvelopePredictor("band", state["config"]["hidden"], state["config"]["layers"])
        model.load_state_dict(state["model"])
        proxy, _ = envelope_fill.render(
            dry, rate, "band_deficit", model=model.eval(), gate_db=GATE_DB
        )
        a, b = native[begin : begin + n], p110.smooth_knee(proxy)[begin : begin + n]
        diff = a - b
        corr[seed] = {
            "error_to_signal_db": float(
                10 * np.log10(np.sum(diff**2) / max(np.sum(b**2), 1e-30) + 1e-30)
            )
        }
        add(f"gated_seed{seed}", a)
        add(f"gated_proxy_seed{seed}", b)
        ungated, _ = read_sxa(P110 / "renders" / case["id"] / f"seed{seed}" / "output.sxa")
        add(f"ungated_seed{seed}", ungated[begin : begin + n])
    p74_path = p108.PANEL / "renders" / case["id"] / "20261013-magnitude" / "output.sxa"
    p74, _ = read_sxa(p74_path)
    add("p74_parent", p74[begin + 128 : begin + 128 + n])
    add("untreated", mask_dry)
    return rows, {case["id"]: corr}


def evaluate(out, workers):
    reg = json.loads((out / "registration.json").read_text())
    if p108.sha(reg["binary"]) != reg["binary_sha256"] or any(
        p108.sha(a["path"]) != a["sha256"] for a in reg["artifacts"].values()
    ):
        raise ValueError("binary or artifact drift")
    cases = json.loads((p108.PANEL / "cases.json").read_text())
    rows, corr = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for r, c in pool.map(_case, [(out, c, reg["binary"], reg["artifacts"]) for c in cases]):
            rows.extend(r)
            corr.update(c)
    with (out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "correspondence.json").write_text(json.dumps(corr, indent=1))
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
        keys = HEADLINE if role == "identity" else p110.HEADLINE
        trimmed = [
            {**r, "scores": {k: r["scores"][k] for k in keys}} for r in rows if r["role"] == role
        ]
        for seed in SEEDS:
            for a, b in (
                (f"gated_seed{seed}", "p74_parent"),
                (f"gated_seed{seed}", f"ungated_seed{seed}"),
                (f"gated_seed{seed}", f"gated_proxy_seed{seed}"),
                (f"ungated_seed{seed}", "p74_parent"),
            ):
                c = strict_paired_recordings(trimmed, a, b)
                contrasts[f"{role}: {a} - {b}"] = {
                    k: {
                        "mean_delta": c[k].get("mean_delta"),
                        "ci95": c[k].get("paired_source_bootstrap_95"),
                        "adverse": len(c[k].get("adverse_sources", [])),
                    }
                    for k in keys
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
    parser.add_argument("--workers", type=int, default=12)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out, a.binary, a.commit)
    else:
        evaluate(a.out, a.workers)


if __name__ == "__main__":
    main()
