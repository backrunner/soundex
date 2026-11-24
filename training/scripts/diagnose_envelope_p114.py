"""P114 diagnosis: where does the envelope mode lose on electronic material?

Zero-training decomposition on the exposed P112 (electronic) and P113 (multi-genre)
panels with the preserved P113 seed-1 model, through the exact Python fill proxy
(SmoothKnee applied, as native output). Variants separate the gate, level
under-prediction, detector region and copy-up source. Diagnostic only: no
candidate, threshold or improvement claim. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import envelope_fill as ef  # noqa: E402
from evaluation import envelope_oracle, missing_band  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402
from models.envelope_predictor import EnvelopePredictor, band_matrix, stft_power  # noqa: E402
from scripts.run_envelope_native_p110 import smooth_knee  # noqa: E402

REPO = ROOT.parent
GATE = (0.0, 6.0)
MODEL = REPO / "data/preserved/p113-envelope-scale/p113-deficit-id25-1.pth"
PANELS = {
    "p112_electronic": REPO / "data/runs/envelope-fresh-p112-20261124/cases.json",
    "p113_multigenre": REPO / "data/runs/envelope-scale-p113-20261124/panel/cases.json",
}
CATALOGS = {
    "p112_electronic": REPO / "data/runs/envelope-fresh-p112-20261124/evaluation-catalog.jsonl",
    "p113_multigenre": REPO / "data/runs/reserved-panel-p113-20261124/evaluation-catalog.jsonl",
}
VARIANTS = (
    "model_gated",
    "model_ungated",
    "model_shape_ref_level",
    "ref_deficit_detector",
    "ref_deficit_detector_gated",
    "oracle_patch_erb",
    "oracle_noise_erb",
)


def _channel(pcm, ref, rate, model, variant):
    """One physical channel; returns filled PCM and detector cutoff statistics."""
    power, spec = stft_power(pcm)
    m = band_matrix(rate)
    present = power @ m
    reference = stft_power(ref)[0] @ m
    true_deficit = np.maximum(reference - present, 0.0)
    if variant.startswith("model"):
        target = ef.predict(model, power, rate)
        if variant == "model_shape_ref_level":
            d = 10**target
            scale = true_deficit.sum(-1, keepdims=True) / np.maximum(
                d.sum(-1, keepdims=True), 1e-30
            )
            target = np.log10(d * scale + 1e-30)
    else:
        target = np.log10(true_deficit + 1e-30)
    gate = (
        GATE
        if variant in ("model_gated", "model_shape_ref_level", "ref_deficit_detector_gated")
        else None
    )
    original = ef.predict
    ef.predict = lambda *_: target
    try:
        out, _ = ef.fill_channel(pcm, rate, "band_deficit", model=None, gate_db=gate)
    finally:
        ef.predict = original
    padded = np.concatenate([np.zeros(128), pcm, np.zeros(len(spec) * 128 - len(pcm))])
    frames = np.lib.stride_tricks.sliding_window_view(padded, 256)[::128][: len(spec)]
    bw, needs = ef.detector(spec, frames, rate)
    return out, float(np.median(bw[needs])) if needs.any() else None


def _case(task):
    panel, case, genre = task
    torch.set_num_threads(1)
    state = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = EnvelopePredictor("band", state["config"]["hidden"], state["config"]["layers"])
    model.load_state_dict(state["model"])
    model.eval()
    dry, rate = read_sxa(case["input"])
    ref, _ = read_sxa(case["ref"])
    begin, n = 89 * 128 - 128, rate * 10
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    rows = []
    for variant in VARIANTS:
        if variant.startswith("oracle"):
            wave = envelope_oracle.render(dry, ref, rate, variant)
            cut = None
        else:
            chans = [_channel(dry[:, c], ref[:, c], rate, model, variant) for c in range(2)]
            wave = np.stack([c[0] for c in chans], 1)
            cut = [c[1] for c in chans]
        wave = smooth_knee(wave)[begin : begin + n]
        scores = missing_band.score(wave, target, mask_dry, rate)
        rows.append(
            {
                "panel": panel,
                "case": case["id"],
                "codec": case["codec"],
                "source": case["source"],
                "genre": genre,
                "variant": variant,
                "detected_cutoff_hz": cut,
                "scores": scores,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    tasks = []
    for panel, path in PANELS.items():
        genres = {
            json.loads(line)["audio_sha256"]: json.loads(line)["genre"]
            for line in CATALOGS[panel].read_text().splitlines()
            if line.strip()
        }
        for case in json.loads(path.read_text()):
            if case["role"] == "paired":
                tasks.append((panel, case, genres[case["source"]]))
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(_case, tasks):
            rows.extend(r)
    with (a.out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {}
    for panel in PANELS:
        for group in ["all", *sorted({r["genre"] for r in rows if r["panel"] == panel})]:
            table = {}
            for v in VARIANTS:
                g = defaultdict(list)
                for r in rows:
                    if (
                        r["panel"] == panel
                        and r["variant"] == v
                        and group in ("all", r["genre"])
                        and r["scores"]["missing_erb_rmse_db"] is not None
                    ):
                        s = r["scores"]
                        g[r["source"]].append(
                            (
                                s["missing_erb_rmse_db"],
                                s["missing_erb_bias_db"],
                                s["missing_lsd_db"],
                            )
                        )
                if g:
                    m = np.mean([np.mean(v_, axis=0) for v_ in g.values()], axis=0)
                    table[v] = {
                        "erb_rmse": float(m[0]),
                        "erb_bias": float(m[1]),
                        "lsd": float(m[2]),
                        "recordings": len(g),
                    }
            summary[f"{panel}/{group}"] = table
        cuts = [
            (c, r["scores"]["mask_cutoff_hz"])
            for r in rows
            if r["panel"] == panel and r["variant"] == "model_gated"
            for c in (r["detected_cutoff_hz"] or [])
            if c and r["scores"]["mask_cutoff_hz"]
        ]
        if cuts:
            diff = np.array([c - m for c, m in cuts])
            summary[f"{panel}/detector"] = {
                "median_detected_minus_true_hz": float(np.median(diff)),
                "abs_median_hz": float(np.median(np.abs(diff))),
                "pairs": len(cuts),
            }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"rows": len(rows)}))


if __name__ == "__main__":
    main()
