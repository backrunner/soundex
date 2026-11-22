"""P107: missing-band baseline and oracle headroom on an existing native panel.

Re-scores frozen native renders (no new training or model output) plus a blind
Rust DSP patch and reference-assisted parametric ceilings. Registration, with
code/binary/panel hashes and every constant, is written before any scoring.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import envelope_oracle, missing_band  # noqa: E402
from evaluation.restoration_evidence import strict_paired_recordings  # noqa: E402
from evaluation.rust_bridge import read_sxa  # noqa: E402

HOP = 128
BURN = 89  # P88 scored-region convention (models.fused_spectrum.BURN).
SECONDS = 10
NATIVE_LAG = HOP  # Frozen context-replay renders report dsp_delay_samples == 128.
HEADLINE = (
    "missing_lsd_db",
    "missing_erb_rmse_db",
    "missing_erb_bias_db",
    "missing_temporal_corr",
    "preserved_lsd_db",
    "preserved_complex_rel",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lowpass(x, rate, hz=4000):
    spec = np.fft.rfft(x, axis=0)
    spec[np.fft.rfftfreq(len(x), 1 / rate) > hz] = 0
    return np.fft.irfft(spec, len(x), axis=0)


def check_lag(wave, dry, rate, lag):
    """Retained-band alignment must hold exactly at `lag`, not one hop away."""
    seg = slice(20000, 150000)
    a, d = lowpass(wave[:200000], rate), lowpass(dry[:200000], rate)

    def err(k):
        return np.sqrt(np.mean(np.square(a[seg.start + k : seg.stop + k] - d[seg])))

    other = min(err(k) for k in (0, HOP, 2 * HOP) if k != lag)
    if not err(lag) < 0.05 * other:
        raise ValueError(f"alignment check failed at lag {lag}")


def model_paths(case_dir):
    out = {}
    for receipt in sorted(case_dir.rglob("output.sxa")):
        out[str(receipt.parent.relative_to(case_dir))] = receipt
    return out


def spectral_render(binary, graph, source, work):
    work.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, SOUNDEX_EVAL_ENHANCEMENT_MODE="spectral", SOUNDEX_EVAL_HIGH_BAND_GAIN="model"
    )
    subprocess.run(
        [binary, graph, source, str(work / "output.sxa"), str(work / "report.json"), "offline"],
        env=env,
        check=True,
        capture_output=True,
    )
    wave, _ = read_sxa(work / "output.sxa")
    return wave


def score_case(task):
    case, panel, out, binary, graph = task
    dry, rate = read_sxa(case["input"])
    ref, sr = read_sxa(case["ref"])
    if rate != sr or dry.shape != ref.shape or rate != case["rate"]:
        raise ValueError("case geometry drift")
    begin = BURN * HOP - HOP
    n = rate * SECONDS
    target, mask_dry = ref[begin : begin + n], dry[begin : begin + n]
    signals = {"untreated": mask_dry}
    for label, path in model_paths(panel / "renders" / case["id"]).items():
        wave, _ = read_sxa(path)
        check_lag(wave, dry, rate, NATIVE_LAG)
        signals[label] = wave[begin + NATIVE_LAG : begin + NATIVE_LAG + n]
    spectral = spectral_render(binary, graph, case["input"], out / "spectral" / case["id"])
    check_lag(spectral, dry, rate, 0)
    signals["spectral_dsp"] = spectral[begin : begin + n]
    for kind in envelope_oracle.ORACLES:
        signals[kind] = envelope_oracle.render(dry, ref, rate, kind)[begin : begin + n]
    rows = []
    for label, wave in signals.items():
        if wave.shape != target.shape or not np.isfinite(wave).all():
            raise ValueError(f"invalid scored signal {label}")
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


def recording_means(rows, path, metric):
    groups = defaultdict(list)
    for r in rows:
        if r["path"] == path and r["scores"][metric] is not None:
            groups[r["source"]].append(r["scores"][metric])
    values = [np.mean(v) for v in groups.values()]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def summarize(rows):
    summary = {}
    for role in sorted({r["role"] for r in rows}):
        for codec in ["all", *sorted({r["codec"] for r in rows if r["role"] == role})]:
            sel = [r for r in rows if r["role"] == role and codec in ("all", r["codec"])]
            paths = sorted({r["path"] for r in sel})
            table = {}
            for p in paths:
                table[p] = {}
                for m in HEADLINE:
                    mean, count = recording_means(sel, p, m)
                    table[p][m] = mean
                    table[p][m + "_recordings"] = count
            if role == "paired":
                for p in paths:
                    for m in ("missing_lsd_db", "missing_erb_rmse_db"):
                        for ref_path, suffix in (
                            ("oracle_patch_erb", "headroom_vs_patch_erb"),
                            ("oracle_complex", "headroom_vs_complex"),
                        ):
                            base, oracle = table["untreated"][m], table[ref_path][m]
                            cur = table[p][m]
                            table[p][f"{m}_{suffix}"] = (
                                (base - cur) / (base - oracle)
                                if None not in (base, oracle, cur) and base != oracle
                                else None
                            )
            summary[f"{role}/{codec}"] = table
    return summary


def contrasts(rows):
    result = {}
    paired = [r for r in rows if r["role"] == "paired"]
    keys = set(HEADLINE)
    trimmed = [{**r, "scores": {k: v for k, v in r["scores"].items() if k in keys}} for r in paired]
    for p in sorted({r["path"] for r in paired} - {"untreated"}):
        result[f"{p} - untreated"] = strict_paired_recordings(trimmed, p, "untreated")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True, help="loaded but unused (spectral)")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cases = json.loads((args.panel / "cases.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    constants = {k: getattr(missing_band, k) for k in dir(missing_band) if k.isupper()}
    constants.update(
        {f"oracle_{k}": getattr(envelope_oracle, k) for k in dir(envelope_oracle) if k.isupper()}
    )
    registration = {
        "study": "p107-missing-band-baseline",
        "version": missing_band.VERSION,
        "panel": str(args.panel),
        "panel_cases_sha256": sha(args.panel / "cases.json"),
        "panel_status": "exposed development recheck (P88 works); not independent testing",
        "binary": str(args.binary),
        "binary_sha256": sha(args.binary),
        "binary_commit": args.commit,
        "graph_sha256": sha(args.graph),
        "code_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                ROOT / "evaluation/missing_band.py",
                ROOT / "evaluation/envelope_oracle.py",
                Path(__file__).resolve(),
            )
        },
        "constants": {k: list(v) if isinstance(v, tuple) else v for k, v in constants.items()},
        "scored_region": {
            "begin_sample": BURN * HOP - HOP,
            "seconds": SECONDS,
            "native_lag": NATIVE_LAG,
            "spectral_lag": 0,
        },
        "headline_metrics": list(HEADLINE),
        "rules": "Descriptive only: no acceptance threshold, candidate selection or "
        "promotion. Oracles use the reference and are not deployable.",
    }
    (args.out / "registration.json").write_text(json.dumps(registration, indent=1))
    tasks = [(c, args.panel, args.out, str(args.binary), str(args.graph)) for c in cases]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for chunk in pool.map(score_case, tasks):
            rows.extend(chunk)
    with (args.out / "records.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=1))
    (args.out / "contrasts.json").write_text(json.dumps(contrasts(rows), indent=1))
    print(json.dumps({"rows": len(rows), "cases": len(cases)}))


if __name__ == "__main__":
    main()
