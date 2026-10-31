"""P67 verify differentiable fusion against every frozen P66 native stage trace.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from models.fused_spectrum import fused_stages, prepare_fusion
from models.losses import build_missing_band_mask
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def snapshot(output):
    source = Path(__file__).resolve().parents[1]
    for path in source.rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".py", ".json", ".toml", ".html", ".yaml"}
            and not set(path.relative_to(source).parts) & {".venv", "outputs", "__pycache__"}
        ):
            target = output / "source/training" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve failures")
    checked_receipt(prior / "complete.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    config = read(prior / "registration.json")
    hashes.update(config["hashes"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "format": "soundex-fused-spectrum-audit-p67-v1",
            "prior": str(prior),
            "source": config["prior"],
            "traces": sorted(str(p) for p in (prior / "renders").glob("*/*/runtime.json")),
            "gates": {
                "complex_relative_rms": 2e-4,
                "pcm_max": 0.00006103515625,
                "pcm_rms": 2e-5,
                "mask_max": 2e-5,
            },
            "rationale": "Existing feature parity v2 complex/inverse-FFT bounds and causal-renderer RMS bound, fixed before scoring. Same native observations isolate DSP; not an end-to-end frontend/model accuracy or quality claim.",
            "region": "all fully supported samples from 128 onward, including warmup and final emitted hop; report startup separately",
            "scope": "All 100 P66 TRAIN roles: four frozen graphs and identity DSP. No optimization, held-out scoring or timing rerun. Original references are not changed.",
            "hashes": hashes,
        },
        output / "registration.json",
    )
    print("registered all 100 P66 traces for fused correspondence", flush=True)


@torch.no_grad()
def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "assessment.json").exists():
        raise ValueError("existing audit retained")
    records = []
    for name in config["traces"]:
        path = Path(name)
        traces = read(path)["traces"]
        label, case = path.parent.parent.name, path.parent.name
        original_label = "20261012-zero" if label == "identity_dsp" else label
        source = Path(config["source"]) / "native-after" / original_label / case / "input.sxa"
        pcm, rate = read_sxa(source)
        if pcm.shape[1] != 1:
            raise ValueError("audit covers independent mono examples only")
        frames = len(traces)
        x = torch.tensor([h["features"] for h in traces]).reshape(frames, 2, 129)
        prediction = torch.tensor([h["prediction"] for h in traces]).reshape(frames, 2, 129)
        condition = torch.tensor([h["condition"] for h in traces])
        cutoff = condition[:, 1] * (rate / 2)
        weights = build_missing_band_mask(
            cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
        ).reshape(1, 1, frames, 129)
        dry = torch.from_numpy(np.pad(pcm[:, 0], (128, 0))[: len(pcm)])[None]
        batch = {
            "x": x.permute(1, 0, 2)[None],
            "mask": weights,
            "active": condition[:, 2].bool()[None],
            "dry": dry,
        }
        plan = prepare_fusion(batch)
        actual_weights = np.asarray([h["dsp"][0]["phase_weights"] for h in traces])
        mask_max = float(np.max(np.abs(plan["weights"].numpy()[0, 0] - actual_weights)))
        result = fused_stages(prediction.permute(1, 0, 2)[None], batch["x"], plan)
        features = result["features"][0].numpy()
        native_features = np.asarray(
            [[h["dsp"][0]["fused_db"], h["dsp"][0]["fused_phase"]] for h in traces]
        ).transpose(1, 0, 2)

        def complex_values(values):
            return 10 ** (values[0] / 20) * np.exp(1j * values[1])

        a, b = complex_values(features), complex_values(native_features)
        relative = float(
            np.sqrt(np.mean(np.abs(a - b) ** 2)) / max(1e-10, np.sqrt(np.mean(np.abs(b) ** 2)))
        )
        native, _ = read_sxa(path.parent / "output.sxa")
        expected = {
            "prelimit": np.asarray([h["dsp"][0]["phase_ola"] for h in traces]).reshape(-1),
            "wet": np.asarray([h["dsp"][0]["wet_limited"] for h in traces]).reshape(-1),
            "final": native[:, 0],
        }
        errors = {}
        for stage, reference in expected.items():
            difference = result[stage][0].numpy().astype(np.float64) - reference
            errors[stage] = {
                "maximum": float(np.max(np.abs(difference[128:]))),
                "rms": float(np.sqrt(np.mean(difference[128:] ** 2))),
                "startup_maximum": float(np.max(np.abs(difference[:128]))),
            }
        gates = config["gates"]
        passed = (
            mask_max <= gates["mask_max"]
            and relative <= gates["complex_relative_rms"]
            and all(
                e["maximum"] <= gates["pcm_max"] and e["rms"] <= gates["pcm_rms"]
                for e in errors.values()
            )
        )
        records.append(
            {
                "label": label,
                "case": case,
                "mask_max": mask_max,
                "complex_relative_rms": relative,
                "errors": errors,
                "passed": passed,
            }
        )
    verify(config["hashes"])
    registered(
        {"records": records, "passed": all(r["passed"] for r in records), "scope": config["scope"]},
        output / "assessment.json",
    )
    if not all(r["passed"] for r in records):
        raise ValueError("fused native correspondence failed; retain all counterexamples")
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run.log" and "__pycache__" not in p.parts
            }
        },
        output / "complete.json",
    )
    print("all 100 native stage correspondence cases passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(args.prior.resolve(), args.output.resolve())
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
