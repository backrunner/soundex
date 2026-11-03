"""P70 frozen baseline identity, metric controls and phase-head interventions.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import copy
import traceback
from pathlib import Path

import numpy as np
import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.longform_review import aligned_output, score_audio
from evaluation.reporting import sha256_file, write_json_report
from evaluation.rust_bridge import read_sxa
from models.direct_spectrum_objective import direct_spectrum_loss
from models.envelope_probe import probe_inputs
from models.fused_spectrum import fused_spectrum_loss
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_cross_phase_training import load
from scripts.run_longform_review import checked_receipt


def independent(wave, reference, rate):
    """Independent NumPy HF rulers: fixed 2048/512 and absolute per-bin floor.

    These are diagnostic distances, not trained perceptual estimators. No shared
    training spectral_loss implementation, normalization or band weights.
    """
    size, hop = 2048, 512
    window = np.hanning(size + 1)[:-1]
    scale = np.sqrt(2 / (size * np.square(window).sum()))
    keep = np.fft.rfftfreq(size, 1 / rate) >= 8000

    def spectrum(pcm):
        frames = np.lib.stride_tricks.sliding_window_view(
            np.asarray(pcm, np.float64).T, size, axis=-1
        )[:, ::hop]
        return np.abs(np.fft.rfft(frames * window, axis=-1))[..., keep] * scale

    a, b = spectrum(wave), spectrum(reference)
    error = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    norm = np.linalg.norm(b.reshape(len(b), -1), axis=1)
    whole_a = np.fft.rfft(wave, axis=0) / len(wave)
    whole_b = np.fft.rfft(reference, axis=0) / len(reference)
    mask = np.fft.rfftfreq(len(wave), 1 / rate) >= 8000
    sides = np.full(len(mask), 2.0)
    sides[0] = sides[-1] = 1

    def power(z):
        return np.sum(np.abs(z[mask]) ** 2 * sides[mask, None], axis=0)

    return {
        "numpy_hf_sc": float(np.mean(error / np.maximum(norm, 1e-12))),
        "numpy_hf_log_rmse_db": float(
            np.sqrt(
                np.mean(
                    (20 * np.log10(np.maximum(a, 1e-7)) - 20 * np.log10(np.maximum(b, 1e-7))) ** 2
                )
            )
        ),
        "numpy_hf_error_rms": float(np.sqrt(power(whole_a - whole_b)).mean()),
        "numpy_hf_energy_error_db": float(
            np.abs(10 * np.log10((power(whole_a) + 1e-20) / (power(whole_b) + 1e-20))).mean()
        ),
    }


def controls(dry, reference, rate, seed):
    """Noncausal reference-informed metric fixtures, never deployable models."""
    x, y = [np.fft.rfft(v, axis=0) for v in (dry, reference)]
    mask = np.fft.rfftfreq(len(dry), 1 / rate) >= 8000
    rng = np.random.default_rng(seed)
    noise = np.abs(y) * np.exp(1j * rng.uniform(-np.pi, np.pi, y.shape))
    # Real Nyquist must retain its magnitude through irfft.
    noise[-1] = np.abs(y[-1])
    mixed = {
        "half_correct_hf": np.where(mask[:, None], (x + y) / 2, x),
        "correct_hf": np.where(mask[:, None], y, x),
        "same_energy_scrambled_hf": np.where(mask[:, None], noise, x),
        "mute_hf": np.where(mask[:, None], 0, x),
    }
    return {
        "reference": reference,
        **{k: np.fft.irfft(v, n=len(dry), axis=0) for k, v in mixed.items()},
    }


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh audit directory required")
    checked_receipt(prior / "p69-complete.json")
    checked_receipt(prior / "longform-recheck/complete.json")
    config = read(prior / "registration.json")
    source = Path(config["p55"])
    cases = [
        c for c in read(source / "cases.json") if c["excerpt"] == 0 and c["codec"] == "mp3-cbr-64"
    ]
    if len(cases) != 20 or len({c["source"] for c in cases}) != 20:
        raise ValueError("expected all twenty first-excerpt MP3 sources")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(config["hashes"])
    hashes.update(read(prior / "p69-complete.json")["files"])
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    hashes[str(prior / "registration.json")] = sha256_file(prior / "registration.json")
    summary = prior / "longform-recheck/summary.json"
    hashes[str(summary)] = sha256_file(summary)
    for e in read(prior / "endpoints.json").values():
        hashes[e["path"]] = e["sha256"]
    models = read(prior / "longform-recheck/registration.json")["models"]
    for case in cases:
        for label in models:
            directory = prior / "longform-recheck/renders" / label / case["id"]
            checked_receipt(directory / "complete.json")
            hashes.update(read(directory / "complete.json")["files"])
    registered(
        {
            "format": "soundex-baseline-metric-audit-p70-v1",
            "prior": str(prior),
            "cases": cases,
            "models": list(models),
            "hashes": hashes,
            "changed_question": "P54 audited short reused crops; P68 audited gradients before full training. P70 audits baseline identities and independent rulers on all twenty fixed first-excerpt MP3 P55 sources, and feature use after completed P69 endpoints. No new score-selected sources, candidate weights, training or latency claim.",
            "metric_protocol": "20 x 10 seconds, all physical channels. Reproduce 160 saved native scores under existing rtol2e-4/atol2e-7. Add independently implemented fixed2048/512 HF spectral convergence and absolute-floor log distance, PCM error and global HF energy. Five analytical controls plus untreated; FFT controls are noncausal/reference-informed diagnostics. Reference stationarity and correct-HF PCM improvement checked numerically; no post-hoc quality threshold or ranking rule.",
            "feature_protocol": "All four P69 endpoints, first eight registered TRAIN draws. Zero only added first-layer columns 645:903; preserve every other tensor. Compare original/ablated losses, hidden contribution and output rotation; inspect masked original-reference phase/magnitude substitutions separately. Substitution is not an optimum or deployable model; this branch reports model/raw and verified fusion proxy only, not newly rendered native quality.",
        },
        output / "registration.json",
    )


@torch.no_grad()
def metrics(config, output):
    prior = Path(config["prior"])
    saved = read(prior / "longform-recheck/summary.json")["records"]
    lookup = {(r["id"], r["path"]): r["scores"] for r in saved}
    results = []
    for i, case in enumerate(config["cases"]):
        reference, rate = read_sxa(Path(case["reference"]))
        dry, dry_rate = read_sxa(Path(case["input"]))
        if rate != dry_rate or rate != case["rate"]:
            raise ValueError("rate mismatch")
        start, count = case["score_start"], case["score_length"]
        reference, dry = reference[start : start + count], dry[start : start + count]
        waves = {"untreated": dry, **controls(dry, reference, rate, 70 + i)}
        for label in config["models"]:
            wave, native_rate = read_sxa(
                prior / "longform-recheck/renders" / label / case["id"] / "native.sxa"
            )
            if native_rate != rate:
                raise ValueError("native rate mismatch")
            waves["native/" + label] = aligned_output(wave, start, count)
        scores = {}
        reproduction = 0.0
        for label, wave in waves.items():
            score = score_audio(wave, reference, rate)
            if label in ("untreated", *("native/" + m for m in config["models"])):
                expected = lookup[(case["id"], label)]
                for key, value in score.items():
                    np.testing.assert_allclose(value, expected[key], rtol=2e-4, atol=2e-7)
                    reproduction = max(reproduction, abs(value - expected[key]))
            scores[label] = {**score, **independent(wave, reference, rate)}
        if scores["reference"]["mr"] > 2e-7 or scores["reference"]["numpy_hf_sc"] > 2e-7:
            raise ValueError("reference not stationary")
        np.testing.assert_allclose(
            scores["half_correct_hf"]["numpy_hf_error_rms"],
            0.5 * scores["untreated"]["numpy_hf_error_rms"],
            rtol=2e-4,
            atol=2e-7,
        )
        if scores["correct_hf"]["numpy_hf_error_rms"] > 2e-7:
            raise ValueError("correct-HF control geometry failure")
        result = {"case": case, "scores": scores, "reproduction_max": reproduction}
        registered(result, output / "metrics" / (case["id"] + ".json"))
        results.append(result)
        print("metric audit", i + 1, "/20", flush=True)
    return results


@torch.no_grad()
def terms(prediction, batch):
    return {
        "direct": direct_spectrum_loss(prediction, batch)["per_example"].tolist(),
        "raw_mr": multires_spectrum_loss(prediction, batch)["per_example"].tolist(),
        "fused_mr": fused_spectrum_loss(prediction, batch)["per_example"].tolist(),
    }


@torch.no_grad()
def features(config, output):
    prior = Path(config["prior"])
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), output / "cache", capacity=100),
    )
    items = [pool.select(d) for d in schedule["draws"][:8]]
    result = {}
    for label, entry in read(prior / "endpoints.json").items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = load(saved)
        ablated = copy.deepcopy(model)
        ablated.phase_head[0].weight[:, 645:903].zero_()
        cases = []
        for index, item in enumerate(items):
            batch, context = item[0].batch, item[1]
            captures = []
            hook = model.phase_head.register_forward_pre_hook(
                lambda _m, args, captures=captures: captures.append(args[0].detach())
            )
            try:
                prediction, _ = predict_budget(model, batch, context)
            finally:
                hook.remove()
            changed, _ = predict_budget(ablated, batch, context)
            parent, _ = predict_budget(model.base, batch, context)
            assert torch.equal(prediction[:, 0], parent[:, 0])
            _, _, condition = probe_inputs(batch, context)
            hz = torch.arange(129)[None, None] * batch["rate"].reshape(-1, 1, 1) / 256
            mask = (
                (hz >= 8000) & model.native_support(condition).bool() & model.base.interior.bool()
            )
            variants = {"trained": prediction, "zero_added_inputs": changed, "parent": parent}
            for which, axes in (
                ("reference_phase", [1]),
                ("reference_magnitude", [0]),
                ("reference_both", [0, 1]),
            ):
                value = prediction.clone()
                for axis in axes:
                    value[:, axis] = torch.where(mask, batch["y"][:, axis], value[:, axis])
                variants[which] = value
            # predict_budget's last call starts at hop 80; exclude its guard
            # region so diagnostic feature statistics start at hop 89 too.
            feature = captures[-1][:, 9:]
            contribution = torch.nn.functional.linear(
                feature[..., 645:903], model.phase_head[0].weight[:, 645:903]
            )
            case = {
                "draw": index,
                "corpora": [
                    rows[schedule["specs"][j]["row_id"]]["corpus"] for j in schedule["draws"][index]
                ]
                * 2,
                "identity": batch["identity"].tolist(),
                "terms": {k: terms(v, batch) for k, v in variants.items()},
                "added_preactivation_rms": float(contribution.square().mean().sqrt()),
                "total_preactivation_rms": float(
                    model.phase_head[0](feature).square().mean().sqrt()
                ),
                "rotation_rms": float(
                    (prediction[:, 1, 89:] - parent[:, 1, 89:]).square().mean().sqrt()
                ),
                "ablation_rotation_difference_rms": float(
                    (changed[:, 1, 89:] - prediction[:, 1, 89:]).square().mean().sqrt()
                ),
                "rotation_tanh_max": float(torch.tanh(model.phase_head(feature) / 64).abs().max()),
            }
            cases.append(case)
            registered(case, output / "features" / label / f"{index}.json")
        if any(not torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items()):
            raise ValueError("frozen endpoint mutated")
        result[label] = cases
        print("feature audit", label, flush=True)
    return result


def run(output):
    config = read(output / "registration.json")
    if (output / "metrics").exists():
        raise ValueError("retain prior attempt; no silent overwrite")
    verify(config["hashes"])
    measured = metrics(config, output)
    feature_results = features(config, output)
    verify(config["hashes"])
    registered({"metrics": measured, "features": feature_results}, output / "assessment.json")
    registered(
        {
            "files": {
                str(p): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.name != "run.log"
            }
        },
        output / "complete.json",
    )


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
    except Exception:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
