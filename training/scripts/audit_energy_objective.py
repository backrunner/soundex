#!/usr/bin/env python3
"""Post-hoc P7 loss audit; no training, threshold tuning or checkpoint selection.

Clean-reference interventions are diagnostics, never inference features. Relative
power strata are descriptive only and do not exclude any recording from scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from data.weak_band_study import identity_specs
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_objective import BANDS, amplitude_metrics
from models.envelope_probe import context_features, predict_probe
from models.factorized_envelope import ABSOLUTE_FLOOR, BURN, RELATIVE_FLOOR, factor_objective
from models.residual_energy import make_energy_model
from scripts.audit_envelope_factorization import factor_oracles


def band_terms(prediction, target, rate, burn=BURN):
    """Exact single-clip decomposition of the existing objective, not a new loss."""
    p, t = prediction[:, burn:], target[:, burn:]
    global_power = t.square().mean()
    floor = (global_power.sqrt() * RELATIVE_FLOOR).clamp_min(ABSOLUTE_FLOOR)
    amplitude_floor = (global_power.sqrt() * 0.01).clamp_min(1e-5)
    frequency = torch.arange(129, device=p.device) * float(rate[0, 0]) / 256
    rows = []
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high)
        a, b = p[..., mask], t[..., mask]
        pp, tp = a.square().mean(-1, keepdim=True), b.square().mean(-1, keepdim=True)
        pn, tn = (pp + floor.square()).sqrt(), (tp + floor.square()).sqrt()
        lp, lt = (pn / floor).log(), (tn / floor).log()
        confidence = tp / (tp + floor.square())
        energy = (lp - lt).abs().mean()
        shape = ((a / pn - b / tn).abs().mean(-1, keepdim=True) * confidence).mean()
        temporal = (lp.diff(dim=1) - lt.diff(dim=1)).abs().mean()
        amplitude = (
            (torch.log1p(a / amplitude_floor) - torch.log1p(b / amplitude_floor)).abs().mean()
        )
        weight = (0.25 if low == 0 else 1) / 4.25
        rows.append(
            {
                "low": low,
                "high": high,
                "weight": weight,
                "band_envelope": energy,
                "band_shape": shape,
                "envelope_temporal": temporal,
                "amplitude": amplitude,
                "total": weight * (energy + 0.5 * shape + 0.25 * temporal + 0.25 * amplitude),
                "reference_relative_power_db": 10
                * (tp.mean().clamp_min(1e-30) / global_power.clamp_min(1e-30)).log10(),
                "predicted_relative_power_db": 10
                * (pp.mean().clamp_min(1e-30) / global_power.clamp_min(1e-30)).log10(),
                "reference_below_floor_fraction": (tp < floor.square()).float().mean(),
                "shape_confidence_mean": confidence.mean(),
            }
        )
    return rows


def synthetic_rankings():
    target = torch.full((1, 8, 129), 0.1, dtype=torch.float64)
    target[..., 86:] = 0
    target[..., 100] = 0.01 * (43**0.5)
    rate = torch.tensor([[48000.0]], dtype=torch.float64)
    rows = []
    for name, factor in (
        ("reference", 1.0),
        ("correct_shape_100x_under", 0.01),
        ("correct_shape_10x_under", 0.1),
        ("correct_shape_10x_over", 10.0),
        ("correct_energy_flat_shape", None),
    ):
        p = target.clone()
        p[..., 86:] = 0.01 if factor is None else target[..., 86:] * factor
        loss = factor_objective(p, target, rate, burn=0)
        rows.append(
            {
                "name": name,
                "loss": {k: float(v) for k, v in loss.items()},
                "metrics": amplitude_metrics(p, target, rate, burn=0),
            }
        )
    return rows


def summarize(records):
    result = {}
    for seed in sorted({r["seed"] for r in records}):
        splits = {}
        for split in ("validation", "identity", "train_probe"):
            groups = {
                arm: [
                    r
                    for r in records
                    if r["seed"] == seed and r["split"] == split and r["arm"] == arm
                ]
                for arm in ("factorized", "rms_residual")
            }
            control = {r["row_id"]: r for r in groups["factorized"]}
            differences = []
            for row in groups["rms_residual"]:
                old = control[row["row_id"]]
                differences.append(
                    {
                        "row_id": row["row_id"],
                        "track_id": row["track_id"],
                        **{
                            key: row["interventions"]["predicted"][key]
                            - old["interventions"]["predicted"][key]
                            for key in ("total", "high_sc", "high_flux", "high_energy_error_db")
                        },
                    }
                )
            # Sum losses within a recording first to avoid multiple clips acting
            # as independent evidence. No exclusion changes official scores.
            by_track = {}
            for row in differences:
                by_track.setdefault(row["track_id"], []).append(row)
            paired = [
                {
                    "track_id": track,
                    **{
                        key: float(np.mean([r[key] for r in values]))
                        for key in ("total", "high_sc", "high_flux", "high_energy_error_db")
                    },
                }
                for track, values in by_track.items()
            ]
            paired.sort(key=lambda r: r["high_sc"], reverse=True)
            splits[split] = {
                "recording_differences": paired,
                "recordings_sc_worse": sum(r["high_sc"] > 0 for r in paired),
                "recordings_loss_worse": sum(r["total"] > 0 for r in paired),
                "top3_positive_sc_share": sum(max(r["high_sc"], 0) for r in paired[:3])
                / max(sum(max(r["high_sc"], 0) for r in paired), 1e-30),
                "remaining_recording_mean_sc_delta": float(
                    np.mean([r["high_sc"] for r in paired[3:]])
                ),
                "arms": {},
            }
            for arm, values in groups.items():
                strata = {}
                for label, lower, upper in (
                    ("below_-80", -1000, -80),
                    ("-80_to_-60", -80, -60),
                    ("-60_to_-40", -60, -40),
                    ("above_-40", -40, 1000),
                ):
                    bands = [
                        b
                        for r in values
                        for b in r["bands"]
                        if b["low"] >= 4000 and lower <= b["reference_relative_power_db"] < upper
                    ]
                    if bands:
                        strata[label] = {
                            "clip_bands": len(bands),
                            **{
                                k: float(np.mean([b[k] for b in bands]))
                                for k in (
                                    "sc",
                                    "energy_bias_db",
                                    "total",
                                    "shape_confidence_mean",
                                    "reference_below_floor_fraction",
                                    "predicted_relative_power_db",
                                )
                            },
                        }
                splits[split]["arms"][arm] = {
                    "clip_means": {
                        intervention: {
                            k: float(np.mean([r["interventions"][intervention][k] for r in values]))
                            for k in ("total", "high_sc", "high_flux", "high_energy_error_db")
                        }
                        for intervention in values[0]["interventions"]
                    },
                    "reference_power_strata": strata,
                }
        result[str(seed)] = splits
    return result


def run(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    config = json.loads((root / "config.json").read_text())
    if json.loads((root / "summary.json").read_text())["status"] != "complete":
        raise ValueError("complete training first")
    rows, hashes = load_sources([Path(p) for p in config["manifests"]])
    if hashes != config["manifests"]:
        raise ValueError("manifests changed")
    clips = EnvelopeClips(rows, root / "binaries/soundex-gain-data", config["cache_path"])
    records, checkpoints = [], {}
    for seed in config["seeds"]:
        for arm in config["arms"]:
            directory = root / f"{arm}-{seed}"
            receipt = json.loads((directory / "complete.json").read_text())
            for name, key in (
                ("best.pth", "checkpoint_sha256"),
                ("evaluation.json", "evaluation_sha256"),
            ):
                if sha256_file(directory / name) != receipt[key]:
                    raise ValueError("completed artifacts changed")
            checkpoints[f"{arm}-{seed}"] = receipt["checkpoint_sha256"]
            existing = json.loads((directory / "evaluation.json").read_text())
            model = make_energy_model(arm).eval()
            model.load_state_dict(
                torch.load(directory / "best.pth", map_location="cpu", weights_only=False)["model"]
            )
            for split, specs in (
                ("validation", config["validation_specs"]),
                ("identity", identity_specs(config["validation_specs"])),
                ("train_probe", config["training_probe_specs"]),
            ):
                old = {r["row_id"]: r for r in existing[split]["rows"]}
                for spec in specs:
                    row = clips.rows[spec["row_id"]]
                    if row["corpus"] != "music_library":
                        continue
                    batch = clips.batch([spec])
                    with torch.no_grad():
                        pred = predict_probe(
                            model, batch, context_features(batch["pcm"], "past"), burn=BURN
                        )
                    target = torch.pow(10.0, batch["y"][:, 0] / 20)
                    metrics = {}
                    with torch.no_grad():
                        for name, amplitude in factor_oracles(
                            pred, target, float(batch["rate"][0, 0])
                        ).items():
                            metrics[name] = {
                                **amplitude_metrics(amplitude, target, batch["rate"], burn=BURN),
                                **{
                                    k: float(v)
                                    for k, v in factor_objective(
                                        amplitude, target, batch["rate"], burn=BURN
                                    ).items()
                                },
                            }
                    for key in ("total", "high_sc", "high_energy_error_db"):
                        if not np.isclose(
                            metrics["predicted"][key],
                            old[row["row_id"]]["metrics"][key],
                            rtol=1e-4,
                            atol=1e-4,
                        ):
                            raise ValueError(
                                f"checkpoint reproduction differs: {row['row_id']} {key}"
                            )
                    p = pred.double().requires_grad_()
                    t = target.double()
                    terms = band_terms(p, t, batch["rate"])
                    loss = factor_objective(p, t, batch["rate"])["total"]
                    torch.testing.assert_close(
                        sum(b["total"] for b in terms), loss, atol=1e-10, rtol=1e-10
                    )
                    gradient = torch.autograd.grad(loss, p)[0][:, BURN:]
                    bands = []
                    for b in terms:
                        prefix = f"band_{b['low']}_{b['high']}"
                        frequency = torch.arange(129) * float(batch["rate"][0, 0]) / 256
                        mask = (frequency >= b["low"]) & (frequency < b["high"])
                        a, ref = p[:, BURN:, mask].detach(), t[:, BURN:, mask]
                        projected = (gradient[..., mask] * a).sum(-1)
                        over = a.square().mean(-1) > ref.square().mean(-1) * 1.01
                        bands.append(
                            {
                                k: float(v.detach()) if isinstance(v, torch.Tensor) else v
                                for k, v in b.items()
                            }
                            | {
                                "sc": metrics["predicted"][prefix + "_sc"],
                                "energy_bias_db": metrics["predicted"][prefix + "_energy_bias_db"],
                                "over_frames": int(over.sum()),
                                "over_frames_gradient_reduces_energy": int(
                                    ((projected > 1e-12) & over).sum()
                                ),
                                "over_frames_gradient_increases_energy": int(
                                    ((projected < -1e-12) & over).sum()
                                ),
                                "mean_log_gain_gradient": float(projected.mean()),
                            }
                        )
                    records.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "split": split,
                            "row_id": row["row_id"],
                            "track_id": row["track_id"],
                            "interventions": metrics,
                            "bands": bands,
                        }
                    )
            print(json.dumps({"seed": seed, "arm": arm, "records": len(records)}), flush=True)
    report = {
        "scope": "post-hoc raw magnitude audit on reused development, identity and training music probes; oracle outputs are not deployable",
        "selection_changed": False,
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoints": checkpoints,
        "synthetic_rankings": synthetic_rankings(),
        "summary": summarize(records),
        "rows": records,
    }
    write_json_report(report, root / "objective-audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    run(parser.parse_args())
