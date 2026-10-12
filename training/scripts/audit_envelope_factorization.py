#!/usr/bin/env python3
"""Post-hoc oracle: separate per-frame band envelope and within-band shape error.

Clean information is used only for offline interventions. These outputs are not
learned predictions, audio-quality measurements, or candidate-selection scores.
"""

import argparse
import json
from pathlib import Path

import torch

from data.causal_study import load_sources
from data.envelope_study import EnvelopeClips
from evaluation.reporting import sha256_file, write_json_report
from models.envelope_objective import BANDS, amplitude_metrics, envelope_objective
from models.envelope_probe import EnvelopeProbe, context_features, predict_probe
from scripts.analyze_envelope_study import paired_comparison


def factor_oracles(prediction, target, rate):
    """Preserve low band; separately provide clean frame/band norm or clean shape."""
    envelope, shape = prediction.clone(), prediction.clone()
    frequency = torch.arange(129, device=prediction.device) * rate / 256
    for low, high in BANDS[1:]:
        mask = (frequency >= low) & (frequency < high)
        p, t = prediction[..., mask], target[..., mask]
        pn, tn = p.norm(dim=-1, keepdim=True), t.norm(dim=-1, keepdim=True)
        envelope[..., mask] = p * tn / pn.clamp_min(1e-20)
        shape[..., mask] = t * pn / tn.clamp_min(1e-20)
    return {
        "predicted": prediction,
        "oracle_band_envelope": envelope,
        "oracle_within_band_shape": shape,
        "oracle_full_amplitude": target,
    }


def main(args):
    torch.set_num_threads(1)
    root = args.study.resolve()
    if json.loads((root / "summary.json").read_text())["status"] != "complete":
        raise ValueError("complete the frozen study before post-hoc diagnosis")
    config = json.loads((root / "config.json").read_text())
    rows, hashes = load_sources([Path(p) for p in config["manifests"]])
    if hashes != config["manifests"]:
        raise ValueError("manifest changed")
    clips = EnvelopeClips(rows, root / "binaries/soundex-gain-data", root / "cache")
    records, checkpoints = [], {}
    with torch.no_grad():
        for seed in config["seeds"]:
            path = root / f"past-{seed}/best.pth"
            checkpoints[str(seed)] = sha256_file(path)
            model = EnvelopeProbe().eval()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            for split, specs in (
                ("train_probe", config["training_probe_specs"]),
                ("validation", config["validation_specs"]),
            ):
                for spec in specs:
                    row = rows[spec["row_id"]]
                    if row["corpus"] != "music_library":
                        continue
                    batch = clips.batch([spec])
                    prediction = predict_probe(model, batch, context_features(batch["pcm"], "past"))
                    target = torch.pow(10.0, batch["y"][:, 0] / 20)
                    for intervention, amplitude in factor_oracles(
                        prediction, target, float(batch["rate"][0, 0])
                    ).items():
                        metrics = amplitude_metrics(amplitude, target, batch["rate"])
                        metrics.update(
                            {
                                k: float(v)
                                for k, v in envelope_objective(
                                    amplitude, target, batch["rate"]
                                ).items()
                            }
                        )
                        records.append(
                            {
                                "seed": seed,
                                "split": split,
                                "row_id": row["row_id"],
                                "track_id": row["track_id"],
                                "intervention": intervention,
                                "metrics": metrics,
                            }
                        )
    write_json_report(
        {
            "post_hoc": True,
            "deployable": False,
            "clean_reference_used": True,
            "checkpoints": checkpoints,
            "rows": records,
            "scope": "per-frame per-band oracle on raw amplitude; no phase/DSP/audio-quality claim",
        },
        root / "factorization-audit.json",
    )
    # The long frontend's startup can propagate through the 64-hop TCN.
    # 80 hops cover the complete composed history; keep the original selection
    # fixed and audit this later crop rather than silently changing the protocol.
    steady = {}
    with torch.no_grad():
        for seed in config["seeds"]:
            arms = {}
            for arm in config["arms"]:
                model = EnvelopeProbe().eval()
                path = root / f"{arm}-{seed}/best.pth"
                model.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=False)["model"]
                )
                scored = []
                for spec in config["validation_specs"]:
                    row = rows[spec["row_id"]]
                    batch = clips.batch([spec])
                    prediction = predict_probe(model, batch, context_features(batch["pcm"], arm))
                    target = torch.pow(10.0, batch["y"][:, 0] / 20)
                    metrics = amplitude_metrics(prediction, target, batch["rate"], burn=80)
                    metrics.update(
                        {
                            k: float(v)
                            for k, v in envelope_objective(
                                prediction, target, batch["rate"], burn=80
                            ).items()
                        }
                    )
                    scored.append(
                        {
                            "row_id": row["row_id"],
                            "track_id": row["track_id"],
                            "corpus": row["corpus"],
                            "metrics": metrics,
                        }
                    )
                arms[arm] = {"checkpoint_sha256": sha256_file(path), "rows": scored}
            steady[str(seed)] = {
                "arms": arms,
                "past_minus_short": paired_comparison(arms["past"], arms["short"]),
                "past_minus_tiled": paired_comparison(arms["past"], arms["tiled"]),
            }
    write_json_report(
        {"post_hoc": True, "selection_changed": False, "metric_burn_hops": 80, "seeds": steady},
        root / "warmup-sensitivity.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    main(parser.parse_args())
