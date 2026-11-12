"""P75 frozen P74 gradient audit of realized high-band complex supervision.

No optimizer updates, inference changes, held-out scoring or timing retries.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.realized_complex import realized_hf_terms
from evaluation.reporting import sha256_file, write_json_report
from models.direct_spectrum_objective import WEIGHTS, direct_spectrum_loss
from models.fused_spectrum import BURN, balanced_risk, fused_stages, phase_objective, prepare_fusion
from models.multires_spectrum import spectral_loss, supervised_waves
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_gradients import cosine, vector
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import load


def prepare(prior, output, previous=None):
    if output.exists():
        raise ValueError("fresh directory required; retain every attempt")
    checked_receipt(prior / "p74-complete.json")
    config = read(prior / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(config["hashes"])
    hashes.update(read(prior / "p74-complete.json")["files"])
    hashes[str(prior / "registration.json")] = sha256_file(prior / "registration.json")
    endpoints = read(prior / "endpoints.json")
    hashes.update({entry["path"]: entry["sha256"] for entry in endpoints.values()})
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    if previous is not None:
        if not (previous / "failure.json").is_file():
            raise ValueError("previous attempt must retain its failure record")
        previous_files = [previous / "registration.json", previous / "failure.json"]
        previous_files.extend((previous / "probes").rglob("*.json"))
        hashes.update({str(p): sha256_file(p) for p in previous_files})
    registered(
        {
            "format": "soundex-realized-supervision-audit-p75-v1",
            "prior": str(prior),
            "draws": list(range(8)),
            "endpoints": endpoints,
            "precision": "float64 diagnostic: lift frozen FP32 tensors exactly; no parameter updates. Retain the original FP32 attempt and unchanged gradient tolerances.",
            "previous_attempt": str(previous) if previous is not None else None,
            "hashes": hashes,
            "question": "Does pre-OLA complex supervision align with realized original-reference HF complex error in the fully trained P74 joint models? Existing raw/fused MR already supervises realized magnitudes; do not claim synthesis supervision is absent.",
            "changed_control": "P68 only studied frozen-magnitude phase heads and MR. P75 uses all four fully trained joint P74 endpoints and a separate multi-resolution realized HF complex diagnostic. Historical V4 already tried complex reconstruction in another model; this is not a first-ever complex-loss claim.",
            "probe": "Same first eight TRAIN draws, detached89-hop model burn, existing synthesis/fusion/unaltered original PCM, 256/1024/2048 MR windows, three >=8kHz bands, existing MR power floors. Balanced paired/identity, all trainable parameters and separate base/phase-head groups.",
            "direction": "Report old gradient and old + 0.1 * log1p(realized fused HF complex), coefficient fixed to the existing direct complex coefficient before probing. No coefficient selection, Adam step, validation tuning or improvement claim; different scales are reported explicitly.",
            "gates": "Exact old objective/gradient decomposition with prior rtol2e-4 atol2e-7; finite gradients; endpoints remain bitwise unchanged. Numerical diagnostic tests precede this registration. Native proxy correctness is inherited from P67; a new native transfer/gradient check is still required before training.",
            "scope": "32 frozen TRAIN endpoint/draw probes, not independent testing, convergence, quality, listening or runtime qualification. No data/source reservation changes or new training.",
        },
        output / "registration.json",
    )
    print("registered 32 frozen original-reference supervision probes", flush=True)


def probe(model, item):
    objective, context = item
    batch = objective.batch
    prediction, _ = predict_budget(model, batch, context)
    direct = direct_spectrum_loss(prediction, batch)
    raw, reference = supervised_waves(prediction, batch)
    final = fused_stages(prediction, batch["x"], prepare_fusion(batch))["final"]
    fused = final[:, BURN * 128 : final.shape[-1] - 128]
    raw_mr = spectral_loss(raw, reference, batch["rate"])["per_example"]
    fused_mr = spectral_loss(fused, reference, batch["rate"])["per_example"]
    rc = realized_hf_terms(raw, reference, batch["rate"])
    fc = realized_hf_terms(fused, reference, batch["rate"])
    terms = {
        "direct": direct["per_example"].log1p(),
        "raw_mr": raw_mr.log1p(),
        "fused_mr": fused_mr.log1p(),
        # This scalar's gradient is exactly the direct complex contribution to
        # log1p(direct); its value is not an independent objective score.
        "direct_complex_contribution": WEIGHTS["complex"]
        * direct["components"]["complex"]
        / (1 + direct["per_example"].detach()),
        "raw_complex": rc["complex"].log1p(),
        "fused_complex": fc["complex"].log1p(),
        "fused_hf_magnitude": fc["magnitude"].log1p(),
    }
    identity = batch["identity"].bool()
    scalars = {name: balanced_risk(value, identity) for name, value in terms.items()}
    old = sum(terms[k] for k in ("direct", "raw_mr", "fused_mr"))
    for name, values in (("old", old), ("fused_complex", terms["fused_complex"])):
        scalars[name + "/paired"] = 0.5 * values[~identity].mean()
        scalars[name + "/identity"] = 0.5 * values[identity].mean()
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = {
        "all": list(range(len(params))),
        "base": [i for i, (k, _) in enumerate(named) if k.startswith("base.")],
        "phase_head": [i for i, (k, _) in enumerate(named) if k.startswith("phase_head.")],
    }
    vectors = {key: {} for key in groups}
    for name, value in scalars.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        gradients = [
            torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients, strict=True)
        ]
        for group, indices in groups.items():
            vectors[group][name] = vector([gradients[i] for i in indices])
    actual = phase_objective(prediction, batch, include_fused=True)
    torch.testing.assert_close(actual, balanced_risk(old, identity), rtol=2e-4, atol=2e-7)
    actual_gradient = vector(torch.autograd.grad(actual, params))
    reconstruction = sum(vectors["all"][k] for k in ("direct", "raw_mr", "fused_mr"))
    torch.testing.assert_close(actual_gradient, reconstruction, rtol=2e-4, atol=2e-7)
    result = {}
    for group, values in vectors.items():
        values["old"] = sum(values[k] for k in ("direct", "raw_mr", "fused_mr"))
        values["proposed_0.1"] = values["old"] + 0.1 * values["fused_complex"]
        result[group] = {
            "norms": {k: float(v.norm()) for k, v in values.items()},
            "cosines": {
                a + "/" + b: cosine(values[a], values[b])
                for a, b in itertools.combinations(values, 2)
            },
        }
    return {
        "scalars": {k: float(v.detach()) for k, v in scalars.items()},
        "gradient_decomposition_max": float((actual_gradient - reconstruction).abs().max()),
        "parameters": result,
        "raw_complex_per_example": rc["complex"].detach().tolist(),
        "fused_complex_per_example": fc["complex"].detach().tolist(),
        "fused_magnitude_per_example": fc["magnitude"].detach().tolist(),
        "identity": identity.tolist(),
        "realized_fused_bands": [
            {k: v.detach().tolist() if isinstance(v, torch.Tensor) else v for k, v in band.items()}
            for band in fc["bands"]
        ],
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "probes").exists():
        raise ValueError("existing probes retained; no silent restart")
    prior = Path(config["prior"])
    old = read(prior / "registration.json")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), prior / "cache", capacity=100),
    )
    items = []
    for i in config["draws"]:
        objective, context = pool.select(schedule["draws"][i])
        batch = {
            k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in objective.batch.items()
        }
        items.append((SimpleNamespace(batch=batch), context.double()))
    results = {}
    for label, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        if saved["step"] != 16384:
            raise ValueError("complete P74 endpoint required")
        model = load(saved).double()
        cases = []
        for index, item in zip(config["draws"], items, strict=True):
            specs = [schedule["specs"][j] for j in schedule["draws"][index]]
            result = {
                "draw": index,
                "samples": [{"spec": s, "row": rows[s["row_id"]]} for s in specs],
                **probe(model, item),
            }
            write_json_report(result, output / "probes" / label / f"{index}.json")
            cases.append(result)
            print("completed", label, index, flush=True)
        if any(not torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items()):
            raise ValueError("read-only audit changed weights")
        results[label] = cases
    verify(config["hashes"])
    registered(
        {
            "cases": results,
            "scope": config["scope"],
            "interpretation": "Local parameter derivatives are not Adam displacements or evidence of restored audio. All weights and original references unchanged.",
        },
        output / "assessment.json",
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--previous-attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.output.resolve(),
                args.previous_attempt.resolve() if args.previous_attempt else None,
            )
        else:
            run(args.output.resolve())
    except BaseException:
        if args.output.exists():
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
