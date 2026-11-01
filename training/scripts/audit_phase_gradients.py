"""P68 frozen P67 phase-head gradient allocation and conflict, no updates.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.reporting import sha256_file, write_json_report
from models.direct_spectrum_objective import direct_spectrum_loss
from models.envelope_probe import probe_inputs
from models.fused_spectrum import balanced_risk, fused_spectrum_loss, prepare_fusion
from models.multires_spectrum import multires_spectrum_loss
from models.restoration_head import predict_budget
from scripts.audit_fused_spectrum import snapshot
from scripts.run_anchored_pilot import verify
from scripts.run_consistency_followup import read, registered
from scripts.run_fused_phase_pilot import make_model
from scripts.run_longform_review import checked_receipt


def vector(parts):
    result = torch.cat([p.detach().flatten() for p in parts]).double()
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite parameter gradient")
    return result


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b)) / denominator if denominator else None


def prepare(prior, output):
    if output.exists():
        raise ValueError("fresh directory required; preserve every attempt")
    checked_receipt(prior / "complete.json")
    config = read(prior / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(config["hashes"])
    hashes.update(read(prior / "complete.json")["files"])
    hashes[str(prior / "complete.json")] = sha256_file(prior / "complete.json")
    hashes.update({str(p): sha256_file(p) for p in (output / "source").rglob("*") if p.is_file()})
    registered(
        {
            "format": "soundex-phase-gradients-p68-v1",
            "prior": str(prior),
            "draws": list(range(8)),
            "endpoints": read(prior / "endpoints.json"),
            "hashes": hashes,
            "scope": "64 frozen endpoint/batch probes: all eight P67 arms and first eight registered TRAIN draws. No parameter update, held-out score, new loss or timing claim.",
            "changed_question": "P39 examined shared full-model gradients. P68 isolates the frozen-magnitude phase readout after P67 support/fusion changes, separating output-radian support from parameter-space and corpus/role conflicts.",
            "gates": "Direct+raw+fused gradients reproduce augmented gradient; phase-coordinate VJP reproduces parameter gradient with existing rtol=2e-4, atol=2e-7. No dB/radian importance ratio, no selection by these local cosines.",
        },
        output / "registration.json",
    )
    print("registered 64 frozen phase-gradient probes", flush=True)


def probe(model, item, corpora):
    objective, context = item
    batch = objective.batch
    captures = []
    hook = model.phase_head.register_forward_pre_hook(
        lambda _module, args: captures.append(args[0].detach())
    )
    try:
        prediction, _ = predict_budget(model, batch, context)
    finally:
        hook.remove()
    plan = prepare_fusion(batch)
    terms = {
        "direct": direct_spectrum_loss(prediction, batch)["per_example"].log1p(),
        "raw": multires_spectrum_loss(prediction, batch)["per_example"].log1p(),
        "fused": fused_spectrum_loss(prediction, batch, plan)["per_example"].log1p(),
    }
    params = list(model.phase_head.parameters())
    identity = batch["identity"].bool()
    masks = {
        corpus + ("/identity" if clean else "/paired"): torch.tensor(
            [c == corpus for c in corpora], device=identity.device
        )
        & (identity == clean)
        for corpus in sorted(set(corpora))
        for clean in (False, True)
    }
    scalars = {name: balanced_risk(value, identity) for name, value in terms.items()}
    total_per_example = sum(terms.values())
    example_weights = torch.where(identity, 0.5 / identity.sum(), 0.5 / (~identity).sum())
    scalars.update(
        {name: (total_per_example * example_weights * mask).sum() for name, mask in masks.items()}
    )
    _, _, condition = probe_inputs(batch, context)
    available = model.native_support(condition).bool()
    hz = torch.arange(129)[None, None] * batch["rate"].reshape(-1, 1, 1) / 256
    interior = model.base.interior.bool().expand_as(available)
    # Include the OLA guard frame; gradient support is determined by the objective,
    # not retrospectively cropped to direct supervision's first frame.
    hf = (hz >= 8000).expand_as(available) & interior
    vectors, output_gradients, partition = {}, {}, {}
    for name, value in scalars.items():
        gradients = torch.autograd.grad(value, [prediction, *params], retain_graph=True)
        g = gradients[0].detach()[:, 1].double()
        vectors[name] = vector(gradients[1:])
        output_gradients[name] = g
        mass = {
            label: float((g.square() * mask).sum())
            for label, mask in (
                ("available_hf", hf & available),
                ("discarded_hf", hf & ~available),
                ("all_hf", hf),
            )
        }
        mass["available_hf_share"] = (
            mass["available_hf"] / mass["all_hf"] if mass["all_hf"] else None
        )
        partition[name] = mass
    total = balanced_risk(total_per_example, identity)
    actual = vector(torch.autograd.grad(total, params, retain_graph=True))
    combined = sum(vectors[k] for k in terms)
    torch.testing.assert_close(combined, actual, rtol=2e-4, atol=2e-7)
    grouped = sum(vectors[k] for k in masks)
    torch.testing.assert_close(grouped, actual, rtol=2e-4, atol=2e-7)
    gphase = sum(output_gradients[k] for k in terms)
    chain = vector(
        torch.autograd.grad(prediction[:, 1], params, grad_outputs=gphase.to(prediction.dtype))
    )
    torch.testing.assert_close(chain, actual, rtol=2e-4, atol=2e-7)
    vectors["common"] = vectors["direct"] + vectors["raw"]
    with torch.no_grad():
        hidden = model.phase_head[1](model.phase_head[0](captures[-1]))
        bounded = torch.tanh(model.phase_head[-1](hidden) / 64)
    return {
        "losses": {k: float(v.detach()) for k, v in scalars.items()},
        "parameter_norms": {k: float(v.norm()) for k, v in vectors.items()},
        "parameter_cosines": {
            a + "/" + b: cosine(vectors[a], vectors[b])
            for a, b in itertools.combinations(vectors, 2)
        },
        "available_phase_cosines": {
            a + "/" + b: cosine(
                (output_gradients[a] * available * hf).flatten(),
                (output_gradients[b] * available * hf).flatten(),
            )
            for a, b in itertools.combinations(terms, 2)
        },
        "output_phase_gradient_power": partition,
        "gradient_reconstruction_max": float((combined - actual).abs().max()),
        "role_reconstruction_max": float((grouped - actual).abs().max()),
        "phase_chain_max": float((chain - actual).abs().max()),
        "hidden_abs_over_099_fraction": float((hidden.abs() > 0.99).double().mean()),
        "rotation_tanh_abs_over_095_fraction": float((bounded.abs() > 0.95).double().mean()),
        "rotation_tanh_max": float(bounded.abs().max()),
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    prior = Path(config["prior"])
    old = read(prior / "registration.json")
    if (output / "probes").exists():
        raise ValueError("existing probe outputs retained; no silent restart")
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), output / "cache", capacity=100),
    )
    items = [pool.select(schedule["draws"][i]) for i in config["draws"]]
    results = {}
    for label, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        if saved["step"] != 512:
            raise ValueError("incomplete endpoint")
        model = make_model(old, saved["seed"], saved["arm"])
        model.load_state_dict(saved["model"])
        cases = []
        for index, item in zip(config["draws"], items, strict=True):
            specs = [schedule["specs"][j] for j in schedule["draws"][index]]
            corpora = [rows[s["row_id"]]["corpus"] for s in specs]
            result = {
                "draw": index,
                "samples": [{"spec": s, "row": rows[s["row_id"]]} for s in specs],
                **probe(model, item, corpora * 2),
            }
            cases.append(result)
            write_json_report(result, output / "probes" / label / f"{index}.json")
        if any(not torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items()):
            raise ValueError("read-only audit changed weights")
        results[label] = cases
        print("completed phase gradient probes", label, flush=True)
    verify(config["hashes"])
    registered(
        {
            "cases": results,
            "scope": config["scope"],
            "interpretation": "Local coordinate-dependent derivatives, not Adam displacements or audio improvements. Output dB/radian norms are not comparable units. All endpoint/data weights unchanged.",
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
