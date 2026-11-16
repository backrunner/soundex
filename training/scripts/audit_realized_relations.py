"""P82 frozen direct/raw/fused relation gradients and saved native correspondence.

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import itertools
import traceback
from pathlib import Path

import torch

from data.broad_coverage import BoundedEnvelopeClips, LazyPool
from data.causal_study import load_sources
from evaluation.realized_complex import realized_hf_terms
from evaluation.reporting import sha256_file, write_json_report
from evaluation.restoration_evidence import extended_panel
from evaluation.rust_bridge import read_sxa
from models.complex_synthesis import analyze
from models.direct_spectrum_objective import direct_spectrum_loss
from models.fused_spectrum import BURN, balanced_risk, fused_stages, phase_objective, prepare_fusion
from models.losses import build_missing_band_mask
from models.multires_spectrum import spectral_loss, supervised_waves
from models.phase_relations import KINDS, phase_relations
from models.realized_phase_relations import (
    COMPONENTS,
    DOMAINS,
    VERSION,
    correspondence,
    domain_spectra,
    prepare_relations,
    relations,
)
from models.restoration_head import predict_budget
from scripts.audit_context_deployment import unpack_trace
from scripts.audit_fused_spectrum import snapshot
from scripts.audit_phase_gradients import cosine, vector
from scripts.audit_phase_routing import exact, samples, scores
from scripts.run_anchored_pilot import verify
from scripts.run_band_training import assert_catalogs_unchanged
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt
from scripts.run_phase_temporal_training import load


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def prepare(prior, controls, failure, output):
    if output.exists():
        raise ValueError("fresh registration required; preserve all previous evidence")
    for root in (prior, controls):
        checked_receipt(root / "complete.json")
    if not (failure / "failure.json").exists():
        raise ValueError("retain wrapped-cusp finite-difference counterexample")
    config = read(prior / "registration.json")
    output.mkdir(parents=True)
    snapshot(output)
    hashes = dict(config["hashes"])
    for root in (prior, controls):
        hashes.update(read(root / "complete.json")["files"])
        hashes[str(root / "complete.json")] = sha256_file(root / "complete.json")
    for root in (failure, output / "source"):
        hashes.update(
            {
                str(p): sha256_file(p)
                for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
        )
    registered(
        {
            **{
                k: config[k]
                for k in (
                    "p74",
                    "endpoints",
                    "fit",
                    "transfer",
                    "native_draws",
                    "coverage",
                    "evaluation_catalogs",
                )
            },
            "version": VERSION,
            "prior": str(prior),
            "hashes": hashes,
            "changed_question": "P75/P76 studied marginal complex error; P80 studied direct phase relations. P82 compares the same circular/wrapped instantaneous/frequency/time kernels on direct, raw-OLA and fused-reanalyzed spectra under one original-PCM target/support. No claim that synthesis supervision was absent.",
            "reference": "Original aligned clean PCM reanalyzed in FP64 with Hann256/hop128; common frames89..T-3 wholly within existing supervised PCM. Same P80 relative1e-3/absolute1e-6 FFT amplitude support, input-only binary native availability and interior>=8k bins. Same reference weights in every domain; no prediction mask, limited/reconstructed target or retimed audio. Historical direct cached-target/full-frame P80 kernels also reported to disclose the target/support distinction.",
            "budget": "Zero optimizer updates. All4 frozen fully trained P74 origins x8 fixed P80 fit draws=32 FP64 gradient probes. All40 saved P80 first-role native traces (8 endpoints x5 existing draws), reused PCM only, no new native execution. Review all kernels/domains/components/role-corpus strata, without selecting origins or tuning coefficients.",
            "derivatives": "Keep old joint spectral objective fixed. Auxiliary0.1*log1p(mean3relations); compare base/head/full vectors and head-only old+aux directions. Separate exact component contributions and music/speech xpaired/identity directions. Component sums must reproduce total auxiliary head derivative at existing rtol2e-4 atol2e-7; every gradient finite; every frozen weight bitwise unchanged. Directions are not finite optimizer steps or quality gains.",
            "native": "Use each saved first physical mono-role trace's observed features/prediction/condition with its original PCM target. Existing PCM max2^-14/RMS2e-5 parity must hold. Bound each relation-risk discrepancy analytically using circularLipschitz1/2 or wrapped1/pi and per-bin phase separation <=pi*min(1,complex_difference/min_magnitude), zero bins bounded bypi, with1e-12 FP64 arithmetic allowance. Report actual differences/bounds, not a universal noise or audible threshold. Reproduce every saved native-v1 metric exactly and add protocol-v2 fields without overriding old scores.",
            "decision": "No phase-sign veto or scalar FP32 epsilon gate; P81 separates spectral direction from coherent tradeoffs. This is definition/gradient/correspondence evidence, not useful model improvement. Full training needs a distinct registered matched finite-update/adequate-budget protocol; do not automatically repeat unchanged scaling or promote any endpoint.",
            "scope": "Fit: same40 recordings and18.724499 unique MR-supervised seconds as P80, with a narrower common relational window. Native correspondence: first role only of each existing draw, hence5 reused recordings, not the full20-source transfer panel. No held-out-independent generalization, physical stereo, listening, latency or device qualification.",
        },
        output / "registration.json",
    )


def probe(model, item, corpora):
    original, context = item
    b = {
        k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in original.batch.items()
    }
    prediction, _ = predict_budget(model, b, context.double())
    plan = prepare_relations(b)
    spectra = domain_spectra(prediction, b, plan)
    relation = {
        f"{domain}/{kind}": relations(spectra[domain], plan, kind)
        for domain, kind in itertools.product(DOMAINS, KINDS)
    }
    raw, reference = supervised_waves(prediction, b)
    final = fused_stages(prediction, b["x"], prepare_fusion(b))["final"]
    final = final[:, BURN * 128 : final.shape[-1] - 128]
    hf = realized_hf_terms(final, reference, b["rate"])
    terms = {
        "fused_mr": spectral_loss(final, reference, b["rate"])["per_example"].log1p(),
        "fused_hf_magnitude": hf["magnitude"].log1p(),
        "fused_complex": hf["complex"].log1p(),
        **{key: 0.1 * value["per_example"].log1p() for key, value in relation.items()},
        **{
            f"legacy/{kind}": 0.1 * phase_relations(prediction, b, kind)["per_example"].log1p()
            for kind in KINDS
        },
    }
    terms["old"] = (
        direct_spectrum_loss(prediction, b)["per_example"].log1p()
        + spectral_loss(raw, reference, b["rate"])["per_example"].log1p()
        + terms["fused_mr"]
    )
    scalars = {key: balanced_risk(value, b["identity"]) for key, value in terms.items()}
    torch.testing.assert_close(
        scalars["old"], phase_objective(prediction, b, include_fused=True), rtol=1e-12, atol=1e-14
    )
    named = list(model.named_parameters())
    params = [p for _, p in named]
    group_ids = {
        "all": list(range(len(named))),
        **{
            group: [i for i, (k, _) in enumerate(named) if k.startswith(group + ".")]
            for group in ("base", "phase_head")
        },
    }
    vectors = {group: {} for group in group_ids}
    for key, scalar in scalars.items():
        gradients = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
        gradients = [
            torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients, strict=True)
        ]
        for group, indices in group_ids.items():
            vectors[group][key] = vector([gradients[i] for i in indices])
    targets = ("old", "fused_mr", "fused_hf_magnitude", "fused_complex")
    head = [params[i] for i in group_ids["phase_head"]]
    components = {}
    for arm, result in relation.items():
        parts = {}
        for key, value in result["components"].items():
            contribution = (0.1 / 3) * value / (1 + result["per_example"].detach())
            g = vector(
                torch.autograd.grad(
                    balanced_risk(contribution, b["identity"]), head, retain_graph=True
                )
            )
            parts[key] = g
        total = sum(parts.values())
        torch.testing.assert_close(total, vectors["phase_head"][arm], rtol=2e-4, atol=2e-7)
        components[arm] = {
            "sum_gradient_max_error": float((total - vectors["phase_head"][arm]).abs().max()),
            "head_norms": {k: float(v.norm()) for k, v in parts.items()},
            "head_cosines": {
                k: {target: cosine(v, vectors["phase_head"][target]) for target in targets}
                for k, v in parts.items()
            },
        }
    strata = {}
    for corpus, identity in itertools.product(sorted(set(corpora)), (False, True)):
        mask = torch.tensor([c == corpus for c in corpora]) & (b["identity"].bool() == identity)
        if not mask.any():
            raise ValueError("registered corpus/role stratum missing")
        gs = {
            key: vector(torch.autograd.grad(value[mask].mean(), head, retain_graph=True))
            for key, value in terms.items()
            if not key.startswith("legacy/")
        }
        strata[corpus + ("/identity" if identity else "/paired")] = {
            "roles": mask.nonzero().flatten().tolist(),
            "scalars": {k: float(v[mask].mean().detach()) for k, v in terms.items()},
            "norms": {k: float(v.norm()) for k, v in gs.items()},
            "cosines": {
                arm: {target: cosine(gs[arm], gs[target]) for target in targets} for arm in relation
            },
            "old_plus_auxiliary_cosines": {
                arm: {target: cosine(gs["old"] + gs[arm], gs[target]) for target in targets}
                for arm in relation
            },
            "balanced_head_direction_cosines": {
                arm: {
                    target: cosine(
                        vectors["phase_head"]["old"]
                        + (vectors["phase_head"][arm] if arm != "old" else 0),
                        gs[target],
                    )
                    for target in targets
                }
                for arm in ("old", *relation)
            },
        }
        target_gradients = {
            key: vector(torch.autograd.grad(terms[key][mask].mean(), params, retain_graph=True))
            for key in targets
        }
        # Actual routed Euclidean field: old full-model gradient plus auxiliary
        # only in the head. This is still not the persistent-Adam direction.
        head_selector = vector(
            [
                torch.ones_like(p) if k.startswith("phase_head.") else torch.zeros_like(p)
                for k, p in named
            ]
        )
        strata[corpus + ("/identity" if identity else "/paired")][
            "balanced_routed_full_direction_cosines"
        ] = {
            arm: {
                target: cosine(
                    vectors["all"]["old"]
                    + (head_selector * vectors["all"][arm] if arm != "old" else 0),
                    gradient,
                )
                for target, gradient in target_gradients.items()
            }
            for arm in ("old", *relation)
        }
    result = {}
    for group, gs in vectors.items():
        result[group] = {
            "norms": {k: float(v.norm()) for k, v in gs.items()},
            "cosines": {
                a + "|" + z: cosine(gs[a], gs[z]) for a, z in itertools.combinations(gs, 2)
            },
            "old_plus_auxiliary_cosines": {
                arm: {target: cosine(gs["old"] + gs[arm], gs[target]) for target in targets}
                for arm in relation
            },
        }
    return {
        "scalars": {k: float(v.detach()) for k, v in scalars.items()},
        "parameters": result,
        "components": components,
        "strata": strata,
        "per_role_relations": plain(relation),
        "roles": b["identity"].tolist(),
        "support": {
            "first_frame": plan["burn"],
            "end_frame_exclusive": plan["end"],
            "frames": plan["weight"].shape[1],
            "mass": plan["weight"].sum((1, 2)).tolist(),
            "predicted_below_reference_phase_floor": {
                k: ((v.abs() < 1e-6) & (plan["weight"] > 0)).sum((1, 2)).tolist()
                for k, v in spectra.items()
            },
        },
    }


@torch.no_grad()
def native_check(root, batch, stored):
    observed = unpack_trace(read(root / "runtime.json"))
    condition = observed["condition"]
    rate = int(batch["rate"][0, 0])
    cutoff = condition[..., 1] * rate / 2
    mask = build_missing_band_mask(
        cutoff, torch.full_like(cutoff, rate), fft_size=256, crossover_width_hz=1000
    ).reshape(1, 1, -1, 129)
    b = {k: v[:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
    b.update(x=observed["features"], mask=mask, active=condition[..., 2].bool())
    proxy = fused_stages(observed["prediction"], b["x"], prepare_fusion(b))["final"]
    pcm, output_rate = read_sxa(root / "output.sxa")
    native = torch.from_numpy(pcm[:, 0])[None]
    if output_rate != rate or native.shape != proxy.shape:
        raise ValueError("native PCM timeline changed")
    difference = (proxy[:, 128:] - native[:, 128:]).double()
    maximum, rms = float(difference.abs().max()), float(difference.square().mean().sqrt())
    if maximum > 2**-14 or rms > 2e-5:
        raise ValueError("native PCM correspondence failure")
    b64 = {k: v.double() if v.is_floating_point() else v for k, v in b.items()}
    plan = prepare_relations(b64)
    a, z = [analyze(v.double())[:, BURN:] for v in (proxy, native)]
    result = {kind: correspondence(a, z, plan, kind) for kind in KINDS}
    for value in result.values():
        if (value["risk_difference"] > value["risk_bound"] + 1e-12).any():
            raise ValueError("phase-risk correspondence bound exceeded")
        for key in COMPONENTS:
            if (value["component_difference"][key] > value["component_bound"][key] + 1e-12).any():
                raise ValueError("phase-component correspondence bound exceeded")
    region = slice(BURN * 128, native.shape[-1] - 128)
    old = scores(native[:, region], b["clean_wave"][:, region], b["rate"])[0]
    if old != stored:
        raise ValueError("frozen native original-reference scores did not reproduce")
    extended = extended_panel(
        native[0, region].numpy()[:, None], b["clean_wave"][0, region].numpy()[:, None], rate
    )
    return {
        "pcm_max": maximum,
        "pcm_rms": rms,
        "phase_correspondence": plain(result),
        "historical_scores_exact": True,
        "native_metrics": {**old, **extended["metrics"]},
        "analysis_coverage": extended["analysis_coverage"],
        "native_relations": {k: plain(relations(z, plan, k)) for k in KINDS},
    }


def run(output):
    config = read(output / "registration.json")
    verify(config["hashes"])
    if (output / "probes").exists():
        raise ValueError("existing execution retained; no silent resume")
    p74, prior = Path(config["p74"]), Path(config["prior"])
    old = read(p74 / "registration.json")
    assert_catalogs_unchanged(
        sorted((Path(old["repository"]) / "data/runs").glob("*/evaluation-catalog.jsonl")),
        config["evaluation_catalogs"],
    )
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    schedule = read(Path(old["schedule"]))["optimization"]
    pool = LazyPool(
        schedule["specs"],
        BoundedEnvelopeClips(rows, Path(old["detector"]), p74 / "cache", capacity=100),
    )
    items = {i: pool.select(schedule["draws"][i]) for i in config["fit"] + config["transfer"]}
    for origin, entry in config["endpoints"].items():
        saved = torch.load(entry["path"], map_location="cpu", weights_only=False)
        model = load(saved).double().eval()
        for draw in config["fit"]:
            sample = samples(schedule, rows, draw)
            corpora = [s["row"]["corpus"] for s in sample] * 2
            result = probe(model, items[draw], corpora)
            registered(
                {"origin": origin, "draw": draw, "samples": sample, **result},
                output / "probes" / origin / f"{draw}.json",
            )
            print("probe", origin, draw, flush=True)
        exact(model.state_dict(), {k: v.double() for k, v in saved["model"].items()})
    records = {(r["path"], r["case"]): r for r in read(prior / "records.json")}
    for origin, arm, draw in itertools.product(
        config["endpoints"], ("circular_relation", "wrapped_relation"), config["native_draws"]
    ):
        root = prior / "native" / origin / arm / str(draw) / "0"
        stored = records[(f"native/{origin}/{arm}", f"{draw}/0")]
        result = native_check(root, items[draw][0].batch, stored["scores"])
        registered(
            {"origin": origin, "arm": arm, "draw": draw, "source": stored["source"], **result},
            output / "native" / origin / arm / f"{draw}.json",
        )
        print("native correspondence", origin, arm, draw, flush=True)
    verify(config["hashes"])
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
    print("complete:32 frozen probes,40 saved native traces,zero optimizer updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--failure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fresh = not args.output.exists()
    try:
        if args.mode == "prepare":
            prepare(
                args.prior.resolve(),
                args.controls.resolve(),
                args.failure.resolve(),
                args.output.resolve(),
            )
        else:
            run(args.output.resolve())
    except BaseException:
        if (
            args.output.exists()
            and (fresh or args.mode == "run")
            and not (args.output / "failure.json").exists()
        ):
            write_json_report({"traceback": traceback.format_exc()}, args.output / "failure.json")
        raise
