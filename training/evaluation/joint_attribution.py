"""P58 descriptive attribution of frozen P57 scores; no selection or re-scoring."""

import math
from collections import defaultdict

BANDS = (0, 4000, 8000, 12000, 16000)
COMPONENTS = {"amplitude": 1.0, "log_shape": 0.25, "envelope": 0.1, "dynamics": 0.25}


def mean(rows):
    keys = set(rows[0])
    if any(set(r) != keys for r in rows):
        raise ValueError("all score fields must match")
    if any(not math.isfinite(v) for r in rows for v in r.values()):
        raise ValueError("nonfinite score")
    return {k: math.fsum(r[k] for r in rows) / len(rows) for k in sorted(keys)}


def difference(candidate, control):
    if candidate.keys() != control.keys():
        raise ValueError("score fields differ")
    return {k: candidate[k] - control[k] for k in candidate}


def decomposition(delta):
    """Signed contributions to the existing MR score, including cancellations."""
    bands = {}
    for low in BANDS:
        weight = (0.25 if low == 0 else 1.0) / 4.25
        bands[str(low)] = {k: weight * w * delta[f"band{low}_{k}"] for k, w in COMPONENTS.items()}
    total = math.fsum(v for band in bands.values() for v in band.values())
    return {"bands": bands, "reconstructed_mr_delta": total, "residual": delta["mr"] - total}


def validate_score(score):
    if any(not math.isfinite(v) for v in score.values()):
        raise ValueError("nonfinite score")
    result = decomposition(score)
    magnitude = math.fsum(abs(v) for band in result["bands"].values() for v in band.values())
    # Bound rounding on each original float32 sum, before subtracting endpoints.
    # A relative bound on a near-zero *difference* incorrectly rejects cancellation.
    # 32 eps covers the weighted band/component/resolution reductions. This is
    # arithmetic validation, not a relaxed audio-quality acceptance tolerance.
    bound = 32 * 2**-23 * max(magnitude, abs(score["mr"]))
    if abs(result["residual"]) > bound:
        raise ValueError("band components do not reproduce the existing MR score")


def index_records(records):
    groups = defaultdict(dict)
    for row in records:
        group = (row["path"], row["role"])
        if row["id"] in groups[group]:
            raise ValueError("duplicate case within path/role")
        groups[group][row["id"]] = row
    return groups


def compare(left, right):
    if left.keys() != right.keys() or not left:
        raise ValueError("unmatched case coverage")
    sources = defaultdict(lambda: [[], []])
    case_deltas = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("source", "rate", "codec", "role", "score_start", "score_length", "channels"):
            if a[field] != b[field]:
                raise ValueError("unmatched case metadata: " + field)
        sources[a["source"]][0].append(a["scores"])
        sources[a["source"]][1].append(b["scores"])
        validate_score(a["scores"])
        validate_score(b["scores"])
        delta = difference(a["scores"], b["scores"])
        case_deltas[key] = {"source": a["source"], "delta": delta}
    per_source = {}
    for source, (aa, bb) in sorted(sources.items()):
        a, b = mean(aa), mean(bb)
        delta = difference(a, b)
        per_source[source] = {"candidate": a, "control": b, "delta": delta}
    a = mean([r["candidate"] for r in per_source.values()])
    b = mean([r["control"] for r in per_source.values()])
    delta = difference(a, b)
    return {
        "cases": len(left),
        "sources": len(per_source),
        "candidate_source_mean": a,
        "control_source_mean": b,
        "source_mean_delta": delta,
        "decomposition": decomposition(delta),
        "per_source": per_source,
        "per_case": case_deltas,
        "mr_better_sources": [s for s, r in per_source.items() if r["delta"]["mr"] < 0],
        "mr_worse_sources": [s for s, r in per_source.items() if r["delta"]["mr"] > 0],
        "spectral_gain_temporal_loss_sources": [
            s for s, r in per_source.items() if r["delta"]["mr"] < 0 < r["delta"]["dynamics"]
        ],
    }


def attribute(records, pairs):
    groups = index_records(records)
    results = {}
    for label, (candidate, control) in pairs.items():
        results[label] = {}
        for role in ("paired", "identity"):
            stages = {
                stage: compare(
                    groups[(stage + "/" + candidate, role)], groups[(stage + "/" + control, role)]
                )
                for stage in ("raw", "native")
            }
            stages["native_minus_raw_delta"] = difference(
                stages["native"]["source_mean_delta"], stages["raw"]["source_mean_delta"]
            )
            results[label][role] = stages
    return results
