"""P49 summaries retain every seed, support, control and preservation counterexample.

SPDX-License-Identifier: Apache-2.0
"""

from collections import Counter

import numpy as np

from evaluation.anchored_review import compare


def changes(current, base):
    result = {}
    for key in base[0]:
        old = np.array([v[key] for v in base])
        new = np.array([v[key] for v in current])
        delta = new - old
        numerical = 8 * np.finfo(np.float32).eps * np.maximum(abs(old), 1e-12)
        result[key] = {
            "count": len(old),
            "improved": int((delta < 0).sum()),
            "worsened": int((delta > 0).sum()),
            "unchanged": int((delta == 0).sum()),
            "improved_beyond_numerical": int((delta < -numerical).sum()),
            "worsened_beyond_numerical": int((delta > numerical).sum()),
            "base_mean": float(old.mean()),
            "current_mean": float(new.mean()),
            "mean_change": float(delta.mean()),
            "median_change": float(np.median(delta)),
            "min_change": float(delta.min()),
            "max_change": float(delta.max()),
        }
    return result


def summarize(config, entries, reports):
    result = {"scope": config["scope"], "coverage": config["coverage"], "arms": {}}
    for seed in config["starts"]:
        for support in config["supports"]:
            cases = [
                reports[label]
                for label, e in entries.items()
                if e["seed"] == seed and e["support"] == support
            ]
            base_fit = [c["fit"]["base"] for c in cases]
            base_guard = [c["guard"]["start"] for c in cases]
            group = {
                "count": len(cases),
                "negative_derivatives": sum(c["fit"]["derivative"] < 0 for c in cases),
                "backtrack_factors": dict(
                    Counter(str(c["fit"]["search"]["selected_scale"]) for c in cases)
                ),
                "controls": {},
            }
            for arm in ("adam", "fixed", "backtrack"):
                fit = []
                for c in cases:
                    if arm != "backtrack":
                        fit.append(c["fit"]["controls"][arm]["fit"])
                    else:
                        accepted = [p for p in c["fit"]["search"]["points"] if p["accepted"]]
                        fit.append(accepted[0]["fit"] if accepted else c["fit"]["base"])
                guard = [c["guard"][arm] for c in cases]
                rulers = {str(b): Counter() for b in (16, 20, 24)}
                for c in guard:
                    for bits, ruler in c["reference_rulers"].items():
                        for violation in ruler["violations"]:
                            rulers[bits][violation["role"]] += 1
                metric_changes = changes(
                    [c["metrics"] for c in guard], [c["metrics"] for c in base_guard]
                )
                spectrum_keys = ("paired_direct_amplitude", "paired_mr")
                checks = (
                    *spectrum_keys,
                    "paired_direct_complex",
                    "paired_mr_dynamics",
                    "identity_mr",
                )
                group["controls"][arm] = {
                    "fit": changes(fit, base_fit),
                    "guard": metric_changes,
                    "guard_spectra_both_improve": sum(
                        all(a["metrics"][k] < b["metrics"][k] for k in spectrum_keys)
                        for a, b in zip(guard, base_guard, strict=True)
                    ),
                    "guard_all_five_improve": sum(
                        all(a["metrics"][k] < b["metrics"][k] for k in checks)
                        for a, b in zip(guard, base_guard, strict=True)
                    ),
                    "raw_guard": compare(guard, base_guard),
                    "reference_ruler_violations": rulers,
                }
            result["arms"][f"{seed}-{support}"] = group
    return result
