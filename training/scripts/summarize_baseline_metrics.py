"""P70 report derivation from frozen evidence; no rescoring or model selection."""

import argparse
import shutil
import statistics
from pathlib import Path

import onnx

from data.causal_study import load_sources
from evaluation.reporting import sha256_file
from evaluation.trainability_audit import assert_reserved_isolation
from scripts.run_band_training import jsonl
from scripts.run_consistency_followup import read, registered
from scripts.run_longform_review import checked_receipt


def summarize(root):
    checked_receipt(root / "complete.json")
    config = read(root / "registration.json")
    prior = Path(config["prior"])
    old = read(prior / "registration.json")
    results = read(root / "assessment.json")
    cases = results["metrics"]
    means = {
        label: {
            metric: statistics.mean(r["scores"][label][metric] for r in cases)
            for metric in cases[0]["scores"][label]
        }
        for label in cases[0]["scores"]
    }
    counts = {
        label: {
            metric: sum(
                r["scores"][label][metric] < r["scores"]["untreated"][metric] for r in cases
            )
            for metric in means[label]
        }
        for label in ("half_correct_hf", "correct_hf", "same_energy_scrambled_hf", "mute_hf")
    }
    features = {}
    for label, rows in results["features"].items():
        terms = {}
        for variant in rows[0]["terms"]:
            terms[variant] = {
                metric: statistics.mean(
                    value
                    for row in rows
                    for i, value in enumerate(row["terms"][variant][metric])
                    if row["corpora"][i] == "music_library" and not row["identity"][i]
                )
                for metric in rows[0]["terms"][variant]
            }
        features[label] = {
            "paired_music_fit_means": terms,
            "rotation_rms_mean": statistics.mean(r["rotation_rms"] for r in rows),
            "ablation_difference_rms_mean": statistics.mean(
                r["ablation_rotation_difference_rms"] for r in rows
            ),
            "rotation_tanh_max": max(r["rotation_tanh_max"] for r in rows),
            "scope": "First eight TRAIN batches; interventions are not new native quality scores or achievable oracle optima.",
        }
    rows, _ = load_sources(list(map(Path, old["manifests"])))
    sources_path = Path(old["p55"]) / "registration.json"
    reserved = read(sources_path)["sources"]
    catalog = Path(old["repository"]) / "data/catalogs/current-approved.jsonl"
    training = [r for r in rows.values() if r["split"] == "train"]
    assert_reserved_isolation(training, list(rows.values()) + jsonl(catalog), reserved)
    original = (
        Path(old["repository"]) / "data/reports/architecture-study-20261011/shape-small/best.onnx"
    )
    metadata = {p.key: p.value for p in onnx.load(old["controls"]["legacy"]).metadata_props}
    if sha256_file(original) != metadata["soundex.legacy_sha256"]:
        raise ValueError("legacy original/wrapper binding changed")
    original_audit = original.parent / "audit.json"
    longform = prior / "longform-recheck/summary.json"
    report = {
        "metric_means": means,
        "controls_better_than_untreated_counts": counts,
        "max_saved_score_reproduction_error": max(r["reproduction_max"] for r in cases),
        "feature_interventions": features,
        "legacy": {
            "original_sha256": sha256_file(original),
            "wrapper_metadata": metadata,
            "historical_parameter_audit": read(original_audit),
        },
        "isolation": {
            "inherited_train_recordings": len({r["track_id"] for r in training}),
            "reserved_sources": len(reserved),
            "connected_alias_check": "passed",
            "scope": "Inherited manifest coverage, not proof against every historical exposure or repeated development selection.",
        },
        "full_p69_groups": read(longform)["groups"],
    }
    registered(report, root / "summary.json")
    frozen = root / "summary-source.py"
    if frozen.exists():
        raise ValueError("retain prior summary source")
    shutil.copy2(Path(__file__).resolve(), frozen)
    paths = [
        root / "summary.json",
        root / "complete.json",
        root / "assessment.json",
        frozen,
        sources_path,
        original,
        original_audit,
        catalog,
        longform,
    ]
    registered({"files": {str(p): sha256_file(p) for p in paths}}, root / "summary-receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root.resolve())
