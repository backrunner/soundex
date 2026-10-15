#!/usr/bin/env python3
"""Cost-only capacity discovery with nonzero diagnostic heads; no quality selection."""

import argparse
from pathlib import Path

import torch

from evaluation.head_export import export_and_measure
from evaluation.reporting import write_json_report
from models.restoration_head import RestorationHead
from scripts.run_consistency_followup import read, registered


def qualifies(report):
    # Reserve 25% of a 48 kHz hop for remaining DSP/scheduling. This is a
    # conservative screen, not proof that a physical audio worker meets deadlines.
    cost = report["cost"]["2"]["combined"]
    return (
        cost["p99_ms"] < 0.75 * 128 / 48
        and cost["max_ms"] < 5
        and cost["mean_ms"] < 0.75 * 128 / 48
        and cost["virtual_final_backlog_ms"] == 0
        and cost["virtual_deadline_misses"] / report["cost"]["2"]["measured_hops"] <= 0.001
    )


def run(output, widths, iterations):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    registered(
        {
            "widths": widths,
            "iterations": iterations,
            "seed": 20261015,
            "selection": "all passing widths; cost only; no audio scores",
            "screen": "stereo mean/p99 <2ms; max <5ms; virtual miss ratio <=0.1%; final backlog zero",
            "scope": "nonzero random output heads; CPU diagnostic, not trained quality or device certification",
        },
        output / "registration.json",
    )
    results = {}
    for width in widths:
        config = {"arm": "complex", "width": width, "residual_slope": 1 / 64}
        checkpoint = output / f"width-{width}.pth"
        if not checkpoint.exists():
            torch.manual_seed(20261015)
            model = RestorationHead(**config).eval()
            with torch.no_grad():
                model.backbone.decoder[-1].weight.normal_(0, 0.01)
                model.backbone.decoder[-1].bias.normal_(0, 0.01)
            torch.save(
                {"arm": "complex", "model_config": config, "model": model.state_dict()}, checkpoint
            )
            del model
        directory = output / f"width-{width}"
        export_and_measure(
            checkpoint, directory, experimental_unbounded=True, iterations=iterations
        )
        report = read(directory / "report.json")
        results[str(width)] = {"passed": qualifies(report), "report": report}
        write_json_report({"results": results}, output / "progress.json")
        print(width, report["parameters"], report["cost"]["2"]["combined"], flush=True)
    write_json_report(
        {
            "results": results,
            "eligible_widths": [int(w) for w, r in results.items() if r["passed"]],
        },
        output / "summary.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[224, 384, 640, 896, 1280])
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    run(args.output, args.widths, args.iterations)
