import copy

import pytest

from evaluation import head_export
from evaluation.reporting import write_json_report
from models.restoration_head import RestorationHead
from scripts.run_consistency_followup import complete
from scripts.run_latency_capacity_study import summarize
from scripts.screen_latency_capacity import qualifies


def report(values):
    return {
        "cost": {
            "2": {
                "combined": {**head_export.distribution(values), **head_export.stream_cost(values)},
                "measured_hops": len(values),
            }
        }
    }


def test_fixed_clock_accumulates_and_recovers_backlog():
    result = head_export.stream_cost([3, 3, 1], period_ms=2)
    assert result["virtual_deadline_misses"] == 3
    assert result["virtual_max_backlog_ms"] == 2
    assert result["virtual_final_backlog_ms"] == 1
    assert head_export.stream_cost([3, 3, 1, 1], period_ms=2)["virtual_final_backlog_ms"] == 0


def test_sub_five_ms_does_not_imply_streaming_acceptance():
    assert qualifies(report([1.2] * 1000))
    assert not qualifies(report([3.0] * 1000))
    assert not qualifies(report([1.0] * 999 + [6]))
    candidate = report([1.0] * 1000)
    for name, value in (
        ("p99_ms", 2.0),
        ("mean_ms", 2.0),
        ("virtual_final_backlog_ms", 0.1),
        ("virtual_deadline_misses", 2),
    ):
        bad = copy.deepcopy(candidate)
        bad["cost"]["2"]["combined"][name] = value
        assert not qualifies(bad)


def test_size_override_is_explicit_and_export_remains_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(head_export, "MAX_ARTIFACT_BYTES", 1)
    model = RestorationHead("complex", width=8, residual_slope=1 / 64).eval()
    with pytest.raises(ValueError, match="budget"):
        head_export.export_graph(model, tmp_path / "limited.onnx")
    session = head_export.export_graph(
        model, tmp_path / "experimental.onnx", experimental_unbounded=True
    )
    assert head_export.verify(model, session)["sequential_hops"] == 320


def test_measurement_receipt_cannot_reuse_different_budget_or_iterations(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    directory = tmp_path / "export"
    write_json_report(
        {"checkpoint_sha256": head_export.sha256_file(checkpoint)}, directory / "report.json"
    )
    complete(directory)
    head_export.export_and_measure(checkpoint, directory)
    with pytest.raises(ValueError, match="policy drift"):
        head_export.export_and_measure(checkpoint, directory, experimental_unbounded=True)
    with pytest.raises(ValueError, match="policy drift"):
        head_export.export_and_measure(checkpoint, directory, iterations=10000)


def test_capacity_summary_preserves_identity_and_second_seed_failures(tmp_path):
    def score(total, paired, identity):
        return {
            "weighted": {
                "total": total,
                "paired_restoration": paired,
                "identity_preservation": identity,
            },
            "combined": total + 0.01,
        }

    initial, improved = score(0.3, 0.6, 0), score(0.28, 0.55, 0.01)
    write_json_report(initial, tmp_path / "initial-bank.json")
    config = {"seeds": [1, 2], "arms": ["anchored", "scaled640"], "scope": "fixture"}
    for seed in config["seeds"]:
        for arm in config["arms"]:
            directory = tmp_path / f"{arm}-{seed}"
            write_json_report({"common": initial}, directory / "training-0.json")
            write_json_report(
                {"common": improved, "arm_bank": improved}, directory / "training-512.json"
            )
            write_json_report([{"preclip_gradient_norm": 1}] * 512, directory / "updates.json")
            (directory / "step-512.pth").write_bytes(b"fixture")
            write_json_report(
                report([1.2] * 1000), tmp_path / "exports" / f"{arm}-{seed}/report.json"
            )
    result = summarize(tmp_path, config)
    assert len(result["results"]) == 4
    assert all(row["passed"] for row in result["results"])
    damaged = score(0.27, 0.54, 0.02)
    write_json_report(
        {"common": improved, "arm_bank": damaged}, tmp_path / "scaled640-2/training-512.json"
    )
    result = summarize(tmp_path, config)
    assert not result["results"][-1]["passed"]
    assert not result["results"][-1]["checks"]["arm_bank_identity_not_worse"]
