import copy

import pytest

from evaluation.reporting import sha256_file, write_json_report
from scripts.run_consistency_followup import complete
from scripts.run_full_capacity_study import ARMS, STEPS, freeze_endpoints, protocol


def parents():
    shared = {"bank": [1, 2], "common": [1], "seeds": [12, 13], "weight": 0.1, "rate": 1e-4}
    parent = {
        **shared,
        "arms": ARMS,
        "model_configs": {
            arm: {
                "arm": "anchored" if arm == "anchored" else "complex",
                "width": width,
                "residual_slope": 1.0 if arm == "anchored" else 1 / 64,
            }
            for arm, width in zip(ARMS, (640, 384, 224, 144), strict=True)
        },
    }
    original = {**shared, "development": [1] * 89, "shifted": [1] * 40, "tolerance": 1e-6}
    return parent, original


def test_full_budget_does_not_depend_on_short_calibration_scores():
    parent, original = parents()
    parent["calibration_passed"] = False
    config = protocol(parent, original)
    assert config["steps"] == 16192
    assert config["endpoints"] == [2048, 8192, 16192]
    assert config["primary_endpoint"] == STEPS
    assert config["arms"] == ["scaled640", "scaled384", "scaled224", "anchored"]
    assert config["native_count"] == 4450
    assert len(config["seeds"]) * len(config["arms"]) * config["steps"] == 129536


@pytest.mark.parametrize("field", ("bank", "common", "seeds", "weight", "rate"))
def test_matched_training_rejects_changed_definition(field):
    parent, original = parents()
    parent[field] = None
    with pytest.raises(ValueError, match="training definition differs"):
        protocol(parent, original)


def test_matched_training_rejects_changed_head_or_missing_control():
    parent, original = parents()
    changed = copy.deepcopy(parent)
    changed["model_configs"]["scaled640"]["residual_slope"] = 1
    with pytest.raises(ValueError, match="model definition changed"):
        protocol(changed, original)
    parent["arms"] = ["scaled640"]
    with pytest.raises(ValueError, match="arms changed"):
        protocol(parent, original)


def test_review_freeze_requires_all_complete_runs_and_verified_endpoints(tmp_path):
    config = {"arms": ["anchored", "scaled640"], "seeds": [12, 13], "steps": 4, "endpoints": [2, 4]}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            directory = tmp_path / f"{arm}-{seed}"
            write_json_report([{}] * 4, directory / "updates.json")
            for step in (2, 4):
                path = directory / f"step-{step}.pth"
                path.write_bytes(b"fixture")
                write_json_report(
                    {"step": step, "checkpoint_sha256": sha256_file(path)},
                    directory / f"registration-{step}.json",
                )
            if arm == "anchored" or seed == 12:
                complete(directory)
    with pytest.raises(FileNotFoundError):
        freeze_endpoints(tmp_path, config)
    assert not (tmp_path / "review-registration.json").exists()
    complete(tmp_path / "scaled640-13")
    assert len(freeze_endpoints(tmp_path, config)) == 8
    (tmp_path / "scaled640-13/step-4.pth").write_bytes(b"changed")
    with pytest.raises(ValueError):
        freeze_endpoints(tmp_path, config)
