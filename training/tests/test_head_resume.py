import copy

import pytest
import torch

from data.causal_study import digest
from evaluation.reporting import write_json_report
from scripts import run_head_study as runner


@pytest.mark.parametrize("configured", (False, True, "joint", "detached", "broad"))
def test_head_study_resumes_exactly_and_rejects_schedule_drift(tmp_path, monkeypatch, configured):
    torch.set_num_threads(1)

    def factory(arm, width=144, residual_slope=1, generation_budget=False):
        return torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1)
        )

    class Objective:
        def __init__(self):
            self.batch = {}

        def __call__(self, prediction):
            return {"total": (prediction - 0.2).square().mean()}

    class Pool:
        def select(self, _):
            return Objective(), torch.ones(5, 2)

    score = {"weighted": {"total": 0.1}, "auxiliary": 0.05, "combined": 0.105}
    monkeypatch.setattr(runner, "RestorationHead", factory)
    monkeypatch.setattr(runner, "shared_state", lambda m: m.state_dict())
    monkeypatch.setattr(runner, "predict", lambda m, o, c: m(c))
    monkeypatch.setattr(runner, "auxiliary", lambda p, b: p.square().mean())
    monkeypatch.setattr(runner, "evaluate", lambda *args: copy.deepcopy(score))
    monkeypatch.setattr(runner, "print", lambda *args, **kwargs: None, raising=False)
    draws = [[0, 1, 2, 3, 4]] * 256
    config = {
        "rate": 0.0001,
        "weight": 0.1,
        "steps": 256,
        "endpoints": [256],
        "schedule_hashes": {"7": digest(draws)},
    }
    if configured:
        config["model_configs"] = {
            "small": {"arm": "complex", "width": 96, "residual_slope": 1 / 64}
        }
        config["rates"] = {"small": 1e-5}
    if configured in ("joint", "detached"):
        config["model_configs"]["small"]["generation_budget"] = True
        config["balanced_arms"] = ["small"]
        config["budget_weights"] = {"small": 0.1}
        if configured == "detached":
            config["detached_budget_arms"] = ["small"]

        def predict_budget(model, batch, context, *, detach_auxiliary=False):
            assert detach_auxiliary == (configured == "detached")
            return model(context), model(context)

        monkeypatch.setattr(runner, "predict_budget", predict_budget)
        monkeypatch.setattr(runner, "budget_risk", lambda p, b: p.square().mean())
        monkeypatch.setattr(runner, "balanced_risk", lambda o, p: o(p)["total"])
        monkeypatch.setattr(runner, "consistency_loss", lambda p, b, **kw: p.square().mean())
    suffix = "7"
    if configured == "broad":
        config["schedule_groups"] = {"small": "broad"}
        config["schedule_hashes"] = {"broad-7": digest(draws)}
        config["score_entire_bank"] = False
        suffix = "broad-7"
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    for root in (full, resumed):
        root.mkdir()
        write_json_report({"draws": draws}, root / f"draws-{suffix}.json")
    runner.train("small", 7, full, config, Pool(), None)
    save = runner.atomic_save

    def interrupt(value, path):
        save(value, path)
        if path.name == "latest.pth" and value["step"] == 64:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "atomic_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train("small", 7, resumed, config, Pool(), None)
    monkeypatch.setattr(runner, "atomic_save", save)
    write_json_report({"draws": [[0, 1, 2, 3, 5]] * 256}, resumed / f"draws-{suffix}.json")
    with pytest.raises(ValueError, match="draw schedule drift"):
        runner.train("small", 7, resumed, config, Pool(), None)
    write_json_report({"draws": draws}, resumed / f"draws-{suffix}.json")
    runner.train("small", 7, resumed, config, Pool(), None)
    a, b = [
        torch.load(root / "small-7/step-256.pth", weights_only=False) for root in (full, resumed)
    ]
    assert a["history"] == b["history"]
    assert a["initial_sha256"] == b["initial_sha256"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for key in a["optimizer"]["state"]:
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                a["optimizer"]["state"][key][name],
                b["optimizer"]["state"][key][name],
                atol=0,
                rtol=0,
            )
