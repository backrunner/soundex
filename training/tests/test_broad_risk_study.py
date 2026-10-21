from types import SimpleNamespace

import pytest
import torch

from models.risk_balance import training_terms as previous_terms
from scripts import run_direct_spectrum_study as runner
from scripts.run_broad_risk_study import ARMS, SEEDS, STEPS, protocol, training_terms
from tests.test_multires_spectrum import feature_batch


def test_full_protocol_fixes_architecture_budget_roles_and_selection():
    config = protocol({"direct_loss": {}, "multires_loss": {}, "weight": 1, "development": []})
    assert config["steps"] == STEPS == 16192
    assert config["seeds"] == list(SEEDS) == [20261012, 20261013]
    assert config["arms"] == list(ARMS)
    assert len({str(v) for v in config["model_configs"].values()}) == 1
    assert config["loss_routes"] == {arm: {arm: 1.0} for arm in ARMS}
    assert "BOTH seeds" in config["selection"] and "No overshoot guard" in config["contrast"]


@pytest.mark.parametrize("arm", ARMS)
def test_lean_callback_preserves_p34_values_and_gradients(arm):
    batch = feature_batch()
    batch["dry"] = batch["clean_wave"].clone()
    prediction = batch["x"].clone()
    prediction[:, 0, 89:, 48:] -= 3
    prediction.requires_grad_()
    budget = torch.full(
        (2, prediction.shape[2] - 80, 5), 0.2, dtype=prediction.dtype, requires_grad=True
    )
    objective = SimpleNamespace(
        batch=batch, region=slice(89 * 128, prediction.shape[2] * 128 - 128)
    )
    old = previous_terms(prediction, budget, objective, 1)
    new = training_terms(prediction, budget, objective, 1)
    assert "overshoot" not in new
    for key in new:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    old_grad = torch.autograd.grad(old[arm], (prediction, budget), retain_graph=True)
    new_grad = torch.autograd.grad(new[arm], (prediction, budget))
    for left, right in zip(old_grad, new_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(new["control"], new["raw"] + new["multires"] + new["budget"])


def test_checkpoint_resume_reproduces_uninterrupted_optimizer_and_absolute_draws(
    tmp_path, monkeypatch
):
    class Tiny(torch.nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.value = torch.nn.Parameter(torch.randn(3))

    monkeypatch.setattr(runner, "RestorationHead", Tiny)
    monkeypatch.setattr(
        runner, "predict_budget", lambda model, batch, _: (model.value * batch, None)
    )

    class Batches:
        def __init__(self, fail=False):
            self.fail, self.seen = fail, []

        def __len__(self):
            return 65

        def __getitem__(self, index):
            self.seen.append(index)
            if self.fail and index == 64:
                raise RuntimeError("simulated interruption after saved update")
            return SimpleNamespace(batch=1 + index / 100), None

    def terms(prediction, *_):
        return {"control": (prediction - 0.4).square().mean()}

    config = {
        "model_configs": {"control": {}},
        "steps": 65,
        "rate": 1e-4,
        "weight": 1,
        "loss_routes": {"control": {"control": 1}},
    }
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir()
    resumed.mkdir()
    runner.train("control", 17, full, config, Batches(), training_terms=terms)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train("control", 17, resumed, config, Batches(fail=True), training_terms=terms)
    batches = Batches()
    runner.train("control", 17, resumed, config, batches, training_terms=terms)
    assert batches.seen == [64]
    a, b = [
        torch.load(root / "control-17/step-65.pth", weights_only=False) for root in (full, resumed)
    ]
    assert a["history"] == b["history"] and a["signature"] == b["signature"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], rtol=0, atol=0)
