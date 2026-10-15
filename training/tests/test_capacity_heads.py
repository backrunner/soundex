import copy

import pytest
import torch

from evaluation.head_export import export_graph, verify
from models.restoration_head import RestorationHead, load_restoration_head
from scripts.run_capacity_study import variants


def example():
    torch.manual_seed(41)
    x = torch.randn(2, 2, 3, 129)
    x[:, 0] = -30 + x[:, 0] * 10
    return x, torch.rand(2, 3, 1538), torch.ones(2, 3, 3)


def test_slope_changes_initial_jacobian_without_changing_range_or_identity():
    torch.manual_seed(11)
    original = RestorationHead("complex", width=8)
    calibrated = RestorationHead("complex", width=8, residual_slope=1 / 64)
    calibrated.load_state_dict(original.state_dict())
    x, context, condition = example()
    gradients = []
    for model in (original, calibrated):
        y, _ = model(x, context, condition, model.initial_state(2))
        torch.testing.assert_close(y, x, atol=0, rtol=0)
        loss = y[:, 0].mean() + y[:, 1].mean()
        gradient = torch.autograd.grad(loss, model.backbone.decoder[-1].bias)[0]
        gradients.append(gradient)
    torch.testing.assert_close(gradients[1], gradients[0] / 64, atol=1e-7, rtol=1e-5)
    with torch.no_grad():
        original.backbone.decoder[-1].bias.fill_(4)
        calibrated.backbone.decoder[-1].bias.fill_(256)
    a, _ = original(x, context, condition, original.initial_state(2))
    b, _ = calibrated(x, context, condition, calibrated.initial_state(2))
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_checkpoint_roundtrip_and_old_default():
    for config in ({"arm": "complex", "width": 8, "residual_slope": 1 / 64}, {"arm": "anchored"}):
        model = RestorationHead(**config)
        checkpoint = {"arm": config["arm"], "model": copy.deepcopy(model.state_dict())}
        if config["arm"] == "complex":
            checkpoint["model_config"] = config
        restored = load_restoration_head(checkpoint)
        assert restored.residual_slope == model.residual_slope
        assert restored.width == model.width
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("slope", (0, -1, float("nan"), float("inf"), 2))
def test_invalid_slope_rejected(slope):
    with pytest.raises(ValueError, match="slope"):
        RestorationHead("complex", residual_slope=slope)


def test_calibrated_nonzero_export_and_silent_backward(tmp_path):
    model = RestorationHead("complex", width=8, residual_slope=1 / 64).eval()
    with torch.no_grad():
        model.backbone.decoder[-1].weight.normal_(0, 0.1)
        model.backbone.decoder[-1].bias.normal_(0, 0.1)
    report = verify(model, export_graph(model, tmp_path / "head.onnx"))
    assert report["sequential_hops"] == 320
    x = torch.zeros(2, 2, 3, 129)
    x[:, 0] = -200
    output, _ = model(x, torch.zeros(2, 3, 1538), torch.ones(2, 3, 3), model.initial_state(2))
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_capacity_variants_fit_parameter_budget():
    counts = [
        sum(p.numel() for p in RestorationHead(**variants()[f"scaled{w}"]).parameters())
        for w in (96, 144, 224)
    ]
    assert counts == [430050, 824658, 1748578]
    assert max(counts) <= 2_000_000


def test_calibration_gate_requires_both_seeds_and_every_size(tmp_path):
    from evaluation.reporting import write_json_report
    from scripts.run_capacity_study import calibration_checks

    def score(total, paired, identity, combined):
        return {
            "weighted": {
                "total": total,
                "paired_restoration": paired,
                "identity_preservation": identity,
            },
            "combined": combined,
        }

    initial = score(0.3, 0.6, 0, 0.31)
    final = score(0.28, 0.55, 0.01, 0.29)
    write_json_report(initial, tmp_path / "initial-bank.json")
    for seed in (1, 2):
        write_json_report(
            {"common": final, "arm_bank": final}, tmp_path / f"anchored-{seed}/training-512.json"
        )
        for arm in ("scaled96", "scaled144", "scaled224"):
            directory = tmp_path / f"{arm}-{seed}"
            write_json_report({"common": initial}, directory / "training-0.json")
            write_json_report({"common": final, "arm_bank": final}, directory / "training-512.json")
            write_json_report([{"preclip_gradient_norm": 1}] * 512, directory / "updates.json")
            (directory / "step-512.pth").write_bytes(b"fixture")
            write_json_report(
                {"cost": {"2": {"combined": {"p99_ms": 0.3}}}},
                tmp_path / "exports" / f"{arm}-{seed}/report.json",
            )
    assert calibration_checks({"seeds": [1, 2]}, tmp_path)["passed"]
    final["weighted"]["identity_preservation"] = 0.1
    write_json_report(
        {"common": final, "arm_bank": final}, tmp_path / "scaled224-2/training-512.json"
    )
    result = calibration_checks({"seeds": [1, 2]}, tmp_path)
    assert not result["passed"]
    assert len(result["results"]) == 6
    assert not result["results"][-1]["checks"]["full_bank_identity_not_worse"]
