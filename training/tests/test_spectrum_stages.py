import torch

from models.causal_objective import limit, render
from scripts.audit_spectrum_stages import stage_waves


def test_stage_counterfactual_keeps_production_limiter_before_crossfade():
    # High positive wet branch and negative dry branch: limiting AFTER mixing
    # would produce a different result, so this fixture detects ordering drift.
    frames = 32
    x = torch.zeros(1, 2, frames, 129)
    x[:, 0] = -100
    prediction = x.clone()
    prediction[:, 0, :, 0] = 50
    batch = {
        "x": x,
        "mask": torch.ones(1, 1, frames, 129),
        "active": torch.ones(1, frames),
        "gate": torch.full((1, frames * 128), 0.5),
        "dry": torch.full((1, frames * 128), -0.9),
    }
    stages = stage_waves(prediction, batch)
    expected, fused = render(
        prediction,
        x,
        batch["mask"],
        batch["active"],
        batch["gate"],
        batch["dry"],
        limiter_mode="smooth-knee",
    )
    torch.testing.assert_close(stages["limited_output"], expected, atol=0, rtol=0)
    torch.testing.assert_close(stages["fused_ola"], fused, atol=0, rtol=0)
    assert (
        stages["limited_output"] - limit(stages["gated_unlimited"], "smooth-knee")
    ).abs().max() > 0.1
    assert stages["gated_unlimited"].abs().max() > 0.95


def test_disabled_gate_keeps_dry_reference_and_exposes_limiter_changes():
    frames = 32
    x = torch.zeros(1, 2, frames, 129)
    x[:, 0] = -100
    dry = torch.linspace(-1, 1, frames * 128)[None]
    batch = {
        "x": x,
        "mask": torch.zeros(1, 1, frames, 129),
        "active": torch.zeros(1, frames),
        "gate": torch.zeros_like(dry),
        "dry": dry,
    }
    stages = stage_waves(x, batch)
    torch.testing.assert_close(stages["gated_unlimited"], dry, atol=0, rtol=0)
    torch.testing.assert_close(stages["limited_output"], limit(dry, "smooth-knee"), atol=0, rtol=0)
    assert (stages["limited_output"] - dry).abs().max() > 0.05
