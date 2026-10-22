import pytest
import torch

from evaluation.fusion_ablation import CASES, FUSIONS, intervene_condition, render_ablation
from models.causal_objective import gate_envelope, render
from models.complex_synthesis import complex_features, synthesize
from tests.test_multires_spectrum import feature_batch


def mono_batch():
    b = {k: v[:1] for k, v in feature_batch().items()}
    frames = b["x"].shape[2]
    b["active"] = torch.ones(1, frames, dtype=torch.bool)
    b["active"][:, :80] = False
    b["gate"] = gate_envelope(b["active"]).to(b["x"].dtype)
    b["mask"] = torch.full((1, 1, frames, 129), 0.5, dtype=b["x"].dtype)
    b["dry"] = synthesize(complex_features(b["x"]))
    return b


def test_current_proxy_is_exactly_existing_render_and_input_is_not_mutated():
    b = mono_batch()
    before = {k: v.clone() for k, v in b.items()}
    p = b["y"]
    expected = render(
        p, b["x"], b["mask"], b["active"], b["gate"], b["dry"], limiter_mode="smooth-knee"
    )
    actual = render_ablation(p, b, "current")
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for k in b:
        torch.testing.assert_close(b[k], before[k], rtol=0, atol=0)


def test_unfused_oracle_recovers_reference_and_all_modes_preserve_zero_residual():
    b = mono_batch()
    # Keep a genuinely degraded observation; x == y would not exercise recovery.
    b["x"][:, 0] -= 6
    b["dry"] = synthesize(complex_features(b["x"]))
    region = slice(89 * 128, -128)
    for mode in FUSIONS:
        _, wet = render_ablation(b["x"], b, mode)
        torch.testing.assert_close(wet[:, region], b["dry"][:, region])
    for mode in ("unfused", "safety-only"):
        _, wet = render_ablation(b["y"], b, mode)
        torch.testing.assert_close(wet[:, region], b["clean_wave"][:, region])
    _, attenuated = render_ablation(b["y"], b, "current")
    assert not torch.allclose(attenuated[:, region], b["clean_wave"][:, region])


def test_safety_only_keeps_warmup_but_removes_content_bypass():
    b = mono_batch()
    b["active"].zero_()
    b["gate"].zero_()
    prediction = b["y"].clone()
    prediction[:, 0] += 3
    current, _ = render_ablation(prediction, b, "unfused")
    safety, _ = render_ablation(prediction, b, "safety-only")
    torch.testing.assert_close(current[:, : 80 * 128], safety[:, : 80 * 128])
    assert not torch.allclose(current[:, 89 * 128 :], safety[:, 89 * 128 :])
    with pytest.raises(ValueError):
        render_ablation(b["y"], b, "typo")


def test_condition_interventions_keep_sample_rate_and_do_not_change_original():
    c = torch.tensor([[[0.91875, 0.8, 0.0]]])
    torch.testing.assert_close(
        intervene_condition(c, "fixed-cutoff"), torch.tensor([[[0.91875, 0.5, 0.0]]])
    )
    torch.testing.assert_close(
        intervene_condition(c, "active-one"), torch.tensor([[[0.91875, 0.8, 1.0]]])
    )
    torch.testing.assert_close(c, torch.tensor([[[0.91875, 0.8, 0.0]]]))
    assert len(CASES) == 8 and len(set(CASES)) == 8
    with pytest.raises(ValueError):
        intervene_condition(c, "typo")
