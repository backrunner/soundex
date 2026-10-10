"""Explicit architecture migration must preserve the parent and frame causality."""

from copy import deepcopy

import pytest
import torch

from export_onnx import _build_generator
from models.generator import SoundExGenerator
from train import initialize_generator_from_checkpoint


def test_zero_interactions_preserve_parent_exactly_then_receive_gradients():
    torch.manual_seed(9)
    legacy = SoundExGenerator(channels=[4, 8, 16, 16], bottleneck_blocks=1).eval()
    model = SoundExGenerator(
        channels=[4, 8, 16, 16], bottleneck_blocks=1, cross_stream_interactions=True
    ).eval()
    model.load_state_dict({**model.state_dict(), **legacy.state_dict()}, strict=True)
    features = torch.randn(2, 2, 3, 129)
    torch.testing.assert_close(model(features), legacy(features), rtol=0, atol=0)
    model(features).square().mean().backward()
    assert all(torch.isfinite(p.weight.grad).all() for pair in model.interactions for p in pair)
    assert any(p.weight.grad.abs().sum() > 0 for pair in model.interactions for p in pair)
    with torch.no_grad():
        for pair in model.interactions:
            for projection in pair:
                projection.weight.add_(0.001)
    independent = torch.cat([model(features[:, :, t:t+1]) for t in range(3)], dim=2)
    torch.testing.assert_close(model(features), independent, rtol=1e-5, atol=1e-6)
    changed = features.clone()
    changed[:, :, 2] += 5
    torch.testing.assert_close(model(changed)[:, :, :2], model(features)[:, :, :2], rtol=0, atol=0)


def test_interaction_migration_is_explicit_and_preserves_provenance(checkpoint_factory):
    path, parent = checkpoint_factory()
    config = deepcopy(parent["resolved_config"])
    config["model"]["generator"]["cross_stream_interactions"] = True
    c = config["model"]["generator"]
    model = SoundExGenerator(
        channels=c["channels"], bottleneck_blocks=c["bottleneck_blocks"],
        expand_ratio=c["expand_ratio"], cross_stream_interactions=True,
    )
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(path, model, config, parent["data"])
    receipt = initialize_generator_from_checkpoint(
        path, model, config, parent["data"], initialize_zero_interactions=True
    )
    assert receipt["mode"] == "generator-only-warm-start-zero-interactions"
    assert len(receipt["zero_initialized_state_keys"]) == 2 * len(c["channels"])
    model.eval()
    inputs = torch.randn(1, 2, 1, 129)
    torch.testing.assert_close(model(inputs), _build_generator(parent)(inputs), rtol=0, atol=0)
    model.interactions[0][0].weight.data.fill_(0.1)
    with pytest.raises(ValueError, match="exactly zero"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_zero_interactions=True
        )
    config["model"]["generator"]["expand_ratio"] += 1
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_zero_interactions=True
        )


def test_export_builder_restores_trained_interactions(checkpoint_factory):
    _, checkpoint = checkpoint_factory()
    config = deepcopy(checkpoint["resolved_config"])
    config["model"]["generator"]["cross_stream_interactions"] = True
    _, interaction_checkpoint = checkpoint_factory("interactions.pth", config=config)
    interaction_checkpoint["model"]["generator_state"]["interactions.0.0.weight"].fill_(0.012)
    assert interaction_checkpoint["model"]["architecture"]["minor"] == 1
    model = _build_generator(interaction_checkpoint)
    assert model.interactions[0][0].weight[0, 0].item() == pytest.approx(0.012)
    assert model(torch.zeros(1, 2, 1, 129)).shape == (1, 2, 1, 129)
