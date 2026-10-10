"""Phase feature expansion preserves warm-start and has bounded deployment cost."""

from copy import deepcopy

import pytest
import torch

from export_onnx import _build_generator
from models.generator import SoundExGenerator
from train import initialize_generator_from_checkpoint


def test_circular_channels_preserve_base_initialization_rng_and_frame_causality():
    torch.manual_seed(19)
    control = SoundExGenerator(channels=[4, 8, 16, 16], bottleneck_blocks=1).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    circular = SoundExGenerator(
        channels=[4, 8, 16, 16], bottleneck_blocks=1, circular_phase_features=True
    ).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    x = torch.randn(2, 2, 3, 129)
    torch.testing.assert_close(control(x), circular(x), rtol=1e-6, atol=1e-7)
    circular(x).square().mean().backward()
    weight = circular.phase_stream.encoders[0].conv[0].weight
    assert torch.isfinite(weight.grad).all() and weight.grad[:, 1:].abs().sum() > 0
    with torch.no_grad():
        weight[:, 1:].fill_(0.01)
    independent = torch.cat([circular(x[:, :, t : t + 1]) for t in range(3)], dim=2)
    torch.testing.assert_close(circular(x), independent, rtol=1e-5, atol=1e-6)
    changed = x.clone()
    changed[:, :, 2] += 10
    torch.testing.assert_close(circular(x)[:, :, :2], circular(changed)[:, :, :2], rtol=0, atol=0)


def test_phase_feature_migration_is_strict_and_records_expanded_channels(checkpoint_factory):
    path, parent = checkpoint_factory()
    config = deepcopy(parent["resolved_config"])
    config["model"]["generator"]["circular_phase_features"] = True
    c = config["model"]["generator"]
    model = SoundExGenerator(
        channels=c["channels"],
        bottleneck_blocks=c["bottleneck_blocks"],
        expand_ratio=c["expand_ratio"],
        circular_phase_features=True,
    ).eval()
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(path, model, config, parent["data"])
    receipt = initialize_generator_from_checkpoint(
        path, model, config, parent["data"], initialize_phase_features=True
    )
    assert receipt["expanded_zero_channel_state_keys"] == ["phase_stream.encoders.0.conv.0.weight"]
    x = torch.randn(2, 2, 1, 129)
    torch.testing.assert_close(model(x), _build_generator(parent)(x), rtol=1e-6, atol=1e-7)
    model.phase_stream.encoders[0].conv[0].weight.data[:, 1:].fill_(0.01)
    with pytest.raises(ValueError, match="exactly zero"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_phase_features=True
        )
    config["model"]["generator"]["expand_ratio"] += 1
    with pytest.raises(ValueError, match="architecture"):
        initialize_generator_from_checkpoint(
            path, model, config, parent["data"], initialize_phase_features=True
        )
