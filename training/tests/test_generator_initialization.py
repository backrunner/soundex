"""Changed objectives start a new, provenance-bound optimizer trajectory."""

from copy import deepcopy

import pytest
import torch

from checkpoint import CheckpointSchemaError, load_checkpoint
from models.generator import SoundExGenerator
from train import initialize_generator_from_checkpoint


def _generator(config):
    c = config["model"]["generator"]
    return SoundExGenerator(
        channels=c["channels"],
        bottleneck_blocks=c["bottleneck_blocks"],
        expand_ratio=c["expand_ratio"],
    )


@pytest.mark.parametrize("change", ["objective", "statistics", "inpainting"])
def test_new_objective_can_initialize_generator_but_cannot_resume(checkpoint_factory, change):
    path, parent = checkpoint_factory()
    config = deepcopy(parent["resolved_config"])
    if change == "objective":
        config["training"]["objective"]["version"] = 2
    elif change == "statistics":
        config["training"]["batch_norm_statistics"] = "frozen"
    else:
        config["training"]["objective"].update(version=3, high_band_spectral_weight=1.0)
    generator = _generator(config)
    receipt = initialize_generator_from_checkpoint(path, generator, config, parent["data"])
    for key, value in generator.state_dict().items():
        torch.testing.assert_close(value, parent["model"]["generator_state"][key], rtol=0, atol=0)
    assert receipt["mode"] == "generator-only-warm-start"
    assert len(receipt["parent_checkpoint_sha256"]) == 64
    assert receipt["parent_epoch"] == 10
    with pytest.raises(CheckpointSchemaError, match="resolved_config"):
        load_checkpoint(path, expected_config=config)


def test_changed_feature_contract_or_manifest_cannot_initialize(checkpoint_factory):
    path, parent = checkpoint_factory()
    config = deepcopy(parent["resolved_config"])
    generator = _generator(config)
    config["audio"]["crossover_width_hz"] += 100
    with pytest.raises(ValueError, match="feature contract"):
        initialize_generator_from_checkpoint(path, generator, config, parent["data"])
    data = deepcopy(parent["data"])
    data["manifest_set_sha256"] = "b" * 64
    with pytest.raises(CheckpointSchemaError, match="manifest_set_sha256"):
        initialize_generator_from_checkpoint(path, generator, parent["resolved_config"], data)


def test_nonfinite_parent_cannot_initialize(checkpoint_factory):
    path, parent = checkpoint_factory()
    next(iter(parent["model"]["generator_state"].values())).fill_(float("nan"))
    torch.save(parent, path)
    with pytest.raises(ValueError, match="non-finite"):
        initialize_generator_from_checkpoint(
            path, _generator(parent["resolved_config"]), parent["resolved_config"], parent["data"]
        )
