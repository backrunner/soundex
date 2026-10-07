"""Real YAML recipes must bind to canonical dataset and checkpoint hashes."""

from pathlib import Path

import pytest

from checkpoint import load_checkpoint
from checkpoint_state import canonical_sha256
from configuration import load_config
from data.protocol import validate_data_config
from export_onnx import _load_optional_config


@pytest.mark.parametrize("profile", ["default.yaml", "titan_xp.yaml"])
def test_real_profile_defaults_survive_checkpoint_save_and_export_comparison(
    profile: str, checkpoint_factory
) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / profile
    config = load_config(path)
    recipe = validate_data_config(config["data"])
    assert config["data"]["recipe"] == recipe.to_dict()
    assert canonical_sha256(config["data"]["recipe"]) == recipe.hash
    assert "preserve_official" in config["data"]["recipe"]["splits"]["medleydb"]

    checkpoint_path, _ = checkpoint_factory(config=config)
    comparison = _load_optional_config(path)
    checkpoint = load_checkpoint(checkpoint_path, expected_config=comparison)

    assert checkpoint["data"]["recipe_sha256"] == recipe.hash
