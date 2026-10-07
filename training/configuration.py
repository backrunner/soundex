"""Load a training profile with canonical preprocessing defaults resolved."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from data.protocol import validate_data_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Resolve recipe defaults before binding configuration to dataset provenance."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a profile whose recipe matches the canonical dataset contract."""
    resolved = deepcopy(config)
    resolved["data"]["recipe"] = validate_data_config(resolved["data"]).to_dict()
    return resolved
