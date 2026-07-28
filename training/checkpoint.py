"""Versioned, reproducible SoundEx training checkpoint protocol."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

from artifact_contract import SUPPORTED_FRAME_CONTRACTS
from checkpoint_state import (
    canonical_sha256,
    capture_provenance,
    capture_rng_state,
    to_plain,
)
from validation import (
    PRIMARY_METRIC,
    TIE_BREAKER_METRIC,
    VALIDATION_SCHEMA_VERSION,
    BestCheckpointTracker,
)

CHECKPOINT_SCHEMA_MAJOR = 1
CHECKPOINT_SCHEMA_MINOR = 2
MODEL_ARCHITECTURE_NAME = "soundex-generator"
MODEL_ARCHITECTURE_MAJOR = 1
MODEL_ARCHITECTURE_MINOR = 0


class CheckpointSchemaError(ValueError):
    """A checkpoint is absent, incompatible, or internally inconsistent."""


class SoundExCheckpoint(TypedDict):
    """Serialized checkpoint shape; values remain weights-only-loadable."""

    schema: dict[str, Any]
    model: dict[str, Any]
    training_state: dict[str, Any]
    resolved_config: dict[str, Any]
    feature_contract: dict[str, Any]
    data: dict[str, Any]
    provenance: dict[str, Any]
    rng_state: dict[str, Any]
    hashes: dict[str, str]


def feature_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the complete training/runtime feature contract from resolved config."""
    audio = _mapping(config.get("audio"), "resolved_config.audio")
    training = _mapping(config.get("training"), "resolved_config.training")
    data = _mapping(config.get("data"), "resolved_config.data")
    recipe = _mapping(data.get("recipe"), "resolved_config.data.recipe")
    fft_size = int(audio.get("fft_size", 0))
    hop_size = int(audio.get("hop_size", 0))
    context_frames = int(training.get("context_frames", 0))
    sample_rates = sorted(int(rate) for rate in recipe.get("sample_rates", []))
    return {
        "schema_version": 1,
        "tensor_layout": "BCTF",
        "input_shape": ["batch", 2, 1, fft_size // 2 + 1],
        "output_shape": ["batch", 2, 1, fft_size // 2 + 1],
        "input_channels": ["log_magnitude_db", "phase_radians"],
        "output_channels": ["log_magnitude_db", "phase_radians"],
        "magnitude": {
            "scale": "decibels",
            "formula": "20*log10(max(abs(stft),1e-10))",
            "floor_db": -200.0,
        },
        "phase": {"units": "radians", "range": "[-pi,pi]"},
        "window": {"name": "hann", "periodic": True},
        "fft_size": fft_size,
        "hop_size": hop_size,
        "crossover_width_hz": float(audio.get("crossover_width_hz", 0.0)),
        "supported_sample_rates": sample_rates,
        "context_frames": context_frames,
        "causal": True,
        "stateless": True,
    }


def build_checkpoint(
    *,
    epoch: int,
    global_step: int,
    generator_state: Mapping[str, Any],
    discriminator_state: Mapping[str, Any],
    generator_optimizer_state: Mapping[str, Any],
    discriminator_optimizer_state: Mapping[str, Any],
    generator_scheduler_state: Mapping[str, Any],
    discriminator_scheduler_state: Mapping[str, Any],
    scaler_state: Mapping[str, Any] | None,
    resolved_config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    validation_state: Mapping[str, Any],
    train_loader: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SoundExCheckpoint:
    """Build a validated schema-1.2 checkpoint from live training state."""
    config = cast(dict[str, Any], to_plain(resolved_config))
    model_config = _mapping(config.get("model"), "resolved_config.model")
    checkpoint: SoundExCheckpoint = {
        "schema": {
            "name": "soundex-checkpoint",
            "major": CHECKPOINT_SCHEMA_MAJOR,
            "minor": CHECKPOINT_SCHEMA_MINOR,
        },
        "model": {
            "architecture": {
                "name": MODEL_ARCHITECTURE_NAME,
                "major": MODEL_ARCHITECTURE_MAJOR,
                "minor": MODEL_ARCHITECTURE_MINOR,
            },
            "generator_config": to_plain(model_config.get("generator")),
            "discriminator_config": to_plain(model_config.get("discriminator")),
            "generator_state": dict(generator_state),
            "discriminator_state": dict(discriminator_state),
        },
        "training_state": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "optimizers": {
                "generator": dict(generator_optimizer_state),
                "discriminator": dict(discriminator_optimizer_state),
            },
            "schedulers": {
                "generator": dict(generator_scheduler_state),
                "discriminator": dict(discriminator_scheduler_state),
            },
            "scaler": dict(scaler_state) if scaler_state is not None else None,
            "validation": cast(dict[str, Any], to_plain(validation_state)),
        },
        "resolved_config": config,
        "feature_contract": feature_contract_from_config(config),
        "data": cast(dict[str, Any], to_plain(data_provenance)),
        "provenance": cast(dict[str, Any], to_plain(provenance or capture_provenance())),
        "rng_state": capture_rng_state(train_loader),
        "hashes": {
            "resolved_config_sha256": canonical_sha256(config),
            "feature_contract_sha256": canonical_sha256(feature_contract_from_config(config)),
        },
    }
    validate_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """Validate and atomically save a checkpoint in the destination directory."""
    validate_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
    )
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        torch.save(dict(checkpoint), staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    expected_config: Mapping[str, Any] | None = None,
    expected_data: Mapping[str, Any] | None = None,
) -> SoundExCheckpoint:
    """Load and strictly validate a schema-1.2 checkpoint."""
    raw = torch.load(path, map_location=map_location, weights_only=True)
    return validate_checkpoint(raw, expected_config=expected_config, expected_data=expected_data)


def validate_checkpoint(
    value: Any,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_data: Mapping[str, Any] | None = None,
) -> SoundExCheckpoint:
    """Validate schema, architecture, feature, data, and hash consistency."""
    checkpoint = _mapping(value, "checkpoint")
    if "schema" not in checkpoint:
        raise CheckpointSchemaError(
            "schema-0 checkpoint is not exportable; retrain with the schema-1 contract"
        )
    schema = _mapping(checkpoint["schema"], "checkpoint.schema")
    if schema.get("name") != "soundex-checkpoint":
        raise CheckpointSchemaError("checkpoint.schema.name: expected 'soundex-checkpoint'")
    major = int(schema.get("major", 0))
    if major != CHECKPOINT_SCHEMA_MAJOR:
        raise CheckpointSchemaError(f"unsupported checkpoint schema major version {major}")
    if "minor" not in schema or int(schema["minor"]) < 0:
        raise CheckpointSchemaError("checkpoint.schema.minor: expected a non-negative integer")
    minor = int(schema["minor"])
    if minor < CHECKPOINT_SCHEMA_MINOR:
        raise CheckpointSchemaError(
            "checkpoint schema 1.2 or newer is required for crossover-bound export/resume"
        )

    required = {
        "model",
        "training_state",
        "resolved_config",
        "feature_contract",
        "data",
        "provenance",
        "rng_state",
        "hashes",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise CheckpointSchemaError(f"checkpoint.{missing[0]}: required field is missing")
    model = _mapping(checkpoint["model"], "checkpoint.model")
    architecture = _mapping(model.get("architecture"), "checkpoint.model.architecture")
    if architecture.get("name") != MODEL_ARCHITECTURE_NAME:
        raise CheckpointSchemaError("checkpoint model architecture name is incompatible")
    if int(architecture.get("major", 0)) != MODEL_ARCHITECTURE_MAJOR:
        raise CheckpointSchemaError("checkpoint model architecture major version is incompatible")
    if "minor" not in architecture or int(architecture["minor"]) < 0:
        raise CheckpointSchemaError(
            "checkpoint.model.architecture.minor: expected a non-negative integer"
        )
    for key in (
        "generator_config",
        "discriminator_config",
        "generator_state",
        "discriminator_state",
    ):
        if key not in model:
            raise CheckpointSchemaError(f"checkpoint.model.{key}: required field is missing")

    config = _mapping(checkpoint["resolved_config"], "checkpoint.resolved_config")
    config_model = _mapping(config.get("model"), "checkpoint.resolved_config.model")
    _compare(
        config_model.get("generator"),
        model["generator_config"],
        "checkpoint.model.generator_config",
    )
    _compare(
        config_model.get("discriminator"),
        model["discriminator_config"],
        "checkpoint.model.discriminator_config",
    )
    feature = _mapping(checkpoint["feature_contract"], "checkpoint.feature_contract")
    derived_feature = feature_contract_from_config(config)
    _compare(derived_feature, feature, "checkpoint.feature_contract")
    _validate_runtime_contract(feature)

    training_state = _mapping(checkpoint["training_state"], "checkpoint.training_state")
    for key in ("epoch", "global_step", "optimizers", "schedulers", "scaler"):
        if key not in training_state:
            raise CheckpointSchemaError(
                f"checkpoint.training_state.{key}: required field is missing"
            )
    if int(training_state["epoch"]) < 0 or int(training_state["global_step"]) < 0:
        raise CheckpointSchemaError("checkpoint training counters must be non-negative")
    if minor >= 1 and "validation" not in training_state:
        raise CheckpointSchemaError(
            "checkpoint.training_state.validation: required for schema 1.2 or newer"
        )

    data = _mapping(checkpoint["data"], "checkpoint.data")
    for key in (
        "recipe",
        "recipe_sha256",
        "manifests",
        "manifest_set_sha256",
        "effective_source_counts",
    ):
        if key not in data:
            raise CheckpointSchemaError(f"checkpoint.data.{key}: required field is missing")
    if canonical_sha256(data["recipe"]) != str(data["recipe_sha256"]):
        raise CheckpointSchemaError("checkpoint.data.recipe_sha256: recipe hash mismatch")
    config_data = _mapping(config.get("data"), "checkpoint.resolved_config.data")
    _compare(config_data.get("recipe"), data["recipe"], "checkpoint.data.recipe")
    _require_sha256(data["manifest_set_sha256"], "checkpoint.data.manifest_set_sha256")
    if minor >= 1:
        if "validation_manifest_sha256" not in data:
            raise CheckpointSchemaError(
                "checkpoint.data.validation_manifest_sha256: required for schema 1.2 or newer"
            )
        _require_sha256(
            data["validation_manifest_sha256"],
            "checkpoint.data.validation_manifest_sha256",
        )
    manifest_bindings: list[dict[str, str]] = []
    for index, manifest in enumerate(cast(Sequence[Any], data["manifests"])):
        item = _mapping(manifest, f"checkpoint.data.manifests[{index}]")
        _require_sha256(item.get("sha256"), f"checkpoint.data.manifests[{index}].sha256")
        for key in ("corpus", "path", "summary"):
            if key not in item:
                raise CheckpointSchemaError(
                    f"checkpoint.data.manifests[{index}].{key}: required field is missing"
                )
        manifest_bindings.append({"corpus": str(item["corpus"]), "sha256": str(item["sha256"])})
    if canonical_sha256(manifest_bindings) != str(data["manifest_set_sha256"]):
        raise CheckpointSchemaError(
            "checkpoint.data.manifest_set_sha256: manifest binding hash mismatch"
        )
    _mapping(data["effective_source_counts"], "checkpoint.data.effective_source_counts")
    validation_row_ids: list[str] | None = None
    if minor >= 1:
        effective_counts = _mapping(
            data["effective_source_counts"], "checkpoint.data.effective_source_counts"
        )
        validation_row_ids = sorted(
            str(row_id) for row_id in effective_counts.get("validation_row_ids", [])
        )
        expected_validation_hash = canonical_sha256(
            {
                "manifest_set_sha256": str(data["manifest_set_sha256"]),
                "row_ids": validation_row_ids,
            }
        )
        if str(data["validation_manifest_sha256"]) != expected_validation_hash:
            raise CheckpointSchemaError(
                "checkpoint.data.validation_manifest_sha256: validation selection hash mismatch"
            )
    if "validation" in training_state:
        _validate_validation_state(
            _mapping(training_state["validation"], "checkpoint.training_state.validation"),
            checkpoint_epoch=int(training_state["epoch"]),
            manifest_sha256=str(
                data.get("validation_manifest_sha256", data["manifest_set_sha256"])
            ),
            expected_row_ids=validation_row_ids,
        )

    hashes = _mapping(checkpoint["hashes"], "checkpoint.hashes")
    expected_hashes = {
        "resolved_config_sha256": canonical_sha256(config),
        "feature_contract_sha256": canonical_sha256(feature),
    }
    _compare(expected_hashes, hashes, "checkpoint.hashes")
    _validate_rng_state(_mapping(checkpoint["rng_state"], "checkpoint.rng_state"))
    provenance = _mapping(checkpoint["provenance"], "checkpoint.provenance")
    for key in ("python_version", "pytorch_version", "ffmpeg_version", "source_git_sha"):
        if key not in provenance:
            raise CheckpointSchemaError(f"checkpoint.provenance.{key}: required field is missing")

    if expected_config is not None:
        _compare(
            _semantic_config(expected_config),
            _semantic_config(config),
            "checkpoint.resolved_config",
        )
    if expected_data is not None:
        for key in (
            "recipe_sha256",
            "manifest_set_sha256",
            "validation_manifest_sha256",
            "effective_source_counts",
        ):
            _compare(expected_data[key], data[key], f"checkpoint.data.{key}")
    return cast(SoundExCheckpoint, checkpoint)


def _validate_runtime_contract(feature: Mapping[str, Any]) -> None:
    fft_size = int(feature.get("fft_size", 0))
    hop_size = int(feature.get("hop_size", 0))
    if (fft_size, hop_size) not in SUPPORTED_FRAME_CONTRACTS:
        raise CheckpointSchemaError(
            "checkpoint.feature_contract FFT/hop pair is unsupported: "
            f"{fft_size}/{hop_size}; expected one of {sorted(SUPPORTED_FRAME_CONTRACTS)}"
        )
    tensor_shape = ["batch", 2, 1, fft_size // 2 + 1]
    expected = {
        "context_frames": 1,
        "supported_sample_rates": [44_100, 48_000],
        "input_shape": tensor_shape,
        "output_shape": tensor_shape,
        "causal": True,
        "stateless": True,
    }
    for key, expected_value in expected.items():
        if feature.get(key) != expected_value:
            raise CheckpointSchemaError(
                f"checkpoint.feature_contract.{key}: expected {expected_value!r}, "
                f"got {feature.get(key)!r}"
            )


def _validate_rng_state(rng: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "dataloader_sampler"}
    missing = sorted(required - set(rng))
    if missing:
        raise CheckpointSchemaError(f"checkpoint.rng_state.{missing[0]}: required field is missing")
    if not isinstance(rng["torch_cpu"], torch.Tensor):
        raise CheckpointSchemaError("checkpoint.rng_state.torch_cpu: expected tensor")
    sampler = _mapping(rng["dataloader_sampler"], "checkpoint.rng_state.dataloader_sampler")
    if sampler.get("source") not in {"torch_global", "sampler_generator"}:
        raise CheckpointSchemaError("checkpoint sampler RNG source is unsupported")
    if not isinstance(sampler.get("state"), torch.Tensor):
        raise CheckpointSchemaError("checkpoint sampler RNG state must be a tensor")


def _validate_validation_state(
    validation: Mapping[str, Any],
    *,
    checkpoint_epoch: int,
    manifest_sha256: str,
    expected_row_ids: list[str] | None,
) -> None:
    path = "checkpoint.training_state.validation"
    required = {"schema_version", "manifest_sha256", "last_epoch", "report", "best", "schedule"}
    missing = sorted(required - set(validation))
    if missing:
        raise CheckpointSchemaError(f"{path}.{missing[0]}: required field is missing")
    if int(validation["schema_version"]) != VALIDATION_SCHEMA_VERSION:
        raise CheckpointSchemaError(f"{path}.schema_version: unsupported version")
    _require_sha256(validation["manifest_sha256"], f"{path}.manifest_sha256")
    if str(validation["manifest_sha256"]) != manifest_sha256:
        raise CheckpointSchemaError(f"{path}.manifest_sha256: does not match checkpoint data")
    last_epoch = int(validation["last_epoch"])
    if last_epoch < 1 or last_epoch > checkpoint_epoch:
        raise CheckpointSchemaError(f"{path}.last_epoch: outside checkpoint history")

    report = _mapping(validation["report"], f"{path}.report")
    if int(report.get("schema_version", 0)) != VALIDATION_SCHEMA_VERSION:
        raise CheckpointSchemaError(f"{path}.report.schema_version: unsupported version")
    overall = _mapping(report.get("overall"), f"{path}.report.overall")
    if int(report.get("row_count", 0)) < 1:
        raise CheckpointSchemaError(f"{path}.report.row_count: must be positive")
    row_ids = report.get("row_ids")
    if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)):
        raise CheckpointSchemaError(f"{path}.report.row_ids: expected a sequence")
    if len(row_ids) != int(report["row_count"]) or len(set(map(str, row_ids))) != len(row_ids):
        raise CheckpointSchemaError(f"{path}.report.row_ids: count or uniqueness mismatch")
    if expected_row_ids is not None and sorted(map(str, row_ids)) != expected_row_ids:
        raise CheckpointSchemaError(f"{path}.report.row_ids: differs from validation selection")
    for metric_name in (PRIMARY_METRIC, TIE_BREAKER_METRIC):
        if metric_name not in overall:
            raise CheckpointSchemaError(f"{path}.report.overall.{metric_name}: required")
    if any(not math.isfinite(float(value)) for value in overall.values()):
        raise CheckpointSchemaError(f"{path}.report.overall: metrics must be finite")
    try:
        tracker = BestCheckpointTracker.from_state_dict(
            dict(_mapping(validation["best"], f"{path}.best"))
        )
    except (TypeError, ValueError) as error:
        raise CheckpointSchemaError(f"{path}.best: {error}") from error
    if tracker.best_epoch is None or tracker.best_epoch > checkpoint_epoch:
        raise CheckpointSchemaError(f"{path}.best.best_epoch: outside checkpoint history")
    schedule = _mapping(validation["schedule"], f"{path}.schedule")
    if int(schedule.get("interval_epochs", 0)) < 1:
        raise CheckpointSchemaError(f"{path}.schedule.interval_epochs: must be positive")


def _semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(config.get("model"), "config.model")
    training = _mapping(config.get("training"), "config.training")
    data = _mapping(config.get("data"), "config.data")
    return {
        "model": to_plain(model),
        "audio": to_plain(_mapping(config.get("audio"), "config.audio")),
        "training": to_plain(training),
        "data_recipe": to_plain(_mapping(data.get("recipe"), "config.data.recipe")),
        "data_sampling": to_plain(_mapping(data.get("sampling"), "config.data.sampling")),
    }


def _compare(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                raise CheckpointSchemaError(f"{path}.{key}: key mismatch")
            _compare(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise CheckpointSchemaError(f"{path}: expected a sequence")
        if len(expected) != len(actual):
            raise CheckpointSchemaError(f"{path}: sequence length mismatch")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]")
        return
    if expected != actual:
        raise CheckpointSchemaError(f"{path}: expected {expected!r}, got {actual!r}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointSchemaError(f"{path}: expected a mapping")
    return dict(value)


def _require_sha256(value: Any, path: str) -> None:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CheckpointSchemaError(f"{path}: expected lowercase SHA-256")
