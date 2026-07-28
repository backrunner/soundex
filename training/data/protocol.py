"""Versioned, split-safe dataset recipe and manifest protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

DATA_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.jsonl"
RECIPE_NAME = "recipe.json"
SUMMARY_NAME = "summary.json"
SPLITS = ("train", "validation", "test")
CORPORA = ("musdb18_hq", "slakh2100", "medleydb")


class DataProtocolError(ValueError):
    """Invalid recipe, manifest, or dataset publication state."""


@dataclass(frozen=True)
class CodecSpec:
    """One bounded codec degradation stratum."""

    id: str
    codec: str
    encoder: str
    container: str
    mode: str
    setting: float
    weight: float
    held_out: bool


@dataclass(frozen=True)
class SegmentPolicy:
    """Deterministic segment generation policy."""

    duration_seconds: tuple[float, float]
    count_per_track: int


@dataclass(frozen=True)
class AlignmentPolicy:
    """Codec-delay estimation and residual-alignment limits."""

    max_offset_samples: int
    correlation_samples: int
    residual_tolerance_samples: int


@dataclass(frozen=True)
class DataRecipe:
    """Complete preprocessing recipe embedded in every dataset version."""

    schema_version: int
    recipe_version: str
    seed: int
    sample_rates: tuple[int, ...]
    segment: SegmentPolicy
    codecs: tuple[CodecSpec, ...]
    channel_roles: tuple[tuple[str, float], ...]
    splits: tuple[tuple[str, tuple[tuple[str, float | bool], ...]], ...]
    alignment: AlignmentPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_version": self.recipe_version,
            "seed": self.seed,
            "sample_rates": list(self.sample_rates),
            "segment": asdict(self.segment),
            "codecs": [asdict(codec) for codec in self.codecs],
            "channel_policy": {"roles": dict(self.channel_roles)},
            "splits": {corpus: dict(policy) for corpus, policy in self.splits},
            "alignment": asdict(self.alignment),
        }

    @property
    def hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def split_policy(self, corpus: str) -> dict[str, float | bool]:
        policies = dict(self.splits)
        if corpus not in policies:
            raise DataProtocolError(f"recipe.splits.{corpus}: missing split policy")
        return dict(policies[corpus])


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataProtocolError(f"{path}: expected a mapping")
    return dict(value)


def _strict_keys(
    value: Any,
    path: str,
    allowed: set[str],
    *,
    required: set[str] | None = None,
) -> dict[str, Any]:
    mapping = _mapping(value, path)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise DataProtocolError(f"{path}.{unknown[0]}: unknown configuration key")
    missing = sorted((required or set()) - set(mapping))
    if missing:
        raise DataProtocolError(f"{path}.{missing[0]}: required configuration key is missing")
    return mapping


def validate_data_config(data_config: Mapping[str, Any]) -> DataRecipe:
    """Validate every ``data`` key and return its authoritative recipe."""
    allowed = {
        "musdb18_hq_path",
        "slakh2100_path",
        "medleydb_path",
        "babyslakh_path",
        "sampling",
        "recipe",
    }
    data = _strict_keys(data_config, "data", allowed, required={"recipe", "sampling"})
    _validate_sampling(data["sampling"])
    return recipe_from_mapping(data["recipe"], path="data.recipe")


def load_recipe_from_profile(path: str | Path) -> DataRecipe:
    """Load a training YAML profile and validate its complete ``data`` section."""
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    if not isinstance(profile, Mapping):
        raise DataProtocolError(f"{profile_path}: expected a YAML mapping")
    return validate_data_config(_mapping(profile.get("data"), "data"))


def _validate_sampling(value: Any) -> None:
    allowed = {
        "strategy",
        "train_ratios",
        "val_ratios",
        "max_train_samples",
        "max_val_samples",
        "validation_source_quotas",
        "samples_per_epoch",
    }
    sampling = _strict_keys(value, "data.sampling", allowed)
    strategy = str(sampling.get("strategy", "weighted"))
    if strategy not in {"weighted", "concat_subsample"}:
        raise DataProtocolError(f"data.sampling.strategy: unsupported value {strategy!r}")
    source_keys = set(CORPORA) | {"babyslakh"}
    for key in (
        "train_ratios",
        "val_ratios",
        "max_train_samples",
        "max_val_samples",
        "validation_source_quotas",
    ):
        if key not in sampling:
            continue
        mapping = _strict_keys(sampling[key], f"data.sampling.{key}", source_keys)
        for source, raw in mapping.items():
            if raw is None and key.startswith("max_"):
                continue
            number = float(raw)
            if number < 0:
                raise DataProtocolError(f"data.sampling.{key}.{source}: must be non-negative")
            if key == "validation_source_quotas" and not number.is_integer():
                raise DataProtocolError(
                    f"data.sampling.{key}.{source}: must be an integer row count"
                )
    samples_per_epoch = sampling.get("samples_per_epoch")
    if samples_per_epoch is not None and int(samples_per_epoch) <= 0:
        raise DataProtocolError("data.sampling.samples_per_epoch: must be positive or null")


def recipe_from_mapping(value: Any, *, path: str = "recipe") -> DataRecipe:
    """Parse and strictly validate a recipe mapping."""
    required = {
        "schema_version",
        "recipe_version",
        "seed",
        "sample_rates",
        "segment",
        "codecs",
        "channel_policy",
        "splits",
        "alignment",
    }
    recipe = _strict_keys(value, path, required, required=required)
    schema_version = int(recipe["schema_version"])
    if schema_version != DATA_SCHEMA_VERSION:
        raise DataProtocolError(
            f"{path}.schema_version: expected {DATA_SCHEMA_VERSION}, got {schema_version}"
        )
    recipe_version = str(recipe["recipe_version"]).strip()
    if not recipe_version:
        raise DataProtocolError(f"{path}.recipe_version: must not be empty")
    sample_rates = tuple(int(rate) for rate in recipe["sample_rates"])
    if sorted(set(sample_rates)) != [44_100, 48_000]:
        raise DataProtocolError(f"{path}.sample_rates: must contain exactly 44100 and 48000")

    segment_raw = _strict_keys(
        recipe["segment"],
        f"{path}.segment",
        {"duration_seconds", "count_per_track"},
        required={"duration_seconds", "count_per_track"},
    )
    durations = tuple(float(item) for item in segment_raw["duration_seconds"])
    if len(durations) != 2 or not 0.0 < durations[0] <= durations[1]:
        raise DataProtocolError(f"{path}.segment.duration_seconds: expected [positive_min, max]")
    segment = SegmentPolicy(durations, int(segment_raw["count_per_track"]))
    if segment.count_per_track <= 0:
        raise DataProtocolError(f"{path}.segment.count_per_track: must be positive")

    codecs_raw = recipe["codecs"]
    if not isinstance(codecs_raw, Sequence) or isinstance(codecs_raw, (str, bytes)):
        raise DataProtocolError(f"{path}.codecs: expected a list")
    codecs: list[CodecSpec] = []
    codec_ids: set[str] = set()
    codec_keys = {"id", "codec", "encoder", "container", "mode", "setting", "weight", "held_out"}
    for index, raw in enumerate(codecs_raw):
        codec_path = f"{path}.codecs[{index}]"
        item = _strict_keys(raw, codec_path, codec_keys, required=codec_keys)
        spec = CodecSpec(
            id=str(item["id"]),
            codec=str(item["codec"]),
            encoder=str(item["encoder"]),
            container=str(item["container"]),
            mode=str(item["mode"]),
            setting=float(item["setting"]),
            weight=float(item["weight"]),
            held_out=bool(item["held_out"]),
        )
        if spec.id in codec_ids:
            raise DataProtocolError(f"{codec_path}.id: duplicate codec id {spec.id!r}")
        codec_ids.add(spec.id)
        if spec.codec not in {"mp3", "aac", "vorbis"}:
            raise DataProtocolError(f"{codec_path}.codec: unsupported codec {spec.codec!r}")
        if spec.mode not in {"cbr", "vbr"}:
            raise DataProtocolError(f"{codec_path}.mode: expected 'cbr' or 'vbr'")
        if spec.setting <= 0 or spec.weight <= 0:
            raise DataProtocolError(f"{codec_path}: setting and weight must be positive")
        codecs.append(spec)
    if not codecs or not any(spec.held_out for spec in codecs):
        raise DataProtocolError(f"{path}.codecs: at least one held-out codec setting is required")
    represented = {(spec.codec, spec.mode) for spec in codecs}
    required_strata = {("mp3", "cbr"), ("mp3", "vbr"), ("aac", "cbr"), ("vorbis", "vbr")}
    if not required_strata <= represented:
        raise DataProtocolError(f"{path}.codecs: missing deployment codec strata")

    channel_raw = _strict_keys(
        recipe["channel_policy"],
        f"{path}.channel_policy",
        {"roles"},
        required={"roles"},
    )
    roles = _strict_keys(
        channel_raw["roles"],
        f"{path}.channel_policy.roles",
        {"mono", "left", "right", "mid", "side"},
        required={"mono", "left", "right", "mid", "side"},
    )
    channel_roles = tuple(sorted((name, float(weight)) for name, weight in roles.items()))
    if any(weight <= 0 for _, weight in channel_roles):
        raise DataProtocolError(f"{path}.channel_policy.roles: weights must be positive")

    split_raw = _strict_keys(
        recipe["splits"], f"{path}.splits", set(CORPORA), required=set(CORPORA)
    )
    splits: list[tuple[str, tuple[tuple[str, float | bool], ...]]] = []
    split_keys = {"train", "validation", "test", "preserve_official", "preserve_official_test"}
    for corpus in CORPORA:
        policy_path = f"{path}.splits.{corpus}"
        policy = _strict_keys(
            split_raw[corpus],
            policy_path,
            split_keys,
            required={"train", "validation", "test"},
        )
        ratios = [float(policy[name]) for name in SPLITS]
        if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
            raise DataProtocolError(f"{policy_path}: split ratios must be non-negative and nonzero")
        normalized: dict[str, float | bool] = {
            name: ratios[index] for index, name in enumerate(SPLITS)
        }
        normalized["preserve_official"] = bool(policy.get("preserve_official", False))
        normalized["preserve_official_test"] = bool(policy.get("preserve_official_test", False))
        splits.append((corpus, tuple(sorted(normalized.items()))))

    alignment_raw = _strict_keys(
        recipe["alignment"],
        f"{path}.alignment",
        {"max_offset_samples", "correlation_samples", "residual_tolerance_samples"},
        required={"max_offset_samples", "correlation_samples", "residual_tolerance_samples"},
    )
    alignment = AlignmentPolicy(
        max_offset_samples=int(alignment_raw["max_offset_samples"]),
        correlation_samples=int(alignment_raw["correlation_samples"]),
        residual_tolerance_samples=int(alignment_raw["residual_tolerance_samples"]),
    )
    if (
        alignment.max_offset_samples < 0
        or alignment.correlation_samples <= 0
        or alignment.residual_tolerance_samples < 0
    ):
        raise DataProtocolError(f"{path}.alignment: values must be non-negative")

    return DataRecipe(
        schema_version=schema_version,
        recipe_version=recipe_version,
        seed=int(recipe["seed"]),
        sample_rates=sample_rates,
        segment=segment,
        codecs=tuple(codecs),
        channel_roles=channel_roles,
        splits=tuple(splits),
        alignment=alignment,
    )


def canonical_track_id(corpus: str, identity: str | Path) -> str:
    """Return a path/codec-independent canonical track identity."""
    if corpus not in CORPORA:
        raise DataProtocolError(f"unsupported corpus {corpus!r}")
    name = Path(str(identity)).stem.casefold()
    name = re.sub(r"(?:_mix|[-_ ]v(?:ersion)?\d+)$", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    if not name:
        raise DataProtocolError(f"cannot derive canonical track id from {identity!r}")
    return f"{corpus}:{name}"


def assign_hashed_split(
    track_id: str,
    seed: int,
    ratios: Mapping[str, float | bool],
    *,
    allowed_splits: Sequence[str] = SPLITS,
) -> str:
    """Assign one canonical track deterministically from normalized ratios."""
    weights = [float(ratios.get(split, 0.0)) for split in allowed_splits]
    total = sum(weights)
    if total <= 0:
        raise DataProtocolError("split ratios must have positive mass")
    digest = hashlib.sha256(f"{seed}:{track_id}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for split, weight in zip(allowed_splits, weights, strict=True):
        cumulative += weight / total
        if point < cumulative:
            return split
    return allowed_splits[-1]


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows, rejecting malformed or duplicate row IDs."""
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataProtocolError(
                    f"{manifest_path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise DataProtocolError(f"{manifest_path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


_REQUIRED_ROW_KEYS = {
    "row_id",
    "schema_version",
    "recipe_hash",
    "corpus",
    "corpus_version",
    "track_id",
    "split",
    "source_id",
    "source_path",
    "source_checksum",
    "sample_rate",
    "source_channels",
    "channel_role",
    "codec_id",
    "codec",
    "encoder",
    "codec_mode",
    "codec_setting",
    "codec_weight",
    "held_out",
    "clean_path",
    "degraded_path",
    "start_sample",
    "length_samples",
    "alignment_offset_samples",
    "residual_alignment_samples",
    "measured_cutoff_hz",
    "clean_checksum",
    "degraded_checksum",
}


def validate_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    recipe: DataRecipe | None = None,
    audit_files: bool = True,
    file_splits: frozenset[str] | None = None,
) -> None:
    """Validate schema and only touch audio files in ``file_splits`` when set."""
    root = root.resolve()
    if not rows:
        raise DataProtocolError("manifest must contain at least one row")
    if file_splits is not None:
        invalid_file_splits = sorted(set(file_splits) - set(SPLITS))
        if invalid_file_splits:
            raise DataProtocolError(
                f"file_splits contains invalid split {invalid_file_splits[0]!r}"
            )
    row_ids: set[str] = set()
    tracks_by_split: dict[str, set[str]] = defaultdict(set)
    recipe_codecs = {spec.id: spec for spec in recipe.codecs} if recipe is not None else {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        missing = sorted(_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise DataProtocolError(f"manifest[{index}].{missing[0]}: required field is missing")
        row_id = str(row["row_id"])
        if row_id in row_ids:
            raise DataProtocolError(f"manifest[{index}].row_id: duplicate {row_id!r}")
        row_ids.add(row_id)
        split = str(row["split"])
        if split not in SPLITS:
            raise DataProtocolError(f"manifest[{index}].split: invalid value {split!r}")
        track_id = str(row["track_id"])
        tracks_by_split[split].add(track_id)
        if int(row["schema_version"]) != DATA_SCHEMA_VERSION:
            raise DataProtocolError(f"manifest[{index}].schema_version: unsupported version")
        if recipe is not None and str(row["recipe_hash"]) != recipe.hash:
            raise DataProtocolError(f"manifest[{index}].recipe_hash: does not match recipe")
        corpus = str(row["corpus"])
        if corpus not in CORPORA or not track_id.startswith(f"{corpus}:"):
            raise DataProtocolError(f"manifest[{index}]: corpus/track_id mismatch")
        sample_rate = int(row["sample_rate"])
        allowed_rates = set(recipe.sample_rates) if recipe is not None else {44_100, 48_000}
        if sample_rate not in allowed_rates:
            raise DataProtocolError(f"manifest[{index}].sample_rate: unsupported rate")
        if int(row["length_samples"]) <= 0 or int(row["start_sample"]) < 0:
            raise DataProtocolError(f"manifest[{index}]: invalid sample interval")
        if split == "train" and bool(row["held_out"]):
            raise DataProtocolError(f"manifest[{index}]: held-out codec appears in training")
        if str(row["channel_role"]) not in {"mono", "left", "right", "mid", "side"}:
            raise DataProtocolError(f"manifest[{index}].channel_role: unsupported role")
        residual = int(row["residual_alignment_samples"])
        if recipe is not None and abs(residual) > recipe.alignment.residual_tolerance_samples:
            raise DataProtocolError(f"manifest[{index}]: residual alignment exceeds recipe")
        cutoff = float(row["measured_cutoff_hz"])
        if not math.isfinite(cutoff) or not 0.0 <= cutoff <= sample_rate / 2.0:
            raise DataProtocolError(f"manifest[{index}].measured_cutoff_hz: invalid cutoff")
        if recipe is not None:
            codec_id = str(row["codec_id"])
            if codec_id not in recipe_codecs:
                raise DataProtocolError(f"manifest[{index}].codec_id: absent from recipe")
            codec = recipe_codecs[codec_id]
            actual_codec = (
                str(row["codec"]),
                str(row["encoder"]),
                str(row["codec_mode"]),
                float(row["codec_setting"]),
                float(row["codec_weight"]),
                bool(row["held_out"]),
            )
            expected_codec = (
                codec.codec,
                codec.encoder,
                codec.mode,
                codec.setting,
                codec.weight,
                codec.held_out,
            )
            if actual_codec != expected_codec:
                raise DataProtocolError(
                    f"manifest[{index}]: codec metadata differs from recipe {codec_id}"
                )

        if file_splits is not None and split not in file_splits:
            continue

        paths: dict[str, Path] = {}
        for key in ("clean_path", "degraded_path"):
            relative = Path(str(row[key]))
            if relative.is_absolute() or ".." in relative.parts:
                raise DataProtocolError(f"manifest[{index}].{key}: must be a safe relative path")
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise DataProtocolError(f"manifest[{index}].{key}: escapes dataset root")
            if not resolved.is_file():
                raise FileNotFoundError(f"manifest[{index}].{key}: missing {resolved}")
            paths[key] = resolved

        if audit_files:
            clean_info = sf.info(paths["clean_path"])
            degraded_info = sf.info(paths["degraded_path"])
            expected_rate = int(row["sample_rate"])
            expected_length = int(row["length_samples"])
            if clean_info.samplerate != expected_rate or degraded_info.samplerate != expected_rate:
                raise DataProtocolError(f"manifest[{index}]: audio sample rate mismatch")
            if clean_info.frames != expected_length or degraded_info.frames != expected_length:
                raise DataProtocolError(f"manifest[{index}]: audio length mismatch")
            for key, checksum_key in (
                ("clean_path", "clean_checksum"),
                ("degraded_path", "degraded_checksum"),
            ):
                if sha256_file(paths[key]) != str(row[checksum_key]):
                    raise DataProtocolError(f"manifest[{index}].{checksum_key}: checksum mismatch")

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = tracks_by_split[left] & tracks_by_split[right]
            if overlap:
                example = sorted(overlap)[0]
                raise DataProtocolError(
                    f"track split leakage between {left} and {right}: {example}"
                )


def manifest_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Create deterministic split and degradation counts for release records."""
    rows = list(rows)
    return {
        "rows": len(rows),
        "tracks_by_split": {
            split: len({str(row["track_id"]) for row in rows if row["split"] == split})
            for split in SPLITS
        },
        "rows_by_split": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "rows_by_codec": dict(sorted(Counter(str(row["codec_id"]) for row in rows).items())),
        "rows_by_sample_rate": dict(
            sorted(Counter(str(row["sample_rate"]) for row in rows).items())
        ),
        "rows_by_channel_role": dict(
            sorted(Counter(str(row["channel_role"]) for row in rows).items())
        ),
    }


def assert_deployment_coverage(rows: Sequence[Mapping[str, Any]], recipe: DataRecipe) -> None:
    """Require every validation/test codec and sample-rate stratum before release."""
    expected_codecs = {spec.id for spec in recipe.codecs}
    expected_rates = set(recipe.sample_rates)
    stereo_roles = {"left", "right", "mid", "side"}
    for split in ("validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise DataProtocolError(f"manifest coverage: split {split!r} has no rows")
        codecs = {str(row["codec_id"]) for row in split_rows}
        rates = {int(row["sample_rate"]) for row in split_rows}
        roles = {str(row["channel_role"]) for row in split_rows}
        if codecs != expected_codecs:
            missing = sorted(expected_codecs - codecs)
            raise DataProtocolError(f"manifest coverage: {split} missing codec strata {missing}")
        if rates != expected_rates:
            missing = sorted(expected_rates - rates)
            raise DataProtocolError(f"manifest coverage: {split} missing sample rates {missing}")
        if not stereo_roles <= roles:
            missing = sorted(stereo_roles - roles)
            raise DataProtocolError(
                f"manifest coverage: {split} missing stereo channel roles {missing}"
            )


class DatasetPublisher:
    """Build a dataset in a fresh staging tree and atomically publish it."""

    def __init__(
        self,
        output_root: str | Path,
        corpus: str,
        recipe: DataRecipe,
        *,
        reuse_existing: bool = False,
    ) -> None:
        if corpus not in CORPORA:
            raise DataProtocolError(f"unsupported corpus {corpus!r}")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.corpus = corpus
        self.recipe = recipe
        self.target = self.output_root / f"{corpus}-{recipe.hash[:16]}"
        self.reused = False
        self._published = False
        self._staging: Path | None = None
        if self.target.exists():
            if not reuse_existing:
                raise FileExistsError(
                    f"dataset version already exists: {self.target}; pass --reuse-existing to verify and skip"
                )
            recipe_path = self.target / RECIPE_NAME
            if not recipe_path.is_file():
                raise DataProtocolError(f"existing dataset has no {RECIPE_NAME}: {self.target}")
            existing = json.loads(recipe_path.read_text(encoding="utf-8"))
            if existing.get("recipe_hash") != recipe.hash:
                raise DataProtocolError(f"existing dataset recipe hash mismatch: {self.target}")
            self.reused = True
            return
        self._staging = Path(
            tempfile.mkdtemp(prefix=f".staging-{corpus}-{recipe.hash[:8]}-", dir=self.output_root)
        )

    @property
    def staging(self) -> Path:
        if self._staging is None:
            raise DataProtocolError("no staging directory for a reused dataset")
        return self._staging

    def publish(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Validate and atomically rename the staging tree into place."""
        if self.reused:
            return self.target
        if self._published:
            raise DataProtocolError("dataset publisher is single-use")
        if any(str(row.get("corpus")) != self.corpus for row in rows):
            raise DataProtocolError(
                f"manifest contains a row outside publisher corpus {self.corpus!r}"
            )
        recipe_payload = self.recipe.to_dict()
        recipe_payload["recipe_hash"] = self.recipe.hash
        (self.staging / RECIPE_NAME).write_text(
            json.dumps(recipe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (self.staging / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        validate_manifest_rows(rows, self.staging, recipe=self.recipe, audit_files=True)
        summary = manifest_summary(rows)
        summary["recipe_hash"] = self.recipe.hash
        (self.staging / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(self.staging, self.target)
        self._published = True
        self._staging = None
        return self.target

    def abort(self) -> None:
        """Remove only this publisher's private staging directory."""
        if self._staging is None:
            return
        if self._staging.parent != self.output_root or not self._staging.name.startswith(
            ".staging-"
        ):
            raise DataProtocolError(f"refusing to remove unexpected staging path {self._staging}")
        shutil.rmtree(self._staging)
        self._staging = None

    def __enter__(self) -> DatasetPublisher:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._published:
            self.abort()
