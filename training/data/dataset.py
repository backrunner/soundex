"""PyTorch Dataset for SoundEx paired audio training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from data.protocol import MANIFEST_NAME, load_manifest, validate_manifest_rows


class SoundExDataset(Dataset):
    """Paired audio dataset whose membership comes only from a JSONL manifest."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        segment_length: int = 88200,  # 2s @ 44100
        sample_rates: Sequence[int] = (44100, 48000),
        augment: bool = True,
        source_name: str = "unknown",
        seed: int = 42,
        audit: bool = False,
    ) -> None:
        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            manifest_path = manifest_path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        all_rows = load_manifest(self.manifest_path)
        validate_manifest_rows(
            all_rows,
            self.root,
            audit_files=audit,
            file_splits=frozenset({split}),
        )
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No manifest rows for split={split} in {manifest_path}")
        self.split = split
        self.segment_length = segment_length
        self.sample_rates = frozenset(int(rate) for rate in sample_rates)
        self.augment = augment
        self.source_name = source_name
        self.seed = seed
        self.epoch = 0
        self.sampling_weights = [float(row.get("sampling_weight", 1.0)) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible training augmentation stream for one epoch."""
        if epoch < 0:
            raise ValueError("dataset epoch must be non-negative")
        self.epoch = int(epoch)

    def _generator(self, idx: int, stream: str) -> np.random.Generator:
        material = f"{self.seed}:{self.epoch}:{idx}:{stream}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        degraded_path = self.root / str(row["degraded_path"])
        clean_path = self.root / str(row["clean_path"])
        degraded, degraded_sr = sf.read(str(degraded_path), dtype="float32", always_2d=True)
        clean, clean_sr = sf.read(str(clean_path), dtype="float32")
        expected_rate = int(row["sample_rate"])
        if (
            expected_rate not in self.sample_rates
            or degraded_sr != expected_rate
            or clean_sr != expected_rate
        ):
            raise ValueError(
                f"Manifest expects {expected_rate} Hz, got degraded={degraded_sr}, "
                f"clean={clean_sr}; allowed={sorted(self.sample_rates)}"
            )
        if degraded.shape[1] != 1:
            raise ValueError(f"Manifest role file must be mono: {degraded_path}")
        degraded = degraded[:, 0]
        if clean.ndim == 2:
            if clean.shape[1] != 1:
                raise ValueError(f"Manifest role file must be mono: {clean_path}")
            clean = clean[:, 0]
        expected_length = int(row["length_samples"])
        if len(degraded) != expected_length or len(clean) != expected_length:
            raise ValueError(
                f"Manifest length mismatch for row {row['row_id']}: "
                f"expected={expected_length}, degraded={len(degraded)}, clean={len(clean)}"
            )
        start = 0
        if expected_length > self.segment_length:
            if self.augment:
                generator = self._generator(idx, "crop")
                start = int(generator.integers(0, expected_length - self.segment_length + 1))
            else:
                start = (expected_length - self.segment_length) // 2
            degraded = degraded[start : start + self.segment_length]
            clean = clean[start : start + self.segment_length]
        elif expected_length < self.segment_length:
            degraded = np.pad(degraded, (0, self.segment_length - expected_length))
            clean = np.pad(clean, (0, self.segment_length - expected_length))

        if self.augment:
            generator = self._generator(idx, "gain")
            gain_db = float(generator.uniform(-6.0, 3.0))
            gain = 10.0 ** (gain_db / 20.0)
            degraded = degraded * gain
            clean = clean * gain

        return {
            "degraded": torch.from_numpy(degraded).float(),
            "clean": torch.from_numpy(clean).float(),
            "metadata": {
                "row_id": str(row["row_id"]),
                "corpus": str(row["corpus"]),
                "track_id": str(row["track_id"]),
                "split": str(row["split"]),
                "source_name": self.source_name,
                "codec_id": str(row["codec_id"]),
                "codec": str(row["codec"]),
                "codec_mode": str(row["codec_mode"]),
                "codec_setting": float(row["codec_setting"]),
                "sample_rate": expected_rate,
                "channel_role": str(row["channel_role"]),
                "cutoff_hz": float(row["measured_cutoff_hz"]),
                "crop_start_sample": start,
                "held_out": bool(row["held_out"]),
            },
        }


def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagate an epoch into nested training datasets without touching validation."""
    if isinstance(dataset, SoundExDataset):
        dataset.set_epoch(epoch)
    elif isinstance(dataset, Subset):
        set_dataset_epoch(dataset.dataset, epoch)
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            set_dataset_epoch(child, epoch)


@dataclass(frozen=True)
class DatasetSourceSpec:
    """One named corpus backed by one immutable version manifest."""

    name: str
    label: str
    manifest_path: Path
    kind: str  # "real" | "synthetic" | "other"


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping, got {type(value)}")


def _positive_float(value: Any, default: float) -> float:
    if value is None:
        return default
    number = float(value)
    return number if number > 0.0 else 0.0


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or value is False:
        return None
    return int(value)


def subsample_dataset(
    dataset: Dataset,
    max_samples: int | None,
    *,
    seed: int,
    label: str,
) -> Dataset:
    """Deterministically keep at most ``max_samples`` items (None = keep all)."""
    total = len(dataset)
    if max_samples is None or max_samples >= total:
        print(f"  {label}: using all {total} segments")
        return dataset
    if max_samples <= 0:
        raise ValueError(f"{label}: max_samples must be positive, got {max_samples}")
    rng = np.random.default_rng(seed)
    if isinstance(dataset, SoundExDataset):
        groups: dict[tuple[str, int, str], list[int]] = {}
        for index, row in enumerate(dataset.rows):
            stratum = (
                str(row["codec_id"]),
                int(row["sample_rate"]),
                str(row["channel_role"]),
            )
            groups.setdefault(stratum, []).append(index)
        if max_samples < len(groups):
            raise ValueError(
                f"{label}: cap {max_samples} cannot preserve {len(groups)} codec/rate/role strata"
            )
        mandatory = [int(rng.choice(group_indices)) for group_indices in groups.values()]
        remaining = np.setdiff1d(np.arange(total), np.asarray(mandatory), assume_unique=False)
        extra_count = max_samples - len(mandatory)
        extras = (
            rng.choice(remaining, size=extra_count, replace=False).tolist() if extra_count else []
        )
        indices = np.asarray(mandatory + extras)
    else:
        indices = rng.choice(total, size=max_samples, replace=False)
    indices = np.sort(indices).tolist()
    print(f"  {label}: subsampled {max_samples} / {total} segments (seed={seed})")
    return Subset(dataset, indices)


def build_source_datasets(
    sources: Sequence[DatasetSourceSpec],
    *,
    segment_length: int,
    sample_rates: Sequence[int],
    split: str,
    augment: bool,
    seed: int,
) -> list[tuple[DatasetSourceSpec, SoundExDataset]]:
    """Load non-empty manifest datasets for one split."""
    built: list[tuple[DatasetSourceSpec, SoundExDataset]] = []
    for source in sources:
        try:
            dataset = SoundExDataset(
                source.manifest_path,
                split,
                segment_length=segment_length,
                sample_rates=sample_rates,
                augment=augment,
                source_name=source.name,
                seed=seed,
            )
        except ValueError as exc:
            print(f"  Skipping {source.label} ({split}): {exc}")
            continue
        built.append((source, dataset))
    if not built:
        raise FileNotFoundError(f"No non-empty datasets for split={split}")
    return built


def _normalize_ratios(
    sources: Sequence[DatasetSourceSpec],
    ratios_cfg: Mapping[str, Any],
) -> dict[str, float]:
    raw = {source.name: _positive_float(ratios_cfg.get(source.name), 0.0) for source in sources}
    # Fallback: equal weight if nothing configured for present sources
    if sum(raw.values()) <= 0.0:
        equal = 1.0 / len(sources)
        return {source.name: equal for source in sources}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items() if value > 0.0}


def _max_samples_map(cfg: Mapping[str, Any]) -> dict[str, int | None]:
    return {str(key): _optional_int(value) for key, value in cfg.items()}


def _sampling_weights(dataset: Dataset) -> list[float]:
    if isinstance(dataset, SoundExDataset):
        return dataset.sampling_weights
    if isinstance(dataset, Subset):
        parent = _sampling_weights(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    return [1.0] * len(dataset)


def create_balanced_dataloaders(
    sources: Sequence[DatasetSourceSpec],
    *,
    batch_size: int = 16,
    segment_length: int = 88200,
    sample_rates: Sequence[int] = (44100, 48000),
    num_workers: int = 4,
    seed: int = 42,
    sampling_config: Mapping[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build train/val loaders with per-source caps and mix ratios.

    Training uses ``WeightedRandomSampler`` so each epoch draws according to
    ``train_ratios`` (renormalized over sources that exist). Synthetic corpora
    can be capped via ``max_train_samples`` so they do not dominate by volume.

    Validation uses a fixed subsampled ConcatDataset (no replacement).
    """
    sampling = _as_mapping(sampling_config)
    strategy = str(sampling.get("strategy", "weighted")).lower()
    train_ratios_cfg = _as_mapping(sampling.get("train_ratios"))
    val_ratios_cfg = _as_mapping(sampling.get("val_ratios"))
    max_train_cfg = _max_samples_map(_as_mapping(sampling.get("max_train_samples")))
    max_val_cfg = _max_samples_map(_as_mapping(sampling.get("max_val_samples")))
    validation_quota_cfg = _max_samples_map(_as_mapping(sampling.get("validation_source_quotas")))
    enforce_validation_quotas = "validation_source_quotas" in sampling
    samples_per_epoch = _optional_int(sampling.get("samples_per_epoch"))

    print("Building training sources…")
    train_parts = build_source_datasets(
        sources,
        segment_length=segment_length,
        sample_rates=sample_rates,
        split="train",
        augment=True,
        seed=seed,
    )
    train_subsets: list[Dataset] = []
    train_meta: list[tuple[str, str, int, float]] = []  # name, kind, n, ratio

    # Only sources with positive ratio (after norm) participate in weighted mix
    present_specs = [spec for spec, _ in train_parts]
    ratios = _normalize_ratios(present_specs, train_ratios_cfg)

    for index, (spec, dataset) in enumerate(train_parts):
        if ratios.get(spec.name, 0.0) <= 0.0:
            print(f"  {spec.label}: ratio=0 — excluded from training mix")
            continue
        capped = subsample_dataset(
            dataset,
            max_train_cfg.get(spec.name),
            seed=seed + index + 17,
            label=f"{spec.label}/train",
        )
        train_subsets.append(capped)
        train_meta.append((spec.name, spec.kind, len(capped), ratios[spec.name]))

    if not train_subsets:
        raise RuntimeError("No training subsets after applying ratios/caps")

    train_concat = train_subsets[0] if len(train_subsets) == 1 else ConcatDataset(train_subsets)

    # Re-normalize ratios for sources that remain after ratio=0 filtering
    ratio_by_name = {name: ratio for name, _kind, _n, ratio in train_meta}
    ratio_sum = sum(ratio_by_name.values())
    ratio_by_name = {k: v / ratio_sum for k, v in ratio_by_name.items()}

    counts = [n for _name, _kind, n, _r in train_meta]
    names = [name for name, _kind, _n, _r in train_meta]
    if strategy == "concat_subsample":
        # Hard rebalance: target counts proportional to ratios, limited by available n
        budget = samples_per_epoch or sum(counts)
        target = []
        for name, n in zip(names, counts, strict=True):
            want = round(budget * ratio_by_name[name])
            target.append(max(1, min(n, want)) if n else 0)
        # Adjust rounding to not exceed budget too much
        rebalanced: list[Dataset] = []
        for offset, (part, want, name) in enumerate(zip(train_subsets, target, names, strict=True)):
            rebalanced.append(
                subsample_dataset(
                    part,
                    want,
                    seed=seed + 100 + offset,
                    label=f"{name}/train-rebalance",
                )
            )
        train_dataset: Dataset = (
            rebalanced[0] if len(rebalanced) == 1 else ConcatDataset(rebalanced)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        effective = {name: len(part) for name, part in zip(names, rebalanced, strict=True)}
    else:
        # weighted (default): each source gets total probability = its ratio
        weights: list[float] = []
        for name, dataset in zip(names, train_subsets, strict=True):
            within_source = _sampling_weights(dataset)
            source_weight = sum(within_source)
            if source_weight <= 0.0:
                raise ValueError(f"{name}: manifest sampling weights must have positive mass")
            weights.extend(
                ratio_by_name[name] * item_weight / source_weight for item_weight in within_source
            )
        weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        epoch_size = samples_per_epoch or int(sum(counts))
        epoch_size = max(epoch_size, batch_size)
        # Make epoch_size divisible-friendly for drop_last
        epoch_size = max(batch_size, (epoch_size // batch_size) * batch_size)
        sampler = WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=epoch_size,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_concat,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        effective = {name: round(epoch_size * ratio_by_name[name]) for name in names}
        train_dataset = train_concat

    print("Train mix (target ratios → expected segments/epoch):")
    real_n = 0
    synth_n = 0
    for name, kind, n_avail, _ in train_meta:
        ratio = ratio_by_name[name]
        exp = effective.get(name, 0)
        print(f"  {name:12} kind={kind:9} available={n_avail:6d}  ratio={ratio:.2%}  ~epoch={exp}")
        if kind == "real":
            real_n += exp
        elif kind == "synthetic":
            synth_n += exp
    denom = max(real_n + synth_n, 1)
    print(
        f"  ⇒ expected real:synthetic ≈ {real_n}:{synth_n} "
        f"({real_n / denom:.1%} real / {synth_n / denom:.1%} synthetic)"
    )

    print("Building validation sources…")
    val_parts = build_source_datasets(
        sources,
        segment_length=segment_length,
        sample_rates=sample_rates,
        split="validation",
        augment=False,
        seed=seed,
    )
    val_specs = [spec for spec, _ in val_parts]
    val_ratios = _normalize_ratios(val_specs, val_ratios_cfg or train_ratios_cfg)
    val_subsets: list[Dataset] = []
    val_source_counts: dict[str, int] = {}
    validation_row_ids: list[str] = []
    for index, (spec, dataset) in enumerate(val_parts):
        if val_ratios.get(spec.name, 0.0) <= 0.0:
            print(f"  {spec.label}: val ratio=0 — excluded")
            continue
        if enforce_validation_quotas:
            if spec.name not in validation_quota_cfg:
                raise ValueError(
                    f"{spec.label}/validation: missing configured validation source quota"
                )
            max_val = validation_quota_cfg[spec.name]
            if max_val is None or max_val <= 0:
                raise ValueError(f"{spec.label}/validation: quota must be a positive integer")
            if len(dataset) < max_val:
                raise ValueError(
                    f"{spec.label}/validation: quota {max_val} exceeds {len(dataset)} rows"
                )
        else:
            # Legacy profiles treat max_val_samples as a cap rather than a quota.
            max_val = max_val_cfg.get(spec.name)
            if max_val is None and samples_per_epoch:
                max_val = max(32, round(0.1 * samples_per_epoch * val_ratios[spec.name]))
        capped = subsample_dataset(
            dataset,
            max_val,
            seed=seed + 1000 + index,
            label=f"{spec.label}/val",
        )
        val_subsets.append(capped)
        val_source_counts[spec.name] = len(capped)
        if isinstance(capped, Subset):
            validation_row_ids.extend(
                str(dataset.rows[int(row_index)]["row_id"]) for row_index in capped.indices
            )
        else:
            validation_row_ids.extend(str(row["row_id"]) for row in dataset.rows)

    if not val_subsets:
        raise RuntimeError("No validation subsets after applying ratios/caps")

    val_dataset: Dataset = val_subsets[0] if len(val_subsets) == 1 else ConcatDataset(val_subsets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    summary = {
        "strategy": strategy,
        "train_ratios": ratio_by_name,
        "train_available": {name: n for name, _k, n, _r in train_meta},
        "train_epoch_expected": effective,
        "val_size": len(val_dataset),
        "validation_source_counts": dict(sorted(val_source_counts.items())),
        "validation_row_ids": sorted(validation_row_ids),
        "samples_per_epoch": getattr(train_loader.sampler, "num_samples", len(train_dataset)),
    }
    return train_loader, val_loader, summary


def create_dataloaders(
    train_manifest: str | Path | Sequence[str | Path],
    val_manifest: str | Path | Sequence[str | Path],
    batch_size: int = 16,
    segment_length: int = 88200,
    sample_rates: Sequence[int] = (44100, 48000),
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build simple train/validation loaders from one or more manifests."""

    def build_dataset(
        manifests: str | Path | Sequence[str | Path], split: str, augment: bool
    ) -> Dataset:
        paths = [manifests] if isinstance(manifests, (str, Path)) else manifests
        datasets = [
            SoundExDataset(
                path,
                split,
                segment_length=segment_length,
                sample_rates=sample_rates,
                augment=augment,
            )
            for path in paths
        ]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    train_dataset = build_dataset(train_manifest, "train", augment=True)
    val_dataset = build_dataset(val_manifest, "validation", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
