"""Train SoundEx on paired degraded and clean audio."""

import argparse
import hashlib
import io
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from checkpoint import (
    build_checkpoint,
    feature_contract_from_config,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from checkpoint_state import build_dataset_provenance, capture_provenance, restore_rng_state
from configuration import load_config
from data.dataset import DatasetSourceSpec, create_balanced_dataloaders, set_dataset_epoch
from data.protocol import MANIFEST_NAME, DataRecipe, validate_data_config
from models.discriminator import MultiScaleDiscriminator
from models.generator import SoundExGenerator
from models.losses import DiscriminatorLoss, GeneratorLoss, build_missing_band_mask
from validation import (
    BestCheckpointTracker,
    aggregate_validation_rows,
    build_validation_state,
    quality_label,
    should_run_validation,
)


def get_device() -> torch.device:
    """Select CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Config keys -> human-readable names for processed mixture datasets.
_DATASET_PATH_KEYS = (
    ("slakh2100_path", "Slakh2100"),
    ("music_library_path", "Music library"),
    ("speech_library_path", "Speech library"),
)

# Environment overrides for data.* paths (Docker / operators).
_PATH_ENV_OVERRIDES = (
    ("MUSDB18_HQ_PATH", "musdb18_hq_path"),
    ("SLAKH2100_PATH", "slakh2100_path"),
    ("MEDLEYDB_PATH", "medleydb_path"),
    ("BABYSLAKH_PATH", "babyslakh_path"),
    ("MUSIC_LIBRARY_PATH", "music_library_path"),
    ("SPEECH_LIBRARY_PATH", "speech_library_path"),
)


def apply_data_path_overrides(
    data_config: dict[str, Any],
    *,
    paths_manifest: Path | None = None,
) -> dict[str, Any]:
    """Return a copy of data_config with env / manifest path overrides applied.

    Priority (highest first): environment variables → paths manifest → YAML.
    """
    merged = dict(data_config)

    if paths_manifest is not None and paths_manifest.is_file():
        with paths_manifest.open(encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        training_data = manifest.get("training_data") or {}
        if isinstance(training_data, dict):
            for key, value in training_data.items():
                if value:
                    merged[key] = str(value)
                    print(f"Path from manifest: {key}={value}")
        # Also accept datasets: musdb18-hq: /path form
        datasets = manifest.get("datasets") or {}
        name_to_key = {
            "musdb18-hq": "musdb18_hq_path",
            "slakh2100": "slakh2100_path",
            "medleydb": "medleydb_path",
            "babyslakh": "babyslakh_path",
            "music-library": "music_library_path",
            "speech-library": "speech_library_path",
        }
        if isinstance(datasets, dict):
            for name, value in datasets.items():
                key = name_to_key.get(str(name))
                if key and value:
                    merged[key] = str(value)
                    print(f"Path from manifest datasets: {key}={value}")

    for env_key, config_key in _PATH_ENV_OVERRIDES:
        value = os.environ.get(env_key)
        if value:
            merged[config_key] = value
            print(f"Path from env: {config_key}={value}")

    return merged


# Flat config path key → (short name, display label, kind)
_SOURCE_META = {
    "musdb18_hq_path": ("musdb18_hq", "MUSDB18-HQ", "real"),
    "slakh2100_path": ("slakh2100", "Slakh2100", "synthetic"),
    "medleydb_path": ("medleydb", "MedleyDB", "real"),
    "babyslakh_path": ("babyslakh", "BabySlakh", "synthetic"),
    "music_library_path": ("music_library", "Music library", "real"),
    "speech_library_path": ("speech_library", "Speech library", "real"),
}


def resolve_dataset_sources(
    data_config: dict[str, Any], recipe: DataRecipe
) -> list[DatasetSourceSpec]:
    """Resolve only the immutable dataset version matching this recipe hash."""
    sources: list[DatasetSourceSpec] = []
    for key, _label in _DATASET_PATH_KEYS:
        raw = data_config.get(key)
        if not raw:
            continue
        name, display, kind = _SOURCE_META[key]
        root = Path(raw).expanduser()
        version_name = f"{name}-{recipe.hash[:16]}"
        candidates = (
            root / MANIFEST_NAME,
            root / "processed" / version_name / MANIFEST_NAME,
            root / version_name / MANIFEST_NAME,
        )
        manifest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if manifest_path is not None:
            sources.append(
                DatasetSourceSpec(
                    name=name,
                    label=display,
                    manifest_path=manifest_path,
                    kind=kind,
                )
            )
            print(f"Using {display} ({kind}): {manifest_path.parent}")
        elif root.is_dir():
            print(
                f"Skipping {display}: recipe version {version_name} was not found. "
                "Run the matching preprocessor with this config."
            )
        else:
            print(f"Skipping {display}: path not found ({root})")

    if not sources:
        raise FileNotFoundError(
            "No processed training data found. Preprocess at least one of "
            "Slakh2100 mixes or audio libraries (no stems required)."
        )
    return sources


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spectral_features(
    audio: torch.Tensor,
    fft_size: int,
    hop_size: int,
) -> torch.Tensor:
    """Convert `[B, samples]` waveforms to `[B, 2, T, F]` dB/phase features."""
    window = torch.hann_window(fft_size, periodic=True, device=audio.device)
    spectrum = torch.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitude_db = 20.0 * spectrum.abs().clamp_min(1e-10).log10()
    phase = torch.angle(spectrum)
    return torch.stack((magnitude_db.transpose(1, 2), phase.transpose(1, 2)), dim=1)


def select_context(
    features: torch.Tensor,
    context_frames: int,
    *,
    randomize: bool = True,
) -> torch.Tensor:
    """Select one deployment-equivalent spectral frame."""
    if context_frames != 1:
        raise ValueError("SoundEx uses stateless single-frame inference; context_frames must be 1")
    available = features.shape[2]
    if available < 1:
        raise ValueError("Spectral input contains no time frames")
    start = int(torch.randint(0, available, ()).item()) if randomize else (available - 1) // 2
    return features[:, :, start : start + 1, :]


def select_causal_sequence(
    features: torch.Tensor,
    sequence_frames: int,
    *,
    randomize: bool = True,
) -> torch.Tensor:
    """Select aligned contiguous frames for causal overlap-add training."""
    available = features.shape[2]
    if sequence_frames < 1 or sequence_frames > available:
        raise ValueError(f"sequence_frames must be in 1..={available}, got {sequence_frames}")
    maximum_start = available - sequence_frames
    if randomize and maximum_start:
        start = int(torch.randint(0, maximum_start + 1, ()).item())
    else:
        start = maximum_start // 2
    return features[:, :, start : start + sequence_frames, :]


def infer_independent_frames(
    generator: SoundExGenerator,
    features: torch.Tensor,
) -> torch.Tensor:
    """Run a sequence as independent fixed-`T=1` deployment calls."""
    if features.ndim != 4 or features.shape[1] != 2:
        raise ValueError("features must have shape [B, 2, T, F]")
    batch, channels, frames, bins = features.shape
    flattened = features.permute(0, 2, 1, 3).reshape(batch * frames, channels, 1, bins)
    predicted = generator(flattened)
    return predicted.reshape(batch, frames, channels, bins).permute(0, 2, 1, 3)


def batch_missing_band_mask(
    batch: dict[str, Any],
    audio_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Build per-row crossover masks from manifest-derived batch metadata."""
    metadata = batch["metadata"]
    # Default collation produces float64 metadata; MPS only accepts float32.
    cutoff_hz = torch.as_tensor(metadata["cutoff_hz"], dtype=torch.float32, device=device)
    sample_rate = torch.as_tensor(metadata["sample_rate"], dtype=torch.float32, device=device)
    return build_missing_band_mask(
        cutoff_hz,
        sample_rate,
        fft_size=int(audio_config["fft_size"]),
        crossover_width_hz=float(audio_config["crossover_width_hz"]),
    )


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    """Enable or disable parameter gradients for a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def initialize_generator_from_checkpoint(
    path: str | Path,
    generator: SoundExGenerator,
    config: dict[str, Any],
    data_provenance: dict[str, Any],
    *,
    initialize_zero_interactions: bool = False,
    initialize_phase_features: bool = False,
) -> dict[str, Any]:
    """Start a new optimizer trajectory with explicit, compatible parent weights."""
    payload = Path(path).read_bytes()
    parent = validate_checkpoint(
        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True),
        expected_data=data_provenance,
    )
    parent_architecture = dict(parent["model"]["generator_config"])
    architecture = dict(config["model"]["generator"])
    if initialize_zero_interactions and initialize_phase_features:
        raise ValueError("migrate one architecture feature at a time")
    if initialize_phase_features:
        if parent_architecture.get("circular_phase_features", False) or not architecture.get(
            "circular_phase_features", False
        ):
            raise ValueError(
                "phase-feature migration needs a legacy phase parent and enabled target"
            )
        parent_architecture.pop("circular_phase_features", None)
        architecture.pop("circular_phase_features", None)
    if initialize_zero_interactions:
        if parent_architecture.get("cross_stream_interactions", False) or not architecture.get(
            "cross_stream_interactions", False
        ):
            raise ValueError("zero-interaction migration needs a legacy parent and enabled target")
        parent_architecture.pop("cross_stream_interactions", None)
        architecture.pop("cross_stream_interactions", None)
    if parent_architecture != architecture:
        raise ValueError("generator initialization architecture differs from parent")
    if parent["feature_contract"] != feature_contract_from_config(config):
        raise ValueError("generator initialization feature contract differs from parent")
    state = parent["model"]["generator_state"]
    if any(
        not torch.isfinite(value).all().item()
        for value in state.values()
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())
    ):
        raise ValueError("generator initialization contains non-finite parameters")
    expanded_keys: list[str] = []
    if initialize_phase_features:
        key = "phase_stream.encoders.0.conv.0.weight"
        target_state = generator.state_dict()
        if set(state) != set(target_state):
            raise ValueError("phase-feature migration has unexpected state keys")
        original, expanded = state[key], target_state[key].clone()
        if original.shape[1] != 1 or expanded.shape != (original.shape[0], 3, *original.shape[2:]):
            raise ValueError("phase-feature migration has incompatible phase encoder shape")
        if torch.count_nonzero(expanded[:, 1:]).item():
            raise ValueError("phase-feature migration requires exactly zero added channels")
        expanded[:, :1] = original
        state = {**state, key: expanded}
        expanded_keys = [key]
    added_keys: list[str] = []
    if initialize_zero_interactions:
        target_state = generator.state_dict()
        added_keys = sorted(key for key in target_state if key.startswith("interactions."))
        if not added_keys or set(target_state) - set(state) != set(added_keys):
            raise ValueError("zero-interaction migration has unexpected state keys")
        if set(state) - set(target_state) or any(
            torch.count_nonzero(target_state[key]).item() for key in added_keys
        ):
            raise ValueError("interaction migration requires exactly zero new projections")
        generator.load_state_dict(
            {**state, **{key: target_state[key] for key in added_keys}}, strict=True
        )
    else:
        generator.load_state_dict(state, strict=True)
    return {
        "mode": "generator-only-warm-start-phase-features"
        if initialize_phase_features
        else "generator-only-warm-start-zero-interactions"
        if initialize_zero_interactions
        else "generator-only-warm-start",
        "zero_initialized_state_keys": added_keys,
        "expanded_zero_channel_state_keys": expanded_keys,
        "parent_checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "parent_epoch": int(parent["training_state"]["epoch"]),
        "parent_recipe_sha256": parent["data"]["recipe_sha256"],
        "parent_manifest_set_sha256": parent["data"]["manifest_set_sha256"],
        "parent_source_git_sha": parent["provenance"]["source_git_sha"],
        "parent_initialization": parent["provenance"].get("initialization"),
        "optimizers_and_discriminator": "fresh; parent state not imported",
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a serializable update-level linear-warmup/cosine schedule."""
    if total_steps <= 0:
        raise ValueError("total scheduler steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in 0..total_steps-1")
    cosine_steps = total_steps - warmup_steps
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    if warmup_steps == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _metadata_value(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value[index].item()
    if isinstance(value, np.ndarray):
        return value[index].item()
    return value[index]


def configure_generator_training_mode(generator: torch.nn.Module, statistics: str) -> None:
    """Optionally keep pretrained BatchNorm buffers fixed during continuation.

    Affine parameters and the rest of the generator remain trainable. This
    changes the optimization trajectory, so the policy belongs in the bound
    training config and cannot be silently applied to an existing resume.
    """
    if statistics not in {"update", "frozen"}:
        raise ValueError("training.batch_norm_statistics must be 'update' or 'frozen'")
    batch_norms = [
        module
        for module in generator.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if statistics == "frozen" and any(not module.track_running_stats for module in batch_norms):
        raise ValueError("frozen BatchNorm statistics require tracked running statistics")
    generator.train()
    if statistics == "frozen":
        for module in batch_norms:
            module.eval()


def train_one_epoch(
    generator: SoundExGenerator,
    discriminator: MultiScaleDiscriminator,
    train_loader: DataLoader,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    gen_scheduler: torch.optim.lr_scheduler.LRScheduler,
    disc_scheduler: torch.optim.lr_scheduler.LRScheduler,
    gen_loss_fn: GeneratorLoss,
    disc_loss_fn: DiscriminatorLoss,
    device: torch.device,
    epoch: int,
    audio_config: dict[str, Any],
    training_config: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """Run one training epoch and return every observable mean loss."""
    configure_generator_training_mode(
        generator, str(training_config.get("batch_norm_statistics", "update"))
    )
    discriminator.train()
    generator_totals: defaultdict[str, float] = defaultdict(float)
    discriminator_total = 0.0
    objective = training_config["objective"]
    use_gan = epoch > int(training_config.get("warmup_epochs", 50)) and (
        float(objective["adversarial_weight"]) > 0
        or float(objective["feature_matching_weight"]) > 0
    )

    for batch_index, batch in enumerate(train_loader):
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        degraded_features = spectral_features(
            degraded, audio_config["fft_size"], audio_config["hop_size"]
        )
        clean_features = spectral_features(
            clean, audio_config["fft_size"], audio_config["hop_size"]
        )
        combined = torch.cat((degraded_features, clean_features), dim=1)
        combined = select_causal_sequence(
            combined,
            int(training_config["objective"]["causal_sequence_frames"]),
        )
        degraded_features, clean_features = combined[:, :2], combined[:, 2:]
        missing_band_mask = batch_missing_band_mask(batch, audio_config, device)

        set_requires_grad(discriminator, False)
        predicted_features = infer_independent_frames(generator, degraded_features)
        if use_gan:
            predicted_disc = discriminator(predicted_features[:, :1])
            with torch.no_grad():
                target_disc = discriminator(clean_features[:, :1])
            generator_losses = gen_loss_fn(
                predicted_features,
                clean_features,
                degraded_features,
                missing_band_mask,
                predicted_disc,
                target_disc,
            )
        else:
            generator_losses = gen_loss_fn(
                predicted_features,
                clean_features,
                degraded_features,
                missing_band_mask,
            )

        gen_optimizer.zero_grad(set_to_none=True)
        generator_losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(), float(training_config["gradient_clip"])
        )
        gen_optimizer.step()
        gen_scheduler.step()

        discriminator_loss = predicted_features.new_zeros(())
        if use_gan:
            set_requires_grad(discriminator, True)
            predicted_disc = discriminator(predicted_features[:, :1].detach())
            target_disc = discriminator(clean_features[:, :1])
            discriminator_loss = disc_loss_fn(predicted_disc, target_disc)
            disc_optimizer.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            disc_optimizer.step()
            disc_scheduler.step()

        for name, value in generator_losses.items():
            generator_totals[name] += float(value.detach())
        discriminator_total += float(discriminator_loss.detach())
        if batch_index % 100 == 0:
            print(
                f"  [{epoch}][{batch_index}/{len(train_loader)}] "
                f"G: {generator_losses['total'].item():.4f} "
                f"D: {discriminator_loss.item():.4f}"
            )

    count = max(len(train_loader), 1)
    return (
        {name: value / count for name, value in generator_totals.items()},
        discriminator_total / count,
    )


@torch.no_grad()
def validate(
    generator: SoundExGenerator,
    val_loader: DataLoader,
    loss_fn: GeneratorLoss,
    device: torch.device,
    audio_config: dict[str, Any],
    context_frames: int,
) -> dict[str, Any]:
    """Compute deterministic row-level and per-stratum validation metrics."""
    generator.eval()
    rows: list[dict[str, Any]] = []
    for batch in val_loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        degraded_features = spectral_features(
            degraded, audio_config["fft_size"], audio_config["hop_size"]
        )
        clean_features = spectral_features(
            clean, audio_config["fft_size"], audio_config["hop_size"]
        )
        combined = select_causal_sequence(
            torch.cat((degraded_features, clean_features), dim=1),
            int(context_frames),
            randomize=False,
        )
        degraded_features, clean_features = combined[:, :2], combined[:, 2:]
        predicted = infer_independent_frames(generator, degraded_features)
        missing_band_mask = batch_missing_band_mask(batch, audio_config, device)
        metadata = batch["metadata"]
        for row_index in range(predicted.shape[0]):
            row_losses = loss_fn(
                predicted[row_index : row_index + 1],
                clean_features[row_index : row_index + 1],
                degraded_features[row_index : row_index + 1],
                missing_band_mask[row_index : row_index + 1],
            )
            codec_mode = str(_metadata_value(metadata, "codec_mode", row_index))
            codec_setting = float(_metadata_value(metadata, "codec_setting", row_index))
            rows.append(
                {
                    "row_id": str(_metadata_value(metadata, "row_id", row_index)),
                    "corpus": str(_metadata_value(metadata, "corpus", row_index)),
                    "codec_id": str(_metadata_value(metadata, "codec_id", row_index)),
                    "quality": quality_label(codec_mode, codec_setting),
                    "sample_rate": int(_metadata_value(metadata, "sample_rate", row_index)),
                    "channel_role": str(_metadata_value(metadata, "channel_role", row_index)),
                    "genre": (
                        str(_metadata_value(metadata, "genre", row_index))
                        if "genre" in metadata
                        else "unlabeled"
                    ),
                    "metrics": {
                        name: float(value.detach().cpu()) for name, value in row_losses.items()
                    },
                }
            )
    return aggregate_validation_rows(rows)


def main() -> None:
    """Parse configuration and run the complete training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument(
        "--initialize-generator-from",
        type=Path,
        help="New run from compatible parent generator only; optimizer and discriminator reset",
    )
    parser.add_argument(
        "--initialize-zero-interactions",
        action="store_true",
        help="Explicitly migrate a legacy parent into zero-initialized cross-stream projections",
    )
    parser.add_argument(
        "--paths-manifest",
        type=Path,
        default=None,
        help=(
            "Optional YAML from download_datasets.py (soundex_data_paths.yaml). "
            "Also read from PATHS_MANIFEST env if unset."
        ),
    )
    parser.add_argument(
        "--initialize-phase-features",
        action="store_true",
        help="Warm-start a legacy phase encoder with zero sin/cos channels",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.initialize_phase_features and not args.initialize_generator_from:
        parser.error("--initialize-phase-features requires --initialize-generator-from")
    if args.initialize_phase_features and args.initialize_zero_interactions:
        parser.error("migrate one architecture feature at a time")
    if args.initialize_zero_interactions and not args.initialize_generator_from:
        parser.error("--initialize-zero-interactions requires --initialize-generator-from")
    if int(config["training"].get("context_frames", 1)) != 1:
        raise ValueError("training.context_frames must be 1 for stateless streaming inference")
    set_seed(int(config["training"].get("seed", 42)))
    device = get_device()
    print(f"Device: {device}")

    manifest = args.paths_manifest
    if manifest is None:
        env_manifest = os.environ.get("PATHS_MANIFEST")
        if env_manifest:
            manifest = Path(env_manifest)
        else:
            data_root = os.environ.get("DATA_ROOT")
            if data_root:
                candidate = Path(data_root) / "soundex_data_paths.yaml"
                if candidate.is_file():
                    manifest = candidate

    data_config = apply_data_path_overrides(
        config.get("data") or {},
        paths_manifest=manifest,
    )
    config["data"] = data_config
    recipe = validate_data_config(data_config)
    print(f"Data recipe: {recipe.recipe_version} ({recipe.hash[:16]})")
    sample_rate = int(config["audio"]["sample_rate"])
    sources = resolve_dataset_sources(data_config, recipe)
    sampling_config = data_config.get("sampling") or {}
    train_loader, val_loader, mix_summary = create_balanced_dataloaders(
        sources,
        batch_size=int(config["training"]["batch_size"]),
        segment_length=sample_rate * 2,
        sample_rates=recipe.sample_rates,
        seed=int(config["training"].get("seed", 42)),
        sampling_config=sampling_config,
    )
    print(
        f"Epoch sampling: strategy={mix_summary['strategy']} "
        f"samples_per_epoch={mix_summary['samples_per_epoch']}"
    )
    data_provenance = build_dataset_provenance(recipe, sources, mix_summary)

    model_config = config["model"]["generator"]
    generator = SoundExGenerator(
        channels=model_config["channels"],
        bottleneck_blocks=model_config["bottleneck_blocks"],
        expand_ratio=model_config["expand_ratio"],
        cross_stream_interactions=model_config.get("cross_stream_interactions", False),
        circular_phase_features=model_config.get("circular_phase_features", False),
    ).to(device)
    configure_generator_training_mode(
        generator, str(config["training"].get("batch_norm_statistics", "update"))
    )
    parameter_count = generator.count_parameters()
    if parameter_count > 2_000_000:
        raise ValueError(f"Generator has {parameter_count:,} parameters; expected <=2M")
    print(f"Generator params: {parameter_count:,}")
    discriminator = MultiScaleDiscriminator(scales=config["model"]["discriminator"]["scales"]).to(
        device
    )

    learning_rate = float(config["training"]["learning_rate"])
    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
    )
    max_epochs = int(config["training"]["max_epochs"])
    scheduler_name = str(config["training"].get("lr_scheduler", "cosine"))
    if scheduler_name != "cosine":
        raise ValueError(f"unsupported training.lr_scheduler {scheduler_name!r}")
    steps_per_epoch = len(train_loader)
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    gan_warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    if gan_warmup_epochs < 0 or gan_warmup_epochs >= max_epochs:
        raise ValueError("training.warmup_epochs must be in 0..max_epochs-1")
    gen_scheduler = build_lr_scheduler(
        gen_optimizer,
        total_steps=max_epochs * steps_per_epoch,
        warmup_steps=warmup_steps,
    )
    disc_scheduler = build_lr_scheduler(
        disc_optimizer,
        total_steps=(max_epochs - gan_warmup_epochs) * steps_per_epoch,
        warmup_steps=warmup_steps,
    )
    validation_interval = int(config["training"].get("validation_interval_epochs", 1))
    checkpoint_interval = int(config["training"].get("checkpoint_interval_epochs", 10))
    if validation_interval < 1 or checkpoint_interval < 1:
        raise ValueError("validation and checkpoint intervals must be positive")
    start_epoch = 1
    global_step = 0
    selection = config["training"].get("validation_selection", {})
    best_tracker = BestCheckpointTracker(**selection)
    provenance = capture_provenance()
    provenance["initialization"] = {"mode": "fresh-random"}
    if args.initialize_generator_from:
        provenance["initialization"] = initialize_generator_from_checkpoint(
            args.initialize_generator_from,
            generator,
            config,
            data_provenance,
            initialize_zero_interactions=args.initialize_zero_interactions,
            initialize_phase_features=args.initialize_phase_features,
        )
        print(f"Generator initialization: {provenance['initialization']}")
    last_validation_report: dict[str, Any] | None = None
    last_validation_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            map_location=device,
            expected_config=config,
            expected_data=data_provenance,
        )
        model_state = checkpoint["model"]
        training_state = checkpoint["training_state"]
        generator.load_state_dict(model_state["generator_state"])
        discriminator.load_state_dict(model_state["discriminator_state"])
        gen_optimizer.load_state_dict(training_state["optimizers"]["generator"])
        disc_optimizer.load_state_dict(training_state["optimizers"]["discriminator"])
        gen_scheduler.load_state_dict(training_state["schedulers"]["generator"])
        disc_scheduler.load_state_dict(training_state["schedulers"]["discriminator"])
        start_epoch = int(training_state["epoch"]) + 1
        global_step = int(training_state["global_step"])
        saved_validation = training_state["validation"]
        saved_interval = int(saved_validation["schedule"]["interval_epochs"])
        if saved_interval != validation_interval:
            raise ValueError(
                "resume validation interval differs from checkpoint: "
                f"{validation_interval} != {saved_interval}"
            )
        best_tracker = BestCheckpointTracker.from_state_dict(saved_validation["best"])
        last_validation_report = dict(saved_validation["report"])
        last_validation_epoch = int(saved_validation["last_epoch"])
        restore_rng_state(checkpoint["rng_state"], train_loader)
        provenance["initialization"] = checkpoint["provenance"].get(
            "initialization", {"mode": "legacy-checkpoint-resume"}
        )

    gen_loss_fn = GeneratorLoss.from_config(
        config["training"]["objective"],
        fft_size=int(config["audio"]["fft_size"]),
        hop_size=int(config["audio"]["hop_size"]),
    )
    disc_loss_fn = DiscriminatorLoss()
    writer = SummaryWriter("logs")
    for epoch in range(start_epoch, max_epochs + 1):
        set_dataset_epoch(train_loader.dataset, epoch)
        generator_losses, discriminator_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            gen_optimizer,
            disc_optimizer,
            gen_scheduler,
            disc_scheduler,
            gen_loss_fn,
            disc_loss_fn,
            device,
            epoch,
            config["audio"],
            config["training"],
        )
        is_best = False
        if should_run_validation(epoch, validation_interval):
            last_validation_report = validate(
                generator,
                val_loader,
                gen_loss_fn,
                device,
                config["audio"],
                int(config["training"]["objective"]["causal_sequence_frames"]),
            )
            last_validation_epoch = epoch
            validation_metrics = last_validation_report["overall"]
            is_best = best_tracker.consider(epoch, validation_metrics)
            writer.add_scalars("loss/validation", validation_metrics, epoch)
        elif last_validation_report is None:
            raise RuntimeError("validation must run before the first checkpoint")
        writer.add_scalars("loss/train_generator", generator_losses, epoch)
        writer.add_scalar("loss/discriminator", discriminator_loss, epoch)
        global_step += len(train_loader)
        if last_validation_epoch == epoch:
            validation_metrics = last_validation_report["overall"]
            print(
                f"Epoch {epoch}: G={generator_losses['total']:.4f} "
                f"D={discriminator_loss:.4f} val={validation_metrics['total']:.4f} "
                f"val_high={validation_metrics['high_band_magnitude']:.4f}"
            )
        else:
            print(f"Epoch {epoch}: G={generator_losses['total']:.4f} D={discriminator_loss:.4f}")

        validation_state = build_validation_state(
            report=last_validation_report,
            manifest_sha256=str(data_provenance["validation_manifest_sha256"]),
            tracker=best_tracker,
            interval_epochs=validation_interval,
            epoch=last_validation_epoch,
        )
        checkpoint = build_checkpoint(
            epoch=epoch,
            global_step=global_step,
            generator_state=generator.state_dict(),
            discriminator_state=discriminator.state_dict(),
            generator_optimizer_state=gen_optimizer.state_dict(),
            discriminator_optimizer_state=disc_optimizer.state_dict(),
            generator_scheduler_state=gen_scheduler.state_dict(),
            discriminator_scheduler_state=disc_scheduler.state_dict(),
            scaler_state=None,
            resolved_config=config,
            data_provenance=data_provenance,
            validation_state=validation_state,
            train_loader=train_loader,
            provenance=provenance,
        )
        checkpoint_dir = Path("checkpoints")
        save_checkpoint(checkpoint, checkpoint_dir / "final-resume.pth")
        if is_best:
            save_checkpoint(checkpoint, checkpoint_dir / "best-validation.pth")
        if epoch % checkpoint_interval == 0:
            save_checkpoint(checkpoint, checkpoint_dir / f"epoch_{epoch:04d}.pth")
    writer.close()


if __name__ == "__main__":
    main()
