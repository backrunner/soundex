"""Checkpoint hashing, provenance, and random-state helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def to_plain(value: Any) -> Any:
    """Convert configuration data to weights-only-safe primitive containers."""
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not checkpoint-serializable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable key and separator semantics."""
    encoded = json.dumps(to_plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_dataset_provenance(
    recipe: Any,
    sources: Sequence[Any],
    effective_source_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a checkpoint to immutable recipe and manifest inputs."""
    from data.protocol import load_manifest, manifest_summary, sha256_file

    manifests: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: item.name):
        manifest_path = Path(source.manifest_path).resolve()
        rows = load_manifest(manifest_path)
        manifests.append(
            {
                "corpus": str(source.name),
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "summary": manifest_summary(rows),
            }
        )
    manifest_bindings = [{"corpus": item["corpus"], "sha256": item["sha256"]} for item in manifests]
    manifest_set_sha256 = canonical_sha256(manifest_bindings)
    validation_row_ids = sorted(
        str(row_id) for row_id in effective_source_counts.get("validation_row_ids", [])
    )
    return {
        "recipe": recipe.to_dict(),
        "recipe_sha256": str(recipe.hash),
        "manifests": manifests,
        "manifest_set_sha256": manifest_set_sha256,
        "validation_manifest_sha256": canonical_sha256(
            {
                "manifest_set_sha256": manifest_set_sha256,
                "row_ids": validation_row_ids,
            }
        ),
        "effective_source_counts": to_plain(effective_source_counts),
    }


def capture_rng_state(train_loader: Any | None = None) -> dict[str, Any]:
    """Capture every random stream that affects training or sample ordering."""
    numpy_state = np.random.get_state()
    sampler = getattr(train_loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if sampler_generator is None:
        sampler_state = {"source": "torch_global", "state": torch.random.get_rng_state()}
    else:
        sampler_state = {"source": "sampler_generator", "state": sampler_generator.get_state()}
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_sampler": sampler_state,
    }


def restore_rng_state(rng_state: Mapping[str, Any], train_loader: Any | None = None) -> None:
    """Restore random streams captured by :func:`capture_rng_state`."""
    random.setstate(tuple(rng_state["python"]))
    numpy_state = cast(Mapping[str, Any], rng_state["numpy"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.random.set_rng_state(cast(torch.Tensor, rng_state["torch_cpu"]).cpu())
    cuda_states = cast(list[torch.Tensor], rng_state["torch_cuda"])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    sampler_state = cast(Mapping[str, Any], rng_state["dataloader_sampler"])
    if sampler_state["source"] == "sampler_generator" and train_loader is not None:
        sampler = getattr(train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is None:
            generator = torch.Generator()
            sampler.generator = generator
        generator.set_state(cast(torch.Tensor, sampler_state["state"]).cpu())


def capture_provenance() -> dict[str, Any]:
    """Record toolchain and source identifiers without requiring Git or FFmpeg."""
    ffmpeg_version: str | None = None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True, check=False, timeout=10
        )
        if result.returncode == 0 and result.stdout:
            ffmpeg_version = result.stdout.splitlines()[0]

    source_sha: str | None = None
    source_origin: str | None = None
    source_receipt_sha256: str | None = None
    repository = Path(__file__).resolve().parents[1]
    receipt = repository.parent / "source-receipt.json"
    archived_source = repository.name == "source" and not (repository / ".git").exists()
    if archived_source and receipt.is_file():
        # git rev-parse in an archive nested inside the live repository would
        # otherwise report a later, unrelated HEAD. The runner freezes this
        # committed archive and writes its receipt before starting train.py.
        payload = receipt.read_bytes()
        source_sha = json.loads(payload).get("code_commit")
        if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
            raise ValueError("frozen source receipt requires a full code_commit SHA")
        source_origin = "frozen-source-receipt"
        source_receipt_sha256 = hashlib.sha256(payload).hexdigest()
    elif archived_source:
        source_origin = "unversioned-source-archive"
    git = shutil.which("git")
    if git is not None and source_origin is None:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            source_sha = result.stdout.strip() or None
            source_origin = "git" if source_sha else None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "ffmpeg_version": ffmpeg_version,
        "source_git_sha": source_sha,
        "source_git_sha_origin": source_origin,
        "source_receipt_sha256": source_receipt_sha256,
        "platform": platform.platform(),
    }
