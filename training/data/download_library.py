# SPDX-License-Identifier: Apache-2.0
"""Fetch original lossless masters from an optional, portable JSONL source catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.input_audio import lossless_source_info


def verify_master(path: Path, entry: dict[str, Any]) -> None:
    """Verify original bytes and completely decode the lossless mono/stereo master."""
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            sha.update(block)
    if sha.hexdigest() != entry["audio_sha256"]:
        raise ValueError(f"original audio checksum mismatch: {entry['id']}")
    info = lossless_source_info(path)
    pcm = hashlib.sha256(f"{info.samplerate}:{info.channels}:".encode())
    frames = 0
    with sf.SoundFile(path) as audio:
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError(f"non-finite decoded master: {entry['id']}")
            frames += len(block)
            pcm.update(block.astype("<f4", copy=False).tobytes())
    if frames != info.frames:
        raise ValueError(f"incomplete decoded master: {entry['id']}")
    expected_pcm = entry.get("decoded_pcm_sha256")
    if expected_pcm and pcm.hexdigest() != expected_pcm:
        raise ValueError(f"decoded audio checksum mismatch: {entry['id']}")


def acquire_master(entry: dict[str, Any], root: Path) -> Path:
    """Resume verified originals; publish a downloaded master only after its checks pass."""
    relative = Path(entry["path"])
    target = root / relative
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError("catalog audio path must be relative and stay inside the output directory")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("catalog audio path resolves outside the output directory")
    url = str(entry["source_url"])
    if urlparse(url).scheme != "https":
        raise ValueError("source URL must use HTTPS")
    if target.exists():
        verify_master(target, entry)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
            pending = Path(output.name)
            with urlopen(url, timeout=60) as response:
                while block := response.read(1 << 20):
                    output.write(block)
        verify_master(pending, entry)
        pending.replace(target)
        return target
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1, 32]")
    entries = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = [entry["path"] for entry in entries]
    if len(targets) != len(set(targets)):
        raise ValueError("catalog contains duplicate target paths")
    with ThreadPoolExecutor(args.workers) as pool:
        for path in pool.map(lambda entry: acquire_master(entry, root), entries):
            print(f"Verified {path}", flush=True)
    # Relative paths remain usable by preprocess_library.py from the saved catalog.
    (root / "catalog.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8"
    )
    print(f"Verified {len(entries)} original lossless recordings")


if __name__ == "__main__":
    main()
