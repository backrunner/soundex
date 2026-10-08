# SPDX-License-Identifier: Apache-2.0
"""Load audio folders or optional source catalogs without a dataset/license whitelist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf

from data.protocol import canonical_track_id, sha256_file

AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"})


def _bind_records(entries: list[dict[str, Any]], root: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    checksums: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("catalog entry requires a recording id")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("catalog entry requires an audio path")
        # Album folders often contain the same basename (e.g. track01.wav).
        identity_hash = hashlib.sha256(entry["id"].encode()).hexdigest()[:16]
        identity = f"{Path(entry['id']).stem}-{identity_hash}"
        track_id = canonical_track_id("music_library", identity)
        if track_id in identities:
            raise ValueError("catalog contains duplicate recording IDs")
        identities.add(track_id)
        audio = (root / entry["path"]).resolve()
        checksum = sha256_file(audio)
        if entry.get("audio_sha256") and entry["audio_sha256"] != checksum:
            raise ValueError("catalog source audio checksum mismatch")
        info = sf.info(audio)
        if info.channels not in {1, 2} or info.frames == 0:
            raise ValueError("recordings require nonempty mono/stereo audio")
        metadata = {key: value for key, value in entry.items() if key not in {"id", "path"}}
        if entry.get("evidence_path"):
            evidence_hash = sha256_file(root / entry["evidence_path"])
            if entry.get("evidence_sha256") and entry["evidence_sha256"] != evidence_hash:
                raise ValueError("catalog evidence checksum mismatch")
            metadata["evidence_sha256"] = evidence_hash
        group = metadata.get("split_group", checksum)
        if not isinstance(group, str) or not group.strip():
            raise ValueError("split_group must be a nonempty string")
        metadata.update(
            audio_sha256=checksum,
            source_file=entry["path"],
            split_group=group,
            original_sample_rate=info.samplerate,
            original_channels=info.channels,
            original_format=info.format,
            original_subtype=info.subtype,
        )
        # Exact copies add no diversity. Keep the first entry and report skipped copies.
        if checksum in checksums:
            print(f"Skipping duplicate audio: {entry['path']}")
            continue
        checksums.add(checksum)
        records.append({"track_id": track_id, "audio": audio, "source_metadata": metadata})
    if not records:
        raise ValueError("audio catalog is empty")
    bindings = [
        {"track_id": row["track_id"], "source_metadata": row["source_metadata"]} for row in records
    ]
    canonical = json.dumps(
        sorted(bindings, key=lambda row: row["track_id"]), sort_keys=True, separators=(",", ":")
    )
    return records, hashlib.sha256(canonical.encode()).hexdigest()


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Only id/path are required; source, license and attribution fields are optional."""
    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return _bind_records(entries, path.parent)


def scan_audio_directory(root: Path) -> tuple[list[dict[str, Any]], str]:
    """Recursively discover recordings; identical files share a split and are deduplicated."""
    if not root.is_dir():
        raise ValueError(f"audio directory does not exist: {root}")
    entries = [
        {"id": path.relative_to(root).as_posix(), "path": path.relative_to(root).as_posix()}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return _bind_records(entries, root)
