# SPDX-License-Identifier: Apache-2.0
"""Publisher receipts reject altered, incomplete and lossy source bytes."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.fetch_archive_masters import acquire, verify_original


def test_original_length_and_md5_bind_entire_file(tmp_path: Path) -> None:
    path = tmp_path / "original.flac"
    sf.write(path, np.sin(np.arange(44100) * 0.1) * 0.1, 44100)
    entry = {
        "id": "original",
        "source_bytes": path.stat().st_size,
        "publisher_original_md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }
    result = verify_original(path, entry)
    assert result["source_quality"]["frames"] == 44100
    assert result["upstream_original_md5_verified"]
    with pytest.raises(ValueError, match="length"):
        verify_original(path, {**entry, "source_bytes": entry["source_bytes"] + 1})
    with pytest.raises(ValueError, match="checksum"):
        verify_original(path, {**entry, "publisher_original_md5": "0" * 32})


def test_lossy_upload_cannot_be_a_clean_master(tmp_path: Path) -> None:
    path = tmp_path / "fake.wav"
    sf.write(path, np.sin(np.arange(44100) * 0.1) * 0.1, 44100, format="OGG", subtype="VORBIS")
    entry = {
        "id": "fake",
        "source_bytes": path.stat().st_size,
        "publisher_original_md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="lossless"):
        verify_original(path, entry)


def test_acquisition_paths_and_urls_are_contained(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        acquire({"path": "../escape.wav"}, tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        acquire({"path": "original.wav", "source_url": "http://example.com/a"}, tmp_path)
