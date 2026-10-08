"""Downloads must not publish bad bytes or trust a misleading audio extension."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data import download_library


@pytest.mark.parametrize("valid", [False, True])
def test_only_verified_originals_are_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid: bool
) -> None:
    buffer = io.BytesIO()
    sf.write(buffer, np.linspace(-0.1, 0.1, 100), 44100, format="FLAC", subtype="PCM_16")
    original = buffer.getvalue()
    entry = {
        "id": "original",
        "path": "album/track.flac",
        "source_url": "https://example.org/original.flac",
        "audio_sha256": hashlib.sha256(original).hexdigest(),
    }
    monkeypatch.setattr(
        download_library, "urlopen", lambda *a, **k: io.BytesIO(original if valid else b"bad bytes")
    )
    if valid:
        path = download_library.acquire_master(entry, tmp_path)
        assert path.read_bytes() == original
        monkeypatch.setattr(download_library, "urlopen", lambda *a, **k: pytest.fail("redownload"))
        assert download_library.acquire_master(entry, tmp_path) == path
    else:
        with pytest.raises(ValueError, match="checksum mismatch"):
            download_library.acquire_master(entry, tmp_path)
        assert not (tmp_path / entry["path"]).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_checksum_valid_lossy_audio_is_not_a_clean_master(tmp_path: Path) -> None:
    path = tmp_path / "disguised.wav"
    sf.write(path, np.linspace(-0.1, 0.1, 100), 44100, format="OGG", subtype="VORBIS")
    entry = {"id": "lossy", "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="lossless"):
        download_library.verify_master(path, entry)


def test_catalog_path_cannot_escape_the_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the output directory"):
        download_library.acquire_master({"path": "../outside.flac"}, tmp_path)
