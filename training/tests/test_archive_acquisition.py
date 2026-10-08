# SPDX-License-Identifier: Apache-2.0
"""Publisher receipts reject altered, incomplete and lossy source bytes."""

import gzip
import hashlib
from io import BytesIO
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


def test_relative_input_receipt_resolves_from_any_consumer_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("raw/native.wav")
    source.parent.mkdir()
    sf.write(source, np.sin(np.arange(44100) * 0.1) * 0.1, 44100)
    result = verify_original(source, {"id": "native"})
    assert Path(result["path"]).is_absolute()
    assert Path(result["path"]).samefile(source)


def test_acquisition_paths_and_urls_are_contained(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        acquire({"path": "../escape.wav"}, tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        acquire({"path": "original.wav", "source_url": "http://example.com/a"}, tmp_path)


def test_http_gzip_is_transport_only_and_not_a_lossy_transcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "native.wav"
    sf.write(path, np.sin(np.arange(44100) * 0.1) * 0.1, 44100)
    payload = path.read_bytes()
    compressed = gzip.compress(payload)

    class Response(BytesIO):
        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.status = 200
            self.headers = {"Content-Encoding": "gzip", "Content-Length": str(len(data))}

    def open_response(url: str, timeout: int) -> Response:
        return Response(compressed)

    monkeypatch.setattr("scripts.fetch_archive_masters.urlopen", open_response)
    result = acquire(
        {
            "id": "kcc-song",
            "path": "raw/native.wav",
            "source_url": "https://example.com/native",
            "source_bytes": len(payload),
        },
        tmp_path,
    )
    assert Path(result["path"]).read_bytes() == payload
    assert result["source_quality"]["frames"] == 44100
    assert not result["upstream_original_md5_verified"]


def test_cached_partial_response_is_retried_and_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "source.wav"
    sf.write(original, np.sin(np.arange(44100) * 0.1) * 0.1, 44100)
    payload = original.read_bytes()
    entry = {
        "id": "song",
        "path": "raw/song.wav",
        "source_url": "https://publisher.example/song.wav",
        "source_bytes": len(payload),
        "publisher_original_md5": hashlib.md5(payload).hexdigest(),
    }
    urls: list[str] = []

    class Response(BytesIO):
        def __init__(self, data: bytes, status: int) -> None:
            super().__init__(data)
            self.status = status
            self.headers = {"Content-Length": str(len(data))}

    def open_response(url: str, timeout: int) -> Response:
        urls.append(url)
        assert timeout > 0
        if len(urls) == 1:
            return Response(payload[:100], 206)
        assert not (tmp_path / entry["path"]).exists()
        return Response(payload, 200)

    monkeypatch.setattr("scripts.fetch_archive_masters.urlopen", open_response)
    monkeypatch.setattr("scripts.fetch_archive_masters.time.sleep", lambda _: None)
    result = acquire(entry, tmp_path)
    assert len(urls) == 2
    assert "soundex_original=" in urls[1]
    assert Path(result["path"]).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))
