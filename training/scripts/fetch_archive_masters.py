# SPDX-License-Identifier: Apache-2.0
"""Acquire reviewed original-file entries with publisher size/MD5 and complete signal audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.source_quality import inspect_master


def verify_original(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Bind publisher bytes and full decoded measurements to the acquired recording."""
    size = path.stat().st_size
    if size != entry["source_bytes"]:
        raise ValueError(f"publisher original length mismatch: {size} != {entry['source_bytes']}")
    md5 = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            md5.update(chunk)
    if md5.hexdigest() != entry["publisher_original_md5"]:
        raise ValueError("publisher original checksum mismatch")
    quality = inspect_master(path, entry.get("audio_sha256"))
    return {
        **entry,
        "path": str(path),
        "audio_sha256": quality["audio_sha256"],
        "decoded_pcm_sha256": quality["decoded_pcm_sha256"],
        "source_quality": quality,
        "upstream_original_md5_verified": True,
    }


def acquire(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resume verified files and atomically publish only fully audited originals."""
    relative = Path(entry["path"])
    target = root / relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("original target path must be relative")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("original path resolves outside acquisition root")
    if urlparse(entry["source_url"]).scheme != "https":
        raise ValueError("original URL must use HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        pending = None
        try:
            if target.exists():
                return verify_original(target, entry)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False
            ) as out:
                pending = Path(out.name)
                url = entry["source_url"]
                if attempt:
                    separator = "&" if "?" in url else "?"
                    url += separator + urlencode(
                        {"soundex_original": entry["publisher_original_md5"], "attempt": attempt}
                    )
                with urlopen(url, timeout=60) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"expected complete original, received HTTP {response.status}"
                        )
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != entry["source_bytes"]:
                        raise ValueError(
                            f"original response length {length} differs from publisher"
                        )
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
            verified = verify_original(pending, entry)
            pending.replace(target)
            verified["path"] = str(target)
            return verified
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    raise RuntimeError("unreachable acquisition state")


def atomic_write(path: Path, value: str) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(value)
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1, 32]")
    entries = [json.loads(line) for line in args.plan.read_text().splitlines() if line.strip()]
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("duplicate acquisition targets")
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors = []
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(acquire, e, args.root.resolve()): e for e in entries}
        for job in as_completed(jobs):
            entry = jobs[job]
            try:
                records.append(job.result())
            except Exception as error:
                errors.append({"id": entry["id"], "error": str(error)})
            progress = {"completed": len(records), "total": len(entries), "errors": errors}
            atomic_write(args.receipt_dir / "progress.json", json.dumps(progress, indent=2) + "\n")
            atomic_write(
                args.receipt_dir / "acquired.jsonl",
                "".join(json.dumps(e) + "\n" for e in sorted(records, key=lambda e: e["id"])),
            )
            print(f"Verified {len(records)}/{len(entries)}; failures {len(errors)}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
