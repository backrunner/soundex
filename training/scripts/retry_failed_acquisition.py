# SPDX-License-Identifier: Apache-2.0
"""Wait for collectors, then retry unresolved downloads without relaxing source checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_archive_masters import atomic_write
from scripts.wait_for_native_training import collection_active, read_catalog


def unresolved_entries(plan: Path, upstream: list[Path]) -> list[dict[str, Any]]:
    """A signal-quality hold is a verified acquisition, not a failed download."""
    failed, acquired = set(), set()
    for directory in upstream:
        progress = json.loads((directory / "progress.json").read_text())
        failed.update(e["id"] for e in progress["errors"])
        acquired.update(e["id"] for e in read_catalog(directory / "acquired.jsonl"))
    entries = [json.loads(line) for line in plan.read_text().splitlines() if line.strip()]
    return [e for e in entries if e["id"] in failed - acquired]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1, 32]")
    receipt = args.receipt_dir.resolve()
    receipt.mkdir(parents=True, exist_ok=True)
    if (receipt / "progress.json").exists():
        raise ValueError("retry receipt directory already contains a collection")
    atomic_write(
        receipt / "progress.json",
        json.dumps({"completed": 0, "total": 1, "errors": [], "stage": "waiting-upstream"}),
    )
    while any(collection_active(d) for d in args.upstream_dir):
        time.sleep(60)
    entries = unresolved_entries(args.plan, args.upstream_dir)
    pending = receipt / "retry-plan.jsonl"
    pending.write_text("".join(json.dumps(e) + "\n" for e in entries))
    if not entries:
        atomic_write(receipt / "acquired.jsonl", "")
        atomic_write(
            receipt / "progress.json", json.dumps({"completed": 0, "total": 0, "errors": []})
        )
        return
    print(
        f"Retrying {len(entries)} failed originals with full publisher and signal checks",
        flush=True,
    )
    with (receipt / "fetch.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("fetch_archive_masters.py")),
                "--plan",
                str(pending),
                "--root",
                str(args.root.resolve()),
                "--receipt-dir",
                str(receipt),
                "--workers",
                str(args.workers),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
