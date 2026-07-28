#!/usr/bin/env python3
"""Keep only full-track mix files; delete stems / RAW / MIDI to free disk.

Run this AFTER unpacking a multitrack dataset (or point at a partial tree).

Supported layouts
-----------------
MUSDB18-HQ
  train|test/<track>/{mixture.wav, *.wav stems...}
  → keeps mixture.wav only

Slakh2100 / redux
  [train|validation|test/]TrackXXXXX/{mix.flac, stems/, MIDI/, ...}
  → keeps mix.flac|mix.wav only (drops stems/, MIDI/, all_src.mid, …)

MedleyDB / 2.0
  …/Artist_Track/{*_MIX.wav, *_STEMS/, *_RAW/, …}
  → keeps *_MIX.wav (+ optional *_METADATA.yaml); drops STEMS/RAW trees

Usage
-----
  # Dry-run (default): print what would be deleted
  python scripts/filter_mix_only.py --root /data/slakh2100

  # Actually delete
  python scripts/filter_mix_only.py --root /data/slakh2100 --delete

  # Dataset hint (optional; auto-detected if omitted)
  python scripts/filter_mix_only.py --root /data/musdb18-hq --dataset musdb --delete
"""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import sys
from pathlib import Path

_MIX_NAMES = frozenset({"mix.flac", "mix.wav", "mixture.flac", "mixture.wav"})
_MEDLEY_MIX = re.compile(r".+_MIX\.wav$", re.IGNORECASE)
_KEEP_META = re.compile(r".+_METADATA\.ya?ml$", re.IGNORECASE)


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            with contextlib.suppress(OSError):
                total += child.stat().st_size
    return total


def detect_dataset(root: Path) -> str:
    if any(root.rglob("mixture.wav")):
        return "musdb"
    if any(root.rglob("mix.flac")) or any(root.rglob("mix.wav")):
        return "slakh"
    if any(p for p in root.rglob("*.wav") if _MEDLEY_MIX.match(p.name)):
        return "medleydb"
    return "unknown"


def collect_musdb_deletions(root: Path) -> list[Path]:
    """Delete everything in track folders except mixture.wav."""
    victims: list[Path] = []
    for mixture in root.rglob("mixture.wav"):
        track_dir = mixture.parent
        for child in track_dir.iterdir():
            if child.name == "mixture.wav":
                continue
            victims.append(child)
    return victims


def collect_slakh_deletions(root: Path) -> list[Path]:
    """Per Track* directory keep only mix.flac / mix.wav."""
    victims: list[Path] = []
    track_dirs: set[Path] = set()
    for name in _MIX_NAMES:
        for mix in root.rglob(name):
            track_dirs.add(mix.parent)

    # Also find Track* dirs that only have stems (broken trees) — leave them;
    # user should re-download mix if missing.
    for track_dir in sorted(track_dirs):
        for child in track_dir.iterdir():
            if child.name.lower() in _MIX_NAMES:
                continue
            victims.append(child)
    return victims


def collect_medleydb_deletions(root: Path) -> list[Path]:
    """Keep *_MIX.wav and optional metadata; drop STEMS/RAW and other audio."""
    victims: list[Path] = []
    # Prefer track folders that contain a MIX file.
    mix_parents = {p.parent for p in root.rglob("*.wav") if _MEDLEY_MIX.match(p.name)}
    for track_dir in sorted(mix_parents):
        for child in track_dir.iterdir():
            name = child.name
            if _MEDLEY_MIX.match(name) or _KEEP_META.match(name):
                continue
            victims.append(child)
    # Orphan STEMS/RAW trees not beside a MIX (still reclaim space).
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        lower = path.name.lower()
        is_stem_tree = lower.endswith(("_stems", "_raw")) or lower in {"stems", "raw"}
        is_nested_victim = any(v == path or path.is_relative_to(v) for v in victims if v.is_dir())
        if is_stem_tree and path not in victims and not is_nested_victim:
            victims.append(path)
    return _dedupe_outermost(victims)


def _dedupe_outermost(paths: list[Path]) -> list[Path]:
    """If both parent and child are listed, keep only the outermost path."""
    resolved = sorted({p.resolve() for p in paths}, key=lambda p: len(p.parts))
    kept: list[Path] = []
    for path in resolved:
        if any(path == k or path.is_relative_to(k) for k in kept):
            continue
        kept.append(path)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter multitrack trees down to mix-only files")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root")
    parser.add_argument(
        "--dataset",
        choices=("auto", "musdb", "slakh", "medleydb"),
        default="auto",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    kind = detect_dataset(root) if args.dataset == "auto" else args.dataset
    if kind == "unknown":
        print(
            "Could not detect dataset type under",
            root,
            file=sys.stderr,
        )
        return 1

    collectors = {
        "musdb": collect_musdb_deletions,
        "slakh": collect_slakh_deletions,
        "medleydb": collect_medleydb_deletions,
    }
    victims = collectors[kind](root)
    # Never delete the root itself.
    victims = [v for v in victims if v.resolve() != root]

    total = sum(path_size(v) for v in victims)
    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"[{mode}] dataset={kind} root={root}")
    print(f"Candidates: {len(victims)} paths, ~{human_bytes(total)}")

    for path in victims[:50]:
        print(f"  {'rm' if args.delete else 'would remove'}: {path}")
    if len(victims) > 50:
        print(f"  ... and {len(victims) - 50} more")

    if not args.delete:
        print("\nRe-run with --delete to reclaim space.")
        return 0

    for path in victims:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Failed: {path}: {exc}", file=sys.stderr)

    print(f"Done. Freed approximately {human_bytes(total)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
