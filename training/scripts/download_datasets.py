#!/usr/bin/env python3
"""Download SoundEx training datasets (mix-only where possible).

Public Zenodo sources:

  musdb18-hq   https://zenodo.org/records/3338373  (~23 GB zip → prune stems)
  slakh2100    https://zenodo.org/records/4599666  (~104 GB tar → extract mix only)
  babyslakh    https://zenodo.org/records/4603870  (tiny smoke subset; 16 kHz — not for full BWE)

MedleyDB is access-controlled on Zenodo; pass a pre-approved URL via
``--medleydb-url`` or place files under the configured medleydb path manually.

Path configuration (priority high → low)
----------------------------------------
For each dataset storage directory:

  1. ``--path NAME=/abs/or/rel``  (repeatable)
  2. ``--musdb18-hq-path`` / ``--slakh2100-path`` / ``--medleydb-path`` / ``--babyslakh-path``
  3. Environment: ``MUSDB18_HQ_PATH``, ``SLAKH2100_PATH``, ``MEDLEYDB_PATH``, ``BABYSLAKH_PATH``
  4. Default: ``$DATA_ROOT/<dest_name>``  (DATA_ROOT from ``--data-root`` / env, default ``/data``)

Archive download cache (zip/tar only, not extracted audio):

  1. ``--cache-dir``
  2. ``CACHE_DIR`` or ``DOWNLOAD_CACHE_DIR``
  3. Default: ``$DATA_ROOT/.cache/downloads``

Examples
--------
  # Default layout under /data
  python scripts/download_datasets.py --datasets musdb18-hq --data-root /data

  # Custom storage + cache on different disks
  python scripts/download_datasets.py --datasets musdb18-hq,slakh2100 \\
      --data-root /mnt/datasets \\
      --cache-dir /mnt/scratch/soundex-cache \\
      --path musdb18-hq=/mnt/datasets/musdb \\
      --path slakh2100=/mnt/huge/slakh

  # Env-based (Docker-friendly)
  export DATA_ROOT=/data
  export MUSDB18_HQ_PATH=/data/custom/musdb
  export CACHE_DIR=/scratch/dl
  python scripts/download_datasets.py --datasets musdb18-hq --preprocess
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in requirements.txt
    yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps dataset id → env var for storage path override
PATH_ENV_KEYS: dict[str, str] = {
    "musdb18-hq": "MUSDB18_HQ_PATH",
    "slakh2100": "SLAKH2100_PATH",
    "babyslakh": "BABYSLAKH_PATH",
    "medleydb": "MEDLEYDB_PATH",
}

# Maps dataset id → training config yaml key
CONFIG_PATH_KEYS: dict[str, str] = {
    "musdb18-hq": "musdb18_hq_path",
    "slakh2100": "slakh2100_path",
    "babyslakh": "babyslakh_path",
    "medleydb": "medleydb_path",
}

REGISTRY: dict[str, dict] = {
    "musdb18-hq": {
        "url": "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1",
        "filename": "musdb18hq.zip",
        "kind": "zip_then_filter",
        "filter_dataset": "musdb",
        "dest_name": "musdb18-hq",
        "ready_marker": ("train", "test"),
        "size_hint_gb": 23,
        "license_note": "MUSDB18-HQ: research-oriented terms (see NOTICE).",
    },
    "slakh2100": {
        "url": ("https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"),
        "filename": "slakh2100_flac_redux.tar.gz",
        "kind": "tar_mix_only",
        "dest_name": "slakh2100",
        "ready_marker_glob": "**/mix.flac",
        "size_hint_gb": 105,
        "license_note": "Slakh2100: CC BY 4.0 (see NOTICE).",
    },
    "babyslakh": {
        "url": "https://zenodo.org/records/4603870/files/babyslakh_16k.zip?download=1",
        "filename": "babyslakh_16k.zip",
        "kind": "zip_then_filter",
        "filter_dataset": "slakh",
        "dest_name": "babyslakh",
        "ready_marker_glob": "**/mix.wav",
        "size_hint_gb": 1,
        "license_note": (
            "BabySlakh: tiny 16 kHz subset for pipeline smoke tests only "
            "(not full-band BWE training)."
        ),
    },
    "medleydb": {
        "url": None,
        "filename": "medleydb.zip",
        "kind": "manual_or_url",
        "filter_dataset": "medleydb",
        "dest_name": "medleydb",
        "ready_marker_glob": "**/*_MIX.wav",
        "size_hint_gb": 20,
        "license_note": (
            "MedleyDB: CC BY-NC-SA; often requires Zenodo access request. "
            "Use --medleydb-url or copy files into the configured medleydb path."
        ),
    },
}


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_path_overrides(items: list[str] | None) -> dict[str, Path]:
    """Parse ``NAME=/path`` pairs from ``--path`` flags."""
    result: dict[str, Path] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --path '{item}'. Expected NAME=/storage/path "
                f"(known names: {', '.join(REGISTRY)})"
            )
        name, raw = item.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if name not in REGISTRY:
            raise SystemExit(
                f"Unknown dataset in --path: '{name}'. Choose from: {', '.join(REGISTRY)}"
            )
        if not raw:
            raise SystemExit(f"Empty path for dataset '{name}'")
        result[name] = Path(raw).expanduser()
    return result


def resolve_dataset_dest(
    name: str,
    data_root: Path,
    cli_paths: dict[str, Path],
) -> Path:
    """Resolve final storage directory for one dataset."""
    if name in cli_paths:
        return cli_paths[name].resolve()
    env_key = PATH_ENV_KEYS.get(name)
    if env_key:
        env_val = os.environ.get(env_key)
        if env_val:
            return Path(env_val).expanduser().resolve()
    return (data_root / REGISTRY[name]["dest_name"]).resolve()


def resolve_cache_dir(cli_cache: Path | None, data_root: Path) -> Path:
    if cli_cache is not None:
        return cli_cache.expanduser().resolve()
    for key in ("CACHE_DIR", "DOWNLOAD_CACHE_DIR"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return (data_root / ".cache" / "downloads").resolve()


def write_paths_manifest(
    path: Path,
    *,
    data_root: Path,
    cache_dir: Path,
    dataset_dirs: dict[str, Path],
) -> None:
    """Write YAML for train.py / operators showing where data lives."""
    payload = {
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "datasets": {name: str(dest) for name, dest in sorted(dataset_dirs.items())},
        # Keys aligned with training/configs/default.yaml → data.*
        "training_data": {
            CONFIG_PATH_KEYS[name]: str(dest)
            for name, dest in dataset_dirs.items()
            if name in CONFIG_PATH_KEYS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    else:
        # Minimal fallback without PyYAML
        lines = [
            f"data_root: {data_root}",
            f"cache_dir: {cache_dir}",
            "datasets:",
        ]
        for name, dest in sorted(dataset_dirs.items()):
            lines.append(f"  {name}: {dest}")
        lines.append("training_data:")
        for name, dest in dataset_dirs.items():
            if name in CONFIG_PATH_KEYS:
                lines.append(f"  {CONFIG_PATH_KEYS[name]}: {dest}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote path manifest: {path}")


# ---------------------------------------------------------------------------
# Download with resume
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """HTTP GET with resume (Range) when the server allows it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.is_file() else 0
    headers = {"User-Agent": "soundex-download/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        log(f"  Resuming {dest.name} from {human_bytes(existing)}")

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=120) as response:
            status = getattr(response, "status", None) or response.getcode()
            mode = "ab" if status == 206 and existing > 0 else "wb"
            if mode == "wb" and existing > 0:
                log("  Server ignored Range; re-downloading from scratch")
                existing = 0
            total_header = response.headers.get("Content-Length")
            total = int(total_header) + (existing if status == 206 else 0) if total_header else None
            written = existing
            with dest.open(mode) as out:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
                    if total:
                        pct = 100.0 * written / total
                        log(
                            f"\r  {dest.name}: {human_bytes(written)} / "
                            f"{human_bytes(total)} ({pct:.1f}%)"
                        )
                    else:
                        log(f"\r  {dest.name}: {human_bytes(written)}")
            log("")
    except Exception as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def try_curl_download(url: str, dest: Path) -> bool:
    """Prefer curl when available (better resume / progress on huge files)."""
    curl = shutil.which("curl")
    if not curl:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    log(f"  curl → {dest}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def fetch(url: str, dest: Path) -> None:
    if try_curl_download(url, dest):
        return
    log("  curl unavailable or failed; using urllib resume downloader")
    download_file(url, dest)


# ---------------------------------------------------------------------------
# Extractors (mix-first)
# ---------------------------------------------------------------------------


def extract_zip(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Unzipping {archive.name} → {dest_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)


def extract_tar_mix_only(archive: Path, dest_dir: Path) -> int:
    """Extract only mix.flac / mix.wav / metadata.yaml from a Slakh-style tarball."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {"mix.flac", "mix.wav", "mixture.flac", "mixture.wav", "metadata.yaml"}
    extracted = 0
    log(f"  Tar mix-only extract from {archive.name} → {dest_dir}")
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base not in keep_names:
                continue
            rel = Path(member.name)
            parts = rel.parts
            if len(parts) > 1 and _looks_like_archive_root(parts[0]):
                target = dest_dir.joinpath(*parts[1:])
            else:
                target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            extracted += 1
            if extracted % 100 == 0:
                log(f"    extracted {extracted} mix/metadata files…")
    log(f"  Extracted {extracted} mix/metadata members (stems skipped)")
    return extracted


def _looks_like_archive_root(name: str) -> bool:
    """True for a single top-level package folder to strip (not split names)."""
    lower = name.lower().strip("/")
    if lower in {"train", "validation", "test", "omitted", "audio"}:
        return False
    return "slakh" in lower or lower.startswith("musdb") or lower.endswith(".tar")


def run_filter_mix_only(root: Path, dataset: str) -> None:
    script = Path(__file__).resolve().parent / "filter_mix_only.py"
    cmd = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--dataset",
        dataset,
        "--delete",
    ]
    log(f"  Pruning non-mix files: {root}")
    subprocess.run(cmd, check=True)


def flatten_if_single_child(dest_dir: Path) -> Path:
    """If zip extracted to a single top-level folder, return that folder for markers."""
    children = [p for p in dest_dir.iterdir() if not p.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest_dir


def is_ready(entry: dict, dest: Path) -> bool:
    if not dest.is_dir():
        return False
    markers = entry.get("ready_marker")
    if markers:
        return all((dest / name).is_dir() for name in markers) or all(
            (flatten_if_single_child(dest) / name).is_dir() for name in markers
        )
    pattern = entry.get("ready_marker_glob")
    if pattern:
        return any(dest.glob(pattern))
    return False


def resolve_ready_root(entry: dict, dest: Path) -> Path:
    if is_ready(entry, dest):
        return dest
    if dest.is_dir():
        nested = flatten_if_single_child(dest)
        if nested != dest and is_ready(entry, nested):
            return nested
    return dest


# ---------------------------------------------------------------------------
# Preprocess hooks
# ---------------------------------------------------------------------------


def maybe_preprocess(
    name: str,
    raw_root: Path,
    config: Path,
    *,
    reuse_existing: bool,
) -> None:
    if name == "musdb18-hq":
        script = "data/preprocess_musdb.py"
        root = raw_root
        flat = flatten_if_single_child(raw_root)
        if (flat / "train").is_dir():
            root = flat
        cmd = [
            sys.executable,
            script,
            "--data-root",
            str(root),
            "--output-dir",
            str(root / "processed"),
        ]
    elif name == "slakh2100":
        script = "data/preprocess_slakh.py"
        cmd = [
            sys.executable,
            script,
            "--data-root",
            str(raw_root),
            "--output-dir",
            str(raw_root / "processed"),
        ]
    elif name == "babyslakh":
        log("  BabySlakh is a 16 kHz smoke fixture and is not published into the release recipe")
        return
    elif name == "medleydb":
        script = "data/preprocess_medleydb.py"
        cmd = [
            sys.executable,
            script,
            "--data-root",
            str(raw_root),
            "--output-dir",
            str(raw_root / "processed"),
        ]
    else:
        return

    cmd.extend(["--config", str(config)])
    if reuse_existing:
        cmd.append("--reuse-existing")

    workspace = Path(__file__).resolve().parents[1]
    output_dir = cmd[cmd.index("--output-dir") + 1]
    log(f"  Preprocess {name} → {output_dir}")
    subprocess.run(cmd, check=True, cwd=str(workspace))


# ---------------------------------------------------------------------------
# Per-dataset pipelines
# ---------------------------------------------------------------------------


def prepare_dataset(
    name: str,
    dest: Path,
    cache_dir: Path,
    *,
    skip_existing: bool,
    keep_archive: bool,
    preprocess: bool,
    medleydb_url: str | None,
    config: Path,
) -> Path:
    if name not in REGISTRY:
        raise SystemExit(f"Unknown dataset: {name}. Choose from {list(REGISTRY)}")

    entry = REGISTRY[name]
    log(f"\n==> {name}  ({entry['license_note']})")
    log(f"    storage: {dest}")
    log(f"    size hint ~{entry['size_hint_gb']} GB download before mix filtering")

    dest.mkdir(parents=True, exist_ok=True)

    if skip_existing and is_ready(entry, dest):
        log(f"  Skip download: already ready at {dest}")
        ready = resolve_ready_root(entry, dest)
        if preprocess:
            maybe_preprocess(name, ready, config, reuse_existing=skip_existing)
        return ready

    if skip_existing and dest.is_dir():
        nested = flatten_if_single_child(dest)
        if nested != dest and is_ready(entry, nested):
            log(f"  Skip download: already ready at {nested}")
            if preprocess:
                maybe_preprocess(name, nested, config, reuse_existing=skip_existing)
            return nested

    url = entry.get("url")
    if name == "medleydb":
        url = medleydb_url or os.environ.get("MEDLEYDB_URL") or url
        if not url:
            log(
                "  MedleyDB is not auto-downloaded (access-controlled).\n"
                "  Options:\n"
                "    1) export MEDLEYDB_URL=https://…/your-approved-archive.zip\n"
                "    2) --medleydb-url URL\n"
                f"    3) Copy files into {dest} then re-run with --skip-existing\n"
            )
            if is_ready(entry, dest):
                log(f"  Found existing tree at {dest}")
                run_filter_mix_only(dest, "medleydb")
                if preprocess:
                    maybe_preprocess(name, dest, config, reuse_existing=skip_existing)
                return dest
            return dest

    assert url
    archive = cache_dir / entry["filename"]
    log(f"  Download cache → {archive}")
    fetch(url, archive)
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError(f"Archive missing or empty: {archive}")

    kind = entry["kind"]
    if kind == "tar_mix_only":
        count = extract_tar_mix_only(archive, dest)
        if count == 0:
            raise RuntimeError(
                f"No mix files extracted from {archive}. Archive layout may have changed."
            )
    elif kind in {"zip_then_filter", "manual_or_url"}:
        extract_zip(archive, dest)
        root = flatten_if_single_child(dest)
        filter_name = entry.get("filter_dataset")
        if filter_name:
            run_filter_mix_only(root, filter_name)
        if name == "musdb18-hq" and root != dest:
            for split in ("train", "test"):
                src = root / split
                if src.is_dir() and not (dest / split).exists():
                    shutil.move(str(src), str(dest / split))
    else:
        raise RuntimeError(f"Unhandled kind: {kind}")

    if not keep_archive:
        log(f"  Removing archive {archive} to free disk")
        archive.unlink(missing_ok=True)

    ready = resolve_ready_root(entry, dest)
    if not is_ready(entry, ready):
        log(f"  WARN: readiness markers not found under {dest}; check layout")
    else:
        log(f"  Ready: {ready}")

    if preprocess:
        maybe_preprocess(name, ready, config, reuse_existing=skip_existing)
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download mix-oriented datasets for SoundEx training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Path env vars: DATA_ROOT, CACHE_DIR, DOWNLOAD_CACHE_DIR,\n"
            "  MUSDB18_HQ_PATH, SLAKH2100_PATH, MEDLEYDB_PATH, BABYSLAKH_PATH"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="musdb18-hq",
        help="Comma-separated: musdb18-hq,slakh2100,babyslakh,medleydb",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/data")),
        help="Default parent for dataset dirs (default: $DATA_ROOT or /data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Archive download cache (default: $CACHE_DIR or $DATA_ROOT/.cache/downloads)",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Per-dataset storage directory (repeatable), e.g. musdb18-hq=/mnt/musdb",
    )
    parser.add_argument(
        "--musdb18-hq-path",
        type=Path,
        default=None,
        help="Storage path for MUSDB18-HQ (alias for --path musdb18-hq=…)",
    )
    parser.add_argument(
        "--slakh2100-path",
        type=Path,
        default=None,
        help="Storage path for Slakh2100",
    )
    parser.add_argument(
        "--medleydb-path",
        type=Path,
        default=None,
        help="Storage path for MedleyDB",
    )
    parser.add_argument(
        "--babyslakh-path",
        type=Path,
        default=None,
        help="Storage path for BabySlakh",
    )
    parser.add_argument(
        "--paths-manifest",
        type=Path,
        default=None,
        help=(
            "Write resolved paths YAML (default: $DATA_ROOT/soundex_data_paths.yaml). "
            "Pass empty string via env PATHS_MANIFEST= to disable."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep zip/tar after extract (default: delete to save disk)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run matching preprocess_*.py after each dataset is ready",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Training profile whose data.recipe drives preprocessing",
    )
    parser.add_argument("--medleydb-url", type=str, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        for key, entry in REGISTRY.items():
            env = PATH_ENV_KEYS.get(key, "")
            log(
                f"{key:12} ~{entry['size_hint_gb']:>4} GB  "
                f"env={env or '-'}  {entry['license_note']}"
            )
        return 0

    data_root = args.data_root.expanduser().resolve()
    cache_dir = resolve_cache_dir(args.cache_dir, data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cli_paths = parse_path_overrides(args.path)
    # Dedicated flags override --path for the same dataset
    for name, flag_val in (
        ("musdb18-hq", args.musdb18_hq_path),
        ("slakh2100", args.slakh2100_path),
        ("medleydb", args.medleydb_path),
        ("babyslakh", args.babyslakh_path),
    ):
        if flag_val is not None:
            cli_paths[name] = flag_val.expanduser()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not names:
        log("No datasets requested")
        return 1

    resolved: dict[str, Path] = {
        name: resolve_dataset_dest(name, data_root, cli_paths) for name in names
    }

    log(f"DATA_ROOT={data_root}")
    log(f"CACHE_DIR={cache_dir}")
    log("Dataset storage paths:")
    for name, dest in resolved.items():
        log(f"  {name}: {dest}")
    log(f"datasets={names}")

    ready_dirs: dict[str, Path] = {}
    for name in names:
        ready_dirs[name] = prepare_dataset(
            name,
            resolved[name],
            cache_dir,
            skip_existing=args.skip_existing,
            keep_archive=args.keep_archive,
            preprocess=args.preprocess,
            medleydb_url=args.medleydb_url,
            config=args.config.expanduser().resolve(),
        )

    # Manifest: default under data_root; disable with PATHS_MANIFEST=
    manifest_env = os.environ.get("PATHS_MANIFEST")
    if args.paths_manifest is not None:
        manifest_path: Path | None = args.paths_manifest.expanduser().resolve()
    elif manifest_env is not None:
        manifest_path = Path(manifest_env).expanduser().resolve() if manifest_env.strip() else None
    else:
        manifest_path = data_root / "soundex_data_paths.yaml"

    if manifest_path is not None:
        # Include all resolved storage dirs (not only ready) for training config sync
        write_paths_manifest(
            manifest_path,
            data_root=data_root,
            cache_dir=cache_dir,
            dataset_dirs=resolved,
        )

    log("\nAll requested dataset steps finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
