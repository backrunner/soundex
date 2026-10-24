"""Source-balanced long-form diagnostics, separate from training and release gates.

SPDX-License-Identifier: Apache-2.0
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from evaluation.fidelity_calibration import pcm_statistics
from models.auraloss_spectrum import spectral_loss as auraloss_loss
from models.complex_synthesis import complex_features, synthesize
from models.multires_spectrum import WEIGHTS, spectral_loss

SEED = 20261024
BANDS = ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, float("inf")))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def aliases(row):
    """Compare file hashes across schemas, decoded PCM and supplied work aliases."""
    result = set()
    for item in (row, row.get("source_metadata", {}), row.get("source_quality", {})):
        for key in ("audio_sha256", "source_checksum", "decoded_pcm_sha256"):
            if item.get(key):
                result.add(("hash", item[key]))
        for key in ("split_group", "work_id", "work_group"):
            if item.get(key):
                result.add(("work", item[key]))
        review = item.get("source_review", {})
        if review.get("work_group"):
            result.add(("work", review["work_group"]))
    return result


def history_files(root):
    """Exclude source-code snapshots; include historical run-specific catalogs."""
    paths = []
    for folder in ("processed", "runs", "archive/runs"):
        for path in (root / folder).rglob("*.jsonl"):
            relative = path.relative_to(root)
            if any(p in {"source", "source_snapshot", ".venv"} for p in relative.parts):
                continue
            if path.name == "manifest.jsonl" or "catalog" in path.name or "catalogs" in path.parts:
                paths.append(path)
    return sorted(set(paths))


def source_candidates(catalog, histories):
    """Conservative exclusion: even unused historical run catalogs count as exposed."""
    used = set()
    for path in histories:
        for line in path.read_text().splitlines():
            if line.strip():
                used.update(aliases(json.loads(line)))
    # A catalog row can bridge a known hash to another work alias. Exclude the
    # entire connected component, not just direct matches with the old manifests.
    previous = -1
    while previous != len(used):
        previous = len(used)
        for row in catalog:
            keys = aliases(row)
            if keys & used:
                used.update(keys)
    candidates, exclusions = [], []
    for row in catalog:
        quality = row.get("source_quality", {})
        reason = None
        if aliases(row) & used:
            reason = "historical_source_or_work_alias"
        elif quality.get("review_flags") or not quality:
            reason = "unresolved_source_quality"
        elif quality.get("sample_rate", 0) < 44100 or quality.get("duration_seconds", 0) < 35:
            reason = "insufficient_rate_or_duration"
        elif quality.get("channels") not in (1, 2):
            reason = "unsupported_channels"
        elif not row.get("upstream_original_md5_verified"):
            reason = "original_checksum_not_verified"
        elif not row.get("source_metadata_path") or not Path(row["source_metadata_path"]).is_file():
            reason = "missing_publisher_evidence"
        elif not row.get("path") or not Path(row["path"]).is_file():
            reason = "missing_audio"
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            candidates.append(row)
    return candidates, exclusions


def license_key(url):
    parsed = urlparse(url)
    if parsed.hostname not in {"creativecommons.org", "www.creativecommons.org"}:
        raise ValueError("unsupported publisher license host")
    path = parsed.path.rstrip("/")
    if path not in {"/licenses/by/3.0", "/licenses/by/4.0", "/publicdomain/zero/1.0"}:
        raise ValueError("license requires separate review")
    return path


def publisher_evidence(row):
    """Bind existing publisher metadata, grant and the exact original file entry."""
    metadata = json.loads(Path(row["source_metadata_path"]).read_text())
    expected = license_key(row["license_url"])
    urls = metadata.get("metadata", {}).get("licenseurl", [])
    if isinstance(urls, str):
        urls = [urls]
    if expected not in [license_key(url) for url in urls]:
        raise ValueError("publisher/catalog license mismatch")
    filename = row["id"].split("/", 1)[1]
    file = next((f for f in metadata.get("files", []) if f["name"] == filename), None)
    if not file or file.get("source") != "original":
        raise ValueError("missing publisher original file")
    if file.get("md5") != row.get("publisher_original_md5"):
        raise ValueError("publisher original checksum mismatch")
    return {"license": expected, "original_file": file, "source_page": row["source_page"]}


def select_sources(candidates, *, per_genre=4, per_author=3):
    """Fixed rare-genre-first/hash order; no waveform or candidate-score selection."""
    groups = defaultdict(list)
    for row in candidates:
        groups[row.get("genre", "unlabeled")].append(row)
    chosen, authors, seen = [], Counter(), set()
    for genre in sorted(groups, key=lambda g: (len(groups[g]), g)):
        count = 0
        for row in sorted(groups[genre], key=lambda r: digest([SEED, r["audio_sha256"]])):
            author = row.get("author", "unknown")
            if authors[author] >= per_author or aliases(row) & seen:
                continue
            chosen.append(row)
            seen.update(aliases(row))
            authors[author] += 1
            count += 1
            if count == per_genre:
                break
    return chosen


def raw_from_trace(traces, channels):
    values = np.asarray([t["prediction"] for t in traces], dtype=np.float32)
    if values.shape != (len(traces), channels * 258) or not np.isfinite(values).all():
        raise ValueError("invalid Rust prediction trace")
    features = torch.from_numpy(values.reshape(-1, channels, 2, 129)).permute(1, 2, 0, 3)
    return synthesize(complex_features(features)).T.numpy()


def aligned_output(wave, score_start, length, *, delay=128):
    """Rust diagnostic OLA has 128-sample delay; the 256-frame adapter is separate."""
    start = score_start + delay
    if wave.ndim != 2 or start < 0 or start + length > len(wave):
        raise ValueError("incomplete output scoring support")
    return wave[start : start + length]


@torch.no_grad()
def score_audio(wave, reference, rate):
    if wave.shape != reference.shape or wave.ndim != 2 or wave.shape[1] not in (1, 2):
        raise ValueError("aligned physical channels required")
    if len(wave) != rate * 10 or not np.isfinite(wave).all() or not np.isfinite(reference).all():
        raise ValueError("ten finite seconds per score required")
    prediction = torch.from_numpy(np.ascontiguousarray(wave.T)).float()
    target = torch.from_numpy(np.ascontiguousarray(reference.T)).float()
    rates = torch.full((len(prediction),), rate)
    result = spectral_loss(prediction, target, rates)
    scores = {"mr": float(result["per_example"].mean())}
    scores.update({name: float(value.mean()) for name, value in result["components"].items()})
    scores["auraloss"] = float(auraloss_loss(prediction, target, rates)["per_example"].mean())
    scores.update(
        {
            k: float(v.mean())
            for k, v in pcm_statistics(prediction.double() - target.double(), rates).items()
        }
    )
    for b, (low, _) in enumerate(BANDS):
        for name in WEIGHTS:
            scores[f"band{low}_{name}"] = float(
                torch.stack([r["bands"][b][name] for r in result["resolutions"]]).mean()
            )
        scores[f"band{low}_mr"] = sum(WEIGHTS[k] * scores[f"band{low}_{k}"] for k in WEIGHTS)
    for name in (*WEIGHTS, "mr"):
        scores["hf8k_" + name] = float(
            np.mean([scores[f"band{low}_{name}"] for low in (8000, 12000, 16000)])
        )
    # Rectangular whole-excerpt Parseval rulers in absolute PCM RMS, per channel.
    reference_fft = np.fft.rfft(reference.astype(np.float64), axis=0) / len(reference)
    error_fft = np.fft.rfft(wave.astype(np.float64) - reference, axis=0) / len(reference)
    hz = np.fft.rfftfreq(len(reference), 1 / rate)
    sides = np.full(len(hz), 2.0)
    sides[0] = sides[-1] = 1.0
    for low, high in BANDS:
        mask = (hz >= low) & (hz < high)
        for label, fft in (("reference", reference_fft), ("error", error_fft)):
            rms = np.sqrt((np.abs(fft[mask]) ** 2 * sides[mask, None]).sum(0))
            scores[f"band{low}_{label}_rms"] = float(rms.mean())
    if wave.shape[1] == 2:
        for name, sign in (("mid", 1), ("side", -1)):
            error = (
                (wave[:, 0].astype(np.float64) - reference[:, 0])
                + sign * (wave[:, 1].astype(np.float64) - reference[:, 1])
            ) / 2
            scores[name + "_error_rms"] = float(np.sqrt(np.mean(error**2)))
    return scores


def average(values):
    keys = set.intersection(*(set(v) for v in values))
    return {key: float(np.mean([v[key] for v in values])) for key in sorted(keys)}


def aggregate(records):
    """Never count rates, codecs, channels or excerpts as independent recordings."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["path"], record["role"])].append(record)
    results = {}
    for (path, role), rows in groups.items():
        sources = defaultdict(list)
        for row in rows:
            sources[row["source"]].append(row["scores"])
        means = {key: average(value) for key, value in sorted(sources.items())}
        results[f"{path}/{role}"] = {
            "sources": len(sources),
            "cases": len(rows),
            "source_mean": average(list(means.values())),
            "clip_mean": average([r["scores"] for r in rows]),
            "per_source": means,
        }
    comparisons = {}
    for key, value in results.items():
        path, role = key.rsplit("/", 1)
        for base in ("native/legacy", "untreated"):
            baseline = results.get(base + "/" + role)
            if baseline is None or base == path:
                continue
            if value["per_source"].keys() != baseline["per_source"].keys():
                raise ValueError("unmatched sources in comparison")
            names = sorted(value["per_source"])
            rng = np.random.default_rng(SEED)
            indices = rng.integers(len(names), size=(5000, len(names)))
            delta = {}
            for metric in value["source_mean"]:
                diffs = np.asarray(
                    [
                        value["per_source"][n][metric] - baseline["per_source"][n][metric]
                        for n in names
                    ]
                )
                delta[metric] = {
                    "mean_delta": float(diffs.mean()),
                    "percentile_95": np.quantile(diffs[indices].mean(1), [0.025, 0.975]).tolist(),
                    "better": int((diffs < 0).sum()),
                    "worse": int((diffs > 0).sum()),
                    "ties": int((diffs == 0).sum()),
                }
            comparisons[key + " versus " + base] = delta
    return {
        "groups": results,
        "comparisons": comparisons,
        "uncertainty_scope": "Paired source bootstrap, 5000 replicates, descriptive marginal 95% intervals; no multiplicity-adjusted or universal superiority claim. Source/work isolation is not artist isolation.",
    }
