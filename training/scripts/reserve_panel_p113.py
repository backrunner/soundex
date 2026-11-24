"""P113 panel: verify downloaded non-electronic originals and reserve an isolated panel.

Candidates are rows of `data/catalogs/new-acquisition-plan.jsonl` that are not
in the approved training catalog and are not electronic (P112 already covers
electronic). Each local file is verified offline with the acquisition audit
(publisher size/MD5, full decode, quality flags). Isolation and balancing follow
P88/P112 (connected file/PCM/work aliases against every history file; genre,
then author, then stable hash). The evaluation catalog is written before any
P113 training so that training excludes these works. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from evaluation.longform_review import (  # noqa: E402
    aliases,
    digest,
    history_files,
    publisher_evidence,
    source_candidates,
)
from scripts.fetch_archive_masters import verify_original  # noqa: E402
from scripts.run_band_training import jsonl  # noqa: E402

PLAN = REPO / "data/catalogs/new-acquisition-plan.jsonl"
APPROVED = REPO / "data/catalogs/current-approved.jsonl"
WORKS = 24
TAG = 113


def _verify(row):
    path = REPO / "data" / row["path"]
    try:
        return {"ok": verify_original(path, row)}
    except Exception as error:  # Every failure is retained with its reason.
        return {"failed": {"id": row["id"], "reason": f"{type(error).__name__}: {error}"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    approved = {r["id"] for r in jsonl(APPROVED)}
    plan = [r for r in jsonl(PLAN) if r["id"] not in approved and r["genre"] != "electronic"]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(_verify, plan, chunksize=4))
    verified = [r["ok"] for r in results if "ok" in r]
    failed = [r["failed"] for r in results if "failed" in r]
    with (a.out / "verified-candidates.jsonl").open("x") as f:
        for row in verified:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (a.out / "verification-failures.json").write_text(json.dumps(failed, indent=1))

    data = REPO / "data"
    histories = history_files(data)
    inventory = sorted(set(histories) | set((data / "catalogs").glob("*.jsonl")))
    universe = list({frozenset(aliases(r)): r for p in inventory for r in jsonl(p)}.values())
    universe += verified
    bridges = {frozenset(aliases(r)) for r in universe}

    def closure(keys):
        blocked, before = set(keys), -1
        while before != len(blocked):
            before = len(blocked)
            for component in bridges:
                if component & blocked:
                    blocked.update(component)
        return blocked

    eligible, exclusions = source_candidates(verified, histories)
    used = closure({k for p in histories for r in jsonl(p) for k in aliases(r)})
    pool, rights_failures = [], []
    for r in eligible:
        if r["source_quality"]["channels"] != 2 or aliases(r) & used:
            continue
        try:
            pool.append({**r, "evaluation_rights_evidence": publisher_evidence(r)})
        except Exception as error:
            rights_failures.append({"id": r["id"], "reason": str(error)})
    count = len(pool)
    chosen, authors, genres = [], Counter(), Counter()
    while pool and len(chosen) < WORKS:
        row = min(
            pool,
            key=lambda r: (
                genres[r["genre"]],
                authors[r["author"]],
                digest([TAG, r["audio_sha256"]]),
            ),
        )
        chosen.append(row)
        authors[row["author"]] += 1
        genres[row["genre"]] += 1
        protected = closure(aliases(row))
        pool = [r for r in pool if not aliases(r) & protected]
    with (a.out / "evaluation-catalog.jsonl").open("x") as f:
        for row in chosen:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "study": "p113-reserved-panel",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidates": len(plan),
        "verified": len(verified),
        "verification_failures": len(failed),
        "eligibility_exclusions": dict(Counter(e["reason"] for e in exclusions)),
        "rights_failures": rights_failures,
        "isolated_stereo_eligible": count,
        "reserved": len(chosen),
        "genres": dict(genres),
        "authors": dict(authors),
        "history_files": [str(p) for p in histories],
    }
    (a.out / "reservation.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "candidates",
                    "verified",
                    "verification_failures",
                    "isolated_stereo_eligible",
                    "reserved",
                    "genres",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
