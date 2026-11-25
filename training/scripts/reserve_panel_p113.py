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


def reserve_panel(out, plan, *, works, tag, study, workers):
    """Verify `plan` rows offline, then reserve up to `works` isolated stereo works.

    Writes verified candidates, failures, the evaluation catalog, the remaining
    isolated eligible works (`unreserved-eligible.jsonl`) and a report into `out`.
    """
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_verify, plan, chunksize=4))
    verified = [r["ok"] for r in results if "ok" in r]
    failed = [r["failed"] for r in results if "failed" in r]
    with (out / "verified-candidates.jsonl").open("x") as f:
        for row in verified:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "verification-failures.json").write_text(json.dumps(failed, indent=1))
    report = reserve_verified(out, verified, works=works, tag=tag, study=study)
    report.update(candidates=len(plan), verification_failures=len(failed))
    (out / "reservation.json").write_text(json.dumps(report, indent=1))
    return report


def reserve_verified(out, verified, *, works, tag, study, accept=None):
    """Reserve from already-verified rows; `accept(row)` optionally filters the pool."""
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
        if accept is not None and not accept(r):
            continue
        try:
            pool.append({**r, "evaluation_rights_evidence": publisher_evidence(r)})
        except Exception as error:
            rights_failures.append({"id": r["id"], "reason": str(error)})
    count = len(pool)
    chosen, authors, genres = [], Counter(), Counter()
    while pool and len(chosen) < works:
        row = min(
            pool,
            key=lambda r: (
                genres[r["genre"]],
                authors[r["author"]],
                digest([tag, r["audio_sha256"]]),
            ),
        )
        chosen.append(row)
        authors[row["author"]] += 1
        genres[row["genre"]] += 1
        protected = closure(aliases(row))
        pool = [r for r in pool if not aliases(r) & protected]
    with (out / "evaluation-catalog.jsonl").open("x") as f:
        for row in chosen:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "unreserved-eligible.jsonl").open("x") as f:
        for row in pool:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "study": study,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verified": len(verified),
        "eligibility_exclusions": dict(Counter(e["reason"] for e in exclusions)),
        "rights_failures": rights_failures,
        "isolated_stereo_eligible": count,
        "reserved": len(chosen),
        "unreserved_eligible": len(pool),
        "genres": dict(genres),
        "authors": dict(authors),
        "history_files": [str(p) for p in histories],
    }
    (out / "reservation.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    a = parser.parse_args()
    approved = {r["id"] for r in jsonl(APPROVED)}
    plan = [r for r in jsonl(PLAN) if r["id"] not in approved and r["genre"] != "electronic"]
    report = reserve_panel(
        a.out, plan, works=WORKS, tag=TAG, study="p113-reserved-panel", workers=a.workers
    )
    keys = (
        "candidates",
        "verified",
        "verification_failures",
        "isolated_stereo_eligible",
        "reserved",
        "genres",
    )
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
