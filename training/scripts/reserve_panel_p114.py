"""P114 panel: verify the unapproved electronic downloads and reserve a fresh electronic panel.

Same offline audit, isolation and balancing as P113 (`reserve_panel`), restricted
to electronic rows of `data/catalogs/new-acquisition-plan.jsonl` that are not in
the approved training catalog. Run before any electronic-targeted change is
designed, so the panel cannot be chosen around it. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reserve_panel_p113 import APPROVED, PLAN, jsonl, reserve_panel  # noqa: E402

WORKS = 16
TAG = 114


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    a = parser.parse_args()
    approved = {r["id"] for r in jsonl(APPROVED)}
    plan = [r for r in jsonl(PLAN) if r["id"] not in approved and r["genre"] == "electronic"]
    report = reserve_panel(
        a.out, plan, works=WORKS, tag=TAG, study="p114-electronic-panel", workers=a.workers
    )
    keys = (
        "candidates",
        "verified",
        "verification_failures",
        "isolated_stereo_eligible",
        "reserved",
        "unreserved_eligible",
        "authors",
    )
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
