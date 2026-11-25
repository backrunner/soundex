"""P115 multi-genre panel: rights-screened, isolated works from the P113 audit.

Reserves up to 24 stereo works from the verified non-electronic candidates that
pass `data.rights_screen` (v3), after P88 isolation against every history file
and catalog (including the P112-P114 panels). Written before P115 training so
training excludes it. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.rights_screen import POLICY_VERSION, screen  # noqa: E402
from scripts.reserve_panel_p113 import REPO, jsonl, reserve_verified  # noqa: E402

SOURCE = REPO / "data/runs/reserved-panel-p113-20261124/verified-candidates.jsonl"


def passes(row):
    return not screen(row, json.loads(Path(row["source_metadata_path"]).read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = reserve_verified(
        a.out, jsonl(SOURCE), works=24, tag=115, study="p115-multigenre-panel", accept=passes
    )
    report["rights_screen"] = POLICY_VERSION
    (a.out / "reservation.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "isolated_stereo_eligible",
                    "reserved",
                    "unreserved_eligible",
                    "genres",
                    "authors",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
