"""Draft per-work rights review records for screened expansion works (status: pending).

Records mirror the accepted library `source_review` fields, filled from publisher
metadata and the automated screen. They are drafts for maintainer review: nothing
here accepts a work. A maintainer copies accepted rows (status "accepted") into
`data/catalogs/expansion-review-acceptance.jsonl`. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import html
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.rights_screen import POLICY_VERSION, screen  # noqa: E402
from evaluation.longform_review import history_files, source_candidates  # noqa: E402
from scripts.run_envelope_diverse_p115 import EXPANSION, REPO  # noqa: E402
from scripts.run_envelope_p108 import jsonl  # noqa: E402


def clean(text, limit=400):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text or "")))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    verified = [r for p in EXPANSION for r in jsonl(p)]
    eligible, _ = source_candidates(verified, history_files(REPO / "data"))
    albums, rows = defaultdict(list), []
    for r in eligible:
        meta = json.loads(Path(r["source_metadata_path"]).read_text())
        if screen(r, meta):
            continue
        item = meta.get("metadata", {})
        rows.append(
            {
                "id": r["id"],
                "audio_sha256": r["audio_sha256"],
                "status": "pending",
                "scope": "commercial-codec-restoration-and-apache-2.0-weight-distribution",
                "grant_basis": f"Publisher item licence {item.get('licenseurl')} on {r['source_page']}",
                "credit_text": f"{r.get('recording_title') or r['id'].split('/')[-1]} by {r.get('author')}",
                "composition_basis": "Publisher original under the item licence; automated screen found "
                "no cover/remix/third-party indicators. Not manually verified.",
                "performers_and_samples_basis": "No sample/feature/Freesound indicators in metadata "
                f"({POLICY_VERSION}). Undisclosed samples cannot be excluded "
                "by metadata.",
                "unresolved_issues": [
                    "maintainer review of album credits and creator terms pending"
                ],
                "source_page": r["source_page"],
                "genre": r.get("genre"),
                "author": r.get("author"),
                "album": item.get("title"),
                "description_excerpt": clean(item.get("description")),
                "screen": POLICY_VERSION,
                "drafted": time.strftime("%Y-%m-%d"),
            }
        )
        albums[r["source_page"]].append(rows[-1])
    with a.out.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = [
        {
            "source_page": page,
            "album": items[0]["album"],
            "author": items[0]["author"],
            "genre": items[0]["genre"],
            "works": len(items),
            "description_excerpt": items[0]["description_excerpt"],
        }
        for page, items in sorted(albums.items(), key=lambda kv: -len(kv[1]))
    ]
    a.out.with_suffix(".albums.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(
        json.dumps(
            {
                "works": len(rows),
                "albums": len(summary),
                "authors": len({r["author"] for r in rows}),
            }
        )
    )


if __name__ == "__main__":
    main()
