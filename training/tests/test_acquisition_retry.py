# SPDX-License-Identifier: Apache-2.0
"""A catch-up pass includes late failures but does not repeatedly fetch quality holds."""

import json
from pathlib import Path

from scripts.retry_failed_acquisition import unresolved_entries


def test_successful_retry_and_signal_hold_do_not_hide_late_download_failure(tmp_path: Path) -> None:
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps({"id": i}) + "\n" for i in ("held", "retried", "late")))
    first, retry = tmp_path / "first", tmp_path / "retry"
    first.mkdir()
    retry.mkdir()
    (first / "progress.json").write_text(
        json.dumps({"errors": [{"id": "retried"}, {"id": "late"}]})
    )
    (first / "acquired.jsonl").write_text(
        json.dumps({"id": "held", "path": "held.flac", "review_flags": ["large_dc_offset"]})
    )
    (retry / "progress.json").write_text(json.dumps({"errors": []}))
    (retry / "acquired.jsonl").write_text(json.dumps({"id": "retried", "path": "retried.wav"}))
    assert [e["id"] for e in unresolved_entries(plan, [first, retry])] == ["late"]
