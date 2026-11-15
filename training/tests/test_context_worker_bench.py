"""Check independent availability accounting and failure retention.

SPDX-License-Identifier: Apache-2.0
"""

import pytest

from scripts.bench_context_worker import HOPS, summarize


def report(missed=()):
    count, run, longest = 0, 0, 0
    events = []
    for hop in range(HOPS):
        bit = hop in missed
        count += bit
        run = run + 1 if bit else 0
        longest = max(longest, run)
        events.append(
            dict(
                hop=hop,
                worker_presentation_missed=bit,
                deadline_misses=count,
                max_consecutive_deadline_misses=longest,
                callback_ns=1000,
                callback_start_late_ns=2000,
            )
        )
    return dict(
        events=events,
        deadline_misses=count,
        max_consecutive_deadline_misses=longest,
        sample_rate=48000,
        max_processing_ns=100000,
        max_queue_wait_ns=10000,
        state_history_warmup_hops=80,
        processing_period_overruns=0,
        finite=True,
        worker_drain_complete=True,
        worker_failed=False,
        output_frames=HOPS * 128,
        fixed_delay_frames=256,
    )


def test_warmup_does_not_hide_availability_failures():
    result = summarize(report(missed=(50, 51)))
    assert result["misses"] == 0
    assert result["normal_miss_ratio"] == 2 / 1098
    assert not result["software_screen_pass"]
    result = summarize(report(missed=(99, 100)))
    assert result["max_miss_run"] == 1
    assert result["whole_run_max_miss_run"] == 2


def test_raw_bits_detect_inconsistent_counters():
    data = report(missed=(300,))
    data["events"][300]["deadline_misses"] = 0
    with pytest.raises(ValueError, match="raw presentation"):
        summarize(data)
    data = report(missed=(300,))
    data["max_consecutive_deadline_misses"] = 0
    with pytest.raises(ValueError, match="whole-run"):
        summarize(data)


def test_final_drained_worker_overrun_cannot_disappear():
    data = report()
    assert summarize(data)["software_screen_pass"]
    # A worker call may complete after the last callback snapshot.
    data["processing_period_overruns"] = 1
    assert not summarize(data)["software_screen_pass"]
    data["processing_period_overruns"] = 0
    data["worker_drain_complete"] = False
    assert not summarize(data)["software_screen_pass"]
