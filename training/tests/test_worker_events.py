"""Trace reconciliation must expose missing evidence and causal ambiguity."""

import copy

import pytest

from evaluation.worker_events import analyze, late_recovery


def report():
    events = []
    for i in range(4):
        events.append(
            dict(
                sequence=i,
                callback_begin_ns=1000 * i + 1,
                callback_decision_ns=1000 * i + 10,
                callback_ready=1 if i >= 2 else 2,
                submitted_ns=1000 * i + 100,
                enqueued=1,
                taken_ns=1000 * i + 110,
                started_ns=1000 * i + 120,
                finished_ns=1000 * i + 420,
                terminal_ns=1000 * i + 430,
                worker_outcome=1,
                history_warming=1,
                history_reset=2,
                consumed_outcome=3 if i == 0 else 1,
            )
        )
    return dict(
        worker_trace=dict(format="soundex-worker-events-v1", ignored_writes=0, events=events),
        events=[
            dict(
                accepted_hops=max(0, i - 1),
                worker_presentation_missed=False,
                nominal_arrival_trace_ns=1000 * i + 1,
                callback_start_late_ns=0,
            )
            for i in range(4)
        ],
        processed_hops=4,
        discarded_input_hops=0,
        discarded_output_hops=0,
        rejected_input_hops=0,
        state_history_warmup_hops=4,
        deadline_misses=0,
        max_processing_ns=300,
        max_queue_wait_ns=20,
    )


def test_counter_crosscheck_rejects_missing_or_truncated_evidence():
    data = report()
    assert analyze(data)["misses"] == []
    bad = copy.deepcopy(data)
    bad["processed_hops"] = 3
    with pytest.raises(ValueError, match="discrepancy"):
        analyze(bad)
    bad = copy.deepcopy(data)
    bad["worker_trace"]["ignored_writes"] = 1
    with pytest.raises(ValueError, match="truncated"):
        analyze(bad)


def test_actual_processing_overlaps_callback_decision():
    data = report()
    packet = data["worker_trace"]["events"][1]
    packet.update(
        started_ns=1900, finished_ns=2200, terminal_ns=2210, worker_outcome=4, consumed_outcome=0
    )
    data["worker_trace"]["events"][2]["callback_ready"] = 2
    data["events"][2]["worker_presentation_missed"] = True
    data["events"][-1]["accepted_hops"] = 1
    data.update(deadline_misses=1, discarded_output_hops=1, max_queue_wait_ns=800)
    result = analyze(data)
    assert result["misses"][0]["input_sequence"] == 1
    assert result["misses"][0]["category"] == "processing_not_finished_by_callback_decision"
    # The elapsed wall time alone does not establish thread CPU saturation.
    assert "thread CPU" in result["scope"]


def test_completed_compute_does_not_prove_result_was_visible_to_consumer():
    data = report()
    data["worker_trace"]["events"][2]["callback_ready"] = 2
    data["events"][2]["worker_presentation_missed"] = True
    data["events"][-1]["accepted_hops"] = 1
    data["deadline_misses"] = 1
    packet = data["worker_trace"]["events"][1]
    packet.update(terminal_ns=2012, consumed_outcome=2)
    result = analyze(data)
    assert result["misses"][0]["category"].endswith("unresolved")


def fault_report():
    """Minimal independent fault/recovery evidence, not trace reconciliation."""
    events = [
        dict(
            sequence=i,
            callback_begin_ns=i * 1000 + 1,
            callback_ready=1,
            wet_enabled=1,
            history_warming=2,
            history_reset=2,
        )
        for i in range(1100)
    ]
    callbacks = [dict(hop=i, worker_presentation_missed=False) for i in range(1100)]
    dry = []
    for start, periods in ((256, 2), (640, 4)):
        callbacks[start].update(
            injected_stall_hops=periods, requested_arrival_trace_ns=start * 1000
        )
        for i in range(start + 1, start + periods + 1):
            callbacks[i]["worker_presentation_missed"] = True
            events[i]["callback_ready"] = 2
        reset = start + periods
        events[reset]["history_reset"] = 1
        for i in range(reset, reset + 80):
            events[i]["history_warming"] = 1
        dry.append(dict(hop=reset + 3, peak=0.0))
    return dict(
        worker_trace=dict(events=events),
        events=callbacks,
        late_producer_faults=True,
        warmup_dry_audit=dict(passed=True, hops=dry),
        finite=True,
        worker_drain_complete=True,
        worker_failed=False,
        fixed_delay_frames=256,
        output_frames=128 * 1100,
    )


def test_late_recovery_requires_observed_fault_and_fully_warmed_acceptance():
    data = fault_report()
    result = late_recovery(data)
    assert result["fault_handling_pass"]
    assert [s["fully_warmed_accepted_hop"] for s in result["segments"]] == [339, 725]
    # Accepted dry output alone cannot count as recovered enhancement availability.
    for event in data["worker_trace"]["events"][640:]:
        event["history_warming"] = 1
    assert not late_recovery(data)["fault_handling_pass"]


@pytest.mark.parametrize("missing", ["alignment", "schedule", "reset", "observed", "frames"])
def test_late_recovery_does_not_silently_pass_missing_evidence(missing):
    data = fault_report()
    if missing == "alignment":
        data["warmup_dry_audit"]["passed"] = False
    elif missing == "schedule":
        data["events"][640]["injected_stall_hops"] = 3
        with pytest.raises(ValueError, match="schedule"):
            late_recovery(data)
        return
    elif missing == "reset":
        data["worker_trace"]["events"][644]["history_reset"] = 2
    elif missing == "observed":
        data["events"][640]["requested_arrival_trace_ns"] = 640002
    else:
        data["output_frames"] -= 128
    assert not late_recovery(data)["fault_handling_pass"]
