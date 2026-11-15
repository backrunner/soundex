"""Reconcile bounded worker events with native counters; retain every missed hop."""

from collections import Counter


def analyze(report):
    trace = report["worker_trace"]
    events, callbacks = trace["events"], report["events"]
    if trace["format"] != "soundex-worker-events-v1" or trace["ignored_writes"]:
        raise ValueError("missing or truncated event trace")
    if len(events) != len(callbacks) or [e["sequence"] for e in events] != list(range(len(events))):
        raise ValueError("event timeline mismatch")
    processed = [e for e in events if e["worker_outcome"] in (1, 4, 5)]
    counts = Counter(e["worker_outcome"] for e in events)
    if counts[0] != sum(e["enqueued"] == 2 for e in events):
        raise ValueError("enqueued input lacks a terminal event")
    assertions = (
        (len(processed), report["processed_hops"]),
        (counts[2] + counts[3], report["discarded_input_hops"]),
        (counts[4] + counts[5], report["discarded_output_hops"]),
        (sum(e["enqueued"] == 2 for e in events), report["rejected_input_hops"]),
        (sum(e["history_warming"] == 1 for e in processed), report["state_history_warmup_hops"]),
        (sum(e["callback_ready"] == 1 for e in events), callbacks[-1]["accepted_hops"]),
        (sum(e["callback_ready"] == 2 for e in events[2:]), report["deadline_misses"]),
        (max(e["finished_ns"] - e["started_ns"] for e in processed), report["max_processing_ns"]),
        (max(e["started_ns"] - e["submitted_ns"] for e in processed), report["max_queue_wait_ns"]),
    )
    if any(a != b for a, b in assertions):
        raise ValueError(f"event/counter discrepancy: {assertions}")
    for e in events:
        if not (0 < e["callback_begin_ns"] <= e["callback_decision_ns"] <= e["submitted_ns"]):
            raise ValueError("callback clock/order invalid")
        if e["enqueued"] not in (1, 2) or e["callback_ready"] not in (1, 2):
            raise ValueError("missing callback event")
    for e in processed:
        if not (
            0
            < e["submitted_ns"]
            <= e["taken_ns"]
            <= e["started_ns"]
            <= e["finished_ns"]
            <= e["terminal_ns"]
        ):
            raise ValueError("worker clock/order invalid")
    misses = []
    for hop in range(2, len(events)):
        callback, packet, host = events[hop], events[hop - 1], callbacks[hop]
        if (callback["callback_ready"] == 2) != host["worker_presentation_missed"]:
            raise ValueError("trace and raw presentation bit disagree")
        if callback["callback_ready"] == 1:
            continue
        start, finish, decision = (
            packet["started_ns"],
            packet["finished_ns"],
            callback["callback_decision_ns"],
        )
        if packet["enqueued"] == 2:
            category = "input_queue_rejected"
        elif packet["worker_outcome"] == 2:
            category = "input_superseded"
        elif packet["worker_outcome"] == 3:
            category = "input_expired_before_processing"
        elif packet["consumed_outcome"] == 3:
            category = "invalid_continuity_after_gap"
        elif start == 0 or start >= decision:
            category = "worker_not_started_by_callback_decision"
        elif finish >= decision:
            category = "processing_not_finished_by_callback_decision"
        else:
            category = "finished_but_publication_or_consumer_window_unresolved"
        misses.append(
            {
                "hop": hop,
                "input_sequence": hop - 1,
                "category": category,
                "available_until_callback_begin_ns": callback["callback_begin_ns"]
                - packet["submitted_ns"],
                "available_until_callback_decision_ns": decision - packet["submitted_ns"],
                "submitted_after_nominal_presentation": packet["submitted_ns"]
                > host["nominal_arrival_trace_ns"],
                "producer_start_late_ns": host["callback_start_late_ns"],
                "packet": packet,
                "callback": callback,
            }
        )
    return {
        "event_counters_match": True,
        "miss_categories": dict(Counter(m["category"] for m in misses)),
        "misses": misses,
        "history_reset_inputs": [e["sequence"] for e in processed if e["history_reset"] == 1],
        "scope": "Actual input/presentation sequence correspondence. Wall events do not isolate thread CPU or OS root cause. Terminal timestamp follows publication/expiry; overlapping consumer windows remain explicitly unresolved. Warmup dry output is distinct from missed presentation and fixed delay.",
    }
