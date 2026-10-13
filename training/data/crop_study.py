"""P13 bounded crop coverage with a reserved same-recording evaluation window."""

import numpy as np

from data.causal_study import SAMPLES
from models.fit_diagnostic import BURN

# Conservatively includes more target support than the loss (which begins at frame 89).
SUPERVISION_START = BURN * 128


def protected_probes(anchors, rows):
    """Prefer the farthest endpoint whose full input has no anchor supervision."""
    probes = []
    for anchor in anchors:
        maximum = rows[anchor["row_id"]]["length_samples"] - SAMPLES
        allowed = [
            start
            for start in (0, maximum)
            if anchor["start"] + SAMPLES <= start
            or anchor["start"] + SUPERVISION_START >= start + SAMPLES
        ]
        if not allowed:
            raise ValueError("no protected same-recording probe")
        probes.append({**anchor, "start": max(allowed, key=lambda s: abs(s - anchor["start"]))})
    return probes


def crop_bank(anchors, reserved, rows, positions):
    if positions < 1 or len(anchors) != len(reserved):
        raise ValueError("invalid crop bank geometry")
    bank, audit = [], []
    for anchor, probe in zip(anchors, reserved, strict=True):
        if anchor["row_id"] != probe["row_id"]:
            raise ValueError("reserved recording mismatch")
        maximum = rows[anchor["row_id"]]["length_samples"] - SAMPLES
        begin, end = probe["start"], probe["start"] + SAMPLES
        # Integer start intervals: training's conservative target support must
        # be wholly before or after the entire reserved evaluation input.
        intervals = [(0, min(maximum, begin - SAMPLES)), (max(0, end - SUPERVISION_START), maximum)]
        intervals = [(a, b) for a, b in intervals if a <= b]
        start = anchor["start"]
        if not any(a <= start <= b for a, b in intervals):
            raise ValueError("anchor supervision overlaps reserved evaluation input")
        # Remove anchor before selecting evenly spaced quantiles, with no
        # outcome-dependent selection and no duplicate starts.
        available = []
        for a, b in intervals:
            available.extend([(a, min(b, start - 1)), (max(a, start + 1), b)])
        available = [(a, b) for a, b in available if a <= b]
        count = sum(b - a + 1 for a, b in available)
        if count < positions - 1:
            raise ValueError("insufficient protected crop positions")
        starts = [start]
        for i in range(positions - 1):
            rank = (2 * i + 1) * count // (2 * (positions - 1))
            for a, b in available:
                length = b - a + 1
                if rank < length:
                    starts.append(a + rank)
                    break
                rank -= length
        bank.extend({**anchor, "start": s} for s in starts)
        audit.append(
            {
                "row_id": anchor["row_id"],
                "reserved_input": [begin, end],
                "allowed_start_intervals_inclusive": intervals,
                "starts": starts,
                "input_context_overlaps_reserved_input": [
                    s < end and s + SAMPLES > begin for s in starts
                ],
                "supervision_overlaps_reserved_input": [
                    s + SUPERVISION_START < end and s + SAMPLES > begin for s in starts
                ],
            }
        )
    return bank, audit


def crop_draws(draws, positions, seed, size):
    """Separate RNG preserves the exact recording schedule and torch RNG."""
    if positions < 1 or size < 1:
        raise ValueError("invalid crop schedule geometry")
    rng = np.random.default_rng(seed ^ 0xC013)
    queues = [[] for _ in range(size)]
    result = []
    for draw in draws:
        chosen = []
        for i in draw:
            if not 0 <= i < size:
                raise ValueError("recording index outside pool")
            if not queues[i]:
                queues[i] = rng.permutation(positions).tolist()
            chosen.append(i * positions + queues[i].pop())
        result.append(chosen)
    return result
