"""Recording-balanced, source-time-isolated schedules and bounded clip loading."""

import math
from collections import OrderedDict, defaultdict
from fractions import Fraction

import numpy as np

from data.causal_study import SAMPLES, digest, source_origin
from data.coverage_study import rng_for
from data.envelope_study import EnvelopeClips
from data.weak_band_study import audit_groups, identity_specs
from models.envelope_probe import context_features
from models.fit_diagnostic import FixedFitObjective


def protected_intervals(specs, rows):
    """Source seconds, shared across codecs, sample rates and all channel views."""
    result = defaultdict(list)
    for spec in specs:
        row = rows[spec["row_id"]]
        start = source_origin(row) + spec["start"]
        result[row["source_checksum"]].append(
            (Fraction(start, row["sample_rate"]), Fraction(start + SAMPLES, row["sample_rate"]))
        )
    return result


def allowed_starts(row, protected):
    """Inclusive integer spans; the entire input including warmup must be disjoint."""
    maximum = row["length_samples"] - SAMPLES
    if maximum < 0:
        return []
    spans = [(0, maximum)]
    for left, right in protected.get(row["source_checksum"], []):
        # Touching endpoints are safe. Fractions prevent cross-rate rounding leaks.
        lo = math.floor(left * row["sample_rate"] - source_origin(row) - SAMPLES) + 1
        hi = math.ceil(right * row["sample_rate"] - source_origin(row)) - 1
        remaining = []
        for a, b in spans:
            if hi < a or lo > b:
                remaining.append((a, b))
            else:
                if a < lo:
                    remaining.append((a, lo - 1))
                if hi < b:
                    remaining.append((hi + 1, b))
        spans = remaining
    return spans


def assert_isolated(specs, rows, protected):
    for spec in specs:
        row = rows[spec["row_id"]]
        if row["split"] != "train" or row["held_out"]:
            raise ValueError("non-training recording in schedule")
        if not any(a <= spec["start"] <= b for a, b in allowed_starts(row, protected)):
            raise ValueError(f"protected source-time overlap or invalid crop: {spec}")


def repair_finite_bank(specs, rows, protected):
    """Preserve valid starts, move overlapping warmups to nearest unused legal start.

    Geometry only, before training; preserve row, recording, codec, view and draw index.
    """
    result = [dict(s) for s in specs]
    used, invalid = defaultdict(set), []
    for i, spec in enumerate(specs):
        spans = allowed_starts(rows[spec["row_id"]], protected)
        if any(a <= spec["start"] <= b for a, b in spans):
            if spec["start"] in used[spec["row_id"]]:
                raise ValueError("duplicate finite-bank crop")
            used[spec["row_id"]].add(spec["start"])
        else:
            invalid.append((i, spans))
    repairs = []
    for i, spans in invalid:
        spec = result[i]
        occupied = used[spec["row_id"]]
        candidates = set()
        for a, b in spans:
            nearest = min(b, max(a, spec["start"]))
            for delta in range(len(occupied) + 1):
                candidates.update(p for p in (nearest - delta, nearest + delta) if a <= p <= b)
        candidates -= occupied
        if not candidates:
            raise ValueError("finite-bank row has insufficient disjoint starts")
        start = min(candidates, key=lambda p: (abs(p - spec["start"]), p))
        repairs.append({"index": i, "before": dict(spec), "after_start": start})
        spec["start"] = start
        occupied.add(start)
    assert_isolated(result, rows, protected)
    return result, repairs


def make_broad_schedule(rows, probes, steps, seed):
    audit_groups(rows)
    protected = protected_intervals(probes, rows)
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows.values(), key=lambda r: r["row_id"]):
        if row["split"] != "train" or row["held_out"]:
            continue
        spans = allowed_starts(row, protected)
        if spans:
            weight = float(row["sampling_weight"])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("invalid row sampling weight")
            groups[row["corpus"]][row["track_id"]].append((row, spans, weight))
    names = ("music_library", "speech_library")
    if any(len(groups[name]) < count for name, count in zip(names, (4, 1), strict=True)):
        raise ValueError("insufficient eligible recordings")
    tracks = {name: sorted(groups[name]) for name in names}
    rngs = {name: rng_for(seed, "broad:" + name) for name in names}
    queues = {name: [] for name in names}
    options = {}
    for name in names:
        for track, values in groups[name].items():
            weights = np.asarray([v[2] for v in values], dtype=np.float64)
            options[track] = (values, weights / weights.sum())
    specs, draws = [], []
    for _ in range(steps):
        draw = []
        for name, count in zip(names, (4, 1), strict=True):
            rng, chosen = rngs[name], set()
            for _ in range(count):
                if not queues[name]:
                    queues[name] = rng.permutation(tracks[name]).tolist()
                index = next(i for i, track in enumerate(queues[name]) if track not in chosen)
                track = queues[name].pop(index)
                chosen.add(track)
                values, weights = options[track]
                row, spans, _ = values[int(rng.choice(len(values), p=weights))]
                position = int(rng.integers(sum(b - a + 1 for a, b in spans)))
                for a, b in spans:
                    if position <= b - a:
                        start = a + position
                        break
                    position -= b - a + 1
                draw.append(len(specs))
                specs.append({"row_id": row["row_id"], "start": start, "gain_db": 0.0})
        draws.append(draw)
    assert_isolated(specs, rows, protected)
    return specs, draws, {name: len(groups[name]) for name in names}


class BoundedEnvelopeClips(EnvelopeClips):
    """Reuse existing disk entries; new random crops have a bounded RAM cache only."""

    def __init__(self, rows, detector, cache, capacity=128):
        super().__init__(rows, detector, cache)
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.write_cache = False
        self.capacity = capacity
        self.memory = OrderedDict()

    def get(self, spec):
        key = digest(spec)
        if key not in self.memory:
            self.memory[key] = super().get(spec)
            if len(self.memory) > self.capacity:
                self.memory.popitem(last=False)
        self.memory.move_to_end(key)
        return self.memory[key]


class LazyPool:
    """Same paired/identity ordering and objective as Pool, materialized per batch."""

    def __init__(self, specs, clips):
        self.specs, self.clips = specs, clips
        self.rows = [clips.rows[s["row_id"]] for s in specs]

    def select(self, indices):
        specs = [self.specs[i] for i in indices]
        batch = self.clips.batch(specs + identity_specs(specs))
        return FixedFitObjective(batch), context_features(batch["pcm"], "past")
