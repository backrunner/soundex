"""P12 nested, corpus-balanced subsets and recording-local probes with disjoint scoring windows."""

import numpy as np

from data.causal_study import SAMPLES
from scripts.run_fit_diagnostic import select_specs


def nested_specs(specs, rows):
    first = select_specs(specs, rows)
    ordered = {}
    for corpus in ("music_library", "speech_library"):
        prefix = [s for s in first if rows[s["row_id"]]["corpus"] == corpus]
        ids = {s["row_id"] for s in prefix}
        ordered[corpus] = prefix + [
            s
            for s in sorted(specs, key=lambda s: s["row_id"])
            if s["row_id"] not in ids and rows[s["row_id"]]["corpus"] == corpus
        ]
    if len(ordered["music_library"]) != 32 or len(ordered["speech_library"]) != 8:
        raise ValueError("registered 32 music / 8 speech probes required")
    if len({rows[s["row_id"]]["track_id"] for s in specs}) != 40:
        raise ValueError("one anchor per recording required")
    return {
        size: ordered["music_library"][: size * 4 // 5] + ordered["speech_library"][: size // 5]
        for size in (5, 10, 40)
    }


def shifted_specs(specs, rows):
    result = []
    for spec in specs:
        maximum = rows[spec["row_id"]]["length_samples"] - SAMPLES
        start = max((0, maximum), key=lambda s: abs(s - spec["start"]))
        if abs(start - spec["start"]) < SAMPLES - 128 - 89 * 128:
            raise ValueError("nonoverlapping scoring window unavailable")
        result.append({**spec, "start": start})
    return result


def make_draws(size, steps, seed):
    if size not in (5, 10, 40) or steps < 1:
        raise ValueError("invalid registered size or step budget")
    music, speech = size * 4 // 5, size // 5
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < steps:
        m = rng.permutation(music).tolist()
        s = (rng.permutation(speech) + music).tolist()
        result.extend([[*m[4 * i : 4 * i + 4], s[i]] for i in range(speech)])
    return result[:steps]


def native_development_specs(specs, rows):
    chosen = []
    for corpus, count in (("music_library", 4), ("speech_library", 1)):
        seen = set()
        for spec in sorted(specs, key=lambda s: s["row_id"]):
            row = rows[spec["row_id"]]
            if row["corpus"] == corpus and row["track_id"] not in seen:
                seen.add(row["track_id"])
                chosen.append(spec)
                if len(seen) == count:
                    break
        if len(seen) != count:
            raise ValueError("insufficient development recordings")
    return chosen
