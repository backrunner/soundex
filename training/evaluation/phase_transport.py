"""P66 phase correction support in calibrated windowed-frame power units.

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np


def correction_support(before, after, weights, active, rate, *, burn=89):
    before, after, weights = (
        np.asarray(x, dtype=np.float64)[burn:] for x in (before, after, weights)
    )
    active = np.asarray(active, dtype=bool)[burn:]
    if before.shape != after.shape or before.ndim != 3 or before.shape[1:] != (2, 129):
        raise ValueError("expected aligned [hops, magnitude/phase, 129] frames")
    if weights.shape != before[:, 0].shape or active.shape != (len(before),) or not len(before):
        raise ValueError("invalid weights/activity/support")
    if rate not in (44100, 48000) or not all(
        np.isfinite(x).all() for x in (before, after, weights)
    ):
        raise ValueError("finite physical-rate frames required")
    if np.any((weights < 0) | (weights > 1)) or np.any(weights[~active] != 0):
        raise ValueError("effective native crossover weights required")
    if not np.array_equal(before[:, 0], after[:, 0]):
        raise ValueError("phase-only attribution requires identical predicted magnitude")
    magnitude = 10 ** (before[:, 0] / 20)
    sides = np.full(129, 2.0)
    sides[[0, -1]] = 1
    # Parseval power of the corresponding 256-sample windowed frame difference.
    # Not unwindowed PCM dBFS and not a measure of audible quality.
    power = (2 * magnitude * np.sin((after[:, 1] - before[:, 1]) / 2)) ** 2 * sides / 256**2
    categories = {
        "inactive": np.broadcast_to(~active[:, None], weights.shape),
        "retained": active[:, None] & (weights == 0),
        "crossover": active[:, None] & (weights > 0) & (weights < 1),
        "generated": active[:, None] & (weights == 1),
    }
    hz = np.arange(129) * rate / 256
    result = {}
    for name, low, high in (
        ("full", 0, np.inf),
        ("hf8k", 8000, np.inf),
        ("4k8k", 4000, 8000),
        ("8k12k", 8000, 12000),
        ("12k16k", 12000, 16000),
        ("16kplus", 16000, np.inf),
    ):
        mask = (hz >= low) & (hz < high)
        total = float((power * mask).sum(-1).mean())
        parts = {k: float((power * mask * v).sum(-1).mean()) for k, v in categories.items()}
        if not np.isclose(sum(parts.values()), total, rtol=1e-12, atol=1e-30):
            raise ValueError("nonpartitioned correction power")
        result[name] = {
            "windowed_frame_power": total,
            "parts": parts,
            "shares": {k: v / total if total else None for k, v in parts.items()},
        }
    return result
