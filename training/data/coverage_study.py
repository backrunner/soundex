"""P19 nested recording coverage with fixed speech and independent sampling streams."""

import hashlib
from collections import defaultdict

import numpy as np

from data.causal_study import SAMPLES
from data.crop_study import crop_bank, protected_probes


def rng_for(seed, name):
    value = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], "little"))


def expanded_anchors(original, rows):
    music = [s for s in original if rows[s["row_id"]]["corpus"] == "music_library"]
    speech = [s for s in original if rows[s["row_id"]]["corpus"] == "speech_library"]
    if len(music) != 32 or len(speech) != 8:
        raise ValueError("requires the registered 32 music / 8 speech anchors")
    groups = defaultdict(list)
    for row in rows.values():
        if (
            row["corpus"] == "music_library"
            and row["split"] == "train"
            and not row.get("held_out", False)
        ):
            groups[row["track_id"]].append(row)
    chosen = {rows[s["row_id"]]["track_id"] for s in music}
    extra, selections = [], []
    for i, track in enumerate(sorted(set(groups) - chosen)):
        template_spec = music[i % len(music)]
        template = rows[template_spec["row_id"]]
        candidates = groups[track]
        role = (
            "mono"
            if candidates[0]["source_channels"] == 1
            else ("mid" if template["channel_role"] == "mono" else template["channel_role"])
        )
        profile = (template["codec_id"], template["sample_rate"], role)
        candidates = [
            r
            for r in candidates
            if (r["codec_id"], r["sample_rate"], r["channel_role"]) == profile
            and r["length_samples"] >= 2 * SAMPLES
        ]
        if not candidates:
            raise ValueError(f"no matched profile for training recording: {track}")
        row = min(candidates, key=lambda r: hashlib.sha256(r["row_id"].encode()).digest())
        fraction = template_spec["start"] / (template["length_samples"] - SAMPLES)
        start = round(fraction * (row["length_samples"] - SAMPLES))
        spec = {"row_id": row["row_id"], "start": start, "gain_db": 0.0}
        fallback = False
        try:
            protected_probes([spec], rows)
        except ValueError:
            spec["start"] = 0
            protected_probes([spec], rows)
            fallback = True
        extra.append(spec)
        selections.append(
            {
                "track_id": track,
                "row_id": row["row_id"],
                "template_row_id": template["row_id"],
                "codec_id": profile[0],
                "sample_rate": profile[1],
                "channel_role": profile[2],
                "channel_role_mapped": role != template["channel_role"],
                "start_fallback": fallback,
            }
        )
    anchors = music + extra + speech
    if len({rows[s["row_id"]]["track_id"] for s in anchors}) != len(anchors) or any(
        rows[s["row_id"]]["split"] != "train" or rows[s["row_id"]].get("held_out", False)
        for s in anchors
    ):
        raise ValueError("anchors must be unique eligible training recordings")
    return anchors, selections


def banks(original, rows):
    expanded, selection = expanded_anchors(original, rows)
    output = {}
    for label, anchors in (("small", original), ("wide", expanded)):
        probes = protected_probes(anchors, rows)
        bank, audit = crop_bank(anchors, probes, rows, 8)
        output[label] = {"anchors": anchors, "bank": bank, "protected": probes, "crop_audit": audit}
    if not all(spec in output["wide"]["bank"] for spec in output["small"]["bank"]):
        raise ValueError("expanded bank must preserve every original crop")
    return output, selection


def schedule(anchors, rows, steps, seed):
    """Balanced corpus streams; speech and per-recording crops do not depend on music size."""
    music = [i for i, s in enumerate(anchors) if rows[s["row_id"]]["corpus"] == "music_library"]
    speech = [i for i, s in enumerate(anchors) if rows[s["row_id"]]["corpus"] == "speech_library"]
    if len(music) < 4 or len(speech) != 8 or steps < 1:
        raise ValueError("invalid coverage schedule")
    mr, sr = rng_for(seed, "music"), rng_for(seed, "speech")
    positions = [rng_for(seed, "crop:" + rows[s["row_id"]]["track_id"]) for s in anchors]
    mq, sq, pq, result = [], [], [[] for _ in anchors], []
    for _ in range(steps):
        selected = []
        for _ in range(4):
            if not mq:
                mq = mr.permutation(music).tolist()
            index = next(i for i, value in enumerate(mq) if value not in selected)
            selected.append(mq.pop(index))
        if not sq:
            sq = sr.permutation(speech).tolist()
        selected.append(sq.pop())
        draw = []
        for index in selected:
            if not pq[index]:
                pq[index] = positions[index].permutation(8).tolist()
            draw.append(index * 8 + pq[index].pop())
        result.append(draw)
    return result


class PoolView:
    """Common scoring bank as an index view, without another multi-GB tensor copy."""

    def __init__(self, pool, specs):
        self.pool, self.specs = pool, specs
        self.indices = [pool.specs.index(s) for s in specs]
        self.rows = [pool.rows[i] for i in self.indices]

    def select(self, indices):
        return self.pool.select([self.indices[i] for i in indices])
