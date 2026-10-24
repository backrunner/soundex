"""Local blinded listening materials; answer keys never enter the listener bundle.

SPDX-License-Identifier: Apache-2.0
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from evaluation.longform_review import SEED, aligned_output
from evaluation.reporting import write_json_report
from evaluation.rust_bridge import read_sxa


def write_trial(directory, reference, conditions, rate, rng):
    if rate != 48000 or reference.shape[0] != 10 * rate:
        raise ValueError("listening trial requires ten seconds at 48 kHz")
    if any(w.shape != reference.shape or not np.isfinite(w).all() for w in conditions.values()):
        raise ValueError("listening condition geometry/nonfinite mismatch")
    anchor = sosfiltfilt(butter(8, 3500, fs=rate, output="sos"), reference, axis=0).astype(
        np.float32
    )
    signals = {**conditions, "hidden_reference": reference, "anchor_3500hz": anchor}
    peak = max(float(np.max(np.abs(w))) for w in [reference, *signals.values()])
    gain = min(1.0, 0.89 / max(peak, 1e-12))
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(directory / "reference.wav", reference * gain, rate, subtype="PCM_24")
    names = list(signals)
    order = rng.permutation(len(names))
    keys, stimuli = {}, []
    for i, index in enumerate(order):
        opaque = f"{i + 1:02d}"
        name = names[int(index)]
        sf.write(directory / f"{opaque}.wav", signals[name] * gain, rate, subtype="PCM_24")
        keys[opaque] = name
        stimuli.append({"id": opaque, "file": f"{directory.name}/{opaque}.wav"})
    return {
        "id": directory.name,
        "reference": f"{directory.name}/reference.wav",
        "conditions": stimuli,
        "channels": reference.shape[1],
        "seconds": 10,
    }, {"mapping": keys, "common_gain": gain}


def build_pack(config, cases, output):
    directory = output / "listening"
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("existing listening pack retained; do not overwrite randomized trials")
    rng = np.random.default_rng(SEED)
    selected = [c for c in cases if c["excerpt"] == 1 and c["codec"] == "mp3-cbr-64"]
    trials, keys = [], {}
    # Do not select a winning endpoint or hide worse seeds from listeners.
    labels = [name for name in config["models"] if not name.endswith("-parent")]
    for index in rng.permutation(len(selected)):
        case = selected[int(index)]
        reference, rate = read_sxa(Path(case["reference"]))
        dry, _ = read_sxa(Path(case["input"]))
        start, length = case["score_start"], case["score_length"]
        conditions = {"untreated": dry[start : start + length]}
        for label in labels:
            native, actual_rate = read_sxa(output / "renders" / label / case["id"] / "native.sxa")
            if rate != actual_rate:
                raise ValueError("listening rate mismatch")
            conditions[label] = aligned_output(native, start, length)
        trial, key = write_trial(
            directory / f"trial-{len(trials) + 1:02d}",
            reference[start : start + length],
            conditions,
            rate,
            rng,
        )
        trials.append(trial)
        keys[trial["id"]] = {**key, "case": case["id"], "source": case["source"]}
    write_json_report(keys, output / "listening-key.json")
    template = Path(__file__).with_name("listening.html").read_text()
    # Embedded data contains no source/model labels or answer key.
    manifest = json.dumps({"protocol": "soundex-longform-p55-v1", "trials": trials}).replace(
        "<", "\\u003c"
    )
    (directory / "index.html").write_text(template.replace("__TRIAL_DATA__", manifest))
    with (directory / "CREDITS.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["title", "author", "source", "license", "changes"])
        for row in config["sources"]:
            writer.writerow(
                [
                    row.get("recording_title", row["id"]),
                    row.get("author", "unknown"),
                    row["source_page"],
                    row["license_url"],
                    "Cropped, resampled, codec degraded, enhanced or filtered for local comparative evaluation; common playback gain. Original license retained, no artist endorsement.",
                ]
            )
    (directory / "README.txt").write_text(
        "Local blinded reference-based listening study, not a completed ITU qualification. Open index.html in a browser. Use the same headphones and volume for all trials. Record listener code, equipment and environment. Compare against the shown reference; rate reference similarity and high-frequency naturalness separately. Report audible defects or no audible difference. Skipped conditions require a reason. Export ratings locally; no network submission. The private condition key is outside this folder. Do not inspect it before finishing. No audio is uploaded or committed. Source grants and attribution are in CREDITS.csv. All conditions share one playback gain per trial; numerical evaluation retains original unscaled references.\n"
    )
