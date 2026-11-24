"""Blinded reference-based listening packs (shared by P107/P111 builders).

Same behavior as the working-tree `evaluation.longform_listening` helpers of
2026-11-24, kept in a tracked module so committed pack builders do not depend
on uncommitted refactors. Answer keys never enter the listener bundle.
SPDX-License-Identifier: Apache-2.0
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def write_trial(directory, reference, conditions, rate, rng):
    if (
        rate not in (44100, 48000)
        or reference.shape[0] != 10 * rate
        or not np.isfinite(reference).all()
    ):
        raise ValueError("listening trial requires ten finite seconds at 44.1 or 48 kHz")
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


def write_listener_files(directory, trials, sources, protocol):
    template = Path(__file__).with_name("listening.html").read_text()
    # Embedded data contains no source/model labels or answer key.
    manifest = json.dumps({"protocol": protocol, "trials": trials}).replace("<", "\\u003c")
    (directory / "index.html").write_text(template.replace("__TRIAL_DATA__", manifest))
    with (directory / "CREDITS.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["title", "author", "source", "license", "changes"])
        for row in sources:
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
