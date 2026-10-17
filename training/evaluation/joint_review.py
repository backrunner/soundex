"""P24 common relative + absolute review, with no post-hoc pass threshold."""

from collections import defaultdict

import numpy as np
import torch

from evaluation.absolute_audio import compare_pcm
from evaluation.head_study import predict
from evaluation.reporting import sha256_file, write_json_report
from models.budget_objective import balanced_risk
from models.restoration_head import load_restoration_head
from scripts.audit_capacity_outliers import native_audio
from scripts.run_consistency_followup import complete, read, verify_receipt


def summarize(rows):
    """Do not average dB or mix music/speech; retain clip and recording weights."""
    groups = defaultdict(list)
    for row in rows:
        for role, value in row["roles"].items():
            groups[(row["corpus"], role)].append((row["track_id"], value))
    result = {}
    for (corpus, role), values in groups.items():
        metrics = {}
        for reference in ("clean", "identity_dsp"):
            recordings = defaultdict(list)
            errors = []
            for track, value in values:
                error = value[reference]["error"]["rms"]
                errors.append(error)
                recordings[track].append(error)
            metrics[reference] = {
                "clip_mean_error_rms": float(np.mean(errors)),
                "recording_mean_error_rms": float(
                    np.mean([np.mean(v) for v in recordings.values()])
                ),
                "p95_error_rms": float(np.percentile(errors, 95)),
                "max_error_rms": max(errors),
                "clips": len(errors),
                "recordings": len(recordings),
            }
        result[f"{corpus}/{role}"] = metrics
    return result


@torch.no_grad()
def joint_review(label, checkpoint, root, pools, binary, onnx, *, observed_binary=None):
    directory = root / "joint-review" / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        if read(directory / "report.json")["checkpoint_sha256"] != (
            sha256_file(checkpoint) if checkpoint else None
        ):
            raise ValueError("joint checkpoint drift")
        return
    model = (
        load_restoration_head(torch.load(checkpoint, map_location="cpu", weights_only=False)).eval()
        if checkpoint
        else None
    )
    groups = {}
    for group, pool in pools.items():
        rows = []
        for i, row in enumerate(pool.rows):
            objective, context = pool.select([i])
            prediction = predict(model, objective, context)
            region, b = objective.region, objective.batch
            baseline = objective.baseline.numpy()
            output = objective.render(prediction).numpy()
            parity = None
            if group == "development":
                output, parity = native_audio(
                    prediction, b, objective, binary, onnx, observed_binary=observed_binary
                )
                # Validate the shared identity DSP control once, then reuse it.
                cache = root / "joint-baseline" / f"{i:03d}.npy"
                cache.parent.mkdir(exist_ok=True)
                if not cache.exists():
                    baseline, _ = native_audio(
                        b["x"], b, objective, binary, onnx, observed_binary=observed_binary
                    )
                    np.save(cache, baseline, allow_pickle=False)
                else:
                    baseline = np.load(cache, allow_pickle=False)
                if (
                    baseline.shape != output.shape
                    or np.max(np.abs(baseline[:, region] - objective.baseline.numpy()[:, region]))
                    > 2e-5
                ):
                    raise ValueError("joint baseline parity drift")
            roles = {}
            for j, role in enumerate(("paired", "identity")):
                rate = int(b["rate"][j, 0])
                roles[role] = {
                    "clean": compare_pcm(
                        output[j, region], b["clean_wave"][j, region].numpy(), rate
                    ),
                    "identity_dsp": compare_pcm(output[j, region], baseline[j, region], rate),
                    "input_to_clean": compare_pcm(
                        b["dry"][j, region].numpy(), b["clean_wave"][j, region].numpy(), rate
                    ),
                }
            rows.append(
                {
                    "spec": pool.specs[i],
                    "track_id": row["track_id"],
                    "corpus": row["corpus"],
                    "balanced_proxy": float(balanced_risk(objective, prediction)),
                    "native_proxy_errors": parity,
                    "roles": roles,
                }
            )
        groups[group] = {
            "scope": "native DSP PCM" if group == "development" else "differentiable DSP proxy PCM",
            "summary": summarize(rows),
            "rows": rows,
        }
    write_json_report(
        {
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            "scope": "joint descriptive evidence; old relative scores retained separately; no automatic quality promotion or audibility threshold",
            "groups": groups,
        },
        directory / "report.json",
    )
    complete(directory)
