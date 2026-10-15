"""P20 fixed head comparison scoring and frozen native replay."""

import numpy as np
import torch

from evaluation.reporting import sha256_file, write_json_report
from models.restoration_head import load_restoration_head, predict_head
from models.spectral_consistency import consistency_loss
from scripts.run_consistency_followup import complete, read, verify_receipt
from scripts.run_fit_diagnostic import metrics, replay
from scripts.run_subset_study import aggregate


def predict(model, objective, context):
    return objective.batch["x"] if model is None else predict_head(model, objective.batch, context)


def auxiliary(prediction, batch):
    return consistency_loss(prediction, batch, phase_reference="clean")


@torch.no_grad()
def evaluate(model, pool, weight):
    records, aux = [], {"music_library": [], "speech_library": []}
    if model is not None:
        model.eval()
    for i, row in enumerate(pool.rows):
        objective, context = pool.select([i])
        prediction = predict(model, objective, context)
        score = metrics(objective(prediction), objective.batch)
        aux[row["corpus"]].append(float(auxiliary(prediction, objective.batch)))
        records.append(
            {
                "spec": pool.specs[i],
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "genre": row.get("source_metadata", {}).get("genre"),
                "metrics": score,
            }
        )
    result = aggregate(records)
    value = 0.8 * float(np.mean(aux["music_library"])) + 0.2 * float(np.mean(aux["speech_library"]))
    return {
        **result,
        "auxiliary": value,
        "combined": result["weighted"]["total"] + weight * value,
        "rows": records,
    }


def review(label, checkpoint, arm, root, pools, binary, onnx, weight):
    directory = root / "review" / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        verify_receipt(directory, "files")
        return
    model = None
    if checkpoint:
        model = load_restoration_head(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        ).eval()
    checkpoint_hash = sha256_file(checkpoint) if checkpoint else None
    proxy = directory / "proxy.json"
    if not proxy.exists():
        write_json_report(
            {
                "checkpoint_sha256": checkpoint_hash,
                "groups": {name: evaluate(model, pool, weight) for name, pool in pools.items()},
            },
            proxy,
        )
    if read(proxy)["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("review checkpoint drift")
    native = directory / "native"
    native.mkdir(exist_ok=True)
    pool = pools["development"]
    for i, spec in enumerate(pool.specs):
        path = native / f"{i:03d}.json"
        if path.exists():
            old = read(path)
            if old["spec"] != spec or old["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("native case drift")
            continue
        objective, context = pool.select([i])
        paired, identity = replay(
            predict(model, objective, context), objective.batch, objective, binary, onnx
        )
        row = pool.rows[i]
        write_json_report(
            {
                "spec": spec,
                "track_id": row["track_id"],
                "corpus": row["corpus"],
                "checkpoint_sha256": checkpoint_hash,
                "paired": paired,
                "identity": identity,
            },
            path,
        )
        if (i + 1) % 8 == 0 or i + 1 == len(pool.specs):
            status = {
                "stage": "native_review",
                "case": label,
                "pairs": i + 1,
                "total_pairs": len(pool.specs),
            }
            write_json_report(status, root / "progress.json")
            print(status, flush=True)
    complete(directory)
