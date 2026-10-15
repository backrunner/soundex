"""Fresh ORT-only process RSS snapshots, excluding torch and training pools."""

import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort


def rss():
    return (
        int(
            subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "rss="], text=True).strip()
        )
        * 1024
    )


def measure(path):
    baseline = rss()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    state_size = session.get_inputs()[3].shape[1]
    feed = {
        "features": np.full((2, 2, 1, 129), -40, np.float32),
        "context": np.ones((2, 1, 1538), np.float32),
        "condition": np.ones((2, 1, 3), np.float32),
        "state": np.zeros((2, state_size), np.float32),
    }
    for _ in range(100):
        _, feed["state"] = session.run(None, feed)
    after = rss()
    return {
        "baseline_rss_bytes": baseline,
        "warmed_rss_bytes": after,
        "increment_rss_bytes": after - baseline,
        "warmup_hops": 100,
        "scope": "fresh ORT-only process; session/allocator/buffers RSS snapshots; excludes FFT frontend and full Rust; not peak memory",
    }


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1])))
