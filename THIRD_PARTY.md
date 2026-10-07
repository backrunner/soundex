# Third-party components

SoundEx's original software is Apache-2.0. Dependencies retain their own licenses
and notices; no dependency source is vendored here. Cargo.lock and the Python
requirements/constraints record the dependency versions, not a change of license.

Important Rust runtime components (license declarations from resolved Cargo metadata):

| Component | License | Use |
|-----------|---------|-----|
| ort | MIT OR Apache-2.0 | ONNX Runtime Rust binding |
| ndarray | MIT OR Apache-2.0 | Tensor representation |
| rustfft | MIT OR Apache-2.0 | FFT |
| realfft | MIT | Real FFT |
| rtrb | MIT OR Apache-2.0 | Bounded audio/worker queues |
| qos-threads | MIT OR Apache-2.0 | macOS worker QoS |
| audio_thread_priority | MPL-2.0 | macOS audio time-constraint scheduling |
| Symphonia | MPL-2.0 | CLI audio decoding |
| hound | Apache-2.0 | CLI/demo WAV output |

ONNX Runtime binaries downloaded by ort retain ONNX Runtime's upstream license
and bundled third-party notices. Python training dependencies similarly retain
their upstream terms. This table highlights runtime components and is not an
exhaustive dependency inventory. Inspect the resolved versions with:

```bash
cargo metadata --locked --format-version 1
python -m pip show torch onnx onnxruntime
```

If distributing packaged executables, preserve dependency license texts/notices
from the exact build and satisfy applicable MPL source obligations. The source
repository does not grant rights to third-party datasets or learned weights;
those are described in [NOTICE](NOTICE) and [legal/LICENSING.md](legal/LICENSING.md).
