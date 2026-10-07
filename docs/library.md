# Library integration


Live audio callbacks should use `RealtimeProcessor`. Construct it before starting
playback, then pass equal-length interleaved input/output slices to `process`.
It writes every sample without model calls, mutexes, or heap allocation in the
callback. A dedicated worker performs inference through bounded SPSC queues;
late results are discarded by timestamp and output falls back to aligned dry
audio with a channel-linked, 128-sample recovery fade. Two consecutive healthy
hops are required before enhancement resumes. Both paths receive final limiting;
non-finite input samples are replaced by a decaying held value. Silence does not
request model enhancement. Poll `stats()` from the owner to observe failures.

```rust,no_run
use soundex_core::{RealtimeProcessor, SoundExConfig};
let mut stream = RealtimeProcessor::new(
    SoundExConfig::with_model("models/soundex-v1.onnx").channels(2),
)?; // control thread: model load and worker creation
let input = [0.0_f32; 128 * 2];
let mut output = [0.0_f32; 128 * 2];
stream.process(&input, &mut output)?; // audio callback
stream.shutdown(); // control thread: stop and join
# Ok::<(), soundex_core::SoundExError>(())
```

The real-time adapter requires 256/128 models at 44.1/48 kHz. Its fixed delay is
**256 samples per channel**, including one worker handoff hop: **5.80/5.33 ms**.
This fits the preferred 8 ms software target; callback execution and any additional
integration buffers must still keep the added delay below the 10 ms redline.
Use hop-aligned callbacks of 128 frames for reliable enhancement availability.
Smaller blocks shorten the worker deadline after hop assembly; larger blocks can
miss worker deadlines inside one callback. All supported block sizes retain
continuous output. Create, destroy, or replace the adapter on a control
thread. A failed worker leaves dry output running; replace the adapter off the
callback to restart enhancement. Device/driver xruns require device testing.

The synchronous `SoundExProcessor` APIs below are for offline processing and
controlled DSP evaluation; their inference calls can wait on ONNX Runtime.

`process_frame` accepts one interleaved hop (`hop_size * channels` samples). With the default
256-point FFT and 128-sample hop, `latency_samples_per_channel()` reports 128 samples (one hop)
of causal STFT delay: about 2.90 ms at 44.1 kHz or 2.67 ms at 48 kHz, before inference and device
buffers. Fixed-hop callers receive startup padding directly and must call `finalize` once to
retrieve that delayed tail.

`process_chunk` accepts any number of complete interleaved sample frames and reports interleaved
samples consumed and produced. It withholds startup padding; concatenating every chunk result and
the single `finalize` result is sample-aligned and the same length as `process_buffer`. A finalized
stream rejects more input until `reset`. A processing failure leaves caller output unchanged and
poisons internal state, which also requires `reset`. Do not mix `process_frame` and `process_chunk`
within one stream.

```rust,no_run
use soundex_core::{SoundExConfig, SoundExProcessor};

let config = SoundExConfig::with_model("models/soundex-v1.onnx")
    .sample_rate(44_100)
    .channels(2);
let mut processor = SoundExProcessor::new(config)?;
let input = vec![0.0_f32; 44_100 * 2];
let output = processor.process_buffer(&input)?;
# Ok::<(), soundex_core::SoundExError>(())
```

Arbitrary streaming chunks append ready samples to an output vector:

```rust,no_run
use soundex_core::{SoundExConfig, SoundExProcessor};

let mut processor = SoundExProcessor::new(
    SoundExConfig::with_model("models/soundex-v1.onnx").channels(2),
)?;
let input = vec![0.0_f32; 4097 * 2];
let mut output = Vec::new();
for chunk in input.chunks(74) {
    processor.process_chunk(chunk, &mut output)?;
}
processor.finalize(&mut output)?;
assert_eq!(output.len(), input.len());
# Ok::<(), soundex_core::SoundExError>(())
```

The added software output latency target is **<8 ms**, with a **<10 ms hard redline**.
Measure the enable/bypass output time difference at integration, including any added
block adaptation or queueing.
The default host callback standard is **128 frames, aligned with hop boundaries**.
See [real-time standard](realtime-standard.md) for callback budgets,
bounded worker fallback and physical-device acceptance. The earlier 5 ms preference
is an optimization reference, not an acceptance gate.
