# Experimental causal artifact contract 2.0

This contract is implemented for the original `soundex-causal-complex` prototypes.
It does not change the default release model, and does not establish audio quality.
Legacy 1.x C1/C9 artifacts retain their existing semantics.

| Tensor (FP32, ordered) | Shape | Meaning |
|---|---|---|
| `input_features` | `[B,2,1,129]` | Current periodic-Hann FFT256 frame: dB, radians |
| `input_state` | `[B,S]` | Previous state for the same physical channels |
| `sample_rate_hz` | `[B,1]` | 44100 or 48000 Hz, fixed for the stream |
| `output_features` | `[B,2,1,129]` | Current restored dB/radian spectrum |
| `output_state` | `[B,S]` | Next state, committed only with finite valid output |

B is 1 or 2. The hop is 128 samples, crossover width 1000 Hz, phase policy `Model`,
high-band gain `Model`. Other policies are rejected for this artifact. Input/output
feature units, window and provenance hashes follow the existing contract. New
metadata declares architecture/version, temporal kind, state size/layout/zero
reset, 64 warmup frames, sample-rate units and parameter budget. Runtime enforces
all shapes/types/names and the state/file budgets. Export checks <=2M parameters,
ONNX <8 MiB and sequential PyTorch/ORT parity before atomically publishing.
Metadata is a declaration, not independent proof of learned parameter provenance.

- GRU: 2 layers, width 192, state 384 floats per physical channel; flattened as
  `[layer,hidden]`. Parameters 643458.
- TCN: width 144, six kernel-2 depthwise blocks, dilation 1/2/4/8/16/32. Each cache
  stores its normalized block inputs, oldest to newest, concatenated by block.
  State 9072 floats per channel; receptive field 64 hops. Parameters 639906.
- Encoder concatenates normalized log magnitude and power-compressed Re/Im,
  plus sample rate. LayerNorm is per frame. Additive Cartesian residuals can
  populate exactly missing bins. DC/Nyquist imaginary components are zero.
- The runtime advances every physical channel on every hop, including silence
  and detector bypass. Never compact active channels into a different state slot.
- Construction starts state at zero. Reset, a worker sequence gap and model reload
  clear state, DSP history and the history counter. ORT construction prewarm is
  discarded and does not count as observed audio.
- First 64 observed hops return aligned dry output while updating state. This is
  185.76 ms at 44.1 kHz / 170.67 ms at 48 kHz of enhancement warmup, **not added
  output delay**. Fixed realtime adapter delay remains 256 samples.
- `RealtimeWorkerStats.state_history_warmup_hops` counts these processed dry hops,
  including recovery after gaps. Deadline availability alone does not demonstrate
  enhancement availability: accepted packets can contain valid dry warmup audio.
- `InferenceEngine` has separate B1/B2 state buffers. Direct callers must keep a
  fixed physical-channel order and reset before reusing a stream. The processor
  always uses its configured physical channel count.

Tests cover sequence/chunk equivalence, no future dependence, per-channel state,
reset, bypass/silence, worker gaps, reload, construction prewarm and Rust heap
allocation. Export compares Cartesian audio rather than raw wrapped phase, with
`atol=1e-5, rtol=1e-4`; state uses `atol=2e-5, rtol=2e-5`. These are numerical
consistency tolerances, not perceptual quality thresholds. Runtime end-to-end
latency, physical device operation and long-duration drift remain separate gates.
