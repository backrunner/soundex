# SoundEx requirements

The root [engineering conventions](../AGENTS.md) and [real-time standard](../docs/realtime-standard.md)
control acceptance. This replaces obsolete early estimates and API sketches; actual
public signatures are documented in the [library guide](../docs/library.md).

## Functional requirements

- Process mono/stereo PCM continuously and offline, preserving the input timeline,
  channel identity and complete tail. The production path supports 44.1/48 kHz.
- Use causal spectral analysis and bandwidth detection to condition enhancement;
  bandwidth detection is a control signal, not a clean-reference target.
- Predict missing/damaged high-band detail against original lossless references.
  DSP handles fusion, phase policy, loudness and safety without obscuring model-only
  versus final-output errors.
- Expose library and CLI integration with explicit configuration, model validation,
  reset and failure behavior. Keep codecs/file I/O outside DSP.
- Model loading and replacement occur outside the audio callback. Reset history
  on replacement, stream reset or dropped-frame recovery; recover smoothly.
- A host callback exchanges preallocated PCM through the streaming adapter; it must
  not infer, allocate, block or change the output timeline.

## Performance and fidelity

Model-plus-DSP compute must be below 5 ms, excluding fixed buffering, and sustain
per-hop deadlines: hop 128 at 48 kHz is 2.667 ms. The fixed 256-frame adapter buffer
adds 5.333 ms at 48 kHz (5.805 ms at 44.1 kHz), separately from compute and hardware
latency. Report p99, maximum, service overruns, paced lateness, queue/fallback behavior,
CPU and RSS. Average RTF is insufficient. Current P38/P39 timing does not pass.

Experimental models have no parameter/file-size cap beyond measured streaming
performance. Production exports retain the <=2M / <8 MiB contract until an explicit
migration; research exports may bypass that size gate explicitly, not silently.

Numerical finiteness, silence, weak input, mono/stereo isolation, block-size invariance,
true transients and smooth fallback are required. Unit/native probes are distinct
from real-device, long-duration and listening qualification.

## Training and reproducibility

Use native lossless sources with documented rights and aligned real-codec pairs.
MUSDB and Slakh are absent from the current route. Report actual recording/source-time
coverage, sample ratios and warmup exclusions rather than catalog counts. Separate
reused held-out rechecks from independent tests. Both raw and actual Rust output
must be evaluated against original references, untreated input and identity DSP.

Freeze candidate rules, full matched budgets, source/input hashes and all endpoints
before scoring. Preserve every failed validation and consult the failure registry
before a repeat. A completed budget does not prove convergence or generalization.

## Maintainability and platform claims

Use small single-purpose modules, typed contracts, meaningful tests and documented
errors. Keep default and no-ORT builds valid. Report macOS CPU, MPS, Linux, Windows
and physical-device evidence separately; successful CPU tests do not imply MPS or
cross-platform runtime qualification. Do not label research artifacts release-ready.
