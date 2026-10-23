# P37: isolate fusion and detector conditions with frozen models

Completed 2026-10-22 following the [model/DSP boundary review](dsp-model-boundary-review.md).
Both P36 `compressed` seeds completed every control without training, changing loss
or promoting weights. **Magnitude remixing offsets some recovery, but removing
fusion provides only small gains; DSP alone does not explain the current bottleneck.**

## Fixed protocol and implementation

Use both frozen 431,015-parameter, 16,192-step endpoints on 89 existing recording-
isolated clips, each in paired and identity roles. This is a reused held-out recheck,
not an independent test. Models, material, 80-hop output warmup, supervised region
and MR v1 remain fixed; the reference is always original unclipped lossless audio.
Score clips first, then aggregate by recording: 26 music and 54 speech recordings.
Clip count is not library coverage.

| Output setting | Magnitude | Phase | Detection/gating |
| --- | --- | --- | --- |
| `current` | Existing remix | Existing unit-circle fusion | Retained |
| `magnitude-only` | Existing remix | Direct prediction | Retained |
| `phase-only` | Direct prediction | Existing fusion | Retained |
| `unfused` | Direct prediction | Direct prediction | Retained |
| `safety-only` | Direct prediction | Direct prediction | Forced wet after warmup, with startup fade retained |

Every path retains OLA, safety limiting and numerical bounds. Separately, current
output tests fixed normalized cutoff=0.5, fixed active=1 and both fixed; sample-rate
conditioning remains real. Changes begin at model warmup; actual DSP decisions are
unchanged. These are frozen-model sensitivity probes, not retraining without
conditions. Condition and output interventions are not combined to select a best case.

Eight settings × two seeds × 89 clips × two roles produce **2,848 Rust frontend/ORT/DSP
replays**. Raw-prediction hashes match exactly across the five output paths, isolating
output processing. Current native per-clip MR differs from P36 by **0**. The mono
proxy uses actual Rust-observed spectra; maximum PCM difference is **2.98e-7**, below
the unchanged `2e-5` tolerance. Real stereo uses native processing; independent mono
proxies do not stand in for shared stereo gating.

Diagnostic options exist only in `soundex-context-replay` and diagnostic constructors,
not production configuration or weight contracts. Environment variables are
`SOUNDEX_DIAGNOSTIC_FUSION` and `SOUNDEX_DIAGNOSTIC_CONDITION`; defaults remain
current/native and unknown values fail.

## Output-side results

Recording-mean changes below compare **the same P36 weights with current DSP**;
negative is better. Cells list seeds 20261012 / 20261013, not gains over untreated input.

| Setting | Music MR | Music reference-dynamics error | Music PCM RMS | Speech MR |
| --- | ---: | ---: | ---: | ---: |
| Magnitude fusion only | −0.031% / −0.058% | +0.041% / +0.029% | +0.017% / +0.024% | −0.106% / −0.126% |
| **Phase fusion only** | **−0.123% / −0.137%** | **−0.109% / −0.107%** | **−0.155% / −0.166%** | **−0.472% / −0.540%** |
| Unfused, gating retained | −0.183% / −0.268% | −0.066% / −0.075% | −0.138% / −0.142% | −0.537% / −0.613% |
| Synthesis plus safety | −0.183% / −0.268% | −0.066% / −0.075% | −0.138% / −0.142% | −0.537% / −0.614% |

Current music MR is `0.3787409035 / 0.3789824394`; phase-only is
`0.3782741786 / 0.3784627003`; unfused is `0.3780471609 / 0.3779681223`.
Phase-only recording outcomes are 20 wins/six losses and 16 wins/10 losses, retaining
counterexamples. Against the contemporaneous P35 control, music dynamics still
regress about 0.10% / 0.13%; the original screen's failures are not all resolved.

Unfused raw music OLA still scores `0.3780627322 / 0.3779570620`. Removing remixing
is far from perfect lossless recovery, so learning itself needs improvement. Prior
oracle spectra still incur fusion error, showing an interface restriction for stronger
models. Limited gains with these weak models do not disprove that restriction, and
oracle scores are not additive shares of total error.

Phase-only reduces 12–16 kHz amplitude error by 0.155% / 0.143% and dynamics error by
0.294% / 0.272%, but above 16 kHz increases dynamics by 0.020% / 0.010% and PCM error
by 0.015% / 0.019%. At 0–4 kHz, PCM error falls 0.272% / 0.289% while dynamics error
rises 0.053% / 0.038%. This is not uniform improvement. All bands, roles and raw/native
stages remain in local reports; thresholds are not fitted to known counterexamples.

## Identity preservation and conditioning

Removing magnitude remixing increases model changes on lossless input. Phase-only
music identity MR rises 0.130% / 0.105%, and speech identity MR 18.754% / 11.297%.
Removing all gating adds almost no paired benefit but changes more previously
bypassed recordings.

Recording-mean native speech-identity PCM RMS:

| Setting | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Current | 4.64e-8 | 5.92e-8 |
| Phase-only | 4.71e-7 | 3.58e-7 |
| Unfused, gating retained | 4.82e-7 | 3.62e-7 |
| Synthesis plus safety | 8.32e-7 | 5.29e-7 |

Worst phase-only speech-clip RMS is `1.88e-6 / 1.67e-6`
(−114.54 / −115.57 dBFS). Relative increases do not establish audible severity;
small absolute values do not qualify the model. Music identity RMS against original
references averages about 8.2e-4, with worst clip about 0.00954. Existing safety-limiter
error is included and cannot all be attributed to new model changes.

With current output, fixed cutoff improves music MR by 0.161% / 0.171%, fixed active
by 0.006% / 0.013%, and both fixed by 0.173% / 0.189%. Fixing both improves speech MR
by 0.151% / 0.195% but worsens speech dynamics by 0.061% / 0.053%. Conditions affect
predictions, with cutoff more influential here. **This does not establish detector
dependence as the main cause or show that retraining without conditions succeeds.**

## Continuity, stereo and timing

A supplement uses the same original four-second interval of one existing true-stereo
recording, actually encoded/aligned as MP3 CBR96 at 44.1/48 kHz. Two seeds and five
paths each cover paired, identity, silence, silent left channel, independent mono
right channel and future-input perturbation: **120 native replays**. This is a limited
diagnostic, not a new independent library test; genre, region and complex stereo
coverage remain insufficient.

- Changing the latter input half changes earlier output by 0; no future reading was found.
- Maximum silence peak is `8.73e-15`; silent-left RMS is at most `3.10e-15`.
- Maximum mono/stereo difference for the same right channel is `5.96e-8`, with no
  evident crosstalk in this probe.
- Phase-only improves paired MR, dynamics and reference-relative seam changes in all
  four seed/rate combinations for this recording, but identity MR still rises.
  Reports include M/S error, physical-channel PCM and reference-relative adjacent
  sample changes. Smoother output is not automatically more faithful to transients;
  this one recording does not replace blind listening.

Two rates, two seeds and five paths yield **120,000 CPU ORT-plus-DSP timing hops**:
100,000 burst and 20,000 paced, using a synthetic stereo multitone. Exclude the first
100 warmup hops in each process.

| Metric | Result |
| --- | ---: |
| Maximum compute | **0.992 ms** |
| Highest scenario p99 | **0.594 ms** |
| ≥5 ms / service over hop / paced late | **0 / 0 / 0** |
| Peak process RSS | **30.16 MiB** |
| Burst summed CPU / wall time | 14.75 s / 14.82 s |
| Paced summed CPU / wall time | 9.35 s / 61.97 s |

Process CPU includes startup/reporting. Diagnostic paths still execute some fusion
work before overriding it, so this is not a ranking of optimized production paths.
Fixed 256-frame buffering remains 5.33 ms at 48 kHz and 5.80 ms at 44.1 kHz; sub-1 ms
compute is not total output delay. Real devices, callback queues, failure recovery
and long-duration operation remain unqualified. Later P38/P39 timing failures must
also be retained, rather than superseded by this passing run.

## Decision at completion

1. Investigate removal of magnitude remixing while retaining phase fusion/gating as
   protective controls. Identity/local-band failures remain, so production defaults
   do not change. Unfused remains a diagnostic; heuristic phase fusion is not assumed
   permanently correct.
2. Preserve explicit full lossless-spectrum/complex-residual output semantics and
   direct supervision. Post-OLA MR targets the original lossless source. Do not train
   through full limiting to learn inverse limiting or compensation for an intensity
   knob. These results justify neither removing safety nor expecting DSP-only changes
   to solve recovery.
3. Isolate conditions in the same small model: identical initialization/schedules,
   two seeds, 16,192 steps, control versus fixed cutoff/active, real rate, and conditions
   applied from the first warmup frame. Keep direct targets, identity 1.0, dynamics,
   size and budget unchanged. Freeze all endpoints, then review current/phase-only
   paths. At P37 completion this training had not started; [P38](detector-condition-study.md)
   subsequently completed it and rejected the fixed-condition route.

The independent MPS addition probe failed with `MTLCompilerService` connection
error 141. P38 startup corrected the blocking inference: P36 already used CPU, so
Metal failure did not prevent the same-device comparison. P37 itself trained no model.

## Engineering checks and retained failures

- Python CPU: **648 passed, seven MPS items unverified**; not an all-platform pass.
  System FFmpeg lacked x265; the existing imageio FFmpeg 7.1 was used only in process
  PATH. Four new ablation tests pass; oracle tests use genuinely different observations
  and references.
- Rust core: **39 passed, one 30-minute stress test ignored**. Default/no-ORT Clippy,
  Rust fmt and relevant Python Ruff/format pass.
- `training/scripts/audit_fusion_boundaries.py` registers hashes, every setting,
  prediction invariance, proxy parity, recording scores and PCM errors. Formal evidence:
  `data/runs/fusion-p37-20261022/`; registration SHA-256
  `872ca412748e3c086007a66d30c48272753360eb28f091b9c6769bfbc86d78e8`.
- Frozen formal binary SHA-256:
  `a2b5539eca1a66214cd63f0bd5ff65e9fc94300ffce635887a435ef2a162896f`.
  Later fixes address no-ORT conditional compilation and reporting scope; formal
  measurements still bind to the original frozen binary.
- The initial formal snapshot contained Python but omitted `parity_policy.json`,
  discovered when importing the supplement. A resource copy and
  `snapshot-resources.json` were added as supplemental evidence without rewriting
  registration. Later executors register/check this resource explicitly. The fixed
  executor remains distinct from the formal source. All-setting preflight in
  `data/runs/fusion-p37-final-preflight-20261022/` completed 64 native roles: matching
  raw/native MR differs from formal clips by 0, maximum proxy error is `2.38e-7`,
  and frozen resource hashes/standalone imports pass.
- Stereo, band summaries and timing are under
  `data/runs/fusion-p37-supplement-20261022/`. Original audio, weights and reports
  remain local and are not pushed.

Use a new output directory; existing evidence cannot be overwritten:

```bash
rtk proxy env PYTHONPATH=training OMP_NUM_THREADS=1 training/.venv/bin/python \
  training/scripts/audit_fusion_boundaries.py \
  --study data/runs/risk-attribution-p36-20261022/study \
  --output data/runs/fusion-audit-new \
  --binary target/release/soundex-context-replay
```
