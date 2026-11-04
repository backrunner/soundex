# P71: calibrated evaluation and twofold band capacity feasibility

Completed 2026-11-04. The [evaluation protocol](evaluation-protocol.md) is
implemented and calibrated; same-depth small/double models pass a local native
feasibility screen. These fixtures have no training updates or quality result.

## Changed hypothesis

[P70](baseline-metric-audit-p70.md) found ambiguous comparator naming, log-distance
blind spots and weak additional-feature use in frozen-magnitude phase heads.
P71 separates spectral shape, energy, temporal behavior, complex fidelity and
absolute errors rather than selecting a new favorable scalar. The prospective
comparison and bootstrap are frozen before candidate training.

Prior [capacity work](full-capacity-study.md) and
[decoder collapse](head-collapse-audit.md) concern a different global representation;
they do not answer whether widening the P63 band representation helps. P71 keeps
depth, history, band partition, decoder interface, phase observation and support
fixed. It also explicitly enables full-model gradients so the magnitude backbone
can learn. This is not a restart of the retired global width-640 route.

## Evaluation calibration

Local evidence: `data/runs/restoration-panel-p71-20261104/`. Registration,
executed source snapshot, every per-case/resolution record, assessment and hash
receipt are retained. All twenty P70 first-excerpt MP3 sources and all fourteen
paths are included: **280 scored paths**, no new score-selected recordings.
The P70 metrics remain unchanged; new independent rulers are added beside them.
All twenty real recordings in this panel are stereo. Mono correctness is covered
by analytic/model/native fixtures and the separate development reviews, not by
a real-mono generalization claim from this calibration. Optional stereo metrics
must be assessed on stereo strata when a future panel mixes channel geometries.

For P69 adjacent versus historical legacy, on this MP3-only calibration:

| Metric | Relative change, seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Existing MR | -2.3575% | -2.5553% |
| New multiresolution HF SC | -3.3454% | -3.5313% |
| Gain-independent HF shape | -6.8948% | -7.5500% |
| HF envelope | -8.9352% | -9.5894% |
| HF dynamics | -4.7635% | -4.7721% |
| Complex HF distance | -2.2088% | -2.2429% |

All corresponding paired recording bootstrap intervals are below zero in this
diagnostic panel. This is selection-exposed evidence, not an independent test or
an overall-quality result. It is also not the full multi-codec P69 mean.

Compared with the matched repeated-input head, adjacent still loses MR, SC,
shape and envelope in both seeds. Their paired intervals are above zero;
dynamics intervals cross zero. Adjacent slightly improves complex distance.
Compared with untreated input, spectral means improve but complex HF distance
worsens **4.3807% / 4.3443%**, with adverse cases on all twenty sources.
The panel exposes these tradeoffs instead of erasing either result.

Correct and half-correct HF controls improve all six listed rulers in 20/20
cases. Energy-matched phase scrambling worsens complex distance in 20/20,
SC in 19/20, envelope in 16/20 and dynamics in 18/20, while shape worsens in only
3/20. Thus shape alone also cannot establish coherent restoration. Muting HF
worsens MR, SC, shape and envelope in all twenty, but retains dynamics/complex
exceptions. Every case and exception is preserved. No fitted weighting or new
perceptual-severity threshold was introduced. ViSQOL and listener ratings remain
missing; the panel is a diagnostic protocol, not a perceptual oracle.

## Same-depth capacity and runtime

Local evidence: `data/runs/band-capacity-p71-20261104/`. Fresh deterministic
nonzero seed-71 fixtures exercise all branches. Full-model training scope changes
gradient connectivity only; historical phase-only construction still freezes
the base by default and its forward arithmetic is preserved.

| Quantity | Small | Approximately double |
| --- | ---: | ---: |
| Parameters, all trainable in joint mode | 204,418 | 408,023 |
| Band token width | 24 | 67 |
| ONNX bytes | 938,708 | 1,746,664 |
| State floats per physical channel | 12,354 | 34,026 |
| Measured hops | 13,000 | 13,000 |
| Maximum frontend + model + DSP compute | 0.558084 ms | 2.242709 ms |
| Service overruns / paced late completions | 0 / 0 | 0 / 0 |
| 48 kHz stereo paced mean / p99 | 0.332893 / 0.481084 ms | 1.423720 / 1.975220 ms |
| Paced process CPU, one-core equivalent | 14.14% | 52.19% |
| Maximum native process peak RSS | 30.36 MiB | 33.59 MiB |
| Fresh ORT-only warmed RSS increment | 14.42 MiB | 17.41 MiB |

The parameter ratio is 1.996x. Same six dilations, eight bands, readout width 96,
phase hidden 32, current/past observations, native support and hop 128; no extra
lookahead. The fixed 256-frame output buffer remains 5.333 ms at 48 kHz, excluded
from compute. Per-hop period is 2.667 ms at 48 kHz and 2.902 ms at 44.1 kHz.

Each model receives four rate/channel unpaced cases (100 warmup + 3,000 measured
hops each) and one 48 kHz stereo paced case (100 + 1,000). ONNX sequential parity
includes reset and silence; actual Rust parity checks both rates and physical
mono/stereo isolation. All gates pass once, without timing retries.

The double model's paced CPU and tails are materially higher than its unpaced
means. These are whole diagnostic-process measurements, including startup/warmup
for CPU/RSS; ORT-only RSS is a separate process and not native peak. Host pacing
is not a real callback/device test. Fresh fixtures establish feasibility, not
trained-endpoint runtime qualification. Historical timing failures remain valid.

## Validation and next step

Analytic tests cover reference zero, gain/shape separation, frequency mismatch,
transients, weak scaling, silent-reference absolute errors and stereo polarity.
Source aggregation rejects duplicates/missing pairs and retains undefined and
adverse sources. Capacity tests cover magnitude gradients, state/chunk parity,
future/channel independence and the frozen historical default.

Proceed to [P72](band-capacity-training-p72.md): fresh matched small/double full
training with identical seeds, exposure, objective and adequate fixed budget.
Do not infer a quality gain from this screen or compare a fresh double model
against a historically pretrained small model as the capacity experiment.
