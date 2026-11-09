# P73: frozen capacity tradeoff attribution

Completed 2026-11-07. P72's larger models improve high-band magnitude while
increasing amplitude-weighted phase-interaction error. This is not a failed
training execution, nor evidence that all new models lose to legacy. It is a
specific reconstruction tradeoff at the four completed matched endpoints.
No new weights, training, audio rendering, timing retries or default promotion.

## Question and evidence

[P64](band-attribution-p64.md) analyzed P63 band versus global representations.
P73 asks whether the fully joint P72 small/double comparison has the same
mechanism, and whether raw-to-native changes explain the other development
panel's weaker result. P64 did not answer this capacity question; no larger
unchanged training budget is proposed here.

Local evidence: `data/runs/capacity-tradeoff-p73-20261107/`.
`registration.json` binds 3,232 inputs and source files, including all four
16,384-update P72 endpoints, original audio/output receipts, prior scores and
timing traces. All bindings verify before and after computation. Frozen analysis
source is retained under `source/`; `complete.json` binds the derived reports.

All 160 P55 cases and 2,080 PCM paths are retained: both seeds/capacities,
raw/native, untreated, historical legacy and identity DSP. There are 640
matched-capacity factor records, covering paired and identity roles and every
codec. This is an exposed-recording diagnostic, not new independent testing.
Every recording and adverse case remains in the detailed files:

- `pcm-records.json` / `pcm-summary.json`: exact band power identities.
- `factors-records.json` / `factors-summary.json`: magnitude/angular accounting.
- `mr-attribution.json`: existing MR components, codecs and source/case deltas.
- `source-stages.json`: matched development case/role populations and codec strata.
- `deadline-accounting.json`: original paced failures, without new measurements.

## Magnitude gains and coherent reconstruction

For complex spectra Y and T, the exact identity is
`|Y-T|² = (|Y|-|T|)² + |Y||T||unit(Y)-unit(T)|²`.
The second term depends on both magnitude and phase. The diagnostic uses a
whole-excerpt rectangular FFT, one-sided Parseval scaling and physical-channel
mean power. Numerical identities pass on all 2,080 paths. It is not an STFT
training window, new score, audibility estimate or reference-phase oracle.

The following table reports **double minus matched small**, recording-equal
paired native means. Units are linear PCM power times 1e6, not dBFS. Negative
means less error. These power means differ from P72's means of per-case RMS.

| HF >=8 kHz error contribution | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| Magnitude-error power change | -1.664789 | -0.651208 |
| Phase-interaction power change | +5.074802 | +2.826155 |
| Total error-power change | +3.410013 | +2.174947 |
| Interaction: symmetric magnitude factor | +4.793044 | +2.645391 |
| Interaction: symmetric angular factor | +0.281758 | +0.180764 |

The magnitude factor accounts for **94.45% / 93.60%** of the interaction increase.
This is a symmetric algebraic attribution, not a causal percentage. It says that
changed predicted magnitudes amplify existing phase mismatch; it does not say
phase is correct or that every bin becomes louder. Simply penalizing magnitude
until enhancement disappears would discard the restoration benefit.

Raw HF total-error power also rises, by 5.063755e-6 / 4.735345e-6. Native fusion
reduces those increments to 3.410013e-6 / 2.174947e-6. Its angular-factor increment
is smaller than the raw 1.478434e-6 / 2.065761e-6. Thus this tradeoff already exists
before DSP; the present results do not justify removing DSP wholesale.

Native magnitude error decreases and phase-interaction error increases in each
MP3-64, AAC-96 and Vorbis-q3 stratum in both seeds. This differs from P64's
codec-specific architecture comparison and must not be generalized from it.
For example, AAC HF total-error changes are +1.238506e-6 / +2.131746e-6;
Vorbis changes are +0.678162e-6 / +0.587459e-6.

## Where the spectral benefit occurs

The existing MR score is reproduced from its registered band/component weights;
no weight or threshold changes. Native paired contributions are:

| Band contribution to MR change | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: |
| 8–12 kHz | -0.000585341 | -0.000220631 |
| 12–16 kHz | -0.001541164 | -0.000725063 |
| >=16 kHz | -0.002290976 | -0.002369099 |
| All bands | -0.004447945 | -0.003332409 |

Most benefit is above 12 kHz, especially above 16 kHz. Amplitude, log-shape and
envelope components improve there, while the existing MR dynamics contribution
worsens in all three high bands in both seeds. This must remain visible alongside
P72's improving independent HF dynamics ruler: the independent panel and existing
MR use different windows, normalizations and aggregation. One is not a replacement
for the other. Gain-independent shape remains mixed as documented in P72.

## Why the other development panel differs

The source review uses independent mono channel views and nine codec settings;
the P55 long-form recheck has twenty stereo recordings and three settings.
Recordings, intervals, duration, channels and codec coverage all differ. These
panels cannot isolate any one factor causally.

P73 matches the actual raw/native case and role before measuring stage changes,
then averages within recording before across recordings. On the 25 paired music
recordings, seed 12's raw MR change is -0.000512438; native processing changes the
capacity difference by +0.000520688, leaving +0.000008250. For seed 13, raw change
is -0.000860800 and the stage difference is -0.000160997, leaving -0.001021797.
There is no consistent sign for a universal DSP penalty.

Seed 12's seven-recording MP3-128 stratum already regresses before DSP:
raw +0.003683641, native +0.002349778. Other strata can lose a raw gain during
fusion. Seed 13's single AAC-96 recording regresses in both paths; it cannot
establish a codec-wide failure. Strata overlap in source membership and must not
be added as if independent. No codec-specific acceptance threshold is introduced.

There are 11/25 and 8/25 adverse native recording means. Concrete retained examples:

- `9f7ddcc81d02db7c61e9907dd45f34fa079d7dec784ae7111ab48a740f8e131a`:
  seed 12 native MR +0.012379885, already raw +0.010243177.
- `12446ae506d9450a239eaa84dac6ce3b135395f33a90aede1bdcefec05776477`:
  seed 12 native +0.008340046, raw +0.013446361.
- `3703455d77e912cdd6014ba5b5fd33c9e380c79c65b463dd3f3adfeeae9955d3`:
  seed 13 native +0.013417363, raw +0.011022806; seed 12 instead has a raw
  improvement that becomes a native regression. Preserve both seeds.

## Runtime diagnosis and remaining boundary

The original four paced traces reproduce 4,000 measured hops and one late
completion. Seed 13 double, hop 784, begins 2.185114 ms late and computes for
0.698375 ms, finishing 0.217405 ms past its 2.667 ms deadline. Hop 783 finishes
on time; hop 785 recovers to an on-time finish. The event is not initiated by a
previous late completion. QoS is accepted; the trace has no accepted time-constraint
request and no thread-CPU probe. The data cannot identify an OS scheduling cause.

The context replay's paced path sleeps until the scheduled arrival and then
invokes frontend/ORT/DSP directly. Production `realtime/worker.rs` instead consumes
a bounded queue, predicts submission timing, uses `worker_wait.rs` to wake early,
and applies deadline discard/history-reset behavior. The retained diagnostic
explicitly excludes callback, transport and device qualification.

Consequently the P72 zero-late paced screen remains failed. It neither proves a
device dropout nor certifies the production worker. No timing retry, scheduler
change, larger buffer, threshold relaxation or claim of a runtime fix was made.
A future runtime comparison must exercise the trained graph through the actual
worker/adapter with balanced fixed runs, callback availability/fallback and CPU
costs. It must retain this failure and distinguish library policy from the
diagnostic's zero-late screen.

## Consequence for the next change

[P74](phase-temporal-p74.md) now implements the small temporal observation
comparison below and registers fresh matched training. Its feasibility result
does not change P73's frozen reconstruction findings or qualification.

Source review shows that P72's band TCN encodes magnitude/context/conditions;
the appended phase readout sees current and one previous input phase frame.
The magnitude/complex-residual backbone does not consume those phase observations,
although joint training connects all parameter gradients. The extra phase head
rotates the output without changing its predicted magnitude. This is an
architectural observation, not proof that this bottleneck causes every failure.

The next useful hypothesis is **phase information inside the small joint temporal
representation**, with a matched parameter/control path, rather than wider
magnitude tokens or another unchanged phase-only continuation. P68/P69 added
observations only to the appended phase head; they do not test this mechanism.
First verify its causal information path and gradient behavior on registered TRAIN
material, then native correspondence and the fixed timing screen before training.

Inspect realized raw/fused complex-spectrum supervision alongside the current
direct FFT256 complex term (weight 0.1): raw/fused MR presently uses magnitude,
log-shape, envelope and dynamics, not a direct complex-distance term. That is a
candidate objective investigation, not a demonstrated loss bug or permission to
replace spectral restoration with waveform minimization. Keep architecture and
objective controls separate. New loss work must pass reference, silence/weak,
finite-difference and deployment correspondence checks before a registered run.

Any subsequent training retains small matched controls, both seeds, equal adequate
budgets, direct original-lossless targets, identity protection and both fixed
development panels. No gain obtained merely by reducing enhancement qualifies.
Listening, unexposed-source testing and actual-device evidence remain missing.

## Validation

Twenty focused tests pass: fifteen existing exact decomposition/deadline tests
and five new source-stage population/weighting tests. Ruff passes. The complete
audit finishes with all 2,080 power identities, 640 factor identities and pre/post
input checks; no registered execution failure. P72's failed quality/runtime
criteria and every counterexample remain unchanged. No cleanup was performed.
