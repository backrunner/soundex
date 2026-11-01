# P69: matched cross-frequency phase training

## Protocol and reason to proceed

[P68](phase-gradients-p68.md) found a concrete observation limitation in the old
phase head and verified the optional adjacent-bin features through actual Rust.
Native output support improved gradient agreement; activation saturation did not
explain the small P67 gains. Four nonzero fixtures passed the synthetic compute
screen, with no promotion or trained-quality conclusion.

P69 tests whether the added information is useful after substantial matched
training. This is not an unchanged P67 extension or a capacity search. Both arms
have the same 204,418 total parameters, 33,281 trainable phase-head parameters and
12,354 state values per physical channel. Their initial tensors match within each
origin. The magnitude backbone is the frozen P63 band model; both phase heads
start at zero correction with fresh optimizer states.

| Arm | Added observation | Output support |
| --- | --- | --- |
| `temporal_repeat` | Repeat the two existing temporal features; no new information | Native DSP availability |
| `adjacent` | Two current-frame adjacent-frequency Cartesian phase features | Same native availability |

The repeated-input control can change optimization conditioning. It controls
parameter count and head width, not a mathematically identical optimizer geometry.
Neither arm adds inference lookahead, history or output buffering.

## Fixed training budget and supervision

Seeds 20261012 and 20261013 each receive both arms: four trajectories, each
**16,384 actual Adam updates**, with four batches per update. Only the phase head
is trainable. Adam uses default betas/epsilon, gradient clipping at 5 and the
existing P63 update-index schedule: 256 updates warming from `1e-5` to `1e-4`,
then cosine decay to `1e-5`. There is no post-step scaling, adaptive loss weight,
moment reset or score-based early stopping.

The fixed P67 fused risk is used for both arms:

`log1p(direct original-reference loss) + log1p(raw multiresolution loss) + log1p(final-DSP multiresolution loss)`

Paired and clean-identity means each receive weight one half. Original aligned,
unclipped lossless audio remains the reference for every term. Direct HF spectral
supervision remains; DSP is a differentiable reconstruction path, not a teacher
target. There is no new waveform-distance or phase loss. Analysis windows do not
add inference delay. History warmup is 80 hops, with supervision starting at frame
89 under the existing OLA support rules.

The budget addresses exposure as well as optimization. P67 used the first 2,048
draws; P69 uses the entire frozen P63 65,536-draw plan. Each trajectory therefore
receives 327,680 paired crops plus their clean identities, rather than recycling
the small calibration bank. Unique supervised MR duration, after removing
warmup/overlap, rises from 3,561.220 to **27,627.408 seconds** (460.457 minutes).
The established 16,384-update fresh-training budget is used equally for both
observations; completing it will not by itself establish convergence.

Coverage remains 182 recordings (157 music, 25 speech), nine codecs and five
views: mono, left, right, mid and side. The draw mixture is 80% music and 20%
speech. This is not 327,680 independent recordings or whole-recording stereo
training. Existing genre/region proportions, unknown labels and coverage gaps are
retained in registration; no claim of equal global coverage is made. Physical
stereo is evaluated separately.

## Isolation, checkpoints and endpoint barrier

Before registration, training sources are checked against every current local
evaluation catalog, connected work/decoded aliases and the P63 review schedule.
Inputs, checkpoints, packages, evaluation reservations, runtime and executable
Python are frozen and verified before launch. The consumed runtime is P68's
frozen P66 binary, not a mutable `target/release` alias.

Each arm/origin first receives two actual diagnostic updates on the first eight
TRAIN draws. Those diagnostic states are retained, exported and checked through
Rust at 44.1/48 kHz, mono and silent-left stereo, then discarded for full training.
Preflight stops on numerical or contract failure; its scores select neither an
arm nor a learning rate.

During full training, checkpoints preserve Adam moments, complete update history,
registration signature and exact update index. Backbone equality is checked at
each checkpoint. Resume verifies the latest hash and restores optimizer history;
it is not a fresh restart. All four final endpoints must reach exactly 16,384
updates and freeze before any comparative review. Intermediate losses are
telemetry only. No control is shortened after seeing a candidate's result.
P67 remains a historical reference with a different budget and learning-rate
schedule; a P69/P67 difference alone cannot isolate the new observation's effect.

The focused implementation checks cover genuine Adam resume with bit-identical
next updates, learning through the added observation columns, unchanged backbone
tensors, equal initialization, zero-parent loading, complete-endpoint rejection,
final report layouts, silence/weak gradients, finite differences and export
contracts. All 25 targeted tests pass; Ruff and formatting checks pass.

## Automatic evaluation after training

The frozen program is configured to run the complete sequence without another
training decision:

1. Freeze all four endpoints and recheck inputs/reservations.
2. Score direct, raw and fused losses on the fixed eight-draw TRAIN panel.
3. Review all four endpoints and both zero-phase parents on P63's source-isolated
   32-batch raw / eight-batch native panel. Include legacy and identity-DSP native
   controls; retain all per-recording spectral, energy, dynamics and PCM outcomes.
4. Measure all four trained endpoints on the same 52,000-hop compute grid, after
   training has stopped. Keep the 5 ms compute limit and hop deadlines separate;
   fixed 256-frame buffering is excluded. Preserve all late calls without retry.
5. Run the P55 20-source / 160-case physical-stereo long-form raw/native recheck,
   including untreated input and controls. These recordings were previously
   exposed and are not an independent test. Prepared listening material is not
   human listening evidence.
6. Write source-equal candidate/control and candidate/zero-parent comparisons and
   the P69 completion receipt.

A useful direction requires lower source-equal native paired-music MR against
both the repeated-input control and zero parent in both seeds on the reserved
review. Report HF shape, band energy, dynamics, identity, relative/absolute PCM
and every counterexample jointly. A small waveform regression alone is not an
automatic rejection; a small spectral mean gain alone is not qualification.
P55 results, training totals and earlier runtime measurements cannot substitute
for independent generalization, listening or actual-device evidence.

## Operational evidence

Local root: `data/runs/cross-phase-training-p69-20261101/`.

Launch verified on 2026-11-01: PID 77959 runs the frozen source from commit
`18a80ed`, with a separate `caffeinate -i -w` process. All 93,288 registered
input/source bindings verify, and all four two-update preflights finish with
sixteen native replays. Maximum native complex discrepancy is `7.689e-6`; the
preflight receipt verifies. The first full-training checkpoint is hash-valid,
finite and exactly preserves every P63 backbone tensor. Full training is active;
no endpoint or quality comparison has completed. The current step belongs to
`progress.json`, not a promise that this launch snapshot stays current.

`registration.json` and its receipt fix the protocol. `source/training/` contains
the executed snapshot. `launch.json`, `run.log` and `progress.json` describe the
live process; `training/*/latest.pth` and accompanying hashes record actual
updates. `preflight-complete.json`, `endpoints.json`, `complete.json` and finally
`p69-complete.json` distinguish the execution stages. An isolated old completion
file is not proof that P69's comparison and final checks finished.

Failures create timestamped records and stop execution. Incomplete native renders
are retained rather than overwritten. Training sources, checkpoints, local
reports and original audio remain ignored; only source and documentation are
committed. No default-weight promotion is part of this run.
