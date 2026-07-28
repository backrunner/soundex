# Plan 005: Add Deployment-Equivalent Training Validation And Release Evaluation

> **Executor instructions**: Complete dependency plans first. Follow every
> verification command and stop on listed conditions. Do not train or publish a
> release model as part of this plan. Update `plans/README.md` when complete.
>
> **Drift check (run first)**: run
> `rtk shasum -a 256 training/train.py training/models/losses.py training/evaluate.py training/data/dataset.py training/configs/default.yaml crates/soundex-core/src/channel.rs crates/soundex-core/tests/processor_test.rs models/MODEL_CARD.template.md`.
> Expected hashes for the first seven files are
> `2c7f9d7437f8f81e8199fd1c107b796c24cf31f1eaa5fac19d04e74f95d37f8c`,
> `ad8fe32a5c07295af79f009136e752a644338f0c07aa07cf1953363a27e03202`,
> `c4748b6b038e37f95ef33b27d06cbb82be9a89b82986780cd38cc5d53a8665c5`,
> `982f5ea64876683577d6c82e352edb8ef19093c148b1384e2b007ace47a4bc87`,
> `5c9d9a19ac729d95708348223cfb873277c15ea5e6a6ed02c43fbf1491195360`,
> `79af24f3769f8023836bd08e42ca9525765102aba352f9215c2eb153c30832f9`, and
> `95f633dde245393b3e2038d35efa97af7e3d69ba0318207c57f9d1b8581912ec`.
> Dependency plans will intentionally change these files. In that case compare
> this plan's assumptions against the new contract and update the plan before
> execution. Git metadata was absent when this plan was written.

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: `plans/001-streaming-model-contract.md`,
  `plans/002-correct-rust-stream.md`, `plans/003-training-data-protocol.md`,
  `plans/004-reproducible-onnx-export.md`
- **Category**: tests
- **Planned at**: workspace snapshot without Git metadata, 2026-07-28

## Why This Matters

Current training loss is averaged over the whole spectrum even though Rust
discards the model's low-band prediction. Validation randomly samples a different
context each epoch and uses final test tracks, while `evaluate.py` compares only
one already-generated file and has no degraded baseline. A checkpoint can
therefore look better without restoring the missing high band, and no existing
gate proves that exported Rust streaming output improves held-out audio.

## Current State

- `training/models/losses.py:41-43` computes full-band magnitude L1 and unweighted
  phase loss across bins with arbitrary low-energy phase.
- `training/data/dataset.py:86-89` returns only waveforms, with no codec/cutoff or
  missing-band mask.
- `training/train.py:287-291` randomly selects validation context and measures
  Python feature loss, not the deployed causal stream.
- `training/train.py:426-440` saves every tenth epoch but no best checkpoint.
- `training/evaluate.py:59-72` evaluates one predicted/reference pair with only
  full-band LSD and SI-SDR.
- `.agents/05-implementation-roadmap.md:166-171` requires significant LSD
  improvement over degraded baseline, subjective quality, and Rust/PyTorch
  parity. Only parity is handled elsewhere; quality gates remain absent.
- `models/MODEL_CARD.template.md:59-64` treats evaluation as optional and has no
  baseline, stratum, latency, or uncertainty fields.

## Commands You Will Need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Python tests | From `training/`: `rtk python3 -m pytest -q` | all pass |
| Eval smoke | From `training/`: `rtk python3 -m pytest -q tests/test_evaluation.py` | fixture report and gates pass |
| Rust tests | `rtk cargo test --workspace --all-targets` | all pass |
| Lint | `rtk ruff check training` | no findings |
| Format | `rtk ruff format --check training` | exit 0 |

## Scope

**In scope**:

- `training/models/losses.py`
- `training/train.py`
- `training/evaluate.py` or a focused `training/evaluation/` package
- `training/data/dataset.py` integration with plan 003 manifests
- `training/configs/default.yaml`
- `training/configs/titan_xp.yaml`
- `training/tests/test_losses.py`
- `training/tests/test_validation.py`
- `training/tests/test_evaluation.py`
- A Rust/Python evaluation bridge or CLI under the existing crates
- `.github/workflows/ci.yml` for synthetic evaluation smoke tests
- `models/MODEL_CARD.template.md`
- Evaluation protocol documentation

**Out of scope**:

- Downloading datasets or training final weights.
- Modifying source splits or codec generation, handled by plan 003.
- Changing the artifact schema, handled by plan 004.
- Hardware latency optimization, handled by plan 006.
- Treating a subjective listening note as a substitute for objective gates.

## Git Workflow

- Preferred branch: `test/deployment-equivalent-evaluation`.
- Suggested commits:
  `train(model): optimize the missing high band`, then
  `test(model): gate releases on Rust streaming evaluation`.
- Do not publish metrics from synthetic fixtures as model-quality claims.

## Steps

### Step 1: Define A Missing-Band-Aware Objective

Consume the degradation metadata/mask from plan 003. The mask must represent the
actually missing high band plus the crossover region, not a bitrate lookup alone.
Implement and log separate losses:

- masked high-band log-magnitude reconstruction;
- magnitude-weighted circular phase or complex-spectrum loss in bins with
  meaningful clean energy;
- a weaker low-band identity loss to prevent needless changes;
- crossover continuity loss; and
- deployment-equivalent waveform/overlap-add loss on short causal sequences.

Keep adversarial and feature matching terms separately observable. Do not let an
unweighted full-band scalar be the checkpoint-selection metric. Add synthetic
tests where only high-band error changes and prove the primary loss changes, and
where only irrelevant zero-energy phase changes and prove it does not dominate.

**Verify**: from `training/`, `rtk python3 -m pytest -q tests/test_losses.py`
passes deterministic numerical expected-value tests.

### Step 2: Make Validation Fixed And Deployment-Equivalent

Use the immutable validation rows/crops from plan 003. Run the exact plan-001
single-frame contract with the same causal zero history, STFT, hop, startup, gate,
crossover, and flush behavior as Rust. Either call the Rust evaluator directly or
share golden fixtures that continuously prove equivalence. Do not randomly choose
validation frames.

Report overall and per-corpus/codec/quality/sample-rate/channel-role metrics.
Enforce configured source quotas rather than merely including sources. Validation
must be bitwise repeatable on CPU for the same checkpoint/manifest.

**Verify**: two validation runs produce identical row IDs and metric JSON; a test
with shuffled DataLoader workers yields the same aggregate.

### Step 3: Preserve Final Test Data And Select The Best Checkpoint

Never access test manifest rows during `train.py`. Select and save
`best-validation.pth` using a documented high-band primary metric plus tie-breaker;
also save the final resume checkpoint. Store the full metric vector and manifest
hash in the checkpoint. Resume must preserve RNG/sampler state from plan 004 and
must continue the same validation schedule.

**Verify**: a synthetic metric sequence selects the known best epoch, resumes at
the next step, and never opens the test manifest.

### Step 4: Replace Single-Pair Evaluation With A Dataset Runner

Turn `evaluate.py` into a manifest-driven runner that takes a checkpoint or ONNX,
exports/validates if necessary, then processes each held-out degraded input through
the actual Rust streaming path using at least two chunking patterns. Compare three
signals: clean reference, degraded baseline, and enhanced output.

Compute at minimum:

- high-band and full-band LSD;
- spectral convergence and recovered high-band energy error;
- SI-SDR as a secondary full-waveform measure;
- low-band preservation error;
- ViSQOLAudio on supported music rows;
- stereo inter-channel correlation/phase and image-width change; and
- gate false-bypass/call rate using known degradation labels.

Use PESQ only on a separately labeled speech subset, never as the primary music
metric. Store per-row results and aggregate by stratum with bootstrap 95% confidence
intervals. Write JSON plus a human-readable Markdown summary.

**Verify**: fixture metrics have hand-computed expected values and two different
chunkings agree within `1e-5` after delay compensation.

### Step 5: Encode Machine-Checkable Release Gates

Put gates in versioned config and include them in reports. Initial minimum gates:

1. Enhanced mean high-band LSD is lower than degraded mean high-band LSD, and the
   bootstrap 95% confidence interval for `enhanced - degraded` is below zero.
2. No required corpus/codec/sample-rate stratum has a statistically supported
   high-band regression.
3. Mean low-band LSD degradation is below 0.25 dB and no non-finite/clipped row is
   accepted.
4. Streaming chunk patterns and offline output agree within `1e-5` after the
   declared latency/flush contract.
5. PyTorch/ORT/Rust parity from plan 004 remains green.

Add a separate mandatory listening protocol for release candidates: randomized,
level-matched, blinded degraded/enhanced/reference clips across strata. Record the
protocol and anonymized aggregate, but do not automate a subjective pass claim.

**Verify**: known-good and known-bad synthetic reports respectively pass and fail
with nonzero exit codes and clear failed-gate names.

### Step 6: Make Evaluation Required Release Metadata

Update the model card so evaluation is required, not optional. Include artifact
hash, checkpoint/data/config hashes, test manifest version, per-stratum baseline
and enhanced metrics with confidence intervals, listening protocol, supported
sample rates/codecs, algorithmic latency, real-model runtime results from plan
006, and known limitations.

**Verify**: add a model-card/release checker that rejects blank required fields
and metric reports whose artifact hash differs from the ONNX file.

## Test Plan

- Unit-test every metric against hand-computed arrays, including silence and
  unequal lengths (which should fail unless alignment metadata permits trimming).
- Test missing-band mask boundaries, phase wrap, zero-energy phase weighting,
  crossover loss, and low-band identity loss.
- Test validation determinism, per-stratum quotas, best checkpoint selection,
  no test-manifest access, bootstrap determinism, and release pass/fail.
- Use generated audio only in CI. Full licensed corpora run in a private release
  job and publish only permitted aggregate metadata.

## Done Criteria

- [ ] Training's primary objective explicitly weights the missing high band.
- [ ] Validation rows, frame positions, and corpus quotas are deterministic.
- [ ] Test rows are never used by the training loop or checkpoint selection.
- [ ] Best checkpoint selection uses a deployment-relevant high-band metric.
- [ ] Evaluation runs the exported model through Rust causal streaming and flush.
- [ ] Every report contains degraded baseline deltas and per-stratum uncertainty.
- [ ] High-band, low-band, perceptual, stereo, gate, and chunk-equivalence metrics
      are present with tested implementations.
- [ ] Release gates exit nonzero on regression and model cards require their output.
- [ ] Python/Rust tests, Ruff, Rust fmt, and clippy pass.
- [ ] Only in-scope files and `plans/README.md` changed.

## STOP Conditions

- Any dependency plan remains incomplete or its contract differs from this plan.
- Held-out test track IDs were previously used for tuning and cannot be replaced.
- ViSQOLAudio licensing/runtime cannot be included in the release environment.
  Report it and choose a reviewed alternative; do not silently omit perceptual
  evaluation.
- Delay/alignment between baseline, enhanced, and clean cannot be proven.
- A proposed aggregate hides a required codec/corpus regression.

## Maintenance Notes

- Metric and gate schema changes must be versioned so scores remain comparable.
- Keep row-level reports private when dataset terms require it; publish permitted
  aggregate results and exact protocol metadata.
- A future stateful model requires new chunk-boundary and state-reset evaluation,
  not reuse of the stateless equivalence claim.
