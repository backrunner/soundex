# P12: fixed-budget and fixed-exposure subset scaling

P11 showed that the current anchored amplitude network can fit five examples,
while P10 failed to improve its full-library common score. P12 isolates subset
size within one training protocol. This is a diagnostic study, not a model release
or an independent test. It does not yet change the generation head or phase model.

## Registered comparisons

The nested sets contain 5, 10 and 40 recording anchors, with music/speech counts
4/1, 8/2 and 32/8. The smallest set is P11's same five cases. Additional anchors
come from the existing train-domain probe list in deterministic row-ID order,
without reading quality results. Work-group/recording/source split checks remain
active. Each paired anchor has its matching clean-identity counterpart.

Each update processes four music examples and one speech example, plus their five
identity counterparts. Within each corpus, every recording is visited once per
shuffled cycle. This fixed 80/20 diagnostic mix is deliberately held constant as
set size grows; it does not replace the full-library 253:1 production sampling
policy. Changes cannot be attributed solely to size by directly comparing with
P10, whose crop/gain sampling, optimizer and corpus weights differ.

All groups use the same fresh P10 anchored model (807,398 parameters), P11 rendered
objective, input phase, 80-hop warmup and smooth DSP. Adam uses fixed learning rate
1e-3, no decay, and gradient clipping at 5. The rate was chosen from P11's training
fit results before any P12 development scoring. Seeds are 20261012 and 20261013.
No weights or optimizer states are transferred from P11.

Two budgets are reported:

| Recording anchors | Common-update checkpoint | Equal-exposure checkpoint | Exposures per anchor at the latter |
|---|---:|---:|---:|
| 5 | 256 updates | 256 updates | 256 |
| 10 | 256 updates | 512 updates | 256 |
| 40 | 256 updates | 2048 updates | 256 |

At the common-update checkpoint, per-anchor exposures are 256, 128 and 32.
At the equal-exposure checkpoint, total compute increases with subset size. These
are separate comparisons; neither alone isolates all effects of data diversity.
Batch size is fixed at ten physical mono-role clips. Frames and histories are
independent; unrelated roles are never presented as stereo pairs.

There is no validation-based checkpoint selection or early stopping. Both
predeclared milestone checkpoints are hashed before scoring new regions or
cross-recording development audio. Step-zero/input-DSP reference is reported.
The full development set is never fitted. Existing development recordings have
been used in earlier studies, so they are not a fresh independent test.

## Evaluation regions and reporting

- Anchors: all 40 fixed train-domain probes; fitted recordings and other train
  recordings are partitioned explicitly. Only the selected pool is trained.
- New regions: alternate crops in those same 40 recordings. The entire scored
  intervals are disjoint from anchor scoring intervals. Three short files cannot
  provide fully disjoint 32768-sample inputs, so some input/history context overlaps;
  no supervised samples overlap. These are same-recording probes, not independent
  recording generalization evidence.
- Development: all 89 existing clips, separated into music and speech. Every pool
  also includes clean-identity controls. Report original-clean restoration and
  incremental identity preservation separately.

All group scores use the same `0.8 * music + 0.2 * speech` weighting, even when the
number of development clips differs. Within each corpus use the mean per-example
P11 loss: equal restoration and identity risk, with original-clean rendered
spectral/dynamics plus 0.25 waveform error for paired restoration. Identity
preservation references existing DSP output; original-reference identity errors
remain in per-row reports. Do not compare these totals with P10's totals.

Native DSP replay uses a fixed five-pair subset from each of the three regions:
P11's five anchors, their shifted versions, and the first four distinct music
recordings plus first speech recording from sorted development rows. The subsets
are fixed before training. Replay both degraded and identity cases: 30 baseline
replays plus 30 for each of ten milestone checkpoints, **330** in total. The other
examples receive proxy evaluation, not native replay. Native parity requires
finite output and sample error <=2e-5; this is numerical correctness, not release
quality. Report high-band SC, energy, flux and low-band errors independently.

## Reproducibility and boundaries

`training/scripts/run_subset_study.py` registers data/source/detector/native-binary
hashes and draw schedules, saves optimizer/RNG state for resume, hashes milestones
before evaluation, and verifies completed artifact receipts. Source is frozen in
a local snapshot before the formal experiment. Artifacts and raw audio stay in
ignored `data/runs/subset-p12-20261013/`; no fitted weights are published.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_subset_study.py \
  --parent data/runs/rendered-p10-20261013/study/config.json \
  --output data/runs/subset-p12-20261013/study \
  --replay-binary target/debug/soundex-prediction-replay \
  --onnx data/runs/causal-p3-20261012/study/control-20261012/best.onnx
```

Interpret fitting, same-recording and cross-recording changes together. Successful
anchor fits with poor shifted-region results suggest crop-specific learning;
successful shifted regions with poor development results suggest a recording/domain
gap. If common-update fits degrade with size but recover with equal exposure,
insufficient exposure contributes. Persistently poor larger-set fits motivate a
separate optimization/head-coordinate experiment. None of these outcomes alone
proves a capacity limit, rules out leakage in unrelated studies, or establishes
listening quality. No runtime changes or new streaming delay are introduced.


The two-update integration check completed all three evaluation regions and 60
native replays (30 baseline, 30 checkpoint), maximum sample error 2.981e-7.
All 424 local Python tests passed, as did Ruff lint/format checks. The formal
six-run queue completed with frozen source `ce6fd15`; `pipeline.json` and
`study/summary.json` both report completion. All milestone/checkpoint/evaluation
receipts and exact per-anchor exposure counts were verified during analysis.


## Completed results

All six runs and 330 native replays completed. The maximum native/proxy sample
error is 1.885e-6. Every anchor receives exactly 256 paired exposures and 256
identity exposures at its equal-exposure endpoint. Both code CI runs passed;
this is software correctness, not restoration qualification.

The following three scores use **identical evaluation cases across all sizes**:
the common five P11 anchors, their shifted scoring regions, and all 89 development
clips. This avoids confusing changes in the fitted set's difficulty with learning
progress. Lower is better; no development checkpoint was selected.

| Size | Seed | Updates | Common anchors | Common shifted regions | Development |
|---|---:|---:|---:|---:|---:|
| Input/DSP | — | 0 | 0.306496 | 0.313313 | 0.271475 |
| 5 | 20261012 | 256 | 0.245245 | 0.714323 | 2.523811 |
| 10 | 20261012 | 256 | 0.294612 | 0.360389 | 0.816346 |
| 10 | 20261012 | 512 | 0.273871 | 0.322062 | 2.016946 |
| 40 | 20261012 | 256 | 0.306103 | 0.316296 | 0.338419 |
| 40 | 20261012 | 2048 | 0.296676 | 0.424947 | 0.511675 |
| 5 | 20261013 | 256 | 0.232373 | 0.728909 | 2.879109 |
| 10 | 20261013 | 256 | 0.297357 | 0.334839 | 0.660467 |
| 10 | 20261013 | 512 | 0.267971 | 0.365623 | 2.032494 |
| 40 | 20261013 | 256 | 0.306262 | 0.319017 | 0.341236 |
| 40 | 20261013 | 2048 | 0.296164 | 0.403989 | 0.446079 |

At the same 256 updates, increasing the training pool reduces the severe
small-set development regression. It does not produce a development win over
input/DSP. With 40 anchors, development is still worse by 24.66% / 25.70% at 256
updates. Extending to 2048 updates worsens that to 88.48% / 64.32%, while the full
fitted-anchor score improves by 6.45% / 6.72% from its own input/DSP baseline.
The ten-anchor runs show the same direction: more exposures improve fitting but
worsen development. Thus insufficient per-anchor exposure does not explain the
missing generalization under this fixed-crop protocol.

These percentages describe this normalized composite score, not perceived quality
or a percentage of recovered information. The protocol uses fixed crops, no gain
augmentation, an 80/20 diagnostic corpus mix and learning rate 1e-3. It cannot
establish that every previous full-library failure has the same cause; P10 had
other training settings, including crop/gain sampling. Both seeds support
fragment-specific fitting and a broader generalization gap in **this** protocol.

Native evidence on the fixed five development pairs independently exposes the
tradeoff. At 40 anchors / 256 updates, mean high-band SC changes by
-0.004019 / -0.009561, but identity SC worsens by +0.033996 / +0.031766. At the
2048-update endpoints, paired SC changes by +0.012544 / -0.002230 and identity SC
by +0.098312 / +0.025745. High-band energy error can improve despite those fidelity
failures; it must not be used as a standalone success measure. These are five-case
native diagnostics, not whole-library native validation or statistical evidence
of listening quality. No candidate is promoted.

The next isolated intervention is fixed versus resampled positions **within the
same selected recordings**, with model, objective, gain policy, corpus ratio and
update budget fixed. First test whether same-recording new regions improve, then
check cross-recording development. Do not simultaneously increase capacity,
change phase or relax identity guards. A later separate head-coordinate study
can investigate remaining generation/optimization limits. Simply extending this
fixed-crop run is not supported by the results.

Reproduce the comparison report after training:

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_subset_study.py \
  --study data/runs/subset-p12-20261013/study
```

`scaling-comparisons.json` verifies all registered checkpoint/evaluation receipts,
compares matching cases, records per-corpus restoration/preservation deltas and
per-clip native deltas, and retains the analysis-script hash. The analysis also
passed a real artifact smoke check. Published weights, model architecture, DSP
and realtime behavior remain unchanged.
