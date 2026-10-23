# P22: Latency-First Capacity Expansion

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

2026-10-15 User lifted the volume ceiling on the experimental model and confirmed **5 ms as the model + DSP computation latency**, excluding fixed output buffering. The existing 256-frame buffer remains at 5.33 / 5.80 ms; model scaling does not extend lookahead. The 2M / 8 MiB production contract for formal artifacts has not yet been migrated. This round uses explicit, size-unbounded diagnostic exports, preserving numerical consistency checks and cannot be directly loaded into production models.

Subsequently, user requests a full post-training evaluation of capacity performance. An independent [P23 Full Budget Comparison](full-capacity-study.md) has been established. Early P22 evaluation results are retained and no longer serve as a barrier to long-training in P23; P22 itself was not re-run or rewritten.

## Cost Measurement

Apple M5 Max, CPU ORT single-thread, stereo, 128-frame hop; 100 warmup iterations measured, then 10,000 hops, including Python causal FFT context processing. During measurement, P11 CPU training remained active. All five scales passed 320-hop PyTorch/ONNX continuous state, mono/stereo, silence, reset, 44.1/48 kHz numerical consistency checks. Output heads used fixed-seed non-zero random weights; these are not training-quality audio results.

| width | parameters | ONNX | average | p99 | maximum | ORT RSS increment | initial screening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 224 | 1,748,578 | 6.70 MiB | 0.261 ms | 0.327 ms | 0.654 ms | 28.39 MiB | passed |
| 384 | 4,594,818 | 17.55 MiB | 0.502 ms | 0.576 ms | 0.726 ms | 47.45 MiB | passed |
| 640 | 11,917,698 | 45.47 MiB | 1.075 ms | 1.196 ms | 1.443 ms | 65.30 MiB | passed |
| 896 | 22,648,450 | 86.39 MiB | 2.397 ms | 3.782 ms | 30.650 ms | 114.39 MiB | on hold |
| 1280 | 45,134,338 | 172.15 MiB | 3.605 ms | 3.979 ms | 5.744 ms | 211.13 MiB | on hold |

RSS is the difference between ORT process memory snapshot before and after 100-hop warmup, including allocator and buffers, excluding PyTorch, FFT frontend, or full Rust—nor is it peak memory. The timing measured does not include full DSP, callbacks, device, wake-up, or queue wait, so it cannot be claimed that model + DSP has passed 5 ms validation. Input signals are fixed synthetic signals; real music, silence, and transients require post-training revalidation for actual deployment cost.

At 48 kHz, a new hop arrives every 2.667 ms, so average <5 ms may still accumulate. Initial screening requires: stereo average and p99 both <2 ms (leaving ~25% hop buffer for remaining DSP/scheduling), with observed maximum <5 ms. When replaying actual service duration at a fixed 2.667 ms hop interval, the virtual timeout ratio must be ≤0.1%, and tail accumulation must be zero. This replay only measures capacity pressure, not actual device deadline records.

All 224/384/640 configurations have zero virtual timeouts; 896 has 5,745/10,000 timeouts, peak backlog 796.86 ms; 1280 has 10,000/10,000 timeouts, ending with 9.38 seconds of backlog. The 896 fluctuations are tied to parallel load, so cannot be used to infer hardware limits. However, this round lacks sufficient headroom—its p99 <5 ms does not justify acceptance.

Full local report: `data/reports/latency-capacity-20261015/summary.json`.

## Calibration Results (2026-10-15 16:03 SGT)

P22 completed all 8 configurations, 4,096 updates, with all 8 training receipts and all 8 export receipts verified. All configurations failed pre-registered training optimization checks, including anchored control; no extended training was initiated, and no qualified weights were selected. 512 steps represent early calibration and do not establish a final capacity ceiling.

Complete 2,088-crop training pool, two seeds averaged (music/voice weighted 0.8/0.2, lower is better):

| Configuration | paired recovery error | identity original damage | main combined error |
| --- | ---: | ---: | ---: |
| Identity initialization | 0.585364 | 0 | 0.292682 |
| Anchored control | 0.584665 | 0.003496 | 0.294080 |
| scaled224 (175 × 10,000 parameters) | 0.582753 | 0.004961 | 0.293857 |
| scaled384 (459 × 10,000 parameters) | 0.582639 | 0.008002 | 0.295320 |
| scaled640 (1,192 × 10,000 parameters) | 0.583808 | 0.018202 | 0.301005 |

The recovery error for the three scaled widths decreased by 0.446% / 0.465% / 0.266% compared to initialization, but original audio damage offset the gains, resulting in a net increase in main error by 0.402% / 0.902% / 2.844%. Both seeds show consistent trends. The main/combined error on the shared 320-crop set did not outperform initialization. These results cannot be interpreted as evidence of completed held-out audio quality, Rust DSP, or perceptual validation. There is no evidence of overfitting to the validation set.

In the P11 equivalent budget at 512 steps, the gradient-clipping rate without complex head scaling was 86.3–89.1%. When changing 144 width to slope=1/64, it dropped to 1.17–2.15%; reducing only learning rate yielded 93.2–93.9%. This supports improved gradient scale sensitivity in output coordinates, but does not indicate that training objectives or final fidelity have been resolved. The 384/640 width gradient-clipping rates were 5.08–6.45% / 9.375%, still below the 25% threshold.

Post-training, each model per seed was re-measured for 10,000 hops, stereo p99: 224 at 0.307–0.308 ms, 384 at 0.565–0.566 ms, 640 at 1.108–1.109 ms; maximum 640 at 1.331 ms. All virtual timeouts under fixed hop intervals were zero. This still does not include full DSP/device validation.

Current evidence shows: under the current calibration budget and parameter settings, widening alone does not improve overall error. Future work should first diagnose the interaction between paired and identity gradients/updates, and the actual output step size as a function of width. Additionally, it must be determined whether short screening is sufficient to evaluate learning progress. Failure of anchored control at 512 steps does not justify dismissing all long training. No loss modification or new learning rate selection has been made based on this.

## Same-Budget Training

This experiment is independent of the frozen [P21](capacity-calibration-study.md), with no changes to P21's parameters or selection rules. P22 was launched and completed after the P21 pipeline exited, avoiding resource contention between full training pools and long training runs. If a prior run fails, it stops and reports failure—not as a successful completion.

- **anchored144 control + scaled224/384/640**, two seeds `20261012 / 20261013`;
  Each configuration per seed for 512 steps, total 4,096 updates; all initialized from scratch.
- Shared 2,088-crop training pool, common 320-crop, P21 calibration sampling order, main/aux loss and weights;
  4 music + 1 voice per step, paired/identity equally weighted.
- Scaled heads continue with `2*tanh(z/64)`, Adam 1e-4, no weight decay, gradient clipping 5;
  First correct known sensitivity issues, then observe expansion effects. Different widths do not share identical initialization tensors.
- Scale selection based solely on pre-training CPU cost, not development set quality. All cost-passing scales are included; no best seed or post-hoc reduction is selected.
- Shared/complete training pool reports main, paired, identity, and combined, preserving music/voice, and gradient clipping ratios. Each scale is re-exported, validated, and measured for 10,000 hops post-training.
- All training optimization checks are recorded; no long training is automatically triggered based on a single scale passing, and no weights are automatically released. Future work must freeze candidates before actual Rust DSP, reusing development set, new recordings, generalization, perceptual, and full chain real-time validation.

Run script: `training/scripts/run_latency_capacity_study.py`
Source code and input hashes are frozen, supporting 64-step checkpoint recovery. Start and end are both validated against input. Training directory:
`data/runs/latency-capacity-p22-20261015/study/`
Progress is tracked via `progress.json`, distinguishing queued, data preparation, training, and calibration completion. Queueing does not imply large model weights have been generated.

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/screen_latency_capacity.py \
  --output data/reports/latency-capacity-20261015
PYTHONPATH=training training/.venv/bin/python training/scripts/run_latency_capacity_study.py \
  --parent data/runs/capacity-p21-20261015/study/calibration \
  --screen data/reports/latency-capacity-20261015 \
  --output data/runs/latency-capacity-p22-20261015/study \
  --wait-for data/runs/capacity-p21-20261015/pipeline.json
```

Formal run uses the frozen source code `d1697c7`, launched at 15:27 SGT on 2026-10-15; historical supervision PID `93749`, child process `93751`; completed at 16:03 SGT and exited. Current status is `calibration_complete`. Logs: `data/runs/latency-capacity-p22-20261015/pipeline.log`. Runtime status is determined by `pipeline.json` and `study/progress.json`. The directory is registered and cannot be directly rerun upon source code modification.

Full Python test suite passed 496 items; subsequent addition of training summary double-seed/identity checks, 15 additional tests passed. Ruff and formatting checks passed. No blind listening or continuous stream device validation was conducted.

The largest scale label is 1,192 × 10,000 parameters (approximately 11.92M); the registered exact model count is in the configuration table. The eight-arm/seed calibration is an early optimization diagnostic, not a capacity ceiling.
