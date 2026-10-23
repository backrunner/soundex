# Real-Time Audio Repair Model Roadmap and Capacity Experiment

Historical protocol and results. See [research status](research.md) and the
[failure registry](failed-experiments.md) for current decisions; translation does
not turn reused validation into an independent test or update old acceptance limits.

Survey Date: 2026-10-11. Objective: Restore compressed audio through high-frequency inpainting, with real-time output in Rust + DSP closely matching the original lossless reference. The results reported in the papers are authors' reports on their respective tasks and cannot be directly ranked into a single SOTA leaderboard across datasets, sample rates, or damage types. Furthermore, no evidence is provided to confirm that SoundEx achieves SOTA performance. Three sets of training and actual output, along with resource measurement results, are detailed in [Experiment Report](architecture-study-20261011.md).

## Project Constraints

Adhering to the [Real-Time Standard](realtime-standard.md): 44.1/48 kHz native sample rate, FFT256/hop128, mono/stereo, fixed 256-frame software timeline. Additional software latency target <8 ms, redline <10 ms. Each hop has only 2.667/2.902 ms processing time—must verify tail latency and enhancement availability. Export limits remain at 2,000,000 parameters and per-file size <8 MiB; increasing these limits does not allow candidates to automatically pass.

The overall audio RTF, GPU throughput, and parameter count do not equate to callback latency. Analysis windows, future information, chunk buffering, model computation, and worker scheduling must be separately accounted for. MAC estimates are used for initial filtering; final evaluation is based on ORT/Rust real-world measurements of p99, deadline miss, CPU usage, peak RSS, and output continuity.

## First-Hand Materials and Adaptation Assessment

| Route | Author Report/Configuration | Assessment for SoundEx |
|---|---|---|
| [Apollo](https://arxiv.org/html/2409.08514v1) | Targeted at compressed music, band-level gain-shape representation, band-level RoFormer and temporal TCN; 16.54M parameters; 20 ms window, 10 ms hop; table reports 53.23 ms as GPU processing time for 1 second of audio | Band-level joint amplitude-phase encoding is worth considering; original scale and window size do not align with project budget |
| [UniverSR，ICASSP 2026](https://arxiv.org/html/2510.00771v2) | Complex STFT flow matching, no independent vocoder; 57M parameters; 1024-point window at 48 kHz; 4-step midpoint ODE with CFG | Complex STFT representation and low-frequency conditioning are instructive; original model exceeds scale budget; 4 steps ≠ 4 simple forward passes |
| [FlashSR](https://arxiv.org/html/2501.10807v1) | Single-step diffusion distillation, with VAE and SR vocoder; student LDM alone ≈ 258M parameters; measured at 5.12 seconds per audio segment on A6000 | Single-step fails to eliminate large model, long block, and vocoder costs; retained as offline quality path; not integrated into real-time pipeline |
| [A2SB](https://arxiv.org/html/2501.11311v2) | 565M parameters; amplitude and circular phase decomposition; bandwidth extension typically uses 50x sampling | Representation consistency and reconstruction fidelity merit investigation; original model cost is incompatible |
| [SwinDS-BWE, 2026-10 preprint](https://arxiv.org/html/2610.09189v1) | 17M generator; local attention, cross-stream interaction; primarily evaluated on English/French speech | "Parameter-efficient" still approximates 8.5x the project budget ceiling; speech results do not validate music restoration quality |
| [Multi-Rate BWE by Token Completion, 2026-09 preprint](https://arxiv.org/html/2609.38502v1) | Freeze codec and predict RVQ tokens; 6-layer, 1024-wide Transformer; codec token rate 93.75 Hz or 25 Hz | Original token period ≈ 10.67/40 ms, which is incompatible as a direct substitute for our 128-frame block-by-block processing; this is not an end-to-end latency measurement |
| [Real-time flow matching demo, ICASSP 2026](https://cmsworkshops.com/ICASSP2026/view_demosession.php?mid=55) | Author reports total latency 48 ms on consumer GPU, with 32 ms algorithm and 16 ms computation; single-channel speech | Indeed considers streaming buffering, but this configuration still exceeds the project's redline |
| [RESTORE, 2026-09 preprint](https://arxiv.org/html/2609.28683v1) | HTDemucs backbone, independent high-frequency stem generation; author reports ~50x real-time on single GPU | Independent high-frequency branch is a viable design reference; throughput figure does not constitute proof of <10 ms streaming; historical audio restoration differs from current codec-paired task |

The official [UL-UNAS implementation](https://github.com/Xiaobin-Rong/ul-unas) provides engineering references for lightweight speech enhancement and streaming, but the denoising performance of the enhancer does not imply capability to recover missing musical high-frequency content. No pre-trained weights, training data, or code from these papers were introduced into this experiment.

License compliance must be reviewed layer by layer. The [Apollo repository](https://github.com/JusperLee/Apollo) declares CC BY-SA 4.0; the [A2SB official repository](https://github.com/NVIDIA/diffusion-audio-restoration) specifies non-commercial terms for both model and code; the [UniverSR repository](https://github.com/woongzip1/UniverSR) shows MIT license for code, though this alone does not complete a review of pre-trained weights or data sources. This round only references publicly available method descriptions, using original implementation and existing review data for training—no external models were used to generate distillation targets.

Note: Another work with the same name, [FlashSR](https://github.com/ysharma3501/FlashSR), uses a different HiERspeech++ path than the Im/Nam diffusion paper and cannot share its license or performance claims.

## Custom Implementation in This Round

Current 806,276-parameter model performs frequency-axis convolution, independently predicting per-frame. Previous rounds of loss adjustment and small cross-stream interactions failed to resolve the trade-offs between high-frequency energy, transients, and fidelity. Thus, this round fixes the V6 loss and DSP, focusing solely on input representation and capacity—without concurrently changing the loss, detector or gain policy.

New optional `SpectralRefiner` branch: all 129 frequency bins of a full frame are passed through a two-layer fully connected encoder, and the same hidden representation predicts amplitude and phase residuals per bin. It directly establishes long-range frequency bin mapping while preserving bin position, without requiring attention or kernel scheduling for short sequences, nor re-synthesizing low-frequency waveforms.

Two equivalent-parameter input encodings:

- `polar`: scaled dB, phase/π, relative square root amplitude
- `gain_shape`: same dB, relative square root amplitude × cos(phase)/sin(phase)

dB is capped to [-120, 60]; relative amplitude is `10^((dB - frame_max_dB)/40)`. This reduces the impact of random phase on weak frequency components as energy decreases. Circular encoding occurs only in the new branch; the parent model retains raw phase. Thus, **the entire model is not strictly 2π-invariant**. This is a simplification of representation, not a full replication of any of the above papers.

Residuals are constrained to ±12 dB and ±0.5 rad; output head is initialized to zero. The parent model's predictions and random number state remain consistent. Residual constraints do not guarantee perceptual quality; low-frequency preservation, fusion, silence gating, and peak control remain under the purview of existing DSP. The new branch does not introduce BatchNorm, time-domain smoothing, future frame access, or new buffering.

| Experiment | Parameters | Relative control parameter growth | stereo MAC/hop estimate |
|---|---:|---:|---:|
| control / V6 | 806,276 | — | 23,389,920 |
| polar-small / width128 | 905,734 | 12.3% | 23,587,808 |
| shape-small / width128 | 905,734 | 12.3% | 23,587,808 |
| shape-large / width512 | 1,399,942 | 73.6% | 24,574,688 |

MAC is an analytical estimate of Conv/Linear operations (including transposed conv border positions, excluding element-wise operations and backend optimizations). The large branch concentrates new parameters in a per-frame, one-time mapping to avoid repeating convolution across all frequency bins. Actual performance must be measured separately; inferences from this table are invalid.

## Training and Selection Criteria

Source code is frozen at `7cfaf78490f385712a65fbcb5f198b49622fc593`.
All new branches are initialized from the same V4 epoch4 parent checkpoint `a152f00c…ccab`:
8 epochs / 1,024 updates, AdamW, learning rate 2e-5, warmup128/cosine, clip1, BatchNorm statistics frozen, no GAN used.
8-frame sequences used for reconstruction loss; model itself remains T=1.
Reuses the same 300 music tracks and VCTK, with sampling ratio 253/254 and 1/254, same recipe and split.

The `control` variant reuses the pre-trained V6 model; its resolved config matches the generated control configuration exactly, with identical parent model, seed, and schedule. This round re-executes CPU FP32 896-line validation and PyTorch/ORT/Rust alignment, with identical export hash and CPU score as the prior V6.
This is not a repeat training with a new random seed—thus, no inference can be drawn about subseed variance.

Each arm is selected by validation total, with high-band magnitude breaking ties.
A single CPU FP32 V6 total is used across all four arms to select candidates; after freezing ONNX hash, a held-out recheck is performed.
Development set, historical held-out, actual Rust output, silence, pulse, stereo isolation, and cost are reported.
Original audio, experimental weights, and reports are preserved in the ignored directory and not pushed.

Prepared configuration:

```bash
cd training
python scripts/prepare_architecture_study.py \
  --parent ../data/runs/consistent-continuation-20261011/checkpoints/best-validation.pth \
  --output-dir ../data/runs/architecture-study-YYYYMMDD/experiments
```

Run the frozen source code with:
`train.py --config <absolute configuration path> --initialize-generator-from <parent checkpoint> --initialize-spectral-refiner`
in each independent run directory.
The `control` variant omits the last flag.
Configuration and selection rules are recorded in `experiments/study.json`.

## Boundary of Future Directions

This experiment does not incorporate temporal context and thus cannot validate or refute the benefit of causal TCN/GRU.
The next structural design should enable worker caching of past frames or hidden states, incorporate causal temporal modeling, and ensure full consistency in training warmup, ONNX state protocol, Rust per-channel reset, and frame drop recovery.
Using only past frames does not require forward-looking, but state maintenance and computational overhead must be measured—cannot be claimed as zero cost.

The short window itself is an information bottleneck: FFT256 has frequency bin spacing of approximately 172.3 Hz (44.1 kHz) and 187.5 Hz (48 kHz).
Expanding network width for the current frame does not extend observation duration.
Future work should prioritize "current frame band encoding + small causal TCN/GRU" to cache past encodings and estimate envelope, harmonic, and transient changes via effective phase variations.
For example, 8 hop history covers ~21–23 ms, but this does not imply a need to wait 21–23 ms for future audio.
Start-up, skip-frame, silence re-entry, and left/right channel state must have clear handling protocols.
This is a pending implementation path; the stateless model and parameter count in this round do not include it.

On the training side, longer paired sequences can be used to add multi-resolution reconstruction supervision to constrain narrowband harmonics and percussion details that are hard to distinguish with short FFTs.
A longer loss analysis window does not require inference of future frames; however, the generator's causal input and effective support region must remain consistently cropped.
This direction should be pursued in a separate, fixed-budget comparison to avoid conflating changes in training objectives with gains in representation or capacity.

Learnable band gain is also a candidate direction: V6 clean-target proxy diagnostics show that fixed MatchEdge is not an identity operation. However, disabling it would cause the current model to underfill high frequencies.
Joint training with actual detector and gain history validation is required—cannot achieve better-looking waveforms by simply attenuating enhancement.

If future distillation is used, the teacher model must originate from reusable data/weights and be recorded in training source documentation.
Quantization, pruning, wider networks, codec tokenization, or generative model substitution must all demonstrate both quality and deployment benefit.

Future implementation order should be determined by results from this round: prioritize adding causal modeling of past frames within the same budget, followed by longer reconstruction supervision, and then independently validate learnable gain.
The current round's performance should be retained as a baseline.
Each step must maintain fixed training budget and be validated on new independent recordings before expanding training.
Further increases in the width of the current stateless branch or simple continuation of the same training should not be prioritized solely based on a minor loss reduction.
