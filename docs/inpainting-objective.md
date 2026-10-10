# 高频频谱 inpainting 与模型/DSP 协作

## 目标与判断标准

SoundEx 以原始无损音频为恢复参考：模型估计压缩裁减的高频谱形、细节、
能量包络及相位，DSP 做频带融合、响度控制、伪影抑制和安全输出。
最终评价对象是经过完整 Rust 流处理后的音频。增加高频、减少 waveform loss、
或者保持压缩输入不变，单独都不能证明成功。

带宽延展需要同时处理谱形、音调属性和时域包络；不恰当的补频可能造成底噪、
嗡鸣、非谐波拍频及瞬态拖尾。原理与反例见
[AudioLabs 的 BWE 教程](https://www.audiolabs-erlangen.de/content/resources/aesCodingTutorial/bwe.html)。
这份资料中的编码器/解码器方法使用高频侧信息；SoundEx 对已解码 PCM 做盲恢复，
没有原始高频侧信息，不能据此声称精确逆转压缩。
神经网络预测参数、DSP 生成信号的协作路线可参考
[Grumiaux / Lagrange 的音乐 DDSP 带宽延展研究](https://arxiv.org/abs/2311.07363)。
当前实现不是该研究模型或某种标准 codec 的复现。

## Objective v3

`training/configs/inpainting_continuation.yaml` 是新实验配置。V1/V2 的公式与历史
checkpoint 保持可读，不把不同版本的 total 直接比较，也不沿用旧优化器续跑新目标。
V3 的损失项如下，所有采样质量比较仍需经过真实部署处理链：

| 项目 | 定义与作用 | 初始权重 |
|---|---|---:|
| 高频 dB 幅度 | 缺失区掩码内，对同一个参考相对底线截断后的预测/参考 dB 做 L1 | 1 / 20 |
| 高频谱收敛 | 缺失区线性幅度残差的 L2，相对该录音的参考高频幅度范数 | 1 |
| 高频能量包络 | 每帧缺失区平均功率相对参考的绝对 dB 差 | 0.5 / 20 |
| 高频时序细节 | 预测/参考的相邻帧**有符号幅度差**之差，相对参考变化范数 | 0.25 |
| 重建后高频 | 分频融合、相位插值、因果 Hann OLA、截取完整支撑样本后，重新分析频谱并比较高频谱收敛 | 0.5 |
| 有效高频相位 | 按参考有效能量加权的 `1 - cos(相位差)` | 0.05 / 2 |
| 波形保真 | 完整支撑区的重建 L1，相对同录音干声误差 | 0.25 |
| 保留频段 | 预测低频相对输入的 dB L1 | 0.1 / 20 |
| 分频接缝 | 过渡区残差的频率变化约束 | 0.05 / 20 |

高频 dB 的共同底线为 `max(参考每帧峰值 - 80 dB, -120 dB)`。它避免仅填入
底噪就因改变数值底线获得明显收益；线性谱、能量与波形项继续惩罚错误的高频。
相对范数有参考全频 RMS 的 1% 与绝对 `1e-4` 底线，能量比较使用对应的功率底线。
尺度逐录音计算并 detach，不能通过改变预测值扩大分母。安静素材附近的绝对底线
有意打破严格增益不变性，以约束梯度；不是静音素材应被忽略的理由。

时序项比较真实参考的变化，**不要求输出高频恒定**。将瞬态抹平或错移，同样
会产生误差。训练取 8 个因果帧，但每帧仍独立送入生成器；导出的推理输入保持
`[batch, 2, 1, 129]`，不会新增运行时前视或上下文缓冲。无 GAN，BatchNorm 统计冻结、
仿射参数继续训练，以减少短程初始化变化带来的干扰。

初始权重已在原生训练集 64 次抽样上测量参数梯度，不是仅依据 loss 数值配置：

| 加权项 | 平均参数梯度 L2 |
|---|---:|
| 高频幅度 / 谱收敛 / 能量 / 时序 | 1.676 / 3.060 / 0.806 / 0.711 |
| 重建后高频 | 2.618 |
| 相位 / 波形 | 1.639 / 1.124 |
| 保留频段 / 接缝 | 0.170 / 0.084 |

这是 MPS、父模型 epoch168、冻结 BN 模式下的初始局部诊断，梯度方向之间会有
抵消，不能直接把范数相加为“贡献比例”，也不能保证整个训练过程或最优权重。
损失单测覆盖参考完美恢复、缺失高频、过量补频、底噪、瞬态抹平、相位/OLA 伪影、
增益缩放与有限梯度；真实音质仍需要开发对照和独立测试/盲听。

### 部署代理的边界

训练的可微重建包含分频、相位插值、OLA 与重新分析，但**不包含**运行时带宽
检测器、响度匹配、限幅器、湿干渐变及异步超时回退状态，也未包含实验性频谱补丁。
其中的掩码来自数据配对的测量截止频率，运行时则来自在线检测。
`reconstruction_high_spectral_convergence` 是可微代理误差，不能叫完整 Rust DSP
误差。候选导出后必须在同一二进制、同一素材上验证最终音频，并分别报告音乐和语音。

## 模型/DSP 对照路径

`EnhancementMode::Neural` 保持已有默认链：检测 → 神经幅度/相位 → 响度匹配 →
分频/相位融合 → OLA → 限幅/湿干渐变。

`Spectral` 是独立的盲 DSP 延展基线，不加载、不调用模型：只转置保留频段上部
的频谱，在补丁边界渐变，随频率衰减，按分析帧位置修正相位旋转；根据供体频段
谱平坦度与截止频率降低纯音/低截止频率情况下的生成量，并跟随输入能量。
不处理 DC/Nyquist，不在静音中生成细节。它不是准确的谐波补全算法。

`Hybrid` 保留模型相位和约 750 Hz 子带的能量上限，由 DSP 提供子带内候选纹理，
按平坦度置信度最多混入 25% 的功率权重。供体候选先受模型子带能量约束，不提高
该子带总候选功率上限，且不修改保留区、分频过渡区或 Nyquist。它是可验证的
实验路径，尚未作为默认音质方案。

这两条新路径都复用现有 STFT，没有新增 FFT 音频缓存或前视延迟；固定 hop API
的 Rust 分配测试、任意分块、静音声道和 reset 测试已覆盖。它们仍须完成听感
和真实设备测试，连续有限输出不代表没有音质伪影。

```bash
# 无需模型的 DSP 基线
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --enhancement-mode spectral --output target/spectral.wav --bits 24
# 同一训练权重的默认 / 组合路径
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --model /path/to/candidate.onnx --enhancement-mode neural --output target/neural.wav --bits 24
cargo run --locked --release -p soundex-cli -- input.mp3 \
  --model /path/to/candidate.onnx --enhancement-mode hybrid --output target/hybrid.wav --bits 24
```

实验结果与默认方案决策见[本次优化记录](optimization-followup-20261010.md)。

使用明确选择的验证行重现三路径对照：

```bash
cargo build --locked --release -p soundex-core --bin soundex-stream-eval
PYTHONPATH=training python -m evaluation.compare_extension \
  --model /path/to/candidate.onnx \
  --manifest /path/to/music/manifest.jsonl --manifest /path/to/speech/manifest.jsonl \
  --selection /path/to/validation-selection.json \
  --binary target/release/soundex-stream-eval --output target/extension-comparison.json
```

选择文件为 `{"split":"validation","row_ids":["..."]}`，应在候选打分前固定。
工具核对验证成员、录音训练泄漏、配方一致性及音频校验值，保持相同选择并输出
配对录音聚类 bootstrap，以及逐语料/codec/采样率/声道角色/品类结果。角色音频是
单声道；这些结果不能代替真正立体声声像测试。所有报告明确标为开发诊断。

## Objective v4：复数重建与频谱一致性

V4 保留 V3 的高频恢复项，另外监督幅度和相位共同形成的复数频谱。令 `C` 为
分频/相位融合后的候选频谱，`R` 为无损参考，`D` 为压缩输入，`P` 为因果 OLA
后重新分析的 STFT，`M` 为缺失区权重。只比较重叠窗口完全支撑且时间对齐的帧：

| 新项 | 残差 | 归一化参考 | 实验权重 |
|---|---|---|---:|
| 重建后高频复数误差 | `sqrt(M) * (P(C) - P(R))` | 无损高频复数范数 | 0.2 |
| 重建后保留频段污染 | `sqrt(1-M) * (P(C) - P(D))` | 输入保留频段复数范数 | 8.0 |
| 高频频谱一致性 | `sqrt(M) * (P(C) - C)` | 无损高频复数范数 | 0.1 |

各残差取逐录音 L2 相对误差；分母使用相应频段范数、该参考全频范数的 1% 和
`1e-4` 的最大值并 detach。一致性项的分母来自无损参考，不能由预测缩放分母。
保留频段项单独约束增强引入的污染；接近无损的目标仍由高频、完整波形等参考项
共同监督，不能通过抑制补频只降低保留区误差。

复杂频谱和重新提取频谱一致性的思路参考
[AP-BWE 论文的谱域训练目标](https://arxiv.org/html/2401.06387v2#S3.SS2.SSS1)。
这里使用 SoundEx 的因果窗、缺失区权重和逐录音尺度，未移植其代码、预训练权重、
GAN 判别器或论文性能结论；原研究是语音任务，不代表音乐恢复已验证。

V4 初始梯度审计仍用真实训练集 64 次抽样、V3 epoch8、MPS、冻结 BN：新增三项
加权参数梯度 L2 为 `0.891 / 0.367 / 0.547`。保留区项初始权重 1 的梯度仅约
0.046，因此本轮在训练前调整到 8。这是局部尺度诊断，不是最佳超参数证明。
单测覆盖有效 STFT 的精确恢复、相位损伤引起的频谱不一致/低频泄漏、静音、空
缺失区、有限梯度，以及高频塌缩不会被奖励。

V4 同样不包含在线检测、实时响度匹配、限幅、湿干渐变或回退。训练与实际 Rust
部署的偏差仍需实测，不能将 V4 代理误差称作完整 DSP 误差。

## 幅度/相位分支交互实验

`cross_stream_interactions: true` 在四个编码层加入同时计算的 `1x1` 双向投影：
幅度分支接收相位特征，相位分支接收幅度特征，然后分别进入后续编码、瓶颈与解码。
投影零初始化，因此从兼容父模型迁移时初始输出精确保留；后续训练学习交互。
构造零投影不消耗对照组之后的 CPU RNG，避免改变训练裁剪和基础随机参数。

该变体为模型结构 `1.1`，参数从 806,276 增至 848,900（+5.29%）。运行契约仍为
`[batch,2,1,129]`，各帧独立；未添加时间卷积、状态、前视或缓存。实际处理成本需
重新测量，不能用参数增长比例代替延迟测量。

`consistent_continuation.yaml` 和 `interacting_continuation.yaml` 使用相同 V4 目标，
每组 6 轮/768 updates、冻结 BN、无 GAN、相同 seed/数据/采样/训练预算，分别隔离
新目标和额外分支交互的影响。抽样沿用既定音乐/语音比例，验证分别报告两类素材。

交互迁移必须显式声明，拒绝其他结构、数据或特征契约的差异：

```bash
# 先将 profile 的 data 路径和抽样配置解析为既有原生数据配置。
PYTHONPATH=training python training/train.py --config /path/to/resolved-interacting.yaml \
  --initialize-generator-from /path/to/parent.pth --initialize-zero-interactions
```

只迁移父生成器，新增投影必须全零，优化器/判别器重新初始化；改变结构或目标
不能以旧 optimizer 直接 resume。结果见[后续优化记录](optimization-20261011.md)。

## 高频电平的部署对照

`HighBandGain::MatchEdge` 保留默认截止边缘匹配（目标高频/边缘 RMS 比为 -6 dB）。
实验性 `HighBandGain::Model` 直接保留模型预测电平，仍经过分频、相位、限幅与
湿干渐变；Spectral 模式不使用神经候选电平匹配。这是显式对照选项，尚未证明
所有素材更接近无损，因此不替换默认。

CLI 用 `--high-band-gain model`，Rust bridge 用 `high_band_gain="model"`，其报告
绑定所选策略；旧报告只可按既有 `match-edge` 默认读取。电平对照结果及限制见
[后续优化记录](optimization-20261011.md)。

## Objective v5：重建能量与相位变化

V4 在部分谱形和瞬态指标上改善，但实际部署的高频能量仍可能不足。V5 增加
两个互补约束：先经过因果 OLA 再测量的能量，以及相对于无损参考的相位变化。
历史 V1–V4 公式不变；比较父模型与新候选时，统一重新用 V5 打分。

| 项目 | 定义 | 权重与尺度 |
|---|---|---:|
| 重建高频能量 | 完整支撑区重新 STFT 后，缺失区平均功率相对无损参考的绝对 dB 差 | 与下一项取平均，再乘 `1 / 20` |
| 重建子带能量 | 每 8 个频点为一子带，先跨帧平均功率，再比较绝对 dB 差；仅计有效子带 | 同上 |
| 相位频率变化 | 相邻频点的预测/参考相位差之差，使用 `1-cos` 的圆周误差 | `0.0025 / 2` |
| 相位时间变化 | 相邻帧的预测/参考相位差之差，使用 `1-cos` 的圆周误差 | `0.025 / 2` |

重建能量的参考底线为无损全频平均功率的 `0.01²` 与 `1e-4²` 的最大值，并
detach；尾部不足 8 点的子带补零权重。相位变化使用相邻点有效参考相位权重的
最小值，不惩罚无损参考中真实存在的瞬态或相位变化。V5 profile 同时将原始
高频逐帧能量项权重由 0.5 提到 1.0，其余 V4 项保留。

相位变化约束的思路参考
[AP-BWE 的相位训练目标](https://arxiv.org/html/2401.06387v2)。这里只使用原理，
仍采用 SoundEx 的有效高频掩码、因果帧和尺度，没有引入该项目代码或权重。
训练代理仍不包含在线截止频率、边缘电平匹配、限幅、湿干渐变与异步回退；
V5 数值不能代替实际 Rust 输出的音质与连续性验证。

在父模型、64 次训练集抽样上审计后，新增高频/子带能量加权梯度 L2 为
2.196 / 2.045，相位频率/时间项为 2.142 / 1.385。初始频率项曾为 21.421，
已在训练前下调；这些是局部梯度诊断，不证明超参数最优。

## 圆周相位输入实验

`circular_phase_features: true` 将相位分支首层输入从原始相位扩为
`[phase, sin(phase), cos(phase)]`。原始通道保留，新通道卷积权重零初始化，
显式迁移父权重后逐步学习；仍保留 raw phase，因此整个网络不保证严格的
`2π` 输入不变性。单测验证新增特征可获得梯度、迁移初始输出在容差内一致、
基础初始化 RNG 不变以及没有跨帧/未来帧依赖。

结构版本为 `1.2`，参数 806,420，比基础结构多 144（0.018%）。推理契约仍是
`[batch,2,1,129]`，没有新增前视、上下文缓存或运行时状态。与 V5 原结构对照
采用同一个 V4 父权重、数据划分、抽样、seed 和各 8 轮/1,024 updates；验证和
实际成本单独测量，不能从参数数目推断音质提升或延迟不变。

```bash
# data 与 sampling 先解析为既有原生数据配置；仅迁移生成器。
PYTHONPATH=training python training/train.py --config /path/to/resolved-circular.yaml \
  --initialize-generator-from /path/to/parent.pth --initialize-phase-features
```

`--initialize-phase-features` 仅允许这一首层从 1 扩为 3 通道；其他配置和参数形状
必须兼容，新增通道必须全零，优化器重新初始化。此开关不能与交互结构迁移混用。
实验决策和结果见[目标复核与改良记录](goal-refinement-20261011.md)。

## V6 deployment-aware continuation (experimental)

`training/configs/deployment_continuation.yaml` retains V5 raw-model terms and
applies the default Neural/MatchEdge gain before every reconstructed waveform,
complex, energy and spectral term. The proxy uses the manifest cutoff/rate and
Rust's rounded edge-bin geometry, -6 dB target ratio, [0.25, 2] gain target, 0.1
linear smoothing, and gain only where crossover weight is positive. This does
not change the exported model architecture or Rust runtime.

A cropped sequence has no prior stream gain. V6 initializes it from the first
frame's **detached** target gain and then follows the causal recurrence. This is
a steady-state approximation; it is not exact full-stream gain history. The
online cutoff detector, enhancement gate, limiter, wet/dry ramp and async
fallback remain outside the proxy. Static manifest cutoffs can disagree with
the detector. Actual Rust audio remains decisive. An identity generator scored
with V6 also passes through this DSP proxy and is not the untreated dry baseline.

V6 adds `reconstruction_temporal_weight = 0.25`: signed adjacent-frame magnitude
changes after causal OLA and reanalysis, compared with the clean reference.
The denominator is the reference change norm, floored by 1% of reference
full-band RMS times the square root of the difference-element count and an
absolute 1e-4 floor; all scales are detached. Both attacks and decays are
supervised. This is distinct from canonical evaluation's positive spectral flux.
It cannot reward constant spectra when the reference changes. Raw and
reconstructed temporal errors are reported separately. V1–V5 formulas remain
unchanged; total losses from different versions must not be compared directly.

The first controlled run uses the same V4 parent, 8 epochs/1,024 updates, frozen
BatchNorm buffers, canonical data recipe/splits/ratios and 896 validation rows.
Select its best checkpoint by fixed V6 validation total, then freeze its hash
before the existing held-out recheck. Compare Rust output with dry input, original
native baseline, V4 and V5; keep experimental unless gains are balanced across
energy, shape, transients, fidelity and continuous output.
