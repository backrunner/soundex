# 实时音频修复的模型路线调研与容量实验

调研日期：2026-10-11。目标是压缩音频的高频 inpainting，并以实际 Rust + DSP
输出接近原始无损参考为准。这里的论文结果是作者在各自任务上的报告，
不能跨数据集、采样率、损伤方式直接排成一个 SOTA 榜，也没有证明 SoundEx 达到 SOTA。
三组训练及实际输出、资源测量结果见[实验报告](architecture-study-20261011.md)。

## 项目约束

沿用 [实时标准](realtime-standard.md)：44.1/48 kHz 原采样率，FFT256/hop128，
mono/stereo，固定 256 帧软件时间轴；新增软件延迟目标 <8 ms、红线 <10 ms。
每 hop 的处理时间只有 2.667/2.902 ms，必须核查尾部时延和增强可用率。
导出上限仍为 2,000,000 参数、单文件 <8 MiB；不能通过提高这些上限让候选自动过关。

整段音频的 RTF、GPU 吞吐和参数量都不等于回调延迟。分析窗、未来信息、分块
缓冲、模型计算和 worker 调度须分开记账。MAC 估计用于筛选，最终以 ORT/Rust
实测的 p99、deadline miss、CPU、峰值 RSS 和输出连续性验收。

## 一手资料与适配判断

| 路线 | 作者报告/配置 | 对 SoundEx 的判断 |
|---|---|---|
| [Apollo](https://arxiv.org/html/2409.08514v1) | 面向压缩音乐，频带 gain-shape 表示、频带 Roformer 与时间 TCN；16.54M 参数；20 ms 窗、10 ms hop；表中 53.23 ms 是 GPU 处理 1 秒音频的时间 | 跨频带与幅相联合编码值得借鉴；原始规模和窗口不能直接满足本项目预算 |
| [UniverSR，ICASSP 2026](https://arxiv.org/html/2510.00771v2) | 复数 STFT 上 flow matching，无独立 vocoder；57M 参数；48 kHz 下 1024 点窗；4 步 midpoint ODE 并带 CFG | 复数表示、低频条件编码可借鉴；原版远超规模预算，4 步不等于 4 次简单前向 |
| [FlashSR](https://arxiv.org/html/2501.10807v1) | 单步扩散蒸馏，加 VAE 与 SR vocoder；仅 student LDM 就约 258M 参数；速度在 A6000 上用 5.12 秒音频测量 | 单步不能消除大模型、长块和 vocoder 成本；保留为离线质量路线，暂不接入实时链 |
| [A2SB](https://arxiv.org/html/2501.11311v2) | 565M 参数；幅度与圆周相位分解；带宽扩展通常用 50 次采样 | 表示与重建一致性值得研究；原模型成本不合适 |
| [SwinDS-BWE，2026-10 预印本](https://arxiv.org/html/2610.09189v1) | 17M 生成器；局部注意力、跨流交互；主要为英/法语音实验 | “参数高效”仍约为项目上限的 8.5 倍；语音结果不能直接证明复杂音乐恢复质量 |
| [Multi-Rate BWE by Token Completion，2026-09 预印本](https://arxiv.org/html/2609.38502v1) | 冻结 codec 后预测 RVQ token；6 层、宽 1024 的 Transformer；codec token 率 93.75 Hz 或 25 Hz | 原 token 周期约 10.67/40 ms，已不适合作为我们 128 帧逐块处理的直接替代；这里并非端到端延迟实测 |
| [实时 flow matching 演示，ICASSP 2026](https://cmsworkshops.com/ICASSP2026/view_demosession.php?mid=55) | 作者报告消费级 GPU 上总延迟 48 ms，其中算法 32 ms、计算 16 ms；单声道语音 | 确实考虑了流式缓存，但该配置仍超出本项目红线 |
| [RESTORE，2026-09 预印本](https://arxiv.org/html/2609.28683v1) | HTDemucs 骨干，独立高频生成 stem；作者报告单 GPU 约 50 倍实时 | 独立高频分支可作为设计参考；吞吐数字不构成 <10 ms 流式证明，历史录音修复也不同于当前配对 codec 任务 |

[UL-UNAS 官方实现](https://github.com/Xiaobin-Rong/ul-unas)提供了轻量语音增强及
流式实现的工程参考，但增强器的降噪成绩不证明它能恢复被裁掉的音乐高频。
没有把这些论文的预训练权重、训练数据或代码引入本轮实验。

授权也需分层核对。[Apollo 仓库](https://github.com/JusperLee/Apollo)声明 CC BY-SA 4.0；
[A2SB 官方仓库](https://github.com/NVIDIA/diffusion-audio-restoration)分别声明模型和代码的
NVIDIA 非商业条款；[UniverSR 仓库](https://github.com/woongzip1/UniverSR)显示 MIT 代码许可，
但这本身不完成预训练权重与来源数据的复用审查。本轮只参考公开方法描述，采用本项目
原创实现和既有审查数据训练，没有以外部模型生成蒸馏目标。
检索中还有 [另一个同名 FlashSR](https://github.com/ysharma3501/FlashSR)，其
HiERspeech++ 路线与 Im/Nam 的扩散论文不同，不能混用它们的许可或性能声明。

## 本轮定制实现

当前 806,276 参数模型仅在频率轴卷积，逐帧独立预测。此前几轮 loss 调整和小型
跨流交互没有解决高频能量、瞬态与保真的全部取舍。因此本轮固定 V6 loss 和 DSP，
单独检查输入表示和容量，而不是继续同时改 loss、检测器、增益和结构。

新增可选 `SpectralRefiner`：完整一帧的 129 个频点进入两层全连接编码器，由同一
隐藏表示预测各频点的幅度/相位残差。它直接建立远距离频段间的映射，保留频点位置，
不需要给短序列增加 attention/kernel 调度，也不重新合成低频波形。

两个等参数量输入编码：

- `polar`：缩放后的 dB、phase/π、相对平方根幅度。
- `gain_shape`：相同 dB、相对平方根幅度乘 cos(phase)/sin(phase)。

其中 dB 限制到 [-120, 60]，相对幅度为 `10^((dB - frame_max_dB)/40)`。
这让弱频点的随机相位影响随能量下降；圆周编码只发生在新增分支，原有父模型仍
包含 raw phase，因此**整个模型并非严格的 2π 不变模型**。这也是我们自己的简化
表示，并非复现上述任何论文的完整模型。

残差分别限制在 ±12 dB 和 ±0.5 rad；输出 head 以零初始化，父模型预测与随机数
状态均保持一致。限制残差本身不保证听感，低频保留、融合、静音门和峰值控制仍由
既有 DSP 负责。新分支不引入 BatchNorm、时域平滑、未来帧或新增缓冲。

| 实验 | 参数 | 相对 control 参数增长 | stereo MAC/hop 估计 |
|---|---:|---:|---:|
| control / V6 | 806,276 | — | 23,389,920 |
| polar-small / width128 | 905,734 | 12.3% | 23,587,808 |
| shape-small / width128 | 905,734 | 12.3% | 23,587,808 |
| shape-large / width512 | 1,399,942 | 73.6% | 24,574,688 |

MAC 为 Conv/Linear 的解析估计（包括 transposed conv 的补边位置，不含逐元素操作
和后端优化）。大分支将新增参数集中在每帧只执行一次的映射上，避免在每个频点
重复扩大所有卷积层；实际性能必须另测，不能由这张表推断。

## 训练与选择规则

源码冻结为 `7cfaf78490f385712a65fbcb5f198b49622fc593`。
所有新增分支从同一 V4 epoch4 父检查点 `a152f00c…ccab` 初始化：
8 epochs / 1,024 updates，AdamW，学习率 2e-5，warmup128/cosine，clip1，
冻结父 BatchNorm 统计，不使用 GAN；8 帧序列用于重建 loss，模型本身仍 T=1。
继续沿用 300 首音乐和 VCTK，抽样比例 253/254 与 1/254，配方和划分不变。

control 复用已完成的 V6 训练，其 resolved config 与此次生成的 control 配置
完全相同、父模型/种子/schedule 相同；本轮重新执行 CPU FP32 896 行验证与
PyTorch/ORT/Rust 对齐，导出哈希和 CPU 分数均与既有 V6 相同。
这不是一个新随机种子的重复训练，不能据此估计多种子方差。

每个 arm 按 validation total 选 epoch，high-band magnitude 打破平局。
用同一 CPU FP32 V6 total 在四个 arm 中选择候选，冻结 ONNX 哈希后再做留出复核。
开发集、历史留出、实际 Rust 输出、静音/脉冲/声道隔离和成本分别报告。
原始音频、实验权重和报告保留在忽略目录，不推送。

准备配置：

```bash
cd training
python scripts/prepare_architecture_study.py \
  --parent ../data/runs/consistent-continuation-20261011/checkpoints/best-validation.pth \
  --output-dir ../data/runs/architecture-study-YYYYMMDD/experiments
```

在每个独立 run 目录运行冻结源码中的 `train.py --config <绝对配置路径>
--initialize-generator-from <父检查点> --initialize-spectral-refiner`。
control 不传最后一个标志。配置和选择规则记录在 `experiments/study.json`。

## 后续路线的边界

这组实验尚未引入时间上下文，不能验证或否定 causal TCN/GRU 的收益。
下一种结构应在 worker 缓存过去帧或隐藏状态、增加因果时间建模，并让训练 warmup、
ONNX 状态协议、Rust 每声道 reset、丢帧恢复完全一致。只使用过去帧可不增加前瞻，
但状态维护和计算开销仍需实测，不能直接宣称零成本。

短窗本身也是信息瓶颈：FFT256 在两种采样率下的频点间隔约 172.3/187.5 Hz。
增加当前帧的网络宽度不会增加观测时长。后续优先考虑“当前帧频带编码 + 小型
因果 TCN/GRU”，以缓存过去的编码和有效频点相位变化来估计包络、谐波与瞬态。
例如 8 个 hop 的历史覆盖约 21–23 ms，但不意味着需要等待 21–23 ms 的未来音频；
启动、跳帧、静音重入和左右声道状态必须有明确的处理协议。这是待实现的路线，
本轮的 stateless 模型和参数统计不包含它。

训练侧还可在更长的成对序列上增加多分辨率重建监督，以约束短 FFT 难以区分的
窄带谐波和打击乐细节。较长的 loss 分析窗本身不要求推理读取未来帧；但必须
保持生成器的因果输入与有效支持区裁剪一致。该方向应另起固定预算对照，避免
把训练目标改变与本轮的表示/容量收益混为一谈。

可学习频带增益也是候选方向：V6 的 clean-target 代理诊断显示固定 MatchEdge
并非正确输出的恒等操作，但直接关掉它会使现有模型高频欠填充。需要联合训练并
用实际检测器/增益历史验收，不能通过衰减增强换取更好看的波形指标。

如后续使用蒸馏，教师也必须来自可复用数据/权重，并计入训练来源记录。
量化、剪枝、扩大宽度、改为 codec token 或生成式模型都须同时证明质量和部署收益。

后续实现顺序应由本轮结果决定：优先在相同预算内加入过去帧的因果建模和更长的
重建监督，再单独验证可学习增益；保留本轮表现作对照。每一步都须固定训练预算，
在新独立录音上确认后再扩大训练。继续增加当前无状态分支宽度或单纯延长同一
续训，不应仅凭总 loss 的小幅下降获得优先级。
