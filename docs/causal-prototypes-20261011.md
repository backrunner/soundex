# 结构重建第一阶段：Oracle、显式状态主干与小样本学习

2026-10-11，实施基点 `64bcbbc`。这是[结构重建计划](architecture-reset-20261011.md)
的 P0/P1 实验记录。两个原创主干和六次小样本训练已完成；尚未完成 P2 全库、
旧主干控制、双种子同预算训练。没有发布新恢复权重或替换默认策略。

## 1. 先检查 DSP 是否破坏正确答案

新增 `soundex-oracle` 仅替换模型输出，保留生产 Rust 的输入检测、增益、相位融合、
交叉频带、OLA、gate 和 limiter。Observed / clean magnitude / clean phase /
clean complex × MatchEdge / Model gain 共八组。Clean 只构成离线答案，
不参与输入检测或 gate，不是可部署模型。

固定旧 `shape-small/best.onnx`：
`bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`。
同一 89 行开发选择，按录音聚合/bootstrap；已经使用过的数据，不是新独立测试。
下面是全片实际 Rust 输出，非训练 loss，HFSC/energy/flux 越低越好。

| Clean complex oracle | 增益 | HFSC | 高频绝对能量误差 dB | flux 误差 | SI-SDR dB |
|---|---|---:|---:|---:|---:|
| 音乐 | MatchEdge | 0.61503 | 2.33333 | 0.91253 | 27.20916 |
| 音乐 | Model | 0.38059 | 0.77544 | 0.70378 | 27.87846 |
| 语音 | MatchEdge | 0.44475 | 2.13039 | 0.52911 | 29.71703 |
| 语音 | Model | 0.32582 | 0.90755 | 0.41828 | 29.89717 |

音乐只有 clean magnitude、保留压缩输入相位、Model gain 时，HFSC 为 0.61204，
能量误差 2.48035 dB。正确幅度仍不够，相位及实际重建路径必须一起优化。
Oracle 不为零也符合预期：保留的低频仍来自压缩输入，检测器、交叉频带和 gate 仍在运行。
这些成绩不能当作新模型质量，也不能据此把 Model gain 应用于所有旧模型。

**决策：新主干采用 Model gain + Model phase；P2 控制也必须采用相同策略和目标。**
旧默认仍为 MatchEdge。可学习增益留作后续独立消融。

## 2. 联合复数主干已经实现

输入为有幅度尺度的压缩 Re/Im、log magnitude 与原生采样率；共享编码后经过真正的
时间主干。输出为加性复数残差，允许填入原本接近零的高频；DC/Nyquist 虚部固定为零。
没有导入外部模型源码、权重、音频或新依赖。具体接口见[状态契约](causal-state-contract.md)。

| 原型 | 参数 | 未训练 ONNX bytes | 每声道状态 floats |
|---|---:|---:|---:|
| 两层 GRU，width 192 | 643458 | 2590696 | 384 |
| 六层缓存 TCN，width 144 | 639906 | 2592496 | 9072 |

两者约 2.47 MiB，均低于 <=2M / <8 MiB 预算。不是上一份文档的简化纸面计算结构。
无长窗、无 GAN、无可学习增益，本阶段先隔离主干可行性。

首次 64 hop 推进历史并输出对齐 dry；静音/bypass 仍更新所有物理声道状态；
reset、序号缺口、reload 清空历史。ORT prewarm 不算真实音频历史。
新增计数区分“按时收到输出”和“仍在状态预热”：后者可能有正常输出包，却没有增强。

验证覆盖：逐帧/序列/不规则分块一致、无未来依赖、batch 与独立声道、相位 2π 等价、
静音梯度有限、64 hop dry 启动、reset/reload、真实 worker 缺帧后等于全新实例，
以及 callback/固定 hop 的 Rust 堆零分配。C/C++ 内部分配由运行成本另测。

96 hop × 44.1/48 kHz × mono/stereo × 两主干，含 silent-left 和第 75 hop reset：
PyTorch/ORT/Rust 全部通过预登记 Cartesian `atol=1e-5, rtol=1e-4`；状态
`atol=2e-5, rtol=2e-5`。最大 Cartesian 绝对差 GRU 1.573e-5、TCN 3.563e-5，
绝对差须结合相对容差解释。实际 Rust 整段/不规则 chunk 差为零。
导出器也在发布文件前运行 80 hop × 两采样率/两 batch 的状态与频谱校验，失败保留原文件。

## 3. 六次小样本诊断，不能当作泛化验证

固定三个已有 **train** clip：两首 Chris Zabriskie 音乐（44.1/48 kHz），一条 VCTK
语音（48 kHz），每条 24576 samples。选择活动高频，来源和完整文件校验和记入报告。
这不是曲风覆盖集；均衡三条样本只用于梯度/表示诊断，**不改变后续全库采样比例**。
全库仍须遵守音乐 253/254、语音 1/254。

每臂随机初始化，seed 20261011，AdamW lr 3e-4，256 updates。64 hop 无梯度真实
burn-in + 128 hop 监督（约 0.34–0.37 秒）。使用已有 **V5** 权重与尺度、FFT256/H128、
steady-state 波形、GAN=0，增益为 Model；没有把 V7 MatchEdge loss 静默改成另一目标。
每条损伤输入通过 Rust 提取实际 cutoff 轨迹，而非直接使用 manifest cutoff 训练。

可微代理包含频带融合/相位/OLA，仍没有 sample gate 和 limiter；V5 重建谱收敛项用
时间平均 mask。实际 Rust 音频另外评分，不能称代理与部署全链完全相同。
理想低通仅用于离线诊断损伤生成，不进入实时处理。三任务分别训练，不混合训练结果。

预登记门槛：最终总 loss <= 初始的 60%；非 identity 还需 raw HFSC <= 初始的 80%，
梯度必须有限。这只是可学习性筛选，不是质量晋级阈值。

| 任务 | 主干 | loss 初始 → 最后 | raw HFSC 初始 → 最后 |
|---|---|---:|---:|
| clean identity | GRU | 1.20234 → 0.00897 | 0.32530 → 0.00259 |
| clean identity | TCN | 2.05912 → 0.00889 | 0.63808 → 0.00251 |
| 理想 8 kHz 低通 | GRU | 2.90070 → 1.45915 | 0.78836 → 0.41865 |
| 理想 8 kHz 低通 | TCN | 3.26906 → 1.53075 | 0.94414 → 0.44895 |
| 真实 MP3 64k | GRU | 2.50053 → 0.97551 | 0.75666 → 0.23964 |
| 真实 MP3 64k | TCN | 2.90624 → 0.99925 | 0.92424 → 0.25213 |

六臂通过门槛，实际 CPU 两线程训练/导出/评分每臂约 23–36 秒。运行结束，无后台训练遗留。
这说明表示和梯度可以有效学习，不说明 256 updates 已收敛，也不能选择泛化胜者。

实际 Rust 对已见 MP3 音乐两条的 flux：原输入 0.92782/0.99935，GRU
0.83654/0.64398，TCN 0.85893/0.65998；GRU 的 SI-SDR 相对输入分别 -0.00043/-0.02901 dB，
语音为 -0.30248 dB。理想低通部分音乐仍损失 1.9–4.0 dB SI-SDR，第一条 flux 也退步。
**不能据此声称全面恢复或超越基线。** Identity Rust 输出 SI-SDR 约 88.7–98.1 dB；
相对原音自身 120 dB 上限的 delta 不应解释为普通有损输入上的同等音质损失。
本阶段没有评分盲听。

## 4. 初步运行成本与失败记录

M5 Max，ORT 单线程，FFT256/H128，Model gain，未训练权重。最初 core 每配置
3 秒是诊断运行，不能通过正式每配置 30 分钟压力门槛：GRU stereo p99
0.128–0.222 ms；TCN 0.260–0.743 ms。GRU 峰值 RSS 36.03 MiB，TCN 38.16 MiB。
TCN 48 kHz 的 core 跑分出现 14 次超 hop 周期尖峰（其中 max 10.501 ms），
不能只引用其 p99 而忽略尾部。原始失败记录保留。

首次每配置 30 秒 paced，producer 普通优先级，worker 请求音频 time constraint：
两主干 worker p99 上界均 <1.5 ms，但 GRU 一组 miss ratio 超 0.1%，两主干各有
callback max 超 500 us 的配置。有限值、长度、边界数值检查通过，仍不等于实际无爆音。
随后只把模拟 producer 也设为音频 time constraint，串行复测，另记录进程 CPU/RSS。
此设置用于区分调度因素，不能代替物理声卡回调，也不会抹掉首次失败。

完整复测数值见本页后附表。固定输出延迟始终是 256 samples（44.1k:5.805 ms，
48k:5.333 ms）；任何 64 hop 预热必须报告为暂不增强，不能增加到播放延迟。

## 5. 复现与后续门槛

本地证据均保留在忽略目录 `data/reports/architecture-reset-20261011/`：
`oracle.json`、`probe/probe.json`、`probe/{gru,tcn}/cost.json`、
`probe/{gru,tcn}/realtime.json`、`probe/rt-repeat-resource.json`、`tiny/report.json`。
代码和来源文档提交，原始音频、实验报告及六个 tiny 权重不推送。

从 training 目录运行 `PYTHONPATH=. .venv/bin/python scripts/probe_causal_core.py --help`
或 `scripts/fit_causal_tiny.py --help`。先编译 release 的 `soundex-state-parity`、
`soundex-stream-eval`、`soundex-gain-data`；oracle 用 `soundex-oracle`。
这些脚本是探针及小样本诊断，**不是完整 P2 训练入口**。

P2 仍须统一旧/新主干的目标、warmup/监督区域、采样、优化搜索和每臂 4096 updates ×2 seeds，
校验代理 gate/重建差异，使用实际 Rust 输出与压缩直通/旧模型对照。
之后再做长历史/纹理/可学习 DSP 消融。现有开发数据不得升级为新独立测试；新测试、
评分盲听、长时状态、物理设备和 x86_64 验收均未完成。

本轮本地检查：285 个 Python 测试通过；Rust workspace 测试通过（一个 30 分钟测试按默认
忽略）；Clippy `-D warnings`、无 ORT 后端构建和 Python lint/format 通过。远端 CI 单独记录。

## 受控复测完成值

每配置 30 秒，producer/worker 的音频 time constraint 请求均被系统接受；
无并行训练/编译。合成信号含 tone/noise/silence/transient，不是音质评测。
CPU 是 `/usr/bin/time -l` 的进程 user+sys / wall，以一个 CPU 核心的 100% 为分母，
包含模拟 producer、worker、ORT 和报告开销；RSS 是该次独立进程峰值。

| 主干 | Hz / ch | worker p99 上界 ms | callback max us | 软件附加 ms | CPU % 单核 | RSS MiB | 状态预热 hops |
|---|---|---:|---:|---:|---:|---:|---:|
| gru | 44100 / 1 | 0.225 | 8.583 | 5.814 | 5.48 | 31.91 | 64 |
| gru | 44100 / 2 | 0.550 | 14.250 | 5.819 | 14.09 | 31.72 | 64 |
| gru | 48000 / 1 | 0.350 | 21.125 | 5.354 | 8.59 | 31.75 | 64 |
| gru | 48000 / 2 | 0.500 | 13.417 | 5.347 | 11.26 | 31.42 | 64 |
| tcn | 44100 / 1 | 0.475 | 12.666 | 5.818 | 11.49 | 33.30 | 64 |
| tcn | 44100 / 2 | 0.875 | 20.167 | 5.825 | 23.18 | 33.61 | 64 |
| tcn | 48000 / 1 | 0.550 | 23.916 | 5.357 | 13.39 | 33.41 | 64 |
| tcn | 48000 / 2 | 0.900 | 21.250 | 5.355 | 25.50 | 33.61 | 64 |

八组均零 deadline miss / queue overflow / worker processing overrun，状态预热均只在开头 64 hop。
这是短时研发可行性结果。首次普通 producer 调度失败仍有效；不能外推所有宿主线程、
物理设备、长时间状态或训练后权重。P1 成本允许进入下一轮训练设计，正式实时门槛尚未全部完成。
