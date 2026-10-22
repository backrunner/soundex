# SoundEx

[English](README.md) · [CI](https://github.com/backrunner/soundex/actions/workflows/ci.yml) · [Apache-2.0](LICENSE)

SoundEx 是实验性的压缩音频恢复库，目标是对被裁减或损伤的高频频谱进行 inpainting，
结合 DSP 融合和伪影抑制，让最终音频尽量接近原始无损参考。项目使用 Rust 实现
DSP、ONNX 推理和实时流适配，同时提供文件处理 CLI 和 PyTorch 训练工程。

**当前状态：**源码与接入 demo 可用，尚未发布合格的音质恢复权重。
仓库附带的合成恒等测试模型只用于验证接入，不代表音质提升。
详细进展见[模型说明](models/README.md)。

[P38 检测条件对照](docs/detector-condition-study.md)已启动 CPU 小模型训练：两组、
两个种子，各 16,192 步。完整预检及 653 项 CPU 测试通过，全部端点冻结后再统一
复核 current 和仅相位融合两条路径。P36 原本也使用 CPU，Metal 故障不阻塞这条训练路线。

[P37 融合边界对照](docs/fusion-boundary-audit.md)已完成 2,848 次冻结模型原生回放。
取消幅度回混、保留相位融合，相对同权重现有 DSP 使音乐 MR 误差下降 0.12%–0.14%，
动态小幅改善，但 identity 仍退步。融合确有限制，尚不足以解决模型恢复问题。
12 万 hop 最大计算耗时 0.992 ms，无 hop 超时或 paced 迟到。未晋级权重；
后续由 P38 进行等预算检测条件训练对照。

[P36 因素对照](docs/risk-attribution-study.md)已完成四条新增等预算轨迹。两个种子中，
仅误差压缩使原生音乐 MR 误差下降 1.27%–1.42%；仅恒等降权下降 1.03%–1.14%，但保真
代价更大。两组仍未通过动态/identity 筛选，未晋级权重。28 万 hop 最大计算耗时
1.278 ms，零次 ≥5 ms，有 1 次 paced 完成迟到；固定缓冲另计。

[P35 广覆盖对照](docs/broad-risk-study.md)已完成四条 16,192 步训练和原生复核。
两个种子的候选部署音乐 MR 误差比同期对照降低 1.12%–2.13%，语音降低 3.07%–4.34%；
动态与部分 identity 指标仍退步，未通过预登记筛选，未晋级权重。28 万 hop 最大计算
耗时 1.948 ms，零次 ≥5 ms，有 1 次 paced 完成迟到；固定缓冲延迟另计。

[P34 小模型调试](docs/risk-balance-study.md)已完成三轮、五个等预算 1,024 步端点，
始终使用同一约 43 万参数结构。平滑风险压缩消除了本轮梯度裁剪，中间恒等权重使
未参与拟合的训练域音乐部署 MR 误差比对照下降 0.305%，但动态／identity 仍有退步。
追加过补约束没有改善整体取舍。这是单种子、有限裁剪池诊断，不是独立泛化证明，
没有晋级正式恢复权重。

[P33 广覆盖频谱训练](docs/broad-spectrum-study.md)已完成四条 16,192 步训练和原生复核。
新增多分辨率监督没有在隔离录音复核上改善音乐；语音频谱均值改善 0.86%–1.26%，
但动态误差退步。音乐恢复仍弱。28 万 hop 中 2 次超过 5 ms（最大 8.445 ms），
未提升默认权重。

[P32 频谱优先监督](docs/spectrum-priority-study.md)已完成六条 1,024 步诊断。
增加多分辨率监督后，拟合训练裁剪的音乐实际部署频谱误差比同预算对照下降
2.02%–2.35%；瞬态、波形保真与跨录音泛化仍待解决。42 万 hop 中 91 次超过
5 ms（最大 45.00 ms），实时长尾尚未达标。预检与 620 项测试通过，未提升默认权重。

[P31 相位输入／复数坐标对照](docs/phase-coordinates-study.md)已完成八条 512 步训练、
320 个完整原生角色与 56 万 hop 计时。相位输入改善部分幅度指标但波形退步，
全局坐标组接近恒等输出，尚无整体恢复收益；9 次计算超过 5 ms（最大 14.10 ms），
79 次 paced 完成迟到。未提升默认权重。

[P30 幅度／相位诊断](docs/phase-swap-audit.md)已完成 1,320 个原生 DSP 回放角色，
确认相位保真与频带融合仍限制恢复；参考辅助结果不代表新模型已取得该收益。

[P29 直接频谱监督对照](docs/direct-spectrum-study.md)已完成。固定训练材料上，
直接监督让模型补出了更多高频并改善能量误差，但复数谱、瞬态与最终波形保真
没有同步改善；未提升默认权重，也尚未验证泛化。

[P28 小模型机制对照](docs/small-mechanism-study.md)已完成：43.1 万～45.6 万参数，
四组、两个种子、每条 2,048 步，分别验证解码器归一化、预算梯度隔离与观测相位输入。
三种干预均未通过双种子改善筛查；诊断样本没有出现解码器失活，输出仍接近恒等 DSP。
56 万 hop 计算均低于 5 ms，但有 5 次服务耗时超过 hop 周期、32 次 paced 完成迟到，
尚无整体音质恢复证据，不提升默认权重。

[P27 诊断](docs/head-collapse-audit.md)发现残差解码器大量失活，旧 1,192 万参数路线
保持停止扩训。DSP 激励与谱形滤波的组合是后续实验，尚未训练出该路线的权重。

[P26](docs/broad-budget-study.md)已完成八条完整训练轨迹、原生复核和 56 万 hop
计时，未建立整体恢复收益。本次计算均低于 5 ms，但仍有逐 hop 超时，没有提升默认权重。
历史对照、结论及复现入口集中在[实验索引](docs/research.md)。

经核查的曲库共 300 首原生无损音乐，另有 VCTK 语音。原始音频、配对数据和实验检查点
保留在本地。使用入口见[训练指南](training/README.md)，来源与授权记录见
[数据文档](docs/datasets.md)，历次方法和结果集中在[实验索引](docs/research.md)。

## 直接运行

需要 **Rust 1.88+** 和 C/C++ 编译工具链。默认构建会下载 ONNX Runtime，首次构建需联网。
在仓库根目录执行：

```bash
git clone https://github.com/backrunner/soundex.git
cd soundex
cargo run --locked -p soundex-core --example offline_demo
cargo run --locked -p soundex-core --example realtime_demo
```

离线 demo 生成两秒 48 kHz 双声道测试音频，通过内置 ONNX 模型处理，保存至
`target/demo/input.wav` 与 `target/demo/output.wav`。实时 demo 模拟每块 128 帧的回调，
输出截止时间、回退和 worker 统计，不打开声卡。

使用自己的合格 **256/128** 模型：

```bash
cargo run --locked --release -p soundex-core --example offline_demo -- /path/to/model.onnx target/demo
cargo run --locked --release -p soundex-core --example realtime_demo -- /path/to/model.onnx
```

## 文件处理

```bash
# 无需模型的分析
cargo run --locked -p soundex-cli -- target/demo/input.wav --dry-run
# 用测试模型验证完整处理流程
cargo run --locked -p soundex-cli -- target/demo/input.wav \
  --model tests/fixtures/low-latency-identity.onnx \
  --output target/demo/cli-output.wav --bits 24
```

实际恢复时应换成通过验证的训练模型。输入支持 MP3、AAC、FLAC、WAV、OGG Vorbis，
输出支持 16/24 位 PCM 和 32 位浮点 WAV。

## 实时流接入

在开始播放前于控制线程创建 `RealtimeProcessor`。默认先完成 worker 推理预热，
仅构造线程等待就绪（最长 10 秒）；音频回调仅调用 `process(input, output)`。
回调不执行推理、不分配堆内存、不加互斥锁；独立 worker 通过有界队列推理。
结果超时或异常时使用时间对齐的原音回退，并通过双声道联动渐变、输入清理与最终限幅
保持输出连续。创建、替换、销毁及 `shutdown` 必须放在控制线程。

实时配置：单/双声道 **44.1/48 kHz，FFT 256 / hop 128**，推荐与 hop 对齐的 **128 帧回调**。
固定新增延迟为 **256 帧**：44.1 kHz 下 **5.80 ms**，48 kHz 下 **5.33 ms**。
软件新增延迟目标 **低于 8 ms**、红线 **低于 10 ms**，包含回调耗时与额外接入缓冲。
声卡与驱动缓冲应另行测量；固定帧延迟不等于真实设备或完整链路的实测延迟。

[库接入示例与契约](docs/library.md) · [实时验收标准](docs/realtime-standard.md) ·
[时延与资源 benchmark](docs/performance.md)

文件处理使用同步 `SoundExProcessor`；它直接调用 ORT，不应放进音频回调。

## 训练与验证

[训练说明](training/README.md)包含环境安装、混音数据预处理、固定数据划分、CPU/MPS/CUDA
与导出命令。默认模型契约为 FP32 `[batch, 2, 1, 129]`，仅 batch 动态，
checkpoint schema 为 1.2，artifact schema 为 1.3。导出必须通过 ONNX 与 ORT 一致性检查；
Rust 在处理前验证模型元数据与张量契约。
可选[因果上下文](docs/causal-context-20261011.md)使用 artifact 1.4：九帧历史/当前输入、
一帧输出，不读取未来；host 保存每声道历史，八个 hop 的增强预热期间保持对齐原音。

[一致性误差预算](docs/parity.md) · [评估流程](training/evaluation/README.md) · [模型卡](models/MODEL_CARD.template.md) ·
[权重发布清单](legal/MODEL_RELEASE_CHECKLIST.md)

构建与检查命令见英文 README。[贡献说明](CONTRIBUTING.md)约定测试与提交格式。

## 开源许可

源码、文档、demo、合成测试图与**正式发布权重**统一采用 **[Apache-2.0](LICENSE)**。
正式权重允许其他开源项目集成、随软件分发、修改、微调和商用；发布时附模型卡、
校验值及适用的署名声明。

训练音频和依赖保留各自许可。现有 MUSDB 候选保留为未发布的研究记录，退出正式权重
路线。训练不设数据集或许可白名单，可直接扩充音频目录；正式发布时核对授权和模型来源。
依赖说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

[权重许可范围](MODEL_WEIGHTS_LICENSE.md) · [NOTICE](NOTICE) · [许可说明](legal/LICENSING.md) ·
[训练数据政策](legal/TRAINING_DATA.md) · [数据来源调查](docs/datasets.md)

Copyright 2026 BackRunner and SoundEx Contributors.
