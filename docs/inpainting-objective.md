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
