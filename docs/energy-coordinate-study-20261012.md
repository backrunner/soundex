# P7：缺失频带的能量参数化

承接 [P6 对照](factorized-envelope-study-20261012.md)，本轮只改变频带能量的输出坐标，
检验严重裁切时相对输入能量的乘法更新是否妨碍学习。以下规则在正式训练前登记。

## 假设与隔离探针

P6 的频带输出为 `s * exp(clamp(log(r/s) + d, -30, 2))`：`s` 是输入全谱 RMS，
`r` 是输入该带的平滑 RMS，`d = 4 * energy_head(latent)`。
当 `r/s = 1e-8` 且未触发限幅时，对 `d` 的导数只有 `r`。
零梯度缺陷已在 P6 修复，但非零梯度仍可能对应很差的优化条件。

候选使用 `s * min(sqrt((r/s + d)^2 + 1e-16), exp(2))`，在全谱 RMS 单位中直接
增加或抑制该带能量。平滑绝对值保证非负并避免零点不定梯度；上限与 P6 相同。
零头初始化保留近似恒等输出，缺失带只存在约 1e-8 的数值地板。
输出头参数数量、带内形状归一化、骨干和输入特征完全相同。
这仍是输入 RMS 上的加法残差，不冒称已经实现无锚点绝对包络预测。

固定已知均匀目标谱形，只训练一个能量偏置，lr=1e-4、AdamW、clip=5、128 更新：
严格缺失带的目标 RMS 为 0.01，乘法头 loss **1.94196→1.94196**，输出 RMS
**8.16e-10→8.60e-10**；加法头 loss **1.94195→0.25063**，RMS **1.15e-9→0.00418**。
这是双精度、单参数合成诊断；其初始输出能量差异仅来自数值地板。
新坐标也会显著放大初始梯度，因此保留梯度裁剪，并在真实训练中检查非有限值。
该结果不能证明骨干充分学习、真实音乐收敛或跨录音泛化。

复现入口：`training/scripts/diagnose_energy_coordinates.py`；本地结果为
`data/runs/energy-p7-20261012/coordinate-probe.json`。

## 正式对照

| 组别 | 能量输出 | 其他部分 |
| --- | --- | --- |
| factorized | P6 相对输入 RMS 的对数增益 | 共同初始化、特征、形状头、loss |
| rms_residual | 全谱 RMS 单位中的加法残差 | 同上 |

两个种子 `20261012/20261013`，每组从头训练 **2048 更新、batch 4**，总计四组。
同种子所有初始参数逐张量相同，均为 **807,398 参数**，不加载 P6 权重。
AdamW lr=1e-4，余弦衰减至 0.1 倍，weight decay=1e-4，clip=5。
采用 `seed+50000` 的新固定抽样，每种子 8192 个，其中 1024 个恒等控制。
沿用核验后的来源与录音隔离；语音抽样比例、恒等替换方式不变。

所有组保持过去 1024/2048 点上下文、80 hop 完整预热排除区和 P6 的能量/谱形/动态 loss。
共同评分、每 512 步验证、初始权重可被选中、音乐/语音及恒等权重均不变。
模型结构测试覆盖同初始化、能量/形状独立、零与弱输入梯度、有限差分、因果性、
分块和声道隔离、MPS/CPU 对应。继承的 loss 不改权重和数值地板。

## 判读与合成验证

预定比较为 `rms_residual - factorized`，两个种子分别做录音级配对 bootstrap。
共同 loss 只负责选模，不负责单独晋级；要求能量恢复与高频谱形共同改善，并审查
flux、保留低带、恒等、语音和压缩输入。区间跨零或种子不一致时保留不确定性。
全部最佳若位于预算边界，仍不能称已收敛。

训练完成后自动执行共同评分、第一种子候选 ONNX 导出、逐 hop CPU 成本与独立 RSS 测量。
诊断模型仍受 <=2M 参数、<8 MiB 限制。benchmark 不与本轮训练并行。
随后对对应种子的冻结 P3 相位做实际 Rust DSP 对照：原 P3、恒等 DSP、两个能量头配
P3 相位，以及候选配输入相位。全部开发 89 片段×5路径×2种子，共 890 次回放。
共同启动区和裁剪同 P6；同时报告动态 HF、固定 4 kHz 以上，以及同 FFT 网格的五频带
原始/输出指标，避免把口径差异误认为合成损伤。原生/代理最大误差门槛仍为 2e-5。

这是复用开发集的实验，不读取独立 test，不通过诊断重新选模或改阈值。
默认权重不替换；真实流式集成、完整延迟、稳定性和盲听仍未被这个幅度实验覆盖。

## 执行与产物

```sh
PYTHONPATH=training training/.venv/bin/python training/scripts/run_energy_study.py \
  --parent data/runs/factor-p6b-20261012/study/config.json \
  --output data/runs/energy-p7-20261012/study \
  --cache data/runs/factor-p6-20261012/study/cache

PYTHONPATH=training training/.venv/bin/python training/scripts/analyze_envelope_study.py \
  --study data/runs/energy-p7-20261012/study --benchmark

PYTHONPATH=training training/.venv/bin/python training/scripts/replay_factorized_study.py \
  --study data/runs/energy-p7-20261012/study \
  --phase-study data/runs/causal-p3-20261012/study \
  --replay-binary target/debug/soundex-prediction-replay
```

正式运行从提交后的冻结源码执行，记录 launch、pipeline 阶段和日志；检查点保存优化器与 RNG。
只共享按输入哈希寻址的缓存，不复用其他实验的权重。原始数据、检查点及报告保留本地忽略目录。
## 正式运行结果

四组均已完成 2048 更新，最佳权重均在末步；这不证明已经收敛。
实际 Rust DSP 完成 890 次回放，最大代理误差 **4.54e-6**，没有晋级默认权重。
同 P3 相位、同窗五频带口径的音乐开发片段平均结果：

| 路径 / 种子 | 输出高频 SC ↓ | 输出 flux ↓ | 输出能量误差 dB ↓ |
| --- | ---: | ---: | ---: |
| factorized / 12 | 0.6964 | 0.8072 | 6.8698 |
| rms_residual / 12 | 3.7355 | 2.7455 | 5.4420 |
| factorized / 13 | 0.6189 | 0.7636 | 7.0865 |
| rms_residual / 13 | 2.7273 | 2.3186 | 5.8591 |

按录音配对，SC 和 flux 的退步在两个种子的 95% bootstrap 区间均高于零；
能量误差改善的区间均跨零。选模分数下降不构成整体恢复收益。
第一种子候选 ONNX **3,277,624 B**，807,398 参数；CPU 前端+推理双声道 P99
为 44.1k **0.22883 ms**、48k **0.22050 ms**，独立 NumPy/ORT RSS 峰值增量 **20.8125 MiB**。
这些沿用上文诊断范围，不是完整部署延迟或内存。360 项本地测试及对应远程 CI 通过。

后续的[loss 与弱高频误补审计](energy-objective-audit.md)表明，原始 SC 退步主要由能量误补
驱动，不能把 SC 当作只反映归一化谱形的指标。新方案未被采用。
