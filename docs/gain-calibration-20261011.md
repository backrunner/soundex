# 输入条件化高频增益校准

本轮针对V7发现的固定MatchEdge目标比例，保持shape-small神经权重不变，
拟合仅10个系数的输入条件化校准器。当前尚未完成质量与资源验证，不替换默认策略。

## 固定协议

- 使用实际ChannelProcessor逐hop分析，包含因果填充、静音reset、检测和context就绪。
  无损信号仅生成标签，不参与特征或门控；不是移植一套近似检测器。
- 特征为bias、cutoff/nyquist、sample_rate/48000、六个保留子带相对边缘的RMS，
  以及残留高频相对边缘RMS；dB特征除20、截断到[-3,3]。
- 训练标签是clean高频/输入edge的dB比相对原−6dB目标的偏移，截断±24dB。
  拟合10系数Huber ridge（3dB阈值、8次IRLS）；输出修正限制±6dB，最终目标[-12,0]dB。
  原[0.25,2]增益边界、0.1平滑、低频融合、相位、限幅和固定输出时间轴保持不变。
- 仅抽取train录音，每录音每采样率最多两种codec，各前4秒，跳过前32hop、每8hop取一帧；
  active且edge>1e-4、clean_high>1e-7才参与拟合。每录音等权，音乐/语音253/254与1/254。
  训练集内按录音hash划出约20%校准验证，选ridge 0.001/0.01/0.1/1，再用全部抽样train重拟合。
- 神经图固定为shape-small，SHA-256 `bdef9d1269ea4b8b1e8e5e04bd6d25032b817f5401e13c51e2c064d721503782`。
  校准器文件哈希在实际Rust质量评估前冻结。既有89行开发集仅用于之后的真实输出评分，
  不以这些结果回改系数或正则选择，也不称为独立测试。
- 同素材比较旧MatchEdge与校准器；质量、数值、连续性、实时成本和听感分别判断。
  这只校准增益目标，没有解决神经预测相位或谱形，也没有改变截止频率检测器。

## 复现与接入

构建`cargo build --release -p soundex-core --bin soundex-gain-data --bin soundex-stream-eval`。
在training目录使用`scripts/fit_gain_calibration.py --manifest <music-manifest> --manifest
<speech-manifest> --binary ../target/release/soundex-gain-data --output <new-directory> --context 1`。
输入必须是项目已经审查并按录音划分的配对清单；原始音频未包含在Git。

实验接入在控制线程调用`calibration::read_gain_calibration(path)`，赋给
`SoundExConfig.gain_calibration`；默认为None，不改变已有应用输出。只支持
Neural/MatchEdge、FFT256/hop128/crossover1000。校准JSON不是ONNX替代品，须同时绑定
对应模型、配方、来源及验证记录；未知schema、错误特征契约或非有限系数会拒绝加载。

本地拟合、开发评分与资源记录写入`data/reports/gain-calibration-20261011/`。
未加入新音频、外部教师或预训练模型；校准参数保留为未发布实验产物。
