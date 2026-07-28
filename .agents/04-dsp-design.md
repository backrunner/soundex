# SoundEx DSP 设计

## 概述

DSP 模块（`soundex-dsp`）负责模型推理之外的所有信号处理工作，包括：
1. STFT/iSTFT 变换
2. 带宽检测门禁（Gatekeeping）
3. 响度匹配
4. 相位一致性修复
5. 频段平滑过渡（Crossover）
6. 限幅保护

## 1. STFT / iSTFT

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| fft_size | 1024 | FFT 点数 |
| hop_size | 512 | 帧移（50% overlap） |
| window | Hann | 分析/合成窗 |
| 分析窗时长 | ~23.2ms @ 44.1kHz | 1024/44100 |
| 算法延迟 | ~11.6ms @ 44.1kHz | (1024-512)/44100 |

### 实现要点

```rust
pub struct StftAnalyzer {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,          // Hann window
    fft: Arc<dyn RealToFft<f32>>,   // realfft forward
    ifft: Arc<dyn ComplexToReal<f32>>, // realfft inverse
}

impl StftAnalyzer {
    /// 时域帧 → (log_magnitude, phase)
    pub fn analyze(&self, frame: &[f32]) -> SpectralFrame {
        // 1. 加窗
        // 2. Real FFT → complex spectrum [N/2+1]
        // 3. 计算 magnitude = |X[k]|, phase = atan2(im, re)
        // 4. log_magnitude = log(magnitude + eps)
    }

    /// (magnitude, phase) → 时域帧
    pub fn synthesize(&self, magnitude: &[f32], phase: &[f32]) -> Vec<f32> {
        // 1. 从 mag + phase 重建 complex spectrum
        // 2. Inverse FFT → 时域
        // 3. 加窗 (合成窗)
        // 4. 归一化 (COLA: Constant Overlap-Add)
    }
}
```

### COLA 条件验证

分析和合成均使用 periodic Hann。合成器逐 sample 累加 `window^2` 权重并显式
归一化，因此 50% overlap 的重建不依赖未验证的常数 COLA 假设：
```
output[n] = sum_m(frame_m[n] * w[n-m*hop]) / sum_m(w[n-m*hop]^2)
```
确保 overlap-add 重建无幅度调制。

## 2. 带宽检测门禁

### 算法原理

有损压缩（MP3、AAC）的典型特征是在某个截止频率以上，频谱能量急剧衰减至噪底。检测逻辑：

1. 计算当前帧的功率谱密度 (PSD)
2. 从高频向低频扫描，找到能量骤降的拐点
3. 若拐点频率显著低于 Nyquist（如 < 0.85 * Nyquist），判定为需要增强
4. 使用多帧平滑避免逐帧抖动

### 检测算法

```rust
pub struct BandwidthDetector {
    threshold_db: f32,          // 能量骤降阈值 (默认 -60 dB)
    min_bandwidth_ratio: f32,   // 最小有效带宽比 (默认 0.85)
    smoothing_alpha: f32,       // 指数平滑系数 (默认 0.1)
    smoothed_bandwidth: f32,    // 平滑后的带宽估计
    hangover_frames: usize,     // 保持帧数（避免频繁切换）
    hangover_counter: usize,
}

impl BandwidthDetector {
    /// 检测有效带宽
    pub fn detect(&mut self, magnitude_db: &[f32], sample_rate: u32) -> f32 {
        let nyquist = sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / (magnitude_db.len() - 1) as f32;

        // 1. 计算全帧平均能量作为参考
        let mean_energy = magnitude_db.iter().sum::<f32>() / magnitude_db.len() as f32;

        // 2. 从高频 bin 向低频扫描
        // 找到第一个能量 > (mean_energy - threshold_db) 的 bin
        let mut cutoff_bin = magnitude_db.len() - 1;
        for i in (1..magnitude_db.len()).rev() {
            if magnitude_db[i] > mean_energy - self.threshold_db {
                cutoff_bin = i;
                break;
            }
        }

        // 3. 转换为频率
        let detected_bw = cutoff_bin as f32 * bin_resolution;

        // 4. 指数平滑
        self.smoothed_bandwidth = self.smoothing_alpha * detected_bw
            + (1.0 - self.smoothing_alpha) * self.smoothed_bandwidth;

        self.smoothed_bandwidth
    }

    /// 判断是否需要增强
    pub fn needs_enhancement(&mut self, detected_bw: f32, nyquist: f32) -> bool {
        let ratio = detected_bw / nyquist;
        let needs = ratio < self.min_bandwidth_ratio;

        // Hangover 逻辑：一旦触发，保持 N 帧
        if needs {
            self.hangover_counter = self.hangover_frames;
        } else if self.hangover_counter > 0 {
            self.hangover_counter -= 1;
            return true; // 仍在 hangover 期间
        }

        needs
    }
}
```

### 门禁参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold_db | -60.0 | 低于均值 60dB 视为噪底 |
| min_bandwidth_ratio | 0.85 | 带宽 < 85% Nyquist 触发增强 |
| smoothing_alpha | 0.1 | 平滑系数 |
| hangover_frames | 10 | 触发后保持 10 帧 (~116ms) |

## 3. 响度匹配

### 目标

确保模型生成的高频成分在响度上与原始低频段自然衔接，避免：
- 高频过响（刺耳）
- 高频过弱（无效增强）

### 算法

```rust
pub struct LoudnessMatcher {
    target_ratio_db: f32,    // 目标高低频能量比 (从训练数据统计得出)
    smoothing_alpha: f32,
    current_gain: f32,
}

impl LoudnessMatcher {
    /// 计算并应用响度修正增益
    pub fn apply(&mut self, low_band_rms: f32, high_band_rms: f32) -> f32 {
        if high_band_rms < 1e-10 {
            return 1.0;
        }

        // 目标：high_band_rms 应约为 low_band_rms * ratio
        let target_rms = low_band_rms * 10.0f32.powf(self.target_ratio_db / 20.0);
        let gain = target_rms / high_band_rms;

        // 限制增益范围 [-12dB, +6dB]
        let gain = gain.clamp(0.25, 2.0);

        // 平滑
        self.current_gain = self.smoothing_alpha * gain
            + (1.0 - self.smoothing_alpha) * self.current_gain;

        self.current_gain
    }
}
```

## 4. 相位一致性修复

### 问题

模型预测的高频相位可能与低频相位在截止频率处不连续，导致：
- 时域波形突变
- 听感上的"金属感"或"相位失真"

### 算法

```rust
pub struct PhaseSmoother {
    transition_bins: usize,  // 过渡区域 bin 数 (默认 20)
}

impl PhaseSmoother {
    /// 在截止频率附近对相位进行平滑过渡
    pub fn smooth_phase(
        &self,
        original_phase: &[f32],   // 原始完整相位谱
        predicted_phase: &[f32],  // 模型预测的高频相位
        cutoff_bin: usize,
    ) -> Vec<f32> {
        let mut result = predicted_phase.to_vec();

        // 在 [cutoff_bin - transition, cutoff_bin + transition] 范围内
        // 使用线性插值混合原始相位和预测相位
        let start = cutoff_bin.saturating_sub(self.transition_bins);
        let end = (cutoff_bin + self.transition_bins).min(result.len());

        for i in start..end {
            let alpha = (i - start) as f32 / (end - start) as f32;
            // 相位插值需要在复数域进行（避免 wrap-around 问题）
            let orig_complex = Complex::from_polar(1.0, original_phase[i]);
            let pred_complex = Complex::from_polar(1.0, result[i]);
            let blended = orig_complex * (1.0 - alpha) + pred_complex * alpha;
            result[i] = blended.arg();
        }

        result
    }
}
```

### 相位传播策略

对于纯谐波信号，高频相位可通过低频相位外推：
```
phase[k_high] ≈ phase[k_low] * (k_high / k_low)
```
作为模型预测的辅助参考或 fallback。

## 5. 频段平滑过渡（Crossover Blend）

### 目标

在原始低频和模型生成的高频之间建立平滑过渡带，避免"拼接感"。

### 算法

```rust
pub struct CrossoverBlend {
    transition_width_hz: f32,  // 过渡带宽度 (默认 1000 Hz)
}

impl CrossoverBlend {
    /// 生成过渡带的混合权重
    /// 返回每个 bin 的 alpha 值：0.0 = 全原始，1.0 = 全生成
    pub fn compute_blend_weights(
        &self,
        cutoff_freq: f32,
        bin_resolution: f32,
        num_bins: usize,
    ) -> Vec<f32> {
        let half_width = self.transition_width_hz / 2.0;
        let start_freq = cutoff_freq - half_width;
        let end_freq = cutoff_freq + half_width;

        (0..num_bins).map(|i| {
            let freq = i as f32 * bin_resolution;
            if freq <= start_freq {
                0.0  // 完全使用原始
            } else if freq >= end_freq {
                1.0  // 完全使用生成
            } else {
                // 平滑过渡 (raised cosine)
                let t = (freq - start_freq) / (end_freq - start_freq);
                0.5 * (1.0 - (std::f32::consts::PI * t).cos())
            }
        }).collect()
    }

    /// 应用混合
    pub fn apply(
        &self,
        original_mag: &[f32],
        generated_mag: &[f32],
        weights: &[f32],
    ) -> Vec<f32> {
        original_mag.iter().zip(generated_mag.iter()).zip(weights.iter())
            .map(|((o, g), w)| o * (1.0 - w) + g * w)
            .collect()
    }
}
```

## 6. 限幅保护

### 目标

防止增强后的信号超出 [-1.0, 1.0] 范围导致削波失真。

### 算法

```rust
pub struct Limiter {
    ceiling: f32,       // 上限 (默认 0.95，留 headroom)
    release_coeff: f32, // 释放系数
    envelope: f32,      // 当前包络
}

impl Limiter {
    /// 软限幅
    pub fn process(&mut self, sample: f32) -> f32 {
        let abs_sample = sample.abs();

        if abs_sample > self.ceiling {
            // 软拐点压缩
            let overshoot = abs_sample - self.ceiling;
            let compressed = self.ceiling + overshoot * 0.1; // 10:1 比率
            let sign = if sample >= 0.0 { 1.0 } else { -1.0 };
            self.envelope = compressed;
            sign * compressed
        } else {
            sample
        }
    }

    /// 批量处理
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}
```

## DSP 处理完整流程

```
1. 输入帧进入 Ring Buffer
2. 当累积够 1024 samples:
   a. STFT 分析 → (log_mag, phase)
   b. BandwidthDetector.detect() → 有效带宽
   c. 若不需要增强 → bypass，直接 overlap-add 输出
   d. 若需要增强:
      i.   分离低频/高频 (基于检测到的截止频率)
      ii.  高频特征送入模型推理
      iii. LoudnessMatcher 计算增益
      iv.  PhaseSmoother 修复相位
      v.   CrossoverBlend 混合低频原始 + 高频生成
      vi.  iSTFT 重建
      vii. Limiter 保护
3. Overlap-add 输出 512 samples
```
