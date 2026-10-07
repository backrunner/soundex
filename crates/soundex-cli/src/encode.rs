//! WAV file encoding via hound.

use std::path::Path;

use anyhow::{Context, Result};

/// WAV output bit depth options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 16-bit signed integer PCM.
    Bits16,
    /// 24-bit signed integer PCM.
    Bits24,
    /// 32-bit float PCM.
    Bits32Float,
}

impl BitDepth {
    /// Parse from CLI argument (16, 24, or 32).
    pub fn from_bits(bits: u16) -> Result<Self> {
        match bits {
            16 => Ok(Self::Bits16),
            24 => Ok(Self::Bits24),
            32 => Ok(Self::Bits32Float),
            _ => anyhow::bail!("Unsupported bit depth: {bits} (use 16, 24, or 32)"),
        }
    }
}

/// Encode f32 samples to a WAV file.
///
/// # Arguments
/// * `path` - Output file path
/// * `samples` - Interleaved f32 samples
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of channels
/// * `bit_depth` - Output bit depth
pub fn encode_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    bit_depth: BitDepth,
) -> Result<()> {
    anyhow::ensure!(channels > 0, "Channel count must be greater than zero");
    anyhow::ensure!(sample_rate > 0, "Sample rate must be greater than zero");
    anyhow::ensure!(
        samples.len().is_multiple_of(channels as usize),
        "Sample count must be divisible by the channel count"
    );
    anyhow::ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "Samples contain non-finite values"
    );

    let spec = match bit_depth {
        BitDepth::Bits16 => hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
        BitDepth::Bits24 => hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        },
        BitDepth::Bits32Float => hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV: {}", path.display()))?;

    match bit_depth {
        BitDepth::Bits16 => {
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * i16::MAX as f32) as i16;
                writer.write_sample(val).context("WAV write error")?;
            }
        }
        BitDepth::Bits24 => {
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                // hound handles i32 → 24-bit conversion internally
                let val = (clamped * 8_388_607.0) as i32; // 2^23 - 1
                writer.write_sample(val).context("WAV write error")?;
            }
        }
        BitDepth::Bits32Float => {
            for &s in samples {
                writer.write_sample(s).context("WAV write error")?;
            }
        }
    }

    writer.finalize().context("WAV finalize error")?;
    Ok(())
}
