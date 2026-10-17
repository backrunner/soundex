//! Offline observed-spectrum trace using the production analyzer and hop clock.
//! This utility allocates and is never part of the audio callback.
mod audio_io;

use soundex_dsp::stft::StftAnalyzer;
use std::{error::Error, fs, path::Path};

fn trace(samples: &[f32], channels: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    if channels == 0
        || !samples.len().is_multiple_of(128 * channels)
        || samples.iter().any(|x| !x.is_finite())
    {
        return Err("trace requires finite complete interleaved hops".into());
    }
    let mut analyzer = StftAnalyzer::new(256, 128);
    let mut buffers = vec![vec![0.0; 256]; channels];
    let mut result = Vec::new();
    for hop in samples.chunks_exact(128 * channels) {
        for (channel, buffer) in buffers.iter_mut().enumerate() {
            buffer.copy_within(128.., 0);
            for (i, sample) in buffer[128..].iter_mut().enumerate() {
                *sample = hop[i * channels + channel];
            }
            let frame = analyzer.analyze(buffer);
            result.extend(frame.log_magnitude);
            result.extend(frame.phase);
        }
    }
    Ok(result)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: soundex-analysis-trace INPUT.sxa OUTPUT.json".into());
    }
    let input = audio_io::read_audio(Path::new(&args[1]))?;
    let features = trace(&input.samples, input.channels as usize)?;
    fs::write(
        &args[2],
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "fft_size": 256, "hop_size": 128,
            "sample_rate": input.sample_rate, "channels": input.channels,
            "layout": "hop,channel,magnitude_db_then_phase,bin",
            "features": features,
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_has_causal_padding_and_independent_channels() {
        let mono: Vec<f32> = (0..384).map(|i| (i as f32 * 0.13).sin()).collect();
        let stereo: Vec<f32> = mono.iter().flat_map(|&x| [0.0, x]).collect();
        let one = trace(&mono, 1).unwrap();
        let two = trace(&stereo, 2).unwrap();
        assert_eq!(one.len(), 3 * 258);
        for hop in 0..3 {
            assert_eq!(
                &two[hop * 516 + 258..(hop + 1) * 516],
                &one[hop * 258..(hop + 1) * 258]
            );
            assert!(two[hop * 516..hop * 516 + 129].iter().all(|&v| v == -200.0));
        }
        let mut analyzer = StftAnalyzer::new(256, 128);
        let mut padded = vec![0.0; 128];
        padded.extend_from_slice(&mono[..128]);
        let first = analyzer.analyze(&padded);
        assert_eq!(&one[..129], first.log_magnitude);
        assert_eq!(&one[129..258], first.phase);
        assert!(trace(&[0.0; 127], 1).is_err());
        assert!(trace(&[f32::NAN; 128], 1).is_err());
        assert!(trace(&[], 0).is_err());
    }
}
