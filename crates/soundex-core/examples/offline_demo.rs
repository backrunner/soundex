// SPDX-License-Identifier: Apache-2.0
//! Generate a stereo test signal and process it with a bundled ONNX fixture.
//! Optionally pass a qualified 256/128 model and an output directory.

use std::{error::Error, f32::consts::TAU, path::PathBuf};

use soundex_core::{SoundExConfig, SoundExProcessor};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 2 || args.first().is_some_and(|arg| arg == "--help") {
        println!("Usage: offline_demo [MODEL.onnx [OUTPUT_DIRECTORY]]");
        println!("Default: synthetic identity fixture; writes target/demo/*.wav");
        return Ok(());
    }
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    let model = args.first().map(PathBuf::from).unwrap_or(fixture);
    let directory = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/demo".into());
    let mut config = SoundExConfig::with_model(&model)
        .sample_rate(48_000)
        .channels(2);
    // Exercise inference on the narrow-band test signal instead of gate bypass.
    config.min_bandwidth_ratio = 1.0;
    let mut processor = SoundExProcessor::new(config)?;
    let input: Vec<f32> = (0..48_000 * 2)
        .flat_map(|frame| {
            let time = frame as f32 / 48_000.0;
            [
                0.2 * (TAU * 440.0 * time).sin(),
                0.2 * (TAU * 660.0 * time).sin(),
            ]
        })
        .collect();
    let output = processor.process_buffer(&input)?;
    if output.len() != input.len() || output.iter().any(|sample| !sample.is_finite()) {
        return Err("invalid output length or non-finite audio".into());
    }
    if processor.inference_run_count() == 0 {
        return Err("demo did not exercise ONNX inference".into());
    }
    std::fs::create_dir_all(&directory)?;
    write_wav(&directory.join("input.wav"), &input)?;
    write_wav(&directory.join("output.wav"), &output)?;
    println!("Model: {}", model.display());
    println!("Output: {} (2 seconds, 48 kHz stereo)", directory.display());
    println!("Inference calls: {}", processor.inference_run_count());
    println!("Bundled identity fixture demonstrates integration, not audio restoration.");
    Ok(())
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()
}
