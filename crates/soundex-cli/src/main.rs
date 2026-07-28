//! SoundEx CLI: decode audio → enhance → output WAV.
//!
//! Usage: soundex <input> [-o output.wav] [options]

mod decode;
mod encode;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use decode::decode_file;
use encode::{encode_wav, BitDepth};

/// SoundEx: AI-powered audio quality enhancement.
///
/// Restores high-frequency content lost in lossy compression (MP3, AAC, etc.)
/// to improve playback experience.
#[derive(Parser, Debug)]
#[command(name = "soundex", version, about)]
struct Args {
    /// Input audio file (MP3, AAC, FLAC, WAV, OGG)
    input: PathBuf,

    /// Output WAV file path (default: <input_stem>_enhanced.wav)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to the ONNX model file
    #[arg(long, default_value = "models/soundex-v1.onnx")]
    model: PathBuf,

    /// Bypass detection threshold in dB
    #[arg(long, default_value_t = -60.0)]
    bypass_threshold: f32,

    /// Artifact FFT size (1024 for the low-latency contract; 2048 for legacy artifacts)
    #[arg(long, default_value_t = 1024)]
    fft_size: usize,

    /// Artifact hop size
    #[arg(long, default_value_t = 512)]
    hop_size: usize,

    /// Artifact crossover transition width in Hz
    #[arg(long, default_value_t = 1000.0)]
    crossover_width_hz: f32,

    /// Only analyze whether enhancement is needed (no output)
    #[arg(long)]
    dry_run: bool,

    /// Output bits per sample (16, 24, or 32)
    #[arg(long, default_value_t = 16)]
    bits: u16,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Validate input
    anyhow::ensure!(
        args.input.exists(),
        "Input file not found: {:?}",
        args.input
    );

    let output_path = args.output.unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        let parent = args.input.parent().unwrap_or(std::path::Path::new("."));
        parent.join(format!("{stem}_enhanced.wav"))
    });

    if args.verbose {
        eprintln!("Input:   {:?}", args.input);
        eprintln!("Model:   {:?}", args.model);
        eprintln!("Output:  {:?}", output_path);
        eprintln!("Bits:    {}", args.bits);
        eprintln!(
            "Frame:   {}/{} ({} samples latency)",
            args.fft_size,
            args.hop_size,
            args.fft_size.saturating_sub(args.hop_size)
        );
    }

    // Step 1: Decode
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{msg}").unwrap());
    pb.set_message("Decoding audio...");

    let decoded = decode_file(&args.input)?;
    let duration_secs =
        decoded.samples.len() as f64 / decoded.sample_rate as f64 / decoded.channels as f64;

    pb.finish_with_message(format!(
        "Decoded: {:.1}s, {}Hz, {}ch, {} samples",
        duration_secs,
        decoded.sample_rate,
        decoded.channels,
        decoded.samples.len()
    ));

    let config = soundex_core::SoundExConfig::with_model(&args.model)
        .sample_rate(decoded.sample_rate)
        .channels(decoded.channels)
        .fft_size(args.fft_size)
        .hop_size(args.hop_size)
        .crossover_width_hz(args.crossover_width_hz)
        .bypass_threshold_db(args.bypass_threshold);

    if args.dry_run {
        let analysis = soundex_core::analyze_buffer(&config, &decoded.samples)?;
        eprintln!(
            "[dry-run] mean bandwidth: {:.0} Hz; enhancement: {} ({:.1}% of frames)",
            analysis.detected_bandwidth_hz,
            if analysis.needs_enhancement() {
                "needed"
            } else {
                "not needed"
            },
            analysis.enhancement_ratio() * 100.0
        );
        return Ok(());
    }

    let bit_depth = BitDepth::from_bits(args.bits)?;

    // Step 2: Enhance
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{msg}").unwrap());
    pb.set_message("Enhancing audio...");

    let mut processor = soundex_core::SoundExProcessor::new(config)
        .with_context(|| format!("Failed to initialize model: {}", args.model.display()))?;
    let enhanced_samples = processor.process_buffer(&decoded.samples)?;

    pb.finish_with_message("Enhancement complete.");

    // Step 3: Encode WAV
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{msg}").unwrap());
    pb.set_message("Writing WAV...");

    encode_wav(
        &output_path,
        &enhanced_samples,
        decoded.sample_rate,
        decoded.channels,
        bit_depth,
    )?;

    pb.finish_with_message(format!("Output: {}", output_path.display()));
    eprintln!("Done! Enhanced audio saved to: {}", output_path.display());

    Ok(())
}
