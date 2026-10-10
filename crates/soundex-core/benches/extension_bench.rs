//! Same-input synchronous neural / DSP / hybrid cost diagnostic.
//! One mode and channel/rate case per process so external RSS/CPU measurements
//! do not inherit another mode's model or PCM allocations. Not a release gate.

use serde_json::json;
use sha2::{Digest, Sha256};
use soundex_core::{EnhancementMode, SoundExConfig, SoundExProcessor};
use std::{env, error::Error, fs, hint::black_box, io::Read, path::PathBuf, time::Instant};

#[path = "realtime_bench/source.rs"]
mod source;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("extension diagnostic requires release build".into());
    }
    if env::var_os("SOUNDEX_REALTIME_PCM_DIR").is_none() {
        return Err("real PCM directory is required".into());
    }
    let model = PathBuf::from(env::var("SOUNDEX_BENCH_MODEL")?).canonicalize()?;
    let mode: EnhancementMode = env::var("SOUNDEX_ENHANCEMENT_MODE")?
        .parse()
        .map_err(|error: &str| error.to_string())?;
    let rate: u32 = env::var("SOUNDEX_EXTENSION_RATE")?.parse()?;
    let channels: u16 = env::var("SOUNDEX_EXTENSION_CHANNELS")?.parse()?;
    if ![44100, 48000].contains(&rate) || ![1, 2].contains(&channels) {
        return Err("unsupported rate/channels".into());
    }
    let seconds: u32 = env::var("SOUNDEX_EXTENSION_SECONDS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;
    if seconds == 0 {
        return Err("seconds must be positive".into());
    }
    let mut reader = fs::File::open(&model)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        digest.update(&buffer[..length]);
    }
    let model_sha256 = format!("{:x}", digest.finalize());
    let mut config = SoundExConfig::with_model(model)
        .sample_rate(rate)
        .channels(channels)
        .enhancement_mode(mode);
    config.min_bandwidth_ratio = 1.0;
    let start = Instant::now();
    let mut processor = SoundExProcessor::new(config)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let hops = (u64::from(seconds) * u64::from(rate)).div_ceil(128) as usize;
    let mut source = source::Input::new(rate, channels, (hops + 32) * 128)?;
    let mut input = vec![0.0; 128 * channels as usize];
    let mut output = vec![0.0; input.len()];
    let mut times = Vec::with_capacity(hops);
    let mut elapsed_ns = 0_u64;
    let mut peak = 0.0_f32;
    for i in 0..hops + 32 {
        source.read_hop(&mut input)?;
        let start = Instant::now();
        processor.process_frame(black_box(&input), black_box(&mut output))?;
        let ns = start.elapsed().as_nanos() as u64;
        if i >= 32 {
            times.push(ns);
            elapsed_ns += ns;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite output".into());
        }
        peak = output.iter().fold(peak, |max, value| max.max(value.abs()));
    }
    times.sort_unstable();
    let percentile = |p: f64| {
        times[((times.len() as f64 * p).ceil() as usize).saturating_sub(1)] as f64 / 1000.0
    };
    let report = json!({"schema_version": 1, "diagnostic": true, "audio_device_tested": false,
        "enhancement_mode": mode.as_str(), "model_sha256": model_sha256,
        "ort_backend_compiled": cfg!(feature = "ort-backend"),
        "model_loaded": mode != EnhancementMode::Spectral, "source": source.evidence(),
        "sample_rate": rate, "channels": channels, "fft_size": 256, "hop_size": 128,
        "hops": hops, "warmup_hops": 32, "construction_ms": load_ms,
        "processing_p50_us": percentile(0.5), "processing_p99_us": percentile(0.99),
        "processing_max_us": *times.last().unwrap() as f64 / 1000.0,
        "processing_rtf": elapsed_ns as f64 / 1e9 / (hops as f64 * 128.0 / rate as f64),
        "inference_runs": processor.inference_run_count(), "output_peak": peak,
        "stft_latency_samples": processor.latency_samples_per_channel(),
        "additional_extension_delay_samples": 0});
    fs::write(
        env::var("SOUNDEX_EXTENSION_REPORT")?,
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
