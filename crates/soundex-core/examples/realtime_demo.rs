// SPDX-License-Identifier: Apache-2.0
//! Simulate 128-frame callbacks. This does not open an audio device or measure xruns.

use std::{
    error::Error,
    f32::consts::TAU,
    path::PathBuf,
    time::{Duration, Instant},
};

use soundex_core::{RealtimeProcessor, SoundExConfig};

const FRAMES: usize = 128;
const CHANNELS: usize = 2;
const SAMPLE_RATE: u32 = 48_000;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 || args.first().is_some_and(|arg| arg == "--help") {
        println!("Usage: realtime_demo [MODEL.onnx]");
        println!("Simulated 128-frame callbacks; default model is a synthetic identity fixture.");
        return Ok(());
    }
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/low-latency-identity.onnx");
    let model = args.first().map(PathBuf::from).unwrap_or(fixture);
    let mut config = SoundExConfig::with_model(&model)
        .sample_rate(SAMPLE_RATE)
        .channels(2);
    config.min_bandwidth_ratio = 1.0;
    // Model loading, thread creation and allocation happen on the control thread.
    let mut stream = RealtimeProcessor::new(config)?;
    let input: Vec<f32> = (0..SAMPLE_RATE as usize * 2)
        .flat_map(|frame| {
            let time = frame as f32 / SAMPLE_RATE as f32;
            [
                0.2 * (TAU * 440.0 * time).sin(),
                0.2 * (TAU * 660.0 * time).sin(),
            ]
        })
        .collect();
    let start = Instant::now();
    let mut output = [0.0; FRAMES * CHANNELS];
    let mut bad_output = false;
    let mut produced = 0;
    for (block, chunk) in input.chunks(FRAMES * CHANNELS).enumerate() {
        // Pacing belongs to this host simulation, never to a device callback.
        let deadline =
            start + Duration::from_secs_f64(block as f64 * FRAMES as f64 / SAMPLE_RATE as f64);
        if let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            std::thread::sleep(remaining);
        }
        let destination = &mut output[..chunk.len()];
        stream.process(chunk, destination)?; // the only operation required in a callback
        bad_output |= destination
            .iter()
            .any(|sample| !sample.is_finite() || sample.abs() > 0.95);
        produced += destination.len();
    }
    let stats = stream.stats();
    let worker = stream.worker_stats();
    let delay = stream.latency_samples_per_channel();
    stream.shutdown(); // join outside the callback
    if bad_output || produced != input.len() || worker.processed_hops == 0 || stats.worker_failed {
        return Err("demo output or model worker check failed".into());
    }
    println!("Model: {}", model.display());
    println!(
        "Fixed software delay: {delay} frames ({:.3} ms)",
        delay as f64 * 1000.0 / SAMPLE_RATE as f64
    );
    println!("Callback counters: {stats:?}");
    println!("Worker counters: {worker:?}");
    println!("Host scheduling can cause misses; aligned dry fallback keeps output moving.");
    println!("Synthetic fixture and host simulation are not quality or device-latency evidence.");
    Ok(())
}
