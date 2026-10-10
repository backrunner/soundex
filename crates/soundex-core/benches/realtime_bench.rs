//! Paced callback diagnostic: model misses must never become output stalls.

use std::{
    env,
    error::Error,
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use serde_json::json;
use sha2::{Digest, Sha256};
use soundex_core::{realtime_policy as policy, EnhancementMode, RealtimeProcessor, SoundExConfig};

const HOP: usize = policy::CALLBACK_FRAMES;

#[path = "realtime_bench/source.rs"]
mod source;

#[path = "realtime_bench/report.rs"]
mod report;

#[path = "realtime_bench/producer.rs"]
mod producer;

fn main() -> Result<(), Box<dyn Error>> {
    let model = PathBuf::from(env::var("SOUNDEX_BENCH_MODEL")?).canonicalize()?;
    let report = PathBuf::from(env::var("SOUNDEX_REALTIME_REPORT")?);
    let selected_case = env::var("SOUNDEX_REALTIME_CASE").ok();
    let enhancement_mode: EnhancementMode = env::var("SOUNDEX_ENHANCEMENT_MODE")
        .unwrap_or_else(|_| "neural".into())
        .parse()
        .map_err(|error: &str| error.to_string())?;
    let producer_rt = match env::var("SOUNDEX_REALTIME_PRODUCER_RT").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        _ => return Err("SOUNDEX_REALTIME_PRODUCER_RT must be 0 or 1".into()),
    };
    let worker_rt = match env::var("SOUNDEX_REALTIME_WORKER_RT").as_deref() {
        Ok("0") => false,
        Ok("1") | Err(_) => true,
        _ => return Err("SOUNDEX_REALTIME_WORKER_RT must be 0 or 1".into()),
    };
    if selected_case
        .as_deref()
        .is_some_and(|case| !matches!(case, "44100:1" | "44100:2" | "48000:1" | "48000:2"))
    {
        return Err("SOUNDEX_REALTIME_CASE must be 44100:1/2 or 48000:1/2".into());
    }
    let seconds: u64 = env::var("SOUNDEX_REALTIME_SECONDS")
        .unwrap_or_else(|_| "60".into())
        .parse()?;
    if seconds == 0 {
        return Err("measurement must be nonzero".into());
    }
    let bytes = fs::read(&model)?;
    if bytes.len() < 1_000_000 {
        return Err("diagnostic requires a trained model, not an identity fixture".into());
    }
    let hash = format!("{:x}", Sha256::digest(&bytes));
    drop(bytes);
    let mut reports = report::Report::new(report);
    for sample_rate in [44100, 48000] {
        for channels in [1, 2] {
            if selected_case
                .as_ref()
                .is_some_and(|selected| selected != &format!("{sample_rate}:{channels}"))
            {
                continue;
            }
            let case = measure(
                &model,
                seconds,
                sample_rate,
                channels,
                producer_rt,
                worker_rt,
                enhancement_mode,
            )?;
            println!(
                "{} Hz / {} channels: {}",
                sample_rate, channels, case.metadata["callback"]
            );
            // Persist every completed case even if a later case fails.
            let evidence = json!({
                "schema_version": 2,
                "report_type": "soundex-paced-realtime-diagnostic",
                "diagnostic": true,
                "enhancement_mode": enhancement_mode.as_str(),
                "model_loaded": enhancement_mode != EnhancementMode::Spectral,
                "model_path": model,
                "model_sha256": hash,
                "model_size_bytes": fs::metadata(&model)?.len(),
                "fft_size": 256, "hop_size": HOP,
                "ort_threads": [1, 1], "parallel_execution": false,
                "audio_device_tested": false,
                "selected_case": selected_case,
                "producer_audio_realtime_requested": producer_rt,
                "worker_audio_realtime_requested": worker_rt,
                "streaming_policy": {
                    "callback_frames": HOP,
                    "added_latency_target_ms": policy::ADDED_LATENCY_TARGET_MS,
                    "added_latency_limit_ms": policy::ADDED_LATENCY_LIMIT_MS,
                    "callback_p99_target_us": policy::CALLBACK_P99_TARGET_US,
                    "callback_max_limit_us": policy::CALLBACK_MAX_LIMIT_US,
                    "worker_miss_ratio_limit": policy::WORKER_MISS_RATIO_LIMIT,
                    "worker_miss_burst_limit_ms": policy::WORKER_MISS_BURST_LIMIT_MS,
                },
            });
            reports.append(&evidence, &case)?;
        }
    }
    Ok(())
}

fn measure(
    model: &std::path::Path,
    seconds: u64,
    sample_rate: u32,
    channels: u16,
    producer_rt: bool,
    worker_rt: bool,
    enhancement_mode: EnhancementMode,
) -> Result<report::Case, Box<dyn Error>> {
    let mut config = SoundExConfig::with_model(model)
        .sample_rate(sample_rate)
        .channels(channels);
    config.enhancement_mode = enhancement_mode;
    config.min_bandwidth_ratio = 1.0;
    config.worker_time_constraint = worker_rt;
    let load_started = Instant::now();
    let mut processor = RealtimeProcessor::new(config)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let hops = (seconds * sample_rate as u64).div_ceil(HOP as u64) as usize;
    let samples = HOP * channels as usize;
    let mut input = vec![0.0; samples];
    let mut output = vec![0.0; samples];
    let mut timings = Vec::with_capacity(hops);
    let mut worker_missed = Vec::with_capacity(hops);
    let mut previous_misses = 0;
    let mut schedule_late_hops = 0;
    let mut callback_overruns = 0;
    let mut nonfinite = 0;
    let mut peak = 0.0_f32;
    let mut boundary_square = 0.0_f64;
    let mut interior_square = 0.0_f64;
    let mut boundary_count = 0;
    let mut interior_count = 0;
    let mut last = [0.0_f32; 2];
    let mut noise = 42_u32;
    let mut source = source::Input::new(sample_rate, channels)?;
    let scheduling = producer::Scheduling::new(producer_rt, sample_rate, HOP as u32);
    let producer_rt_accepted = scheduling.accepted();
    let origin = Instant::now();
    let period = Duration::from_secs_f64(HOP as f64 / sample_rate as f64);
    for hop in 0..hops {
        let scheduled = origin + Duration::from_secs_f64((hop * HOP) as f64 / sample_rate as f64);
        if let Some(wait) = scheduled.checked_duration_since(Instant::now()) {
            thread::sleep(wait);
        }
        if Instant::now().saturating_duration_since(scheduled) >= period {
            schedule_late_hops += 1;
        }
        if !source.read_hop(&mut input)? {
            fill_input(
                &mut input,
                hop * HOP,
                sample_rate,
                channels as usize,
                &mut noise,
            );
        }
        let start = Instant::now();
        let callback_stats = processor.process(&input, &mut output)?;
        let elapsed = start.elapsed();
        timings.push(elapsed.as_nanos() as u64);
        worker_missed.push(callback_stats.deadline_misses > previous_misses);
        previous_misses = callback_stats.deadline_misses;
        if elapsed >= period {
            callback_overruns += 1;
        }
        for (frame, values) in output.chunks_exact(channels as usize).enumerate() {
            for (channel, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    nonfinite += 1;
                }
                peak = peak.max(value.abs());
                let delta = (value - last[channel]) as f64;
                if frame == 0 {
                    boundary_square += delta * delta;
                    boundary_count += 1;
                } else {
                    interior_square += delta * delta;
                    interior_count += 1;
                }
                last[channel] = value;
            }
        }
    }
    let wall_seconds = origin.elapsed().as_secs_f64();
    // Restore before allocation/serialization/join on the control thread.
    drop(scheduling);
    let stats = processor.stats();
    let worker = processor.worker_stats();
    processor.shutdown();
    let mut sorted = timings.clone();
    sorted.sort_unstable();
    let percentile = |p: f64| {
        sorted[((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1)] as f64 / 1000.0
    };
    let maximum_us = *sorted.last().ok_or("empty timings")? as f64 / 1000.0;
    let fixed_ms = 256_000.0 / sample_rate as f64;
    let boundary_rms = (boundary_square / boundary_count as f64).sqrt();
    let interior_rms = (interior_square / interior_count as f64).sqrt();
    let assessment = policy::assess(stats, sample_rate, percentile(0.99), maximum_us, 0)
        .ok_or("invalid real-time policy measurement")?;
    let redline_met = assessment.latency_limit_met;
    Ok(report::Case {
        metadata: json!({
            "sample_rate_hz": sample_rate, "channels": channels,
            "construction_ms": load_ms, "hops": hops,
            "source": source.evidence(),
            "measurement_wall_seconds": wall_seconds,
            "output_frames": stats.output_frames,
            "expected_output_frames": hops * HOP,
            "fixed_delay_samples_per_channel": 256,
            "fixed_delay_ms": fixed_ms,
            "callback": {"p50_us": percentile(0.50), "p99_us": percentile(0.99), "max_us": maximum_us},
            "conservative_max_added_latency_ms": fixed_ms + maximum_us / 1000.0,
            "redline_10ms_met": redline_met,
            "target_8ms_met": assessment.latency_target_met,
            "callback_budget_met": assessment.callback_budget_met,
            "enhancement_availability_met": assessment.enhancement_availability_met,
            "worker_miss_ratio": assessment.worker_miss_ratio,
            "worker_miss_burst_ms": assessment.worker_miss_burst_ms,
            "max_consecutive_worker_misses": stats.max_consecutive_deadline_misses,
            "preferred_5ms_met": fixed_ms + maximum_us / 1000.0 < 5.0,
            "callback_deadline_overruns": callback_overruns,
        "producer_schedule_late_hops": schedule_late_hops,
        "producer_audio_realtime_request_accepted": producer_rt_accepted,
            "worker_deadline_misses": stats.deadline_misses,
            "worker_accepted_hops": stats.accepted_hops,
            "queue_overflows": stats.queue_overflows,
            "stale_results": stats.stale_results,
            "invalid_samples": stats.invalid_samples,
            "rejected_results": stats.rejected_results,
            "worker_failed": stats.worker_failed,
            "worker": {
                "processed_hops": worker.processed_hops,
                "discarded_input_hops": worker.discarded_input_hops,
                "discarded_output_hops": worker.discarded_output_hops,
                "max_queue_wait_us": worker.max_queue_wait_ns as f64 / 1000.0,
                "max_processing_us": worker.max_processing_ns as f64 / 1000.0,
                "queue_wait_p99_upper_bound_us": worker.queue_wait_p99_upper_bound_ns as f64 / 1000.0,
                "processing_p99_upper_bound_us": worker.processing_p99_upper_bound_ns as f64 / 1000.0,
                "processing_period_overruns": worker.processing_period_overruns,
            "macos_qos_applied": worker.macos_qos_applied,
            "macos_realtime_request_accepted": worker.macos_realtime_request_accepted,
            },
            "nonfinite_output_samples": nonfinite, "output_peak": peak,
            "boundary_delta_rms": boundary_rms,
            "interior_delta_rms": interior_rms,
            "boundary_to_interior_rms_ratio": boundary_rms / interior_rms.max(1e-12),
            "continuous_output_checks_passed": nonfinite == 0 && peak <= 0.95
                && stats.output_frames == (hops * HOP) as u64 && callback_overruns == 0 && redline_met,
        }),
        callback_samples_ns: timings,
        worker_presentation_missed: worker_missed,
    })
}

fn fill_input(input: &mut [f32], start: usize, rate: u32, channels: usize, noise: &mut u32) {
    for (frame, values) in input.chunks_exact_mut(channels).enumerate() {
        let sample = start + frame;
        let t = sample as f64 / rate as f64;
        let segment = (t as u64 / 3) % 6;
        *noise ^= *noise << 13;
        *noise ^= *noise >> 17;
        *noise ^= *noise << 5;
        let broadband = (*noise as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
        let left = match segment {
            0 => 0.25 * (std::f64::consts::TAU * 440.0 * t).sin() as f32,
            1 => 0.15 * (std::f64::consts::TAU * 440.0 * t).sin() as f32 + 0.08 * broadband,
            2 => 0.0,
            3 => 0.001 * (std::f64::consts::TAU * 880.0 * t).sin() as f32,
            4 => 0.6 * (std::f64::consts::TAU * 220.0 * t).sin() as f32,
            _ => {
                if sample.is_multiple_of(rate as usize / 2) {
                    0.9
                } else {
                    0.0
                }
            }
        };
        values[0] = left;
        if channels == 2 {
            values[1] = if segment == 2 {
                0.12 * (std::f64::consts::TAU * 880.0 * t).sin() as f32
            } else {
                -left * 0.7
            };
        }
    }
}
