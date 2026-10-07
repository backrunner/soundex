//! Fail-closed production-model latency and deadline benchmark.

use std::{
    env,
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{self, Command},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(target_os = "macos")]
use std::mem::MaybeUninit;

use sha2::{Digest, Sha256};
use soundex_core::{realtime_policy, SoundExConfig, SoundExProcessor};

#[path = "core_bench/report.rs"]
mod report;
#[path = "core_bench/support.rs"]
mod support;
use report::*;
use support::*;

const MIN_PRODUCTION_MODEL_BYTES: u64 = 1_000_000;
const MAX_PRODUCTION_MODEL_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PEAK_RSS_BYTES: u64 = 50 * 1024 * 1024;
const RELEASE_SECONDS_PER_CASE: u64 = 30 * 60;
const DEFAULT_WARMUP_HOPS: usize = 32;
const FFT_SIZE: usize = 256;
const HOP_SIZE: usize = 128;

fn main() {
    if cfg!(debug_assertions) {
        return;
    }
    match run() {
        Ok(true) => {}
        Ok(false) => process::exit(2),
        Err(error) => {
            eprintln!("soundex production benchmark failed: {error}");
            process::exit(2);
        }
    }
}

fn run() -> Result<bool, Box<dyn Error>> {
    let model_path = required_path("SOUNDEX_BENCH_MODEL")?;
    let artifact = validate_artifact(&model_path)?;
    let diagnostic = env_flag("SOUNDEX_BENCH_DIAGNOSTIC");
    let measurement_seconds =
        env_number("SOUNDEX_BENCH_SECONDS_PER_CASE", RELEASE_SECONDS_PER_CASE)?;
    if measurement_seconds == 0 {
        return Err("SOUNDEX_BENCH_SECONDS_PER_CASE must be positive".into());
    }
    if measurement_seconds < RELEASE_SECONDS_PER_CASE && !diagnostic {
        return Err(format!(
            "runs shorter than {RELEASE_SECONDS_PER_CASE}s per case require SOUNDEX_BENCH_DIAGNOSTIC=1"
        )
        .into());
    }
    let warmup_hops = env_number("SOUNDEX_BENCH_WARMUP_HOPS", DEFAULT_WARMUP_HOPS)?;
    let output_path = env::var_os("SOUNDEX_BENCH_REPORT")
        .map(PathBuf::from)
        .map(workspace_relative)
        .unwrap_or_else(|| workspace_root().join("target/soundex-performance.json"));
    let hardware = hardware_evidence();

    let mut cases = Vec::with_capacity(4);
    for sample_rate in [44_100, 48_000] {
        for channels in [1, 2] {
            cases.push(run_case(
                &model_path,
                sample_rate,
                channels,
                warmup_hops,
                measurement_seconds,
            )?);
        }
    }
    let cold_model_load_ms = cases[0].processor_construction_ms;
    let peak_rss_bytes = cases.iter().filter_map(|case| case.rss_bytes).max();
    let performance_gates = evaluate_gates(&cases, diagnostic, &hardware);
    let passed = performance_gates.passed;
    let report = BenchmarkReport {
        schema_version: 4,
        report_type: "soundex-performance",
        generated_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        diagnostic,
        artifact,
        protocol: FrameProtocol {
            fft_size: FFT_SIZE,
            hop_size: HOP_SIZE,
            frequency_bins: FFT_SIZE / 2 + 1,
        },
        hardware,
        ort: OrtEvidence {
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
            graph_optimization: "level3".to_owned(),
            output_preallocation: true,
        },
        cold_model_load_ms,
        peak_rss_bytes,
        cases,
        performance_gates,
    };
    write_report(&output_path, &report)?;
    println!("wrote benchmark evidence to {}", output_path.display());
    if !passed {
        eprintln!(
            "performance gates failed: {}",
            report.performance_gates.failed.join(", ")
        );
    }
    Ok(passed)
}

fn run_case(
    model_path: &Path,
    sample_rate: u32,
    channels: u16,
    warmup_hops: usize,
    measurement_seconds: u64,
) -> Result<CaseReport, Box<dyn Error>> {
    let mut config = SoundExConfig::with_model(model_path)
        .sample_rate(sample_rate)
        .channels(channels)
        .fft_size(FFT_SIZE)
        .hop_size(HOP_SIZE)
        .ort_threads(1, 1)
        .ort_parallel_execution(false);
    config.min_bandwidth_ratio = 1.0;
    let construction_started = Instant::now();
    let mut processor = SoundExProcessor::new(config)?;
    let processor_construction_ms = construction_started.elapsed().as_secs_f64() * 1_000.0;
    let input = forced_enhancement_input(sample_rate, channels);
    let mut output = vec![0.0; input.len()];

    let first_started = Instant::now();
    let first_info = processor.process_frame(black_box(&input), black_box(&mut output))?;
    let first_frame_ns = duration_ns(first_started.elapsed());
    if first_info.bypassed {
        return Err("benchmark stimulus unexpectedly bypassed inference".into());
    }
    for _ in 0..warmup_hops {
        let info = processor.process_frame(black_box(&input), black_box(&mut output))?;
        if info.bypassed {
            return Err("benchmark stimulus bypassed inference during warm-up".into());
        }
    }

    let deadline = Duration::from_secs_f64(HOP_SIZE as f64 / sample_rate as f64);
    let measurement_target = Duration::from_secs(measurement_seconds);
    let measurement_started = Instant::now();
    let mut samples_ns = Vec::new();
    let mut deadline_misses = 0;
    let mut nonfinite_output_samples = 0;
    while measurement_started.elapsed() < measurement_target {
        let frame_started = Instant::now();
        let info = processor.process_frame(black_box(&input), black_box(&mut output))?;
        let elapsed = frame_started.elapsed();
        if info.bypassed {
            return Err("benchmark stimulus bypassed inference during measurement".into());
        }
        if elapsed > deadline {
            deadline_misses += 1;
        }
        nonfinite_output_samples += output.iter().filter(|sample| !sample.is_finite()).count();
        samples_ns.push(duration_ns(elapsed));
    }
    let measurement_wall_seconds = measurement_started.elapsed().as_secs_f64();
    let measured_hops = samples_ns.len();
    let processing_seconds = samples_ns.iter().sum::<u64>() as f64 / 1_000_000_000.0;
    let audio_seconds = measured_hops as f64 * HOP_SIZE as f64 / sample_rate as f64;
    let real_time_factor = processing_seconds / audio_seconds;
    let session_runs = processor.inference_run_count();
    let expected_session_runs = 1 + warmup_hops as u64 + measured_hops as u64;
    let latency = latency_summary(&samples_ns);
    let (rss_bytes, rss_measurement) = resident_set_size();

    Ok(CaseReport {
        sample_rate_hz: sample_rate,
        channels,
        algorithmic_latency_samples_per_channel: processor.latency_samples_per_channel(),
        algorithmic_latency_ms: processor.latency_samples_per_channel() as f64 / sample_rate as f64
            * 1_000.0,
        processor_construction_ms,
        first_frame_us: first_frame_ns as f64 / 1_000.0,
        warmup_hops,
        measured_hops,
        configured_measurement_seconds: measurement_seconds,
        measurement_wall_seconds,
        processing_seconds,
        audio_seconds,
        real_time_factor,
        throughput_x_realtime: 1.0 / real_time_factor,
        deadline_us: deadline.as_secs_f64() * 1_000_000.0,
        deadline_misses,
        nonfinite_output_samples,
        session_runs,
        expected_session_runs,
        rss_bytes,
        rss_measurement,
        serial_latency_p99_ms: (FFT_SIZE - HOP_SIZE) as f64 / sample_rate as f64 * 1000.0
            + latency.p99_us / 1000.0,
        serial_latency_max_ms: (FFT_SIZE - HOP_SIZE) as f64 / sample_rate as f64 * 1000.0
            + latency.max_us.max(first_frame_ns as f64 / 1000.0) / 1000.0,
        latency_target_met: (FFT_SIZE - HOP_SIZE) as f64 / sample_rate as f64 * 1_000_000.0
            + latency.max_us.max(first_frame_ns as f64 / 1000.0)
            < realtime_policy::ADDED_LATENCY_TARGET_MS * 1000.0,
        latency,
        samples_ns,
    })
}

fn evaluate_gates(
    cases: &[CaseReport],
    diagnostic: bool,
    hardware: &HardwareEvidence,
) -> PerformanceGates {
    let mut failed = Vec::new();
    if diagnostic {
        failed.push("diagnostic_run".to_owned());
    }
    if hardware.power_mode == "not-recorded" {
        failed.push("power_mode_not_recorded".to_owned());
    }
    for case in cases {
        let label = format!("{}hz_{}ch", case.sample_rate_hz, case.channels);
        if case.configured_measurement_seconds < RELEASE_SECONDS_PER_CASE {
            failed.push(format!("{label}_stress_duration"));
        }
        if case.measurement_wall_seconds < case.configured_measurement_seconds as f64
            || case.audio_seconds < case.configured_measurement_seconds as f64
        {
            failed.push(format!("{label}_stress_coverage"));
        }
        if case.warmup_hops == 0 {
            failed.push(format!("{label}_warmup"));
        }
        if case.algorithmic_latency_samples_per_channel > HOP_SIZE {
            failed.push(format!("{label}_algorithmic_latency"));
        }
        if case.channels == 2 && case.latency.p99_us >= 5_000.0 {
            failed.push(format!("{label}_p99"));
        }
        if case.serial_latency_max_ms >= realtime_policy::ADDED_LATENCY_LIMIT_MS {
            failed.push(format!("{label}_serial_latency_redline"));
        }
        if case.deadline_misses != 0 {
            failed.push(format!("{label}_deadline_misses"));
        }
        if case.real_time_factor >= 1.0 {
            failed.push(format!("{label}_rtf"));
        }
        if case.nonfinite_output_samples != 0 {
            failed.push(format!("{label}_nonfinite_output"));
        }
        if case.session_runs != case.expected_session_runs {
            failed.push(format!("{label}_session_run_count"));
        }
        if !matches!(case.rss_bytes, Some(rss) if rss <= MAX_PEAK_RSS_BYTES) {
            failed.push(format!("{label}_peak_rss"));
        }
    }
    PerformanceGates {
        passed: failed.is_empty(),
        preferred_target_met: cases.iter().all(|case| case.latency_target_met),
        failed,
        thresholds: GateThresholds {
            maximum_algorithmic_latency_samples: HOP_SIZE,
            maximum_stereo_p99_us: 5_000.0,
            target_serial_latency_us: realtime_policy::ADDED_LATENCY_TARGET_MS * 1000.0,
            maximum_serial_latency_us: realtime_policy::ADDED_LATENCY_LIMIT_MS * 1000.0,
            maximum_deadline_misses: 0,
            maximum_real_time_factor: 1.0,
            maximum_peak_rss_bytes: MAX_PEAK_RSS_BYTES,
            minimum_warmup_hops: 1,
            minimum_measurement_seconds_per_case: RELEASE_SECONDS_PER_CASE,
        },
    }
}
