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

use serde::Serialize;
use sha2::{Digest, Sha256};
use soundex_core::{SoundExConfig, SoundExProcessor};

const MIN_PRODUCTION_MODEL_BYTES: u64 = 1_000_000;
const MAX_PRODUCTION_MODEL_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PEAK_RSS_BYTES: u64 = 50 * 1024 * 1024;
const RELEASE_SECONDS_PER_CASE: u64 = 30 * 60;
const DEFAULT_WARMUP_HOPS: usize = 32;
const FFT_SIZE: usize = 1024;
const HOP_SIZE: usize = 512;

#[derive(Serialize)]
struct ArtifactEvidence {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Serialize)]
struct HardwareEvidence {
    label: String,
    cpu: String,
    architecture: String,
    operating_system: String,
    host: String,
    power_mode: String,
    rustc: String,
}

#[derive(Serialize)]
struct OrtEvidence {
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
    graph_optimization: String,
    output_preallocation: bool,
}

#[derive(Serialize)]
struct FrameProtocol {
    fft_size: usize,
    hop_size: usize,
    frequency_bins: usize,
}

#[derive(Serialize)]
struct LatencySummary {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

#[derive(Serialize)]
struct CaseReport {
    sample_rate_hz: u32,
    channels: u16,
    algorithmic_latency_samples_per_channel: usize,
    algorithmic_latency_ms: f64,
    processor_construction_ms: f64,
    first_frame_us: f64,
    warmup_hops: usize,
    measured_hops: usize,
    configured_measurement_seconds: u64,
    measurement_wall_seconds: f64,
    processing_seconds: f64,
    audio_seconds: f64,
    real_time_factor: f64,
    throughput_x_realtime: f64,
    deadline_us: f64,
    deadline_misses: usize,
    nonfinite_output_samples: usize,
    session_runs: u64,
    expected_session_runs: u64,
    rss_bytes: Option<u64>,
    rss_measurement: &'static str,
    latency: LatencySummary,
    samples_ns: Vec<u64>,
}

#[derive(Serialize)]
struct GateThresholds {
    maximum_algorithmic_latency_samples: usize,
    maximum_stereo_p99_us: f64,
    maximum_deadline_misses: usize,
    maximum_real_time_factor: f64,
    maximum_peak_rss_bytes: u64,
    minimum_warmup_hops: usize,
    minimum_measurement_seconds_per_case: u64,
}

#[derive(Serialize)]
struct PerformanceGates {
    passed: bool,
    failed: Vec<String>,
    thresholds: GateThresholds,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    report_type: &'static str,
    generated_unix_seconds: u64,
    diagnostic: bool,
    artifact: ArtifactEvidence,
    protocol: FrameProtocol,
    hardware: HardwareEvidence,
    ort: OrtEvidence,
    cold_model_load_ms: f64,
    peak_rss_bytes: Option<u64>,
    cases: Vec<CaseReport>,
    performance_gates: PerformanceGates,
}

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
        schema_version: 2,
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
        failed,
        thresholds: GateThresholds {
            maximum_algorithmic_latency_samples: HOP_SIZE,
            maximum_stereo_p99_us: 5_000.0,
            maximum_deadline_misses: 0,
            maximum_real_time_factor: 1.0,
            maximum_peak_rss_bytes: MAX_PEAK_RSS_BYTES,
            minimum_warmup_hops: 1,
            minimum_measurement_seconds_per_case: RELEASE_SECONDS_PER_CASE,
        },
    }
}

fn validate_artifact(path: &Path) -> Result<ArtifactEvidence, Box<dyn Error>> {
    let canonical = path.canonicalize().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("cannot resolve benchmark model {}: {error}", path.display()),
        )
    })?;
    let metadata = canonical.metadata()?;
    if !metadata.is_file() {
        return Err("SOUNDEX_BENCH_MODEL must name a regular ONNX file".into());
    }
    let sha256 = sha256_file(&canonical)?;
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx");
    let fixture_hash = fixture
        .is_file()
        .then(|| sha256_file(&fixture))
        .transpose()?;
    if fixture_hash.as_deref() == Some(&sha256)
        || canonical.file_name().and_then(|name| name.to_str()) == Some("identity.onnx")
    {
        return Err(
            "identity.onnx is a correctness fixture, not a production benchmark model".into(),
        );
    }
    if metadata.len() < MIN_PRODUCTION_MODEL_BYTES {
        return Err(format!(
            "model is only {} bytes; production benchmark requires a non-trivial artifact of at least {MIN_PRODUCTION_MODEL_BYTES} bytes",
            metadata.len()
        )
        .into());
    }
    if metadata.len() >= MAX_PRODUCTION_MODEL_BYTES {
        return Err(format!(
            "model is {} bytes; production benchmark requires the FP32 artifact to be smaller than {MAX_PRODUCTION_MODEL_BYTES} bytes",
            metadata.len()
        )
        .into());
    }
    Ok(ArtifactEvidence {
        path: canonical.display().to_string(),
        sha256,
        size_bytes: metadata.len(),
    })
}

fn forced_enhancement_input(sample_rate: u32, channels: u16) -> Vec<f32> {
    (0..HOP_SIZE)
        .flat_map(|sample| {
            let time = sample as f32 / sample_rate as f32;
            let left = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * time).sin();
            let right = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * time).sin();
            [left, right].into_iter().take(channels as usize)
        })
        .collect()
}

fn latency_summary(samples_ns: &[u64]) -> LatencySummary {
    let mut sorted = samples_ns.to_vec();
    sorted.sort_unstable();
    LatencySummary {
        p50_us: percentile_ns(&sorted, 0.50) / 1_000.0,
        p95_us: percentile_ns(&sorted, 0.95) / 1_000.0,
        p99_us: percentile_ns(&sorted, 0.99) / 1_000.0,
        max_us: sorted.last().copied().unwrap_or(0) as f64 / 1_000.0,
    }
}

fn percentile_ns(sorted: &[u64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)] as f64
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_report(path: &Path, report: &BenchmarkReport) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .map(workspace_relative)
        .ok_or_else(|| {
            format!("{name} is required and must name the production ONNX artifact").into()
        })
}

fn workspace_relative(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace_root().join(path)
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_number<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|error| format!("invalid {name}={value:?}: {error}").into()),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "yes"))
}

fn hardware_evidence() -> HardwareEvidence {
    let cpu = cpu_name();
    HardwareEvidence {
        label: env::var("SOUNDEX_BENCH_HARDWARE_LABEL").unwrap_or_else(|_| cpu.clone()),
        cpu,
        architecture: env::consts::ARCH.to_owned(),
        operating_system: env::consts::OS.to_owned(),
        host: env::var("HOSTNAME")
            .or_else(|_| env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_owned()),
        power_mode: env::var("SOUNDEX_BENCH_POWER_MODE")
            .unwrap_or_else(|_| "not-recorded".to_owned()),
        rustc: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
    }
}

fn cpu_name() -> String {
    if cfg!(target_os = "macos") {
        return command_output("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown Apple CPU".to_owned());
    }
    if cfg!(target_os = "linux") {
        if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(value) = contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                matches!(key.trim(), "model name" | "Processor").then(|| value.trim().to_owned())
            }) {
                return value;
            }
        }
    }
    env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_owned())
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(target_os = "linux")]
fn resident_set_size() -> (Option<u64>, &'static str) {
    let value = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let value = line.strip_prefix("VmHWM:")?;
                value.split_whitespace().next()?.parse::<u64>().ok()
            })
        })
        .map(|kilobytes| kilobytes * 1024);
    (value, "linux-vmhwm")
}

#[cfg(target_os = "macos")]
fn resident_set_size() -> (Option<u64>, &'static str) {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let value = (result == 0).then(|| unsafe { usage.assume_init().ru_maxrss as u64 });
    (value, "macos-getrusage-ru_maxrss")
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn resident_set_size() -> (Option<u64>, &'static str) {
    let process_id = process::id().to_string();
    let value = command_output("ps", &["-o", "rss=", "-p", &process_id])
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|kilobytes| kilobytes * 1024);
    (value, "ps-current-rss")
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
