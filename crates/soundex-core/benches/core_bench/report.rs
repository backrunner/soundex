//! Artifact-bound benchmark report schema.

use serde::Serialize;

#[derive(Serialize)]
pub(super) struct ArtifactEvidence {
    pub(super) path: String,
    pub(super) sha256: String,
    pub(super) size_bytes: u64,
}

#[derive(Serialize)]
pub(super) struct HardwareEvidence {
    pub(super) label: String,
    pub(super) cpu: String,
    pub(super) architecture: String,
    pub(super) operating_system: String,
    pub(super) host: String,
    pub(super) power_mode: String,
    pub(super) rustc: String,
}

#[derive(Serialize)]
pub(super) struct OrtEvidence {
    pub(super) intra_threads: usize,
    pub(super) inter_threads: usize,
    pub(super) parallel_execution: bool,
    pub(super) graph_optimization: String,
    pub(super) output_preallocation: bool,
}

#[derive(Serialize)]
pub(super) struct FrameProtocol {
    pub(super) fft_size: usize,
    pub(super) hop_size: usize,
    pub(super) frequency_bins: usize,
}

#[derive(Serialize)]
pub(super) struct LatencySummary {
    pub(super) p50_us: f64,
    pub(super) p95_us: f64,
    pub(super) p99_us: f64,
    pub(super) max_us: f64,
}

#[derive(Serialize)]
pub(super) struct CaseReport {
    pub(super) sample_rate_hz: u32,
    pub(super) channels: u16,
    pub(super) algorithmic_latency_samples_per_channel: usize,
    pub(super) algorithmic_latency_ms: f64,
    pub(super) processor_construction_ms: f64,
    pub(super) first_frame_us: f64,
    pub(super) warmup_hops: usize,
    pub(super) measured_hops: usize,
    pub(super) configured_measurement_seconds: u64,
    pub(super) measurement_wall_seconds: f64,
    pub(super) processing_seconds: f64,
    pub(super) audio_seconds: f64,
    pub(super) real_time_factor: f64,
    pub(super) throughput_x_realtime: f64,
    pub(super) deadline_us: f64,
    pub(super) deadline_misses: usize,
    pub(super) nonfinite_output_samples: usize,
    pub(super) session_runs: u64,
    pub(super) expected_session_runs: u64,
    pub(super) rss_bytes: Option<u64>,
    pub(super) rss_measurement: &'static str,
    pub(super) serial_latency_p99_ms: f64,
    pub(super) serial_latency_max_ms: f64,
    pub(super) latency_target_met: bool,
    pub(super) latency: LatencySummary,
    pub(super) samples_ns: Vec<u64>,
}

#[derive(Serialize)]
pub(super) struct GateThresholds {
    pub(super) maximum_algorithmic_latency_samples: usize,
    pub(super) maximum_stereo_p99_us: f64,
    pub(super) target_serial_latency_us: f64,
    pub(super) maximum_serial_latency_us: f64,
    pub(super) maximum_deadline_misses: usize,
    pub(super) maximum_real_time_factor: f64,
    pub(super) maximum_peak_rss_bytes: u64,
    pub(super) minimum_warmup_hops: usize,
    pub(super) minimum_measurement_seconds_per_case: u64,
}

#[derive(Serialize)]
pub(super) struct PerformanceGates {
    pub(super) passed: bool,
    pub(super) preferred_target_met: bool,
    pub(super) failed: Vec<String>,
    pub(super) thresholds: GateThresholds,
}

#[derive(Serialize)]
pub(super) struct BenchmarkReport {
    pub(super) schema_version: u32,
    pub(super) report_type: &'static str,
    pub(super) generated_unix_seconds: u64,
    pub(super) diagnostic: bool,
    pub(super) artifact: ArtifactEvidence,
    pub(super) protocol: FrameProtocol,
    pub(super) hardware: HardwareEvidence,
    pub(super) ort: OrtEvidence,
    pub(super) cold_model_load_ms: f64,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) cases: Vec<CaseReport>,
    pub(super) performance_gates: PerformanceGates,
}
