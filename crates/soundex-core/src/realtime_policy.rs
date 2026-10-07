//! Engineering acceptance policy for continuous real-time playback enhancement.

use crate::RealtimeStats;

/// Default host callback size; align callbacks with the model's hop boundaries.
pub const CALLBACK_FRAMES: usize = 128;
/// Preferred added software delay, including any integration buffering.
pub const ADDED_LATENCY_TARGET_MS: f64 = 8.0;
/// Strict added software delay redline; hardware's existing delay is separate.
pub const ADDED_LATENCY_LIMIT_MS: f64 = 10.0;
/// Preferred 99th percentile audio-side callback execution time.
pub const CALLBACK_P99_TARGET_US: f64 = 100.0;
/// Strict maximum observed audio-side callback execution time.
pub const CALLBACK_MAX_LIMIT_US: f64 = 500.0;
/// Maximum missed worker presentation ratio, excluding startup padding.
pub const WORKER_MISS_RATIO_LIMIT: f64 = 0.001;
/// Maximum contiguous missed worker presentation duration.
pub const WORKER_MISS_BURST_LIMIT_MS: f64 = 20.0;

/// Component assessment, independent of device xruns and model quality gates.
#[derive(Clone, Copy, Debug)]
pub struct Assessment {
    /// Fixed adapter delay + integration buffering + maximum callback time.
    pub added_latency_ms: f64,
    /// Fraction of valid presentation deadlines missed by the worker.
    pub worker_miss_ratio: f64,
    /// Duration of the longest contiguous presentation miss sequence.
    pub worker_miss_burst_ms: f64,
    /// Whether the preferred 8ms software budget is met.
    pub latency_target_met: bool,
    /// Whether the strict 10ms software redline is met.
    pub latency_limit_met: bool,
    /// Whether the preferred p99 and strict maximum callback budgets are met.
    pub callback_budget_met: bool,
    /// Whether enough results arrive on time without a long outage or failure.
    pub enhancement_availability_met: bool,
}

/// Assess measured callback/worker evidence without allocating.
///
/// `extra_buffer_frames` includes buffering added outside this adapter. Returns
/// `None` for unsupported rates, invalid timing values, or overflowing counts.
/// Empty measurement sets never establish enhancement availability. This is
/// not release certification: sample integrity, device xruns, long-duration
/// stress, model parity and listening evidence must be validated separately.
pub fn assess(
    stats: RealtimeStats,
    sample_rate: u32,
    callback_p99_us: f64,
    callback_max_us: f64,
    extra_buffer_frames: usize,
) -> Option<Assessment> {
    if !matches!(sample_rate, 44100 | 48000)
        || !callback_p99_us.is_finite()
        || !callback_max_us.is_finite()
        || callback_p99_us < 0.0
        || callback_max_us < callback_p99_us
    {
        return None;
    }
    let frames = (2 * CALLBACK_FRAMES).checked_add(extra_buffer_frames)?;
    let added_latency_ms = frames as f64 / sample_rate as f64 * 1000.0 + callback_max_us / 1000.0;
    let measured = stats.accepted_hops.checked_add(stats.deadline_misses)?;
    let worker_miss_ratio = if measured == 0 {
        1.0
    } else {
        stats.deadline_misses as f64 / measured as f64
    };
    let worker_miss_burst_ms =
        stats.max_consecutive_deadline_misses as f64 * CALLBACK_FRAMES as f64 / sample_rate as f64
            * 1000.0;
    Some(Assessment {
        added_latency_ms,
        worker_miss_ratio,
        worker_miss_burst_ms,
        latency_target_met: added_latency_ms < ADDED_LATENCY_TARGET_MS,
        latency_limit_met: added_latency_ms < ADDED_LATENCY_LIMIT_MS,
        callback_budget_met: callback_p99_us < CALLBACK_P99_TARGET_US
            && callback_max_us < CALLBACK_MAX_LIMIT_US,
        enhancement_availability_met: measured > 0
            && !stats.worker_failed
            && stats.invalid_samples == 0
            && stats.rejected_results == 0
            && worker_miss_ratio <= WORKER_MISS_RATIO_LIMIT
            && worker_miss_burst_ms <= WORKER_MISS_BURST_LIMIT_MS,
    })
}
