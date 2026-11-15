//! Fixed-clock host simulation through the actual worker, never device qualification.
mod audio_io;

use soundex_core::{diagnostics::realtime_context_head, SoundExConfig};
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn input_accounting_complete(
    attempted: u64,
    rejected: u64,
    processed: u64,
    discarded: u64,
) -> bool {
    match (
        attempted.checked_sub(rejected),
        processed.checked_add(discarded),
    ) {
        (Some(expected), Some(observed)) => expected == observed,
        _ => false,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: soundex-worker-replay MODEL INPUT.sxa OUTPUT.sxa REPORT.json QOS|RT".into(),
        );
    }
    let input = audio_io::read_audio(Path::new(&args[2]))?;
    if input.frames < 256 || !input.frames.is_multiple_of(128) {
        return Err("complete nonempty hop sequence required".into());
    }
    let rt = match args[5].as_str() {
        "QOS" => false,
        "RT" => true,
        _ => return Err("QOS or RT required".into()),
    };
    let mut config = SoundExConfig::default()
        .sample_rate(input.sample_rate)
        .channels(input.channels);
    config.worker_time_constraint = rt;
    let mut stream = realtime_context_head(Path::new(&args[1]), config)?;
    let samples = 128 * input.channels as usize;
    let mut output = vec![0.0; input.samples.len()];
    let mut events = Vec::with_capacity(input.frames / 128);
    let mut previous_misses = 0;
    let start = Instant::now();
    for (hop, (source, destination)) in input
        .samples
        .chunks(samples)
        .zip(output.chunks_mut(samples))
        .enumerate()
    {
        let arrival =
            start + Duration::from_secs_f64(hop as f64 * 128.0 / input.sample_rate as f64);
        if let Some(wait) = arrival.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        let began = Instant::now();
        let callback = stream.process(source, destination)?;
        let elapsed = began.elapsed();
        // Everything below is host instrumentation, outside the measured callback.
        let worker = stream.worker_stats();
        events.push(serde_json::json!({"hop": hop, "callback_start_late_ns": began.saturating_duration_since(arrival).as_nanos() as u64,
            "callback_ns": elapsed.as_nanos() as u64, "startup_padding": hop < 2,
            "worker_presentation_missed": callback.deadline_misses > previous_misses,
            "accepted_hops": callback.accepted_hops,
            "deadline_misses": callback.deadline_misses, "max_consecutive_deadline_misses": callback.max_consecutive_deadline_misses,
            "queue_overflows": callback.queue_overflows, "stale_results": callback.stale_results,
            "worker_failed": callback.worker_failed, "worker_processed_hops": worker.processed_hops,
            "worker_discarded_inputs": worker.discarded_input_hops,
            "worker_state_warmup_hops": worker.state_history_warmup_hops,
            "max_queue_wait_ns": worker.max_queue_wait_ns, "max_processing_ns": worker.max_processing_ns,
            "processing_period_overruns": worker.processing_period_overruns}));
        previous_misses = callback.deadline_misses;
    }
    // Drain the submitted work off the callback timeline so the final worker
    // maximum includes its last call. Never extend or change the output clock.
    let drain_until = Instant::now() + Duration::from_secs(1);
    let drained = loop {
        let w = stream.worker_stats();
        if input_accounting_complete(
            (input.frames / 128) as u64,
            stream.stats().queue_overflows,
            w.processed_hops,
            w.discarded_input_hops,
        ) {
            break true;
        }
        if stream.stats().worker_failed || Instant::now() >= drain_until {
            break false;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    let latency = stream.latency_samples_per_channel();
    // Join first: the processed counter can precede the worker's final metric stores.
    let (stats, worker) = stream.shutdown_with_stats();
    let finite = output.iter().all(|x| x.is_finite() && x.abs() <= 0.95);
    audio_io::write_audio(
        Path::new(&args[3]),
        &audio_io::BinaryAudio {
            samples: output,
            ..input
        },
    )?;
    let report = serde_json::json!({"scope": "Actual callback/worker host simulation. No physical device or xrun measurement. Worker statistics are cumulative snapshots, not per-hop CPU attribution. All startup/warmup/failure counters retained.",
        "sample_rate": input.sample_rate, "channels": input.channels, "fixed_delay_frames": latency,
        "time_constraint_requested": rt, "time_constraint_accepted": worker.macos_realtime_request_accepted,
        "qos_applied": worker.macos_qos_applied, "finite": finite, "worker_failed": stats.worker_failed,
        "worker_drain_complete": drained, "output_frames": stats.output_frames,
        "attempted_input_hops": input.frames / 128, "rejected_input_hops": stats.queue_overflows,
        "processed_hops": worker.processed_hops, "discarded_input_hops": worker.discarded_input_hops,
        "discarded_output_hops": worker.discarded_output_hops,
        "state_history_warmup_hops": worker.state_history_warmup_hops,
        "processing_period_overruns": worker.processing_period_overruns,
        "warmup_runs": worker.warmup_runs, "deadline_misses": stats.deadline_misses,
        "max_consecutive_deadline_misses": stats.max_consecutive_deadline_misses,
        "max_processing_ns": worker.max_processing_ns, "max_queue_wait_ns": worker.max_queue_wait_ns,
        "processing_p99_upper_bound_ns": worker.processing_p99_upper_bound_ns,
        "queue_wait_p99_upper_bound_ns": worker.queue_wait_p99_upper_bound_ns,
        "events": events});
    std::fs::write(&args[4], serde_json::to_vec_pretty(&report)?)?;
    if !finite || stats.worker_failed || !drained {
        return Err("callback output/worker failure retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::input_accounting_complete;

    #[test]
    fn queue_rejections_are_not_pending_worker_input() {
        assert!(input_accounting_complete(1100, 1, 1097, 2));
        assert!(!input_accounting_complete(1100, 0, 1097, 2));
        assert!(input_accounting_complete(1100, 0, 1100, 0));
        assert!(!input_accounting_complete(1100, 0, 1099, 0));
        assert!(!input_accounting_complete(1100, 0, 1101, 0));
        assert!(!input_accounting_complete(0, 1, u64::MAX, 1));
    }
}
