//! Fixed-clock host simulation through the actual worker, never device qualification.
mod audio_io;

use soundex_core::{
    diagnostics::{realtime_context_head, realtime_context_head_traced},
    SoundExConfig,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn injected_hops(hop: usize) -> u32 {
    match hop {
        256 => 2,
        640 => 4,
        _ => 0,
    }
}

fn warmup_dry_audit(
    input: &audio_io::BinaryAudio,
    output: &[f32],
    trace: &serde_json::Value,
) -> serde_json::Value {
    let Some(events) = trace["events"].as_array() else {
        return serde_json::Value::Null;
    };
    let limiter = soundex_dsp::limiter::Limiter::with_mode(
        0.95,
        10.0,
        soundex_dsp::limiter::LimiterMode::SmoothKnee,
    );
    let samples = 128 * input.channels as usize;
    let mut rows = Vec::new();
    let mut maximum = 0.0_f64;
    for hop in 3..events.len() {
        // Two consecutive accepted dry-history outputs exclude the preceding
        // transition hop. The expected dry clock is always input hop minus two.
        if events[hop]["callback_ready"] != 1
            || events[hop - 1]["callback_ready"] != 1
            || events[hop - 1]["history_warming"] != 1
            || events[hop - 2]["history_warming"] != 1
        {
            continue;
        }
        let mut peak = 0.0_f64;
        let mut power = 0.0_f64;
        for (i, value) in output[hop * samples..(hop + 1) * samples]
            .iter()
            .enumerate()
        {
            let expected =
                limiter.process_sample(input.samples[(hop - 2) * samples + i].clamp(-1.0, 1.0));
            let error = f64::from(*value) - f64::from(expected);
            peak = peak.max(error.abs());
            power += error * error;
        }
        maximum = maximum.max(peak);
        rows.push(
            serde_json::json!({"hop": hop, "peak": peak, "rms": (power / samples as f64).sqrt()}),
        );
    }
    serde_json::json!({"eligible_hops": rows.len(), "maximum_pcm_error": maximum,
        "passed": !rows.is_empty() && maximum <= 2e-6, "hops": rows,
        "scope": "Protected aligned dry fidelity after two accepted history-warmup hops; existing functional PCM tolerance2e-6. This is not a limited replacement for the lossless restoration target, or a transient/listening criterion."})
}

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
    if args.len() != 6 && args.len() != 7 {
        return Err(
            "usage: soundex-worker-replay MODEL INPUT.sxa OUTPUT.sxa REPORT.json QOS|RT [TRACE|TRACE-LATE]"
                .into(),
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
    let late = args.get(6).is_some_and(|arg| arg == "TRACE-LATE");
    if late && input.frames / 128 < 1100 {
        return Err(
            "late-producer diagnostic requires at least1100 hops including recovery".into(),
        );
    }
    let (mut stream, epoch) = if args.len() == 7 {
        if args[6] != "TRACE" && !late {
            return Err("optional mode must be TRACE or TRACE-LATE".into());
        }
        let (stream, epoch) =
            realtime_context_head_traced(Path::new(&args[1]), config, input.frames / 128)?;
        (stream, Some(epoch))
    } else {
        (realtime_context_head(Path::new(&args[1]), config)?, None)
    };
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
        let stall_hops = if late { injected_hops(hop) } else { 0 };
        let requested =
            arrival + Duration::from_secs_f64(stall_hops as f64 * 128.0 / input.sample_rate as f64);
        if let Some(wait) = requested.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        let began = Instant::now();
        let callback = stream.process(source, destination)?;
        let elapsed = began.elapsed();
        // Everything below is host instrumentation, outside the measured callback.
        let worker = stream.worker_stats();
        events.push(serde_json::json!({"hop": hop, "callback_start_late_ns": began.saturating_duration_since(arrival).as_nanos() as u64,
            "nominal_arrival_trace_ns": epoch.map(|e| arrival.saturating_duration_since(e).as_nanos() as u64 + 1),
            "injected_stall_hops": stall_hops,
            "requested_arrival_trace_ns": epoch.map(|e| requested.saturating_duration_since(e).as_nanos() as u64 + 1),
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
    let (stats, worker, trace) = stream.shutdown_with_trace();
    let finite = output.iter().all(|x| x.is_finite() && x.abs() <= 0.95);
    let dry_audit = warmup_dry_audit(&input, &output, &trace);
    let dry_pass = dry_audit["passed"].as_bool().unwrap_or(!late);
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
        "events": events, "worker_trace": trace, "late_producer_faults": late, "warmup_dry_audit": dry_audit});
    std::fs::write(&args[4], serde_json::to_vec_pretty(&report)?)?;
    if !finite || stats.worker_failed || !drained || (late && !dry_pass) {
        return Err("callback output/worker failure retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{audio_io, injected_hops, input_accounting_complete, warmup_dry_audit};

    #[test]
    fn queue_rejections_are_not_pending_worker_input() {
        assert!(input_accounting_complete(1100, 1, 1097, 2));
        assert!(!input_accounting_complete(1100, 0, 1097, 2));
        assert!(input_accounting_complete(1100, 0, 1100, 0));
        assert!(!input_accounting_complete(1100, 0, 1099, 0));
        assert!(!input_accounting_complete(1100, 0, 1101, 0));
        assert!(!input_accounting_complete(0, 1, u64::MAX, 1));
    }

    #[test]
    fn faults_are_fixed_and_do_not_shift_subsequent_nominal_arrivals() {
        let faults: Vec<_> = (0..1100)
            .filter(|&h| injected_hops(h) > 0)
            .map(|h| (h, injected_hops(h)))
            .collect();
        assert_eq!(faults, [(256, 2), (640, 4)]);
        assert_eq!(injected_hops(257), 0);
        assert_eq!(injected_hops(641), 0);
    }

    #[test]
    fn dry_audit_detects_wrong_sample_timeline_for_both_channels() {
        let samples: Vec<_> = (0..512)
            .flat_map(|i| [i as f32 / 1000.0, -(i as f32) / 2000.0])
            .collect();
        let input = audio_io::BinaryAudio {
            sample_rate: 48000,
            channels: 2,
            frames: 512,
            samples,
        };
        let mut output = vec![0.0; 1024];
        output[512..].copy_from_slice(&input.samples[..512]);
        let trace = serde_json::json!({"events": vec![serde_json::json!({"callback_ready":1,"history_warming":1});4]});
        assert_eq!(warmup_dry_audit(&input, &output, &trace)["passed"], true);
        output.rotate_right(2);
        assert_eq!(warmup_dry_audit(&input, &output, &trace)["passed"], false);
    }
}
