//! Experimental end-to-end CPU replay; not the realtime callback/device adapter.
#[cfg(feature = "ort-backend")]
mod audio_io;

#[cfg(feature = "ort-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use soundex_core::{diagnostics::context_head::ContextHeadStream, SoundExConfig};
    use std::{
        path::Path,
        time::{Duration, Instant},
    };
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 || args.len() > 7 {
        return Err("usage: soundex-context-replay MODEL INPUT.sxa OUTPUT.sxa REPORT.json [REPEATS] [trace]".into());
    }
    let input = audio_io::read_audio(Path::new(&args[2]))?;
    let repeats: usize = args.get(5).map(|x| x.parse()).transpose()?.unwrap_or(1);
    let trace = args.get(6).map(String::as_str) == Some("trace");
    let paced = std::env::var("SOUNDEX_DIAGNOSTIC_PACED").as_deref() == Ok("1");
    let qos_requested = std::env::var("SOUNDEX_DIAGNOSTIC_QOS").as_deref() == Ok("1");
    #[cfg(target_os = "macos")]
    let qos_applied =
        qos_requested && qos_threads::set_current_thread(qos_threads::Qos::High).is_ok();
    #[cfg(not(target_os = "macos"))]
    let qos_applied = false;
    if repeats == 0
        || (trace && (repeats != 1 || paced))
        || input.frames == 0
        || !input.frames.is_multiple_of(128)
        || input.samples.iter().any(|x| !x.is_finite())
    {
        return Err("expected nonempty finite whole hops; traces require one repeat".into());
    }
    let config = SoundExConfig::default()
        .sample_rate(input.sample_rate)
        .channels(input.channels);
    let load = Instant::now();
    let mut stream = ContextHeadStream::new(Path::new(&args[1]), config)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let width = input.channels as usize * 128;
    let mut output = vec![0.0; input.samples.len()];
    let mut times = Vec::with_capacity(input.frames as usize / 128 * repeats);
    let mut traces = Vec::new();
    let mut active_hops = 0;
    let period = Duration::from_secs_f64(128.0 / input.sample_rate as f64);
    let mut paced_finish_lateness_ms = Vec::new();
    let started = Instant::now();
    // Repeats preserve context/state; the loop seam is a diagnostic stimulus,
    // never presented as a natural song or a listening-quality measurement.
    for _ in 0..repeats {
        for (pcm, out) in input
            .samples
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            if paced {
                let arrival = started + period.mul_f64(times.len() as f64);
                if let Some(wait) = arrival.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
            times.push(stream.step(pcm, out)?);
            active_hops += usize::from(stream.enhancing());
            if paced {
                let deadline = started + period.mul_f64(times.len() as f64);
                paced_finish_lateness_ms.push(
                    Instant::now()
                        .saturating_duration_since(deadline)
                        .as_secs_f64()
                        * 1000.0,
                );
            }
            if trace {
                traces.push(stream.trace());
            }
        }
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let report = serde_json::json!({
        "schema_version": 1, "scope": "experimental Rust frontend + CPU ORT + production DSP; unpaced process; no realtime callback/transport/device qualification",
        "sample_rate": input.sample_rate, "channels": input.channels,
        "hop_samples": 128, "lookahead_samples": 0, "dsp_delay_samples": 128,
        "realtime_fixed_buffer_samples": 256, "warmup_hops": 80,
        "timing_columns": ["frontend_ms", "ort_ms", "dsp_ms", "total_ms"],
        "load_ms": load_ms, "wall_seconds": wall_seconds, "repeats": repeats,
        "paced": paced, "qos_requested": qos_requested, "qos_applied": qos_applied,
        "paced_finish_lateness_ms": paced_finish_lateness_ms,
        "timings": times, "traces": traces, "output_is_last_repeat": true,
        "finite": output.iter().all(|x| x.is_finite()),
        "active_hops": active_hops, "inference_hops": times.len(),
        "output_peak": output.iter().fold(0.0_f32, |a,x| a.max(x.abs()))
    });
    std::fs::write(&args[4], serde_json::to_vec(&report)?)?;
    audio_io::write_audio(
        Path::new(&args[3]),
        &audio_io::BinaryAudio {
            samples: output,
            ..input
        },
    )?;
    Ok(())
}

#[cfg(not(feature = "ort-backend"))]
fn main() {
    eprintln!("requires ort-backend");
    std::process::exit(1);
}
