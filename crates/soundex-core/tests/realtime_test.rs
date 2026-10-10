//! Actual-worker failure and recovery contracts for the audio callback adapter.
#![cfg(feature = "ort-backend")]

use soundex_core::{RealtimeProcessor, SoundExConfig, SoundExProcessor};
use std::{
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

fn config(name: &str) -> SoundExConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    SoundExConfig::with_model(path).channels(2)
}

#[test]
fn actual_model_failure_leaves_continuous_aligned_dry_audio() {
    let mut settings = config("nonfinite-output.onnx");
    settings.min_bandwidth_ratio = 1.0;
    let mut processor = RealtimeProcessor::new(settings).unwrap();
    let input: Vec<f32> = (0..128)
        .flat_map(|n| {
            let sample = 0.2 * (n as f32 * 0.057).sin();
            [sample, -sample]
        })
        .collect();
    let mut output = [f32::NAN; 256];
    processor.process(&input, &mut output).unwrap();
    let until = Instant::now() + Duration::from_secs(2);
    while !processor.stats().worker_failed && Instant::now() < until {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(processor.stats().worker_failed);
    assert_eq!(processor.latency_samples_per_channel(), 256);
    for n in 0..100 {
        processor.process(&input, &mut output).unwrap();
        if n >= 1 {
            assert_eq!(&output, input.as_slice());
        }
        assert!(output.iter().all(|sample| sample.is_finite()));
    }
    assert_eq!(processor.stats().output_frames, 101 * 128);
    assert!(processor.stats().queue_overflows > 0);
    processor.shutdown();
}

#[test]
fn actual_worker_rejoins_after_saturation_without_changing_dry_delay() {
    let mut settings = config("low-latency-identity.onnx");
    settings.min_bandwidth_ratio = 0.0;
    let mut processor = RealtimeProcessor::new(settings).unwrap();
    let input = [0.125; 256];
    let mut output = [f32::NAN; 256];
    // Saturate deliberately; the callback must not wait for the worker.
    for n in 0..2000 {
        processor.process(&input, &mut output).unwrap();
        assert_eq!(output, [if n < 2 { 0.0 } else { 0.125 }; 256]);
    }
    let before = processor.stats().accepted_hops;
    // Control-thread pacing, not part of callback processing.
    for _ in 0..40 {
        thread::sleep(Duration::from_millis(5));
        processor.process(&input, &mut output).unwrap();
        assert_eq!(output, input);
    }
    assert!(processor.stats().accepted_hops > before + 20);
    assert!(!processor.stats().worker_failed);
    processor.shutdown();
}

#[test]
fn incompatible_realtime_profile_is_rejected_before_starting_worker() {
    let legacy = config("identity.onnx").fft_size(1024).hop_size(512);
    assert!(RealtimeProcessor::new(legacy).is_err());
    assert!(
        RealtimeProcessor::new(config("low-latency-identity.onnx").sample_rate(96000)).is_err()
    );
}

#[test]
fn caller_shape_error_clears_output_and_does_not_poison_the_stream() {
    let mut processor = RealtimeProcessor::new(config("low-latency-identity.onnx")).unwrap();
    let mut malformed_output = [0.5; 3];
    assert!(processor.process(&[0.0; 3], &mut malformed_output).is_err());
    assert_eq!(malformed_output, [0.0; 3]);
    let mut output = [1.0; 2];
    processor.process(&[0.0; 2], &mut output).unwrap();
    assert_eq!(output, [0.0; 2]);
    processor.shutdown();
}

#[test]
fn silence_clears_detector_hangover_and_stops_generator_runs() {
    let mut settings = config("low-latency-identity.onnx");
    settings.min_bandwidth_ratio = 1.0;
    let mut processor = SoundExProcessor::new(settings).unwrap();
    let input: Vec<f32> = (0..128)
        .flat_map(|n| {
            let sample = 0.2 * (n as f32 * 0.057).sin();
            [sample, -sample]
        })
        .collect();
    let mut output = [0.0; 256];
    for _ in 0..10 {
        processor.process_frame(&input, &mut output).unwrap();
    }
    let before = processor.inference_run_count();
    for _ in 0..20 {
        processor.process_frame(&[0.0; 256], &mut output).unwrap();
    }
    assert!(processor.inference_run_count() - before <= 1);
    assert_eq!(output, [0.0; 256]);
    assert!(processor.is_bypassed());
}

#[test]
fn construction_prepares_both_batches_before_accepting_audio() {
    let processor = RealtimeProcessor::new(config("low-latency-identity.onnx")).unwrap();
    let worker = processor.worker_stats();
    assert_eq!(worker.warmup_runs, 8);
    assert!(worker.warmup_ns > 0);
    assert_eq!(worker.processed_hops, 0);
    assert_eq!(worker.max_processing_ns, 0);
    assert_eq!(processor.stats().output_frames, 0);
    assert!(!processor.stats().worker_failed);
    processor.shutdown();

    let mut cold = config("low-latency-identity.onnx");
    cold.realtime_warmup = false;
    let processor = RealtimeProcessor::new(cold).unwrap();
    assert_eq!(processor.worker_stats().warmup_runs, 0);
    processor.shutdown();
}
