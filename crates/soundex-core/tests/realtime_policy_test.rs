//! Boundary checks for timing, availability and integration delay accounting.

use soundex_core::{realtime_policy::assess, RealtimeStats};

fn healthy() -> RealtimeStats {
    RealtimeStats {
        accepted_hops: 9990,
        deadline_misses: 10,
        max_consecutive_deadline_misses: 2,
        ..RealtimeStats::default()
    }
}

#[test]
fn software_budget_counts_extra_buffers_and_has_separate_target_and_redline() {
    let normal = assess(healthy(), 44100, 3.0, 61.0, 0).unwrap();
    assert!(normal.latency_target_met && normal.latency_limit_met);
    assert!((normal.added_latency_ms - 5.865988662).abs() < 1e-8);
    let one_buffer = assess(healthy(), 44100, 3.0, 61.0, 128).unwrap();
    assert!(!one_buffer.latency_target_met && one_buffer.latency_limit_met);
    let two_buffers = assess(healthy(), 44100, 3.0, 61.0, 256).unwrap();
    assert!(!two_buffers.latency_limit_met);
}

#[test]
fn callback_and_worker_availability_cannot_be_hidden_by_low_added_delay() {
    assert!(
        !assess(healthy(), 48000, 100.0, 100.0, 0)
            .unwrap()
            .callback_budget_met
    );
    assert!(
        !assess(healthy(), 48000, 3.0, 500.0, 0)
            .unwrap()
            .callback_budget_met
    );
    assert!(
        assess(healthy(), 48000, 3.0, 61.0, 0)
            .unwrap()
            .enhancement_availability_met
    );
    let mut stats = healthy();
    stats.deadline_misses = 11; // More than 0.1 percent.
    assert!(
        !assess(stats, 48000, 3.0, 61.0, 0)
            .unwrap()
            .enhancement_availability_met
    );
    stats = healthy();
    stats.max_consecutive_deadline_misses = 7; // 20.317ms at 44.1kHz.
    assert!(
        !assess(stats, 44100, 3.0, 61.0, 0)
            .unwrap()
            .enhancement_availability_met
    );
    stats = healthy();
    stats.worker_failed = true;
    assert!(
        !assess(stats, 48000, 3.0, 61.0, 0)
            .unwrap()
            .enhancement_availability_met
    );
}

#[test]
fn no_data_or_invalid_timing_cannot_establish_a_pass() {
    assert!(
        !assess(RealtimeStats::default(), 48000, 0.0, 0.0, 0)
            .unwrap()
            .enhancement_availability_met
    );
    for (rate, p99, maximum) in [
        (96000, 1.0, 2.0),
        (48000, f64::NAN, 2.0),
        (48000, 1.0, f64::INFINITY),
        (48000, 2.0, 1.0),
        (48000, -1.0, 1.0),
    ] {
        assert!(assess(healthy(), rate, p99, maximum, 0).is_none());
    }
    assert!(assess(healthy(), 48000, 1.0, 2.0, usize::MAX).is_none());
}
