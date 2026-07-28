//! Benchmarks for soundex-dsp algorithms.

use criterion::{criterion_group, criterion_main, Criterion};
use soundex_dsp::bandwidth::{BandwidthDetector, BandwidthDetectorConfig};
use soundex_dsp::stft::{StftAnalyzer, StftSynthesizer};
use soundex_dsp::window;

fn bench_hann_window(c: &mut Criterion) {
    c.bench_function("hann_window_1024", |b| b.iter(|| window::hann(1024)));
}

fn bench_stft_analyze(c: &mut Criterion) {
    let mut analyzer = StftAnalyzer::new(1024, 512);
    let frame = vec![0.0f32; 1024];
    c.bench_function("stft_analyze_1024", |b| b.iter(|| analyzer.analyze(&frame)));
}

fn bench_stft_roundtrip(c: &mut Criterion) {
    let mut analyzer = StftAnalyzer::new(1024, 512);
    let mut synthesizer = StftSynthesizer::new(1024, 512);
    let frame = vec![0.1f32; 1024];
    c.bench_function("stft_roundtrip_1024", |b| {
        b.iter(|| {
            let spectral = analyzer.analyze(&frame);
            synthesizer.synthesize(&spectral)
        })
    });
}

fn bench_bandwidth_detection(c: &mut Criterion) {
    let mut detector = BandwidthDetector::new(BandwidthDetectorConfig::default());
    let spectrum = vec![-20.0; 513];
    c.bench_function("bandwidth_detect_513", |b| {
        b.iter(|| detector.detect(&spectrum, 44_100));
    });
}

criterion_group!(
    benches,
    bench_hann_window,
    bench_stft_analyze,
    bench_stft_roundtrip,
    bench_bandwidth_detection
);
criterion_main!(benches);
