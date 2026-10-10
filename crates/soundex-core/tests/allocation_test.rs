#![cfg(feature = "ort-backend")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    path::PathBuf,
};

use soundex_core::{EnhancementMode, RealtimeProcessor, SoundExConfig, SoundExProcessor};

struct CountingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

fn record_allocation() {
    let tracking = TRACKING.try_with(Cell::get).unwrap_or(false);
    if tracking {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.with(|count| count.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    let result = operation();
    TRACKING.with(|tracking| tracking.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    (result, allocations)
}

fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn stereo_tone(length: usize) -> Vec<f32> {
    (0..length)
        .flat_map(|sample| {
            let time = sample as f32 / 44_100.0;
            let left = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * time).sin();
            let right = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * time).sin();
            [left, right]
        })
        .collect()
}

fn processor(enhance: bool, name: &str, fft_size: usize, hop_size: usize) -> SoundExProcessor {
    let mut config = SoundExConfig::with_model(model_path(name))
        .fft_size(fft_size)
        .hop_size(hop_size)
        .channels(2);
    config.min_bandwidth_ratio = if enhance { 1.0 } else { 0.0 };
    SoundExProcessor::new(config).unwrap()
}

#[test]
fn warmed_process_frame_has_no_rust_heap_allocations() {
    for (name, fft_size, hop_size) in [
        ("identity.onnx", 1024, 512),
        ("low-latency-identity.onnx", 256, 128),
    ] {
        let input = stereo_tone(hop_size);
        let mut output = vec![0.0; input.len()];
        for enhance in [true, false] {
            let mut processor = processor(enhance, name, fft_size, hop_size);
            for _ in 0..4 {
                processor.process_frame(&input, &mut output).unwrap();
            }
            // Rust allocator only; ORT C/C++ allocations require the RSS stress run.
            let (result, allocations) =
                count_allocations(|| processor.process_frame(&input, &mut output));
            result.unwrap();
            assert_eq!(allocations, 0, "{fft_size}/{hop_size} enhance={enhance}");
        }
    }
}

#[test]
fn realtime_callback_allocates_nothing_including_startup_and_saturated_fallback() {
    for mode in [
        EnhancementMode::Neural,
        EnhancementMode::Spectral,
        EnhancementMode::Hybrid,
    ] {
        let config = SoundExConfig::with_model(model_path("low-latency-identity.onnx"))
            .channels(2)
            .enhancement_mode(mode);
        let mut processor = RealtimeProcessor::new(config).unwrap();
        let mut output = [0.0; 256];
        let mut input = [0.125; 256];
        input[9] = f32::NAN;
        input[12] = f32::INFINITY;
        let (result, allocations) = count_allocations(|| {
            for _ in 0..1000 {
                processor.process(&input, &mut output)?;
            }
            Ok::<_, soundex_core::SoundExError>(())
        });
        result.unwrap();
        assert_eq!(allocations, 0, "{mode:?}");
        assert!(output.iter().all(|sample| sample.is_finite()));
        processor.shutdown();
    }
}

#[test]
fn extension_paths_allocate_nothing_in_fixed_hop_processing() {
    for mode in [EnhancementMode::Spectral, EnhancementMode::Hybrid] {
        let mut config = SoundExConfig::with_model(model_path("low-latency-identity.onnx"))
            .channels(2)
            .enhancement_mode(mode);
        config.min_bandwidth_ratio = 1.0;
        let mut processor = SoundExProcessor::new(config).unwrap();
        let input = stereo_tone(128);
        let mut output = [0.0; 256];
        for _ in 0..4 {
            processor.process_frame(&input, &mut output).unwrap();
        }
        let (result, allocations) =
            count_allocations(|| processor.process_frame(&input, &mut output));
        result.unwrap();
        assert_eq!(allocations, 0, "{mode:?}");
    }
}
