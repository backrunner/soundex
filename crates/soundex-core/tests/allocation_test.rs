#![cfg(feature = "ort-backend")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    path::PathBuf,
};

use soundex_core::{SoundExConfig, SoundExProcessor};

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

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx")
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

fn processor(enhance: bool) -> SoundExProcessor {
    let mut config = SoundExConfig::with_model(model_path()).channels(2);
    config.min_bandwidth_ratio = if enhance { 1.0 } else { 0.0 };
    SoundExProcessor::new(config).unwrap()
}

#[test]
fn warmed_process_frame_has_no_rust_heap_allocations() {
    let input = stereo_tone(512);
    let mut output = vec![0.0; input.len()];

    for enhance in [true, false] {
        let mut processor = processor(enhance);
        for _ in 0..4 {
            processor.process_frame(&input, &mut output).unwrap();
        }

        // This observes Rust's global allocator. Allocations made internally by
        // ONNX Runtime's C/C++ allocator require the production benchmark/RSS run.
        let (result, allocations) =
            count_allocations(|| processor.process_frame(&input, &mut output));

        result.unwrap();
        assert_eq!(
            allocations,
            0,
            "warmed {} path made {allocations} Rust heap allocation(s)",
            if enhance { "enhancement" } else { "bypass" }
        );
    }
}
