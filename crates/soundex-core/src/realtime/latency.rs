//! Bounded worker-only timing observations; snapshots stay off the callback.

use std::sync::atomic::{AtomicU64, Ordering};

const WIDTH_NS: u64 = 25_000;
const BINS: usize = 128;

pub(super) struct Histogram {
    bins: [AtomicU64; BINS],
    maximum: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            bins: std::array::from_fn(|_| AtomicU64::new(0)),
            maximum: AtomicU64::new(0),
        }
    }
}

impl Histogram {
    pub fn record(&self, nanoseconds: u64) {
        let index = (nanoseconds.saturating_sub(1) / WIDTH_NS).min((BINS - 1) as u64);
        self.maximum.fetch_max(nanoseconds, Ordering::Relaxed);
        self.bins[index as usize].fetch_add(1, Ordering::Release);
    }

    /// Nearest-rank p99 upper bound, 25us resolution below the overflow bin.
    /// A live snapshot is approximate; take it after the worker stops for a
    /// stable distribution. The overflow bin uses the observed maximum.
    pub fn p99_upper_bound_ns(&self) -> u64 {
        let bins = self.bins.each_ref().map(|bin| bin.load(Ordering::Acquire));
        let total: u64 = bins.iter().sum();
        if total == 0 {
            return 0;
        }
        let rank = total - total / 100;
        let mut cumulative = 0;
        for (index, count) in bins.into_iter().enumerate() {
            cumulative += count;
            if cumulative >= rank {
                return if index == BINS - 1 {
                    self.maximum.load(Ordering::Relaxed)
                } else {
                    ((index + 1) as u64 * WIDTH_NS).min(self.maximum.load(Ordering::Relaxed))
                };
            }
        }
        unreachable!("rank is within the snapshot sample count")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p99_bounds_exact_rank_including_boundaries_and_overflow() {
        let histogram = Histogram::default();
        assert_eq!(histogram.p99_upper_bound_ns(), 0);
        let mut values = vec![0, 1, 24_999, 25_000, 25_001, 3_175_000, 3_175_001];
        values.extend((1..=1000).map(|value| value * 1234));
        values.extend([u64::MAX, 8_000_000]);
        for &value in &values {
            histogram.record(value);
        }
        values.sort_unstable();
        let exact = values[values.len() - values.len() / 100 - 1];
        let bound = histogram.p99_upper_bound_ns();
        assert!(bound >= exact);
        assert!(bound - exact < WIDTH_NS);

        let overflow = Histogram::default();
        overflow.record(3_175_001);
        overflow.record(u64::MAX);
        assert_eq!(overflow.p99_upper_bound_ns(), u64::MAX);
    }
}
