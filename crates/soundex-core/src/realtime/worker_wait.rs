//! Predictive, bounded idle waiting; only the worker sleeps or spins.

use std::{
    hint, thread,
    time::{Duration, Instant},
};

const LEAD: Duration = Duration::from_micros(150);
const GRACE: Duration = Duration::from_micros(50);
const POLL: Duration = Duration::from_micros(100);

pub(super) struct Wait {
    arrival: Option<Instant>,
    period: Duration,
    spin: bool,
}

impl Wait {
    pub fn new(period: Duration, spin: bool) -> Self {
        Self {
            arrival: None,
            period,
            spin,
        }
    }

    pub fn submitted(&mut self, at: Option<Instant>) {
        self.arrival = at.and_then(|at| at.checked_add(self.period));
    }

    pub fn idle(&self) {
        let now = Instant::now();
        match self.arrival {
            Some(at) if at > now + LEAD => thread::sleep(at - now - LEAD),
            // At most 200us per predicted arrival. Each caller iteration checks
            // both the bounded queue and shutdown; no unbounded busy loop.
            Some(at) if now <= at + GRACE && self.spin => hint::spin_loop(),
            Some(at) if now.saturating_duration_since(at) > self.period => {
                // No incoming stream: don't keep polling at 10k wakeups/sec.
                thread::sleep(Duration::from_millis(1));
            }
            _ => thread::sleep(POLL),
        }
    }
}

pub(super) fn apply_qos() -> bool {
    #[cfg(target_os = "macos")]
    {
        qos_threads::set_current_thread(qos_threads::Qos::High).is_ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Restore priority on the same worker before model teardown, including errors.
pub(super) struct Scheduling {
    #[cfg(target_os = "macos")]
    handle: Option<audio_thread_priority::RtPriorityHandle>,
}

impl Scheduling {
    pub fn new(sample_rate: u32) -> Self {
        #[cfg(target_os = "macos")]
        {
            Self {
                handle: audio_thread_priority::promote_current_thread_to_real_time(
                    super::transport::HOP as u32,
                    sample_rate,
                )
                .ok(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sample_rate;
            Self {}
        }
    }

    pub fn accepted(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.handle.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

impl Drop for Scheduling {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(handle) = self.handle.take() {
            let _ = audio_thread_priority::demote_current_thread_from_real_time(handle);
        }
    }
}
