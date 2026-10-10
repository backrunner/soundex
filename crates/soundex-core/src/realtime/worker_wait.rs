//! Predictive, bounded idle waiting; only the worker sleeps or spins.

use std::{
    hint, thread,
    time::{Duration, Instant},
};

const LEAD: Duration = Duration::from_micros(500);
const POLL: Duration = Duration::from_micros(50);

#[derive(Debug, PartialEq, Eq)]
enum Idle {
    Sleep(Duration),
    Spin,
}

pub(super) struct Wait {
    arrival: Option<Instant>,
    period: Duration,
    spin: bool,
    spin_lead: Duration,
    spin_grace: Duration,
}

impl Wait {
    pub fn new(period: Duration, wide_spin: bool) -> Self {
        Self {
            arrival: None,
            period,
            // Native RT preserves its computation quota for inference; only
            // normal scheduling uses the bounded 200us spin window.
            spin: wide_spin,
            spin_lead: Duration::from_micros(if wide_spin { 150 } else { 50 }),
            spin_grace: Duration::from_micros(if wide_spin { 50 } else { 25 }),
        }
    }

    pub fn submitted(&mut self, at: Option<Instant>) {
        self.arrival = at.and_then(|at| at.checked_add(self.period));
    }

    pub fn idle(&self) {
        match self.action(Instant::now()) {
            Idle::Sleep(duration) => thread::sleep(duration),
            Idle::Spin => hint::spin_loop(),
        }
    }

    fn action(&self, now: Instant) -> Idle {
        match self.arrival {
            Some(at) if at > now + LEAD => Idle::Sleep(at - now - LEAD),
            // Each iteration checks the bounded queue and shutdown. An absent
            // input cannot extend this fixed prediction window.
            Some(at) if self.spin && now + self.spin_lead >= at && now <= at + self.spin_grace => {
                Idle::Spin
            }
            Some(at) if now.saturating_duration_since(at) > self.period => {
                // No incoming stream: don't keep polling at 10k wakeups/sec.
                Idle::Sleep(Duration::from_millis(1))
            }
            _ => Idle::Sleep(POLL),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_wait_preserves_quota_and_idle_backoff() {
        let origin = Instant::now();
        let period = Duration::from_micros(2667);
        let mut wait = Wait::new(period, false);
        assert_eq!(wait.action(origin), Idle::Sleep(POLL));
        wait.submitted(Some(origin));
        let at = origin + period;
        assert_eq!(
            wait.action(at - LEAD - Duration::from_micros(1)),
            Idle::Sleep(Duration::from_micros(1))
        );
        for offset in [0, 25, 50, 100, 250, 500] {
            assert_eq!(
                wait.action(at - Duration::from_micros(offset)),
                Idle::Sleep(POLL)
            );
        }
        assert_eq!(
            wait.action(at + period + Duration::from_micros(1)),
            Idle::Sleep(Duration::from_millis(1))
        );
        wait.submitted(None);
        assert_eq!(wait.action(at), Idle::Sleep(POLL));
    }

    #[test]
    fn normal_spin_is_bounded() {
        let origin = Instant::now();
        let period = Duration::from_micros(2667);
        let mut wait = Wait::new(period, true);
        wait.submitted(Some(origin));
        let at = origin + period;
        assert_eq!(
            wait.action(at - Duration::from_micros(151)),
            Idle::Sleep(POLL)
        );
        assert_eq!(wait.action(at - Duration::from_micros(150)), Idle::Spin);
        assert_eq!(wait.action(at + Duration::from_micros(50)), Idle::Spin);
        assert_eq!(
            wait.action(at + Duration::from_micros(51)),
            Idle::Sleep(POLL)
        );
    }
}
