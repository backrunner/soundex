//! Opt-in CPU-clock probes for the standalone replay, never an audio callback.

use std::{
    io,
    time::{Duration, Instant},
};

/// CPU time charged to the calling thread; does not include time asleep or other threads.
pub fn thread_cpu_time() -> io::Result<Duration> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut value = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: value is a valid, aligned writable timespec for the duration of
        // this call. CLOCK_THREAD_CPUTIME_ID names the current thread's CPU clock.
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut value) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if value.tv_sec < 0 || !(0..1_000_000_000).contains(&value.tv_nsec) {
            return Err(io::Error::other("invalid thread CPU clock value"));
        }
        Ok(Duration::new(value.tv_sec as u64, value.tv_nsec as u32))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "thread CPU diagnostic supports macOS/Linux",
        ))
    }
}

/// Matched enclosing wall/CPU interval. Difference also includes blocking and
/// clock/probe error; it is not a direct OS scheduler or preemption event counter.
pub struct Sample {
    wall: Instant,
    cpu: Duration,
}

impl Sample {
    /// Start an enclosing wall/CPU interval; propagate unsupported-clock/OS errors.
    pub fn start() -> io::Result<Self> {
        Ok(Self {
            wall: Instant::now(),
            cpu: thread_cpu_time()?,
        })
    }

    /// Return wall, CPU and signed difference in milliseconds; reject clock reversal.
    pub fn finish(self) -> io::Result<[f64; 3]> {
        let cpu = thread_cpu_time()?
            .checked_sub(self.cpu)
            .ok_or_else(|| io::Error::other("thread CPU clock moved backwards"))?;
        let wall = self.wall.elapsed().as_secs_f64() * 1000.0;
        let cpu = cpu.as_secs_f64() * 1000.0;
        Ok([wall, cpu, wall - cpu])
    }
}

/// Keep all empty probe measurements, without subtracting an estimated overhead.
pub fn calibrate() -> io::Result<Vec<[f64; 3]>> {
    (0..1000).map(|_| Sample::start()?.finish()).collect()
}

#[cfg(all(test, not(miri), any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;

    #[test]
    fn sleeping_does_not_charge_wall_time_to_thread_cpu() {
        let probe = Sample::start().unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let [wall, cpu, gap] = probe.finish().unwrap();
        assert!(
            wall >= 25.0 && cpu >= 0.0 && gap > 10.0,
            "{wall} {cpu} {gap}"
        );
    }

    #[test]
    fn busy_work_advances_cpu_clock_and_probes_are_finite() {
        let probe = Sample::start().unwrap();
        let until = Instant::now() + Duration::from_millis(15);
        while Instant::now() < until {
            std::hint::black_box((0..100).fold(0_u64, |a, b| a.wrapping_add(b)));
        }
        let [wall, cpu, _] = probe.finish().unwrap();
        assert!(wall > 0.0 && cpu > 0.0);
        let calibration = calibrate().unwrap();
        assert_eq!(calibration.len(), 1000);
        assert!(calibration.iter().flatten().all(|v| v.is_finite()));
    }
}
