//! Optional native audio-priority pacing; this still has no audio device.

pub(super) struct Scheduling {
    #[cfg(target_os = "macos")]
    handle: Option<audio_thread_priority::RtPriorityHandle>,
}

impl Scheduling {
    pub fn new(requested: bool, sample_rate: u32, frames: u32) -> Self {
        #[cfg(target_os = "macos")]
        {
            Self {
                handle: requested
                    .then(|| {
                        audio_thread_priority::promote_current_thread_to_real_time(
                            frames,
                            sample_rate,
                        )
                        .ok()
                    })
                    .flatten(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (requested, sample_rate, frames);
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
