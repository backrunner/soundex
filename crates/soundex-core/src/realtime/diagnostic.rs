//! Opt-in experimental processor, using the same callback/worker transport.

use std::path::Path;

use super::{worker::Processor, RealtimeProcessor};
use crate::{diagnostics::context_head::ContextHeadStream, Result, SoundExConfig, SoundExError};

impl Processor for ContextHeadStream {
    fn warm_up_inference(&mut self) -> Result<u64> {
        self.prewarm()
            .map_err(|e| SoundExError::ModelLoad(e.to_string()))
    }
    fn reset(&mut self) {
        ContextHeadStream::reset(self);
    }
    fn state_history_warming(&self) -> bool {
        ContextHeadStream::state_history_warming(self)
    }
    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.step(input, output)
            .map(|_| ())
            .map_err(|e| SoundExError::Inference(e.to_string()))
    }
}

impl RealtimeProcessor {
    pub(crate) fn context_diagnostic(path: &Path, mut config: SoundExConfig) -> Result<Self> {
        if config.fft_size != 256
            || config.hop_size != 128
            || !matches!(config.sample_rate, 44100 | 48000)
        {
            return Err(SoundExError::InvalidInput(
                "diagnostic worker requires FFT256/hop128 at 44100 or 48000 Hz".into(),
            ));
        }
        config.high_band_gain = crate::HighBandGain::Model;
        config.limiter_mode = crate::LimiterMode::SmoothKnee;
        let stream = ContextHeadStream::new(path, config.clone())
            .map_err(|e| SoundExError::ModelLoad(e.to_string()))?;
        Self::from_worker(config, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::super::transport::{queues, Packet};
    use super::*;
    use std::sync::{atomic::Ordering, Arc};
    use std::time::{Duration, Instant};

    fn fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/context-worker-probe.onnx")
    }

    fn wait_until(mut done: impl FnMut() -> bool) {
        let until = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(
                Instant::now() < until,
                "worker handshake timed out; not a hop-latency test"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn callbacks_preserve_context_stream_timeline_and_silent_channel() {
        for rate in [44100, 48000] {
            for channels in [1, 2] {
                let config = SoundExConfig::default()
                    .sample_rate(rate)
                    .channels(channels);
                let mut direct = ContextHeadStream::new(&fixture(), config.clone()).unwrap();
                let mut callback =
                    RealtimeProcessor::context_diagnostic(&fixture(), config).unwrap();
                assert_eq!(callback.latency_samples_per_channel(), 256);
                assert_eq!(callback.worker_stats().warmup_runs, 1);
                let mut previous = vec![0.0; 128 * channels as usize];
                for hop in 0..120 {
                    let input: Vec<_> = (0..128)
                        .flat_map(|n| {
                            let value = 0.15 * (((hop * 128 + n) as f32) * 0.13).sin();
                            if channels == 1 {
                                vec![value]
                            } else {
                                vec![0.0, value]
                            }
                        })
                        .collect();
                    let mut expected = vec![0.0; input.len()];
                    let mut actual = vec![0.0; input.len()];
                    direct.step(&input, &mut expected).unwrap();
                    callback.process(&input, &mut actual).unwrap();
                    wait_until(|| callback.worker_stats().processed_hops > hop as u64);
                    // Handshaked functional sequence, deliberately not CPU/device qualification.
                    assert!(actual
                        .iter()
                        .zip(&previous)
                        .all(|(a, b)| (a - b).abs() < 2e-6));
                    if channels == 2 {
                        assert!(actual.iter().step_by(2).all(|x| *x == 0.0));
                    }
                    previous = expected;
                }
                assert_eq!(callback.worker_stats().state_history_warmup_hops, 80);
                assert!(!callback.stats().worker_failed);
                callback.shutdown();
            }
        }
    }

    #[test]
    fn actual_worker_gap_resets_experimental_history() {
        let config = SoundExConfig::default().channels(1);
        let stream = ContextHeadStream::new(&fixture(), config.clone()).unwrap();
        let mut fresh = ContextHeadStream::new(&fixture(), config).unwrap();
        let (mut audio, worker_queues) = queues();
        let control = Arc::clone(&audio.control);
        let handle = super::super::worker::spawn(
            stream,
            worker_queues,
            Arc::clone(&control),
            128,
            48000,
            false,
            true,
        )
        .unwrap();
        for sequence in (0..85).chain([100]) {
            let mut packet = Packet {
                sequence,
                ..Packet::default()
            };
            for (n, value) in packet.samples[..128].iter_mut().enumerate() {
                *value = 0.2 * ((sequence * 128 + n as u64) as f32 * 0.11).sin();
            }
            audio.input.push(packet).unwrap();
            let mut result = None;
            wait_until(|| {
                result = audio.output.pop().ok();
                result.is_some()
            });
            let result = result.unwrap();
            if sequence == 100 {
                let mut expected = [0.0; 128];
                fresh.step(&packet.samples[..128], &mut expected).unwrap();
                assert_eq!(&result.samples[..128], &expected);
                assert!(!result.valid);
            }
        }
        assert_eq!(control.state_warmup.load(Ordering::Relaxed), 81);
        control.stop.store(true, Ordering::Release);
        handle.join().unwrap();
    }

    #[test]
    fn experimental_graph_stays_rejected_by_production_loading() {
        assert!(RealtimeProcessor::new(SoundExConfig::with_model(fixture())).is_err());
        assert!(RealtimeProcessor::context_diagnostic(
            &fixture(),
            SoundExConfig::default().sample_rate(96000)
        )
        .is_err());
    }
}
