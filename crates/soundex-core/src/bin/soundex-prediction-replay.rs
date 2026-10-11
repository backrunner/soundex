//! Offline training proxy verification, never called by the audio callback.
mod audio_io;
use soundex_core::{diagnostics::replay_predictions, HighBandGain, SoundExConfig};
use std::{error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: soundex-prediction-replay MODEL INPUT.sxa FEATURES.json OUTPUT.sxa".into(),
        );
    }
    let input = audio_io::read_audio(Path::new(&args[2]))?;
    let features: Vec<f32> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let mut config = SoundExConfig::with_model(&args[1])
        .sample_rate(input.sample_rate)
        .channels(input.channels);
    config.high_band_gain = HighBandGain::Model;
    let samples = replay_predictions(config, &input.samples, &features)?;
    audio_io::write_audio(
        Path::new(&args[4]),
        &audio_io::BinaryAudio { samples, ..input },
    )?;
    Ok(())
}
