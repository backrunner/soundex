//! Collect calibration pairs using production frame analysis; never a realtime callback.
mod audio_io;
use soundex_core::{calibration::collect_gain_observations, SoundExConfig};
use std::{error::Error, fs};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: soundex-gain-data INPUT.sxa CLEAN.sxa OUTPUT.json CONTEXT_FRAMES".into(),
        );
    }
    let input = audio_io::read_audio(std::path::Path::new(&args[1]))?;
    let clean = audio_io::read_audio(std::path::Path::new(&args[2]))?;
    if input.sample_rate != clean.sample_rate
        || input.channels != clean.channels
        || input.frames != clean.frames
    {
        return Err("unaligned pair".into());
    }
    let config = SoundExConfig::default()
        .sample_rate(input.sample_rate)
        .channels(input.channels);
    let rows =
        collect_gain_observations(&config, &input.samples, &clean.samples, args[4].parse()?)?;
    let values: Vec<_> = rows.iter().map(|r| serde_json::json!({"hop":r.hop,"channel":r.channel,"cutoff_hz":r.cutoff_hz,"active":r.active,"features":r.features,"edge_rms":r.edge_rms,"clean_high_rms":r.clean_high_rms})).collect();
    fs::write(&args[3], serde_json::to_vec(&values)?)?;
    Ok(())
}
