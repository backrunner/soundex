//! Offline upper-bound diagnostic, never a deployable reference-conditioned model.
mod audio_io;
use soundex_core::{
    diagnostics::{process_oracle, OracleSpectrum},
    HighBandGain, SoundExConfig,
};
use std::{error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: soundex-oracle MODEL INPUT.sxa CLEAN.sxa OUTPUT.sxa REPORT.json observed|magnitude|phase|complex match-edge|model".into());
    }
    let input = audio_io::read_audio(Path::new(&args[2]))?;
    let clean = audio_io::read_audio(Path::new(&args[3]))?;
    if (input.sample_rate, input.channels, input.frames)
        != (clean.sample_rate, clean.channels, clean.frames)
    {
        return Err("unaligned oracle pair".into());
    }
    let mode = match args[6].as_str() {
        "observed" => OracleSpectrum::Observed,
        "magnitude" => OracleSpectrum::CleanMagnitude,
        "phase" => OracleSpectrum::CleanPhase,
        "complex" => OracleSpectrum::CleanComplex,
        _ => return Err("unknown oracle spectrum".into()),
    };
    let mut config = SoundExConfig::with_model(&args[1])
        .sample_rate(input.sample_rate)
        .channels(input.channels);
    config.high_band_gain = args[7].parse::<HighBandGain>()?;
    let result = process_oracle(config, &input.samples, &clean.samples, mode)?;
    let hops: Vec<_> = result.hops.iter().map(|h| serde_json::json!({"bypassed": h.bypassed, "cutoff_hz": h.detected_bandwidth_hz, "gain_db": h.enhancement_gain_db})).collect();
    audio_io::write_audio(
        Path::new(&args[4]),
        &audio_io::BinaryAudio {
            samples: result.audio,
            ..input
        },
    )?;
    fs::write(
        &args[5],
        serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"diagnostic_only":true,"oracle":args[6],"high_band_gain":args[7],"phase_policy":"model","hops":hops}),
        )?,
    )?;
    Ok(())
}
