//! Sequential golden inference diagnostic for explicit-state artifacts.
use ndarray::Array4;
use soundex_core::{inference::InferenceEngine, HighBandGain, SoundExConfig};
use std::{error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: soundex-state-parity MODEL INPUT.json OUTPUT.json".into());
    }
    let input: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let channels = input["channels"]
        .as_u64()
        .filter(|c| (1..=2).contains(c))
        .ok_or("invalid channel count")? as usize;
    let rate = input["sample_rate"]
        .as_u64()
        .filter(|r| [44100, 48000].contains(r))
        .ok_or("invalid rate")? as u32;
    let mut config = SoundExConfig::with_model(&args[1])
        .sample_rate(rate)
        .channels(channels as u16);
    config.high_band_gain = HighBandGain::Model;
    let mut engine = InferenceEngine::load(Path::new(&args[1]), &config)?;
    if !engine.is_stateful() {
        return Err("expected explicit-state model".into());
    }
    let mut results = Vec::new();
    let resets = input["resets"].as_array().ok_or("missing resets")?;
    for (index, frame) in input["frames"]
        .as_array()
        .ok_or("missing frames")?
        .iter()
        .enumerate()
    {
        if resets.iter().any(|x| x.as_u64() == Some(index as u64)) {
            engine.reset_state();
        }
        let values: Vec<f32> = frame
            .as_array()
            .ok_or("invalid frame")?
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|v| v as f32)
                    .filter(|v| v.is_finite())
                    .ok_or("invalid feature")
            })
            .collect::<Result<_, _>>()?;
        let frame = Array4::from_shape_vec((channels, 2, 1, 129), values)?;
        results.push(engine.infer(&frame)?.into_raw_vec_and_offset().0);
    }
    fs::write(
        &args[3],
        serde_json::to_vec(&serde_json::json!({"frames":results,"runs":engine.run_count()}))?,
    )?;
    Ok(())
}
