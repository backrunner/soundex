//! Process strict raw audio fixtures through the production SoundEx stream.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use soundex_core::{analyze_buffer, SoundExConfig, SoundExProcessor};

mod audio_io;
use audio_io::{read_audio, write_audio, BinaryAudio};

enum ProcessingMode {
    Offline,
    Chunked(Vec<usize>),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 6 && arguments.len() != 7 {
        return Err(
            "usage: soundex-stream-eval MODEL.onnx INPUT.sxa OUTPUT.sxa REPORT.json \
             offline|chunked [CHUNK_FRAMES_CSV]"
                .into(),
        );
    }
    let model_path = PathBuf::from(&arguments[1]);
    let input_path = PathBuf::from(&arguments[2]);
    let output_path = PathBuf::from(&arguments[3]);
    let report_path = PathBuf::from(&arguments[4]);
    let mode_name = arguments[5]
        .to_str()
        .ok_or("processing mode must be valid UTF-8")?;
    let mode = match (mode_name, arguments.get(6)) {
        ("offline", None) => ProcessingMode::Offline,
        ("chunked", Some(raw)) => ProcessingMode::Chunked(parse_chunk_frames(
            raw.to_str().ok_or("chunk pattern must be valid UTF-8")?,
        )?),
        ("offline", Some(_)) => return Err("offline mode does not accept a chunk pattern".into()),
        ("chunked", None) => return Err("chunked mode requires CHUNK_FRAMES_CSV".into()),
        _ => return Err("processing mode must be 'offline' or 'chunked'".into()),
    };

    let audio = read_audio(&input_path)?;
    let config = SoundExConfig::with_model(model_path)
        .sample_rate(audio.sample_rate)
        .channels(audio.channels)
        .fft_size(env_value("SOUNDEX_EVAL_FFT_SIZE", 256_usize)?)
        .hop_size(env_value("SOUNDEX_EVAL_HOP_SIZE", 128_usize)?);
    let mut config = config;
    config.crossover_width_hz = env_value("SOUNDEX_EVAL_CROSSOVER_WIDTH_HZ", 1000.0_f32)?;
    config.enhancement_mode = std::env::var("SOUNDEX_EVAL_ENHANCEMENT_MODE")
        .unwrap_or_else(|_| "neural".into())
        .parse()
        .map_err(|error: &str| error.to_string())?;
    config.limiter_mode = std::env::var("SOUNDEX_EVAL_LIMITER_MODE")
        .unwrap_or_else(|_| "legacy".into())
        .parse()
        .map_err(|error: &str| error.to_string())?;
    config.high_band_gain = std::env::var("SOUNDEX_EVAL_HIGH_BAND_GAIN")
        .unwrap_or_else(|_| "match-edge".into())
        .parse()
        .map_err(|error: &str| error.to_string())?;
    if let Some(path) = std::env::var_os("SOUNDEX_EVAL_GAIN_CALIBRATION") {
        config.gain_calibration = Some(soundex_core::calibration::read_gain_calibration(path)?);
    }
    config.phase_policy = std::env::var("SOUNDEX_EVAL_PHASE_POLICY")
        .unwrap_or_else(|_| "model".into())
        .parse()
        .map_err(|error: &str| error.to_string())?;
    let phase_policy = config.phase_policy.as_str();
    let limiter_mode = config.limiter_mode.as_str();
    let enhancement_mode = config.enhancement_mode.as_str();
    let high_band_gain = config.high_band_gain.as_str();
    let analysis = analyze_buffer(&config, &audio.samples)?;
    let gain_calibration = config
        .gain_calibration
        .as_ref()
        .map(|c| c.coefficients().to_vec());
    let mut processor = SoundExProcessor::new(config)?;
    let latency = processor.latency_samples_per_channel();
    let output = match &mode {
        ProcessingMode::Offline => processor.process_buffer(&audio.samples)?,
        ProcessingMode::Chunked(pattern) => process_chunked(&mut processor, &audio, pattern)?,
    };
    if output.len() != audio.samples.len() {
        return Err(format!(
            "processor returned {} samples for {} input samples",
            output.len(),
            audio.samples.len()
        )
        .into());
    }
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("processor output contains non-finite samples".into());
    }
    let clipped_samples = output.iter().filter(|sample| sample.abs() > 1.0).count();
    write_audio(
        &output_path,
        &BinaryAudio {
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            frames: audio.frames,
            samples: output,
        },
    )?;
    let pattern = match &mode {
        ProcessingMode::Offline => Vec::new(),
        ProcessingMode::Chunked(pattern) => pattern.clone(),
    };
    let report = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"mode\": \"{}\",\n",
            "  \"enhancement_mode\": \"{}\",\n",
            "  \"high_band_gain\": \"{}\",\n",
            "  \"sample_rate\": {},\n",
            "  \"channels\": {},\n",
            "  \"frames\": {},\n",
            "  \"chunk_frames\": [{}],\n",
            "  \"analysis\": {{\n",
            "    \"frames_analyzed\": {},\n",
            "    \"frames_needing_enhancement\": {},\n",
            "    \"enhancement_ratio\": {:.9},\n",
            "    \"detected_bandwidth_hz\": {:.9}\n",
            "  }},\n",
            "  \"processor\": {{\n",
            "    \"latency_samples_per_channel\": {},\n",
            "    \"output_finite\": true,\n",
            "    \"clipped_samples\": {}\n",
            "  }}\n",
            "}}\n"
        ),
        mode_name,
        enhancement_mode,
        high_band_gain,
        audio.sample_rate,
        audio.channels,
        audio.frames,
        pattern
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        analysis.frames_analyzed,
        analysis.frames_needing_enhancement,
        analysis.enhancement_ratio(),
        analysis.detected_bandwidth_hz,
        latency,
        clipped_samples,
    );
    let mut report: serde_json::Value = serde_json::from_str(&report)?;
    report["limiter_mode"] = serde_json::json!(limiter_mode);
    report["phase_policy"] = serde_json::json!(phase_policy);
    report["gain_calibration_coefficients"] = serde_json::json!(gain_calibration);
    fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn process_chunked(
    processor: &mut SoundExProcessor,
    audio: &BinaryAudio,
    pattern: &[usize],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let channels = usize::from(audio.channels);
    let mut output = Vec::with_capacity(audio.samples.len());
    let mut frame_offset = 0;
    let mut pattern_index = 0;
    while frame_offset < audio.frames {
        let chunk_frames = pattern[pattern_index % pattern.len()].min(audio.frames - frame_offset);
        let start = frame_offset * channels;
        let end = (frame_offset + chunk_frames) * channels;
        let progress = processor.process_chunk(&audio.samples[start..end], &mut output)?;
        if progress.consumed != end - start {
            return Err("processor did not consume the complete chunk".into());
        }
        frame_offset += chunk_frames;
        pattern_index += 1;
    }
    processor.finalize(&mut output)?;
    Ok(output)
}

fn parse_chunk_frames(raw: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let chunks: Vec<usize> = raw
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("invalid chunk frame count {item:?}"))
        })
        .collect::<Result<_, _>>()?;
    if chunks.is_empty() || chunks.contains(&0) {
        return Err("chunk frame counts must be non-empty positive integers".into());
    }
    Ok(chunks)
}

fn env_value<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match std::env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|error| format!("invalid {name}={value:?}: {error}").into()),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pattern_is_strict() {
        assert_eq!(parse_chunk_frames("1,257,509").unwrap(), [1, 257, 509]);
        assert!(parse_chunk_frames("").is_err());
        assert!(parse_chunk_frames("0,1").is_err());
        assert!(parse_chunk_frames("1,nope").is_err());
    }
}
