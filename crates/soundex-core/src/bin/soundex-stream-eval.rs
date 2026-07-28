//! Process strict raw audio fixtures through the production SoundEx stream.

use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use soundex_core::{analyze_buffer, SoundExConfig, SoundExProcessor};

const MAGIC: &[u8; 4] = b"SXA1";
const HEADER_BYTES: usize = 20;

struct BinaryAudio {
    sample_rate: u32,
    channels: u16,
    frames: usize,
    samples: Vec<f32>,
}

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
        .fft_size(env_value("SOUNDEX_EVAL_FFT_SIZE", 1024_usize)?)
        .hop_size(env_value("SOUNDEX_EVAL_HOP_SIZE", 512_usize)?);
    let mut config = config;
    config.crossover_width_hz = env_value("SOUNDEX_EVAL_CROSSOVER_WIDTH_HZ", 1000.0_f32)?;
    let analysis = analyze_buffer(&config, &audio.samples)?;
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
    fs::write(report_path, report)?;
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

fn read_audio(path: &Path) -> Result<BinaryAudio, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_BYTES || &bytes[..4] != MAGIC {
        return Err(invalid_data(path, "missing SXA1 header").into());
    }
    let sample_rate = u32::from_le_bytes(bytes[4..8].try_into()?);
    let channels = u16::from_le_bytes(bytes[8..10].try_into()?);
    let flags = u16::from_le_bytes(bytes[10..12].try_into()?);
    let frame_count = u64::from_le_bytes(bytes[12..20].try_into()?);
    if !matches!(sample_rate, 44_100 | 48_000) {
        return Err(invalid_data(path, "sample rate must be 44100 or 48000").into());
    }
    if !matches!(channels, 1 | 2) {
        return Err(invalid_data(path, "channel count must be 1 or 2").into());
    }
    if flags != 0 {
        return Err(invalid_data(path, "unsupported SXA1 flags").into());
    }
    let frames = usize::try_from(frame_count)
        .map_err(|_| invalid_data(path, "frame count does not fit this platform"))?;
    let sample_count = frames
        .checked_mul(usize::from(channels))
        .ok_or_else(|| invalid_data(path, "sample count overflows usize"))?;
    let payload_bytes = sample_count
        .checked_mul(4)
        .ok_or_else(|| invalid_data(path, "payload size overflows usize"))?;
    if bytes.len() != HEADER_BYTES + payload_bytes {
        return Err(invalid_data(path, "payload length does not match header").into());
    }
    let samples: Vec<f32> = bytes[HEADER_BYTES..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid_data(path, "payload contains non-finite samples").into());
    }
    Ok(BinaryAudio {
        sample_rate,
        channels,
        frames,
        samples,
    })
}

fn write_audio(path: &Path, audio: &BinaryAudio) -> Result<(), Box<dyn Error>> {
    if audio.samples.len() != audio.frames * usize::from(audio.channels) {
        return Err("output sample count does not match audio shape".into());
    }
    let mut bytes = Vec::with_capacity(HEADER_BYTES + audio.samples.len() * 4);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&audio.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&audio.channels.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&(audio.frames as u64).to_le_bytes());
    for sample in &audio.samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn invalid_data(path: &Path, message: &str) -> IoError {
    IoError::new(
        ErrorKind::InvalidData,
        format!("{}: {message}", path.display()),
    )
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
