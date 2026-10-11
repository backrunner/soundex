use std::{
    error::Error,
    fs,
    io::{Error as IoError, ErrorKind},
    path::Path,
};
const MAGIC: &[u8; 4] = b"SXA1";
const HEADER_BYTES: usize = 20;

pub struct BinaryAudio {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
    pub samples: Vec<f32>,
}

pub fn read_audio(path: &Path) -> Result<BinaryAudio, Box<dyn Error>> {
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
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

#[allow(dead_code)]
pub fn write_audio(path: &Path, audio: &BinaryAudio) -> Result<(), Box<dyn Error>> {
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
