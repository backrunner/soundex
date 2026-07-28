//! Audio file decoding via Symphonia.

use std::path::Path;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio information.
pub struct DecodedAudio {
    /// Interleaved f32 samples (mono or stereo).
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

/// Decode an audio file to interleaved f32 PCM samples.
///
/// Supports: MP3, AAC, FLAC, WAV, OGG Vorbis.
pub fn decode_file(path: &Path) -> Result<DecodedAudio> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open: {}", path.display()))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Probe format
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let probe_result = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe audio format")?;

    let mut format = probe_result.format;

    // Select the first audio track
    let track = format.default_track().context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    // Create decoder
    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    // Decode all packets
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(anyhow::anyhow!("Decode error: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let duration = decoded.capacity() as u64;
                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(sample_buf.samples());
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip malformed packets
                continue;
            }
            Err(e) => return Err(anyhow::anyhow!("Decode error: {e}")),
        }
    }

    anyhow::ensure!(
        !all_samples.is_empty(),
        "Audio file contains no decodable samples"
    );
    anyhow::ensure!(
        all_samples.len() % channels as usize == 0,
        "Decoded sample count is not aligned to the channel count"
    );

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate,
        channels,
    })
}
