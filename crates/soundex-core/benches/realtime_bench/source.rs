//! Optional streamed real-music PCM input with exact provenance binding.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

pub(super) struct Input {
    reader: Option<BufReader<File>>,
    evidence: Value,
}

impl Input {
    pub fn new(rate: u32, channels: u16, required_frames: usize) -> Result<Self, Box<dyn Error>> {
        let Some(directory) = env::var_os("SOUNDEX_REALTIME_PCM_DIR") else {
            return Ok(Self {
                reader: None,
                evidence: json!({"type": "synthetic", "generator": "tone-noise-silence-transients-v1"}),
            });
        };
        let path = PathBuf::from(directory)
            .join(format!("{rate}-{channels}.f32"))
            .canonicalize()?;
        let mut file = File::open(&path)?;
        let size = file.metadata()?.len();
        if size % (4 * channels as u64) != 0 {
            return Err("PCM has an incomplete channel frame".into());
        }
        let available_frames = size / (4 * channels as u64);
        if available_frames < required_frames as u64 {
            return Err(format!(
                "PCM has {available_frames} frames; benchmark requires {required_frames} \
                 frames including complete hops and warmup: {}",
                path.display()
            )
            .into());
        }
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        Ok(Self {
            reader: Some(BufReader::new(File::open(&path)?)),
            evidence: json!({"type": "f32le-interleaved", "path": path,
                "size_bytes": size, "sha256": format!("{:x}", hash.finalize())}),
        })
    }

    pub fn read_hop(&mut self, output: &mut [f32]) -> Result<bool, Box<dyn Error>> {
        let Some(reader) = self.reader.as_mut() else {
            return Ok(false);
        };
        let mut bytes = [0_u8; 1024];
        reader.read_exact(&mut bytes[..output.len() * 4])?;
        for (sample, data) in output.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
            *sample = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        }
        Ok(true)
    }

    pub fn evidence(&self) -> &Value {
        &self.evidence
    }
}
