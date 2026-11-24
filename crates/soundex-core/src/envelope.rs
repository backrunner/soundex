//! Experimental native envelope predictor (P108/P109 causal GRU).
//!
//! Artifact `.sxenv`: magic `SXENV001`, little-endian u32 header length, a JSON
//! header, then little-endian f32 tensors in PyTorch order:
//! `inp.0.weight [H,I]`, `inp.0.bias [H]`, per layer `weight_ih [3H,in]`,
//! `weight_hh [3H,H]`, `bias_ih [3H]`, `bias_hh [3H]`, then `out.weight [O,H]`,
//! `out.bias [O]`. GRU gates are ordered r, z, n as in `torch.nn.GRU`. This is
//! an experimental opt-in format, not the production ONNX artifact contract.

use std::path::Path;

use sha2::{Digest, Sha256};
use soundex_dsp::envelope::{EnvelopeLayout, FillTarget};

use crate::error::{Result, SoundExError};

const MAGIC: &[u8; 8] = b"SXENV001";
const MAX_PARAMETERS: usize = 2_000_000;

fn invalid(message: impl Into<String>) -> SoundExError {
    SoundExError::ModelContract(message.into())
}

/// Loaded, validated envelope predictor weights.
#[derive(Debug)]
pub struct EnvelopeModel {
    input: usize,
    hidden: usize,
    layers: usize,
    output: usize,
    target: FillTarget,
    warmup_frames: usize,
    w_in: Vec<f32>,
    b_in: Vec<f32>,
    w_ih: Vec<Vec<f32>>,
    w_hh: Vec<Vec<f32>>,
    b_ih: Vec<Vec<f32>>,
    b_hh: Vec<Vec<f32>>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    payload_sha256: String,
}

/// Preallocated per-call buffers; one per physical channel.
#[derive(Debug, Clone)]
pub struct EnvelopeScratch {
    hidden_in: Vec<f32>,
    gates_x: Vec<f32>,
    gates_h: Vec<f32>,
}

fn take(values: &mut std::slice::Iter<'_, f32>, count: usize, name: &str) -> Result<Vec<f32>> {
    let out: Vec<f32> = values.by_ref().take(count).copied().collect();
    if out.len() != count {
        return Err(invalid(format!("truncated tensor {name}")));
    }
    Ok(out)
}

impl EnvelopeModel {
    /// Load and validate an `.sxenv` artifact for one FFT/hop and sample rate.
    pub fn load(path: &Path, fft_size: usize, hop_size: usize) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| SoundExError::ModelNotFound(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes, fft_size, hop_size)
    }

    pub fn from_bytes(bytes: &[u8], fft_size: usize, hop_size: usize) -> Result<Self> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(invalid("not an SXENV001 envelope artifact"));
        }
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
        let header_end = 12usize
            .checked_add(header_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| invalid("truncated envelope header"))?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[12..header_end])
            .map_err(|e| invalid(format!("envelope header: {e}")))?;
        let payload = &bytes[header_end..];
        let digest = Sha256::digest(payload);
        let payload_sha256: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        let field = |key: &str| {
            header
                .get(key)
                .ok_or_else(|| invalid(format!("missing header {key}")))
        };
        let number = |key: &str| -> Result<usize> {
            field(key)?
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("header {key} must be an integer")))
        };
        let text = |key: &str| -> Result<String> {
            field(key)?
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("header {key} must be a string")))
        };
        if text("format")? != "soundex-envelope" || number("version")? != 1 {
            return Err(invalid("unsupported envelope format/version"));
        }
        if text("payload_sha256")? != payload_sha256 {
            return Err(invalid("envelope payload SHA-256 mismatch"));
        }
        if number("fft_size")? != fft_size || number("hop_size")? != hop_size {
            return Err(invalid("envelope artifact FFT/hop differs from the stream"));
        }
        if text("activation")? != "gelu-erf" || text("band_layout")? != "half-erb-4k-24k" {
            return Err(invalid("unsupported envelope activation or band layout"));
        }
        let target = match text("target")?.as_str() {
            "deficit" => FillTarget::Deficit,
            "level" => FillTarget::Level,
            _ => return Err(invalid("envelope target must be deficit or level")),
        };
        let (input, hidden, layers, output) = (
            number("input")?,
            number("hidden")?,
            number("layers")?,
            number("output")?,
        );
        if input != fft_size / 2 + 3
            || output != EnvelopeLayout::band_count()
            || !(1..=1024).contains(&hidden)
            || !(1..=4).contains(&layers)
        {
            return Err(invalid(
                "envelope tensor geometry differs from the stream layout",
            ));
        }
        let mut expected = hidden * input + hidden + output * hidden + output;
        // The input projection maps to `hidden`, so every GRU layer has width `hidden`.
        expected += layers * (6 * hidden * hidden + 6 * hidden);
        if expected > MAX_PARAMETERS || payload.len() != expected * 4 {
            return Err(invalid(format!(
                "envelope payload has {} bytes, expected {}",
                payload.len(),
                expected * 4
            )));
        }
        let values: Vec<f32> = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("envelope weights must be finite"));
        }
        let mut it = values.iter();
        let w_in = take(&mut it, hidden * input, "inp.weight")?;
        let b_in = take(&mut it, hidden, "inp.bias")?;
        let (mut w_ih, mut w_hh, mut b_ih, mut b_hh) = (vec![], vec![], vec![], vec![]);
        for _ in 0..layers {
            w_ih.push(take(&mut it, 3 * hidden * hidden, "weight_ih")?);
            w_hh.push(take(&mut it, 3 * hidden * hidden, "weight_hh")?);
            b_ih.push(take(&mut it, 3 * hidden, "bias_ih")?);
            b_hh.push(take(&mut it, 3 * hidden, "bias_hh")?);
        }
        let w_out = take(&mut it, output * hidden, "out.weight")?;
        let b_out = take(&mut it, output, "out.bias")?;
        Ok(Self {
            input,
            hidden,
            layers,
            output,
            target,
            warmup_frames: number("warmup_frames")?,
            w_in,
            b_in,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            w_out,
            b_out,
            payload_sha256,
        })
    }

    pub fn input_len(&self) -> usize {
        self.input
    }
    pub fn output_len(&self) -> usize {
        self.output
    }
    pub fn state_len(&self) -> usize {
        self.hidden * self.layers
    }
    pub fn target(&self) -> FillTarget {
        self.target
    }
    pub fn warmup_frames(&self) -> usize {
        self.warmup_frames
    }
    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn scratch(&self) -> EnvelopeScratch {
        EnvelopeScratch {
            hidden_in: vec![0.0; self.hidden],
            gates_x: vec![0.0; 3 * self.hidden],
            gates_h: vec![0.0; 3 * self.hidden],
        }
    }

    /// One causal frame: `x` -> `out` (log10 energies relative to the loudness
    /// reference), updating `state` ([layer][hidden]) in place. No allocation.
    pub fn step(
        &self,
        x: &[f32],
        state: &mut [f32],
        scratch: &mut EnvelopeScratch,
        out: &mut [f32],
    ) {
        let h = self.hidden;
        debug_assert_eq!(x.len(), self.input);
        debug_assert_eq!(state.len(), self.state_len());
        debug_assert_eq!(out.len(), self.output);
        affine(&self.w_in, &self.b_in, x, &mut scratch.hidden_in);
        for value in &mut scratch.hidden_in {
            *value = gelu(*value);
        }
        for layer in 0..self.layers {
            let hidden = &mut state[layer * h..(layer + 1) * h];
            affine(
                &self.w_ih[layer],
                &self.b_ih[layer],
                &scratch.hidden_in,
                &mut scratch.gates_x,
            );
            affine(
                &self.w_hh[layer],
                &self.b_hh[layer],
                hidden,
                &mut scratch.gates_h,
            );
            let (xr, rest) = scratch.gates_x.split_at(h);
            let (xz, xn) = rest.split_at(h);
            let (hr, rest) = scratch.gates_h.split_at(h);
            let (hz, hn) = rest.split_at(h);
            for (j, value) in hidden.iter_mut().enumerate() {
                let r = sigmoid(xr[j] + hr[j]);
                let z = sigmoid(xz[j] + hz[j]);
                let n = (xn[j] + r * hn[j]).tanh();
                *value = (1.0 - z) * n + z * *value;
            }
            scratch.hidden_in.copy_from_slice(hidden);
        }
        affine(&self.w_out, &self.b_out, &scratch.hidden_in, out);
    }
}

fn affine(weight: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    for ((row, &b), value) in weight.chunks_exact(cols).zip(bias).zip(out.iter_mut()) {
        *value = b + row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>();
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Exact-form GELU 0.5 x (1 + erf(x / sqrt 2)), as `torch.nn.GELU()`.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x as f64 * std::f64::consts::FRAC_1_SQRT_2) as f32)
}

/// erf via the Numerical Recipes erfc Chebyshev fit (|error| < 1.2e-7).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (-2.0, -0.995_322_265),
        ] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x})");
        }
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(EnvelopeModel::from_bytes(b"NOTSXENV0000", 256, 128).is_err());
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&100u32.to_le_bytes());
        assert!(EnvelopeModel::from_bytes(&bytes, 256, 128).is_err());
    }
}
