//! Compare a SoundEx ONNX model with a golden tensor file.

use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ndarray::Array4;
use soundex_core::inference::InferenceEngine;
use soundex_core::SoundExConfig;

const MAGIC: &[u8; 4] = b"SXT1";
const HEADER_BYTES: usize = 20;
const TOLERANCE: f32 = 1e-5;

struct BinaryTensor {
    shape: [usize; 4],
    values: Vec<f32>,
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
    if arguments.len() != 4 {
        return Err("usage: soundex-model-parity MODEL.onnx INPUT.sxt EXPECTED.sxt".into());
    }
    let model_path = PathBuf::from(&arguments[1]);
    let input = read_tensor(Path::new(&arguments[2]))?;
    let expected = read_tensor(Path::new(&arguments[3]))?;
    if input.shape != expected.shape {
        return Err(format!(
            "input shape {:?} does not match expected shape {:?}",
            input.shape, expected.shape
        )
        .into());
    }

    let input_array = Array4::from_shape_vec(
        (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        ),
        input.values,
    )?;
    if input.shape[1] != 2 || input.shape[2] != 1 || input.shape[3] < 2 {
        return Err(format!(
            "input shape {:?} is not a SoundEx [B, 2, 1, F] tensor",
            input.shape
        )
        .into());
    }
    let fft_size = input.shape[3]
        .checked_sub(1)
        .and_then(|bins| bins.checked_mul(2))
        .ok_or("input frequency dimension cannot be converted to an FFT size")?;
    let config = SoundExConfig::with_model(&model_path).fft_size(fft_size);
    let mut engine = InferenceEngine::load(&model_path, &config)?;
    let actual = engine.infer(&input_array)?;
    if actual.shape() != expected.shape {
        return Err(format!(
            "runtime output shape {:?} does not match golden {:?}",
            actual.shape(),
            expected.shape
        )
        .into());
    }

    let mut maximum = 0.0_f32;
    let mut total = 0.0_f64;
    for (&actual_value, &expected_value) in actual.iter().zip(&expected.values) {
        if !actual_value.is_finite() || !expected_value.is_finite() {
            return Err("runtime or golden tensor contains a non-finite value".into());
        }
        let difference = (actual_value - expected_value).abs();
        maximum = maximum.max(difference);
        total += f64::from(difference);
    }
    let mean = total / expected.values.len() as f64;
    if maximum >= TOLERANCE || mean >= f64::from(TOLERANCE) {
        return Err(format!(
            "parity mismatch: max={maximum:.8e}, mean={mean:.8e}, tolerance={TOLERANCE:.1e}"
        )
        .into());
    }
    println!(
        "parity passed: elements={}, max={maximum:.8e}, mean={mean:.8e}",
        expected.values.len()
    );
    Ok(())
}

fn read_tensor(path: &Path) -> Result<BinaryTensor, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_BYTES || &bytes[..4] != MAGIC {
        return Err(invalid_data(path, "missing SXT1 header").into());
    }
    let mut shape = [0_usize; 4];
    for (index, dimension) in shape.iter_mut().enumerate() {
        let start = 4 + index * 4;
        let raw: [u8; 4] = bytes[start..start + 4]
            .try_into()
            .map_err(|_| invalid_data(path, "truncated shape"))?;
        *dimension = u32::from_le_bytes(raw) as usize;
    }
    let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
        product.checked_mul(dimension)
    });
    let elements = elements.ok_or_else(|| invalid_data(path, "shape overflows usize"))?;
    let payload_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| invalid_data(path, "payload size overflows usize"))?;
    if bytes.len() != HEADER_BYTES + payload_bytes {
        return Err(invalid_data(path, "payload length does not match shape").into());
    }
    let values = bytes[HEADER_BYTES..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    Ok(BinaryTensor { shape, values })
}

fn invalid_data(path: &Path, message: &str) -> IoError {
    IoError::new(
        ErrorKind::InvalidData,
        format!("{}: {message}", path.display()),
    )
}
