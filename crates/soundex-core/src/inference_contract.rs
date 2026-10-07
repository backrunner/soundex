//! Validation for the versioned SoundEx ONNX tensor and metadata contract.

use std::collections::HashMap;

use ort::value::{TensorElementType, ValueType};

use crate::error::{Result, SoundExError};
use crate::SoundExConfig;

const INPUT_NAME: &str = "input_features";
const OUTPUT_NAME: &str = "output_features";
const CHANNELS: i64 = 2;
const FRAMES: i64 = 1;
mod numbers;
use numbers::{parse_float, parse_number, parse_numbers};

const LEGACY_FRAME_CONTRACT: (usize, usize) = (2048, 512);
const SUPPORTED_FRAME_CONTRACTS: &[(usize, usize)] =
    &[(256, 128), (512, 256), (1024, 512), LEGACY_FRAME_CONTRACT];
const REQUIRED_METADATA: &[&str] = &[
    "soundex.artifact_schema",
    "soundex.model_architecture",
    "soundex.model_architecture_version",
    "soundex.input_name",
    "soundex.output_name",
    "soundex.tensor_layout",
    "soundex.input_shape",
    "soundex.output_shape",
    "soundex.input_channels",
    "soundex.output_channels",
    "soundex.sample_rates",
    "soundex.fft_size",
    "soundex.hop_size",
    "soundex.crossover_width_hz",
    "soundex.window",
    "soundex.window_periodic",
    "soundex.magnitude_scale",
    "soundex.db_formula",
    "soundex.db_floor",
    "soundex.phase_units",
    "soundex.phase_range",
    "soundex.context_frames",
    "soundex.causal",
    "soundex.stateless",
    "soundex.source_checkpoint_sha256",
    "soundex.resolved_config_sha256",
    "soundex.data_recipe_sha256",
    "soundex.manifest_set_sha256",
];

pub(crate) struct ModelFrameContract {
    pub(crate) frequency_bins: usize,
}

pub(crate) fn validate_session_contract(
    session: &ort::session::Session,
    config: &SoundExConfig,
) -> Result<ModelFrameContract> {
    if session.inputs().len() != 1 || session.outputs().len() != 1 {
        return contract_error(format!(
            "expected one input and one output, got {} inputs and {} outputs",
            session.inputs().len(),
            session.outputs().len()
        ));
    }
    if session.inputs()[0].name() != INPUT_NAME || session.outputs()[0].name() != OUTPUT_NAME {
        return contract_error(format!(
            "expected tensors {INPUT_NAME:?}/{OUTPUT_NAME:?}, got {:?}/{:?}",
            session.inputs()[0].name(),
            session.outputs()[0].name()
        ));
    }
    let model_metadata = session
        .metadata()
        .map_err(|error| SoundExError::ModelLoad(error.to_string()))?;
    let mut metadata = HashMap::with_capacity(REQUIRED_METADATA.len());
    for &key in REQUIRED_METADATA {
        let value = model_metadata
            .custom(key)
            .ok_or_else(|| SoundExError::ModelContract(format!("missing metadata {key}")))?;
        metadata.insert(key.to_owned(), value);
    }
    let frame_contract = validate_metadata(&metadata, config)?;
    validate_tensor_type(
        session.inputs()[0].dtype(),
        "input",
        frame_contract.frequency_bins,
    )?;
    validate_tensor_type(
        session.outputs()[0].dtype(),
        "output",
        frame_contract.frequency_bins,
    )?;
    Ok(frame_contract)
}

fn validate_tensor_type(value_type: &ValueType, label: &str, frequency_bins: usize) -> Result<()> {
    match value_type {
        ValueType::Tensor { ty, shape, .. }
            if *ty == TensorElementType::Float32 && shape.len() == 4 =>
        {
            validate_dimensions(shape, label, frequency_bins)
        }
        _ => contract_error(format!(
            "{label} must be a rank-4 float32 tensor, got {value_type:?}"
        )),
    }
}

fn validate_dimensions(shape: &[i64], label: &str, frequency_bins: usize) -> Result<()> {
    if shape[0] != -1 {
        return contract_error(format!(
            "{label} batch dimension must be dynamic so one artifact supports batch 1 and 2; got {}",
            shape[0]
        ));
    }
    let frequency_bins = i64::try_from(frequency_bins)
        .map_err(|_| SoundExError::ModelContract("frequency-bin count exceeds i64".into()))?;
    let expected = [CHANNELS, FRAMES, frequency_bins];
    if shape[1..] != expected {
        return contract_error(format!(
            "{label} shape must be [B, 2, 1, {frequency_bins}], got {shape:?}"
        ));
    }
    Ok(())
}

fn validate_metadata(
    metadata: &HashMap<String, String>,
    config: &SoundExConfig,
) -> Result<ModelFrameContract> {
    let (_, artifact_minor) = expect_version_at_least(metadata, "soundex.artifact_schema", 1, 1)?;
    expect_value(metadata, "soundex.model_architecture", "soundex-generator")?;
    expect_major(metadata, "soundex.model_architecture_version", 1)?;
    for (key, expected) in [
        ("soundex.input_name", INPUT_NAME),
        ("soundex.output_name", OUTPUT_NAME),
        ("soundex.tensor_layout", "BCTF"),
        ("soundex.input_channels", "log_magnitude_db,phase_radians"),
        ("soundex.output_channels", "log_magnitude_db,phase_radians"),
        ("soundex.window", "hann"),
        ("soundex.window_periodic", "true"),
        ("soundex.magnitude_scale", "decibels"),
        ("soundex.db_formula", "20*log10(max(abs(stft),1e-10))"),
        ("soundex.db_floor", "-200.0"),
        ("soundex.phase_units", "radians"),
        ("soundex.phase_range", "[-pi,pi]"),
        ("soundex.context_frames", "1"),
        ("soundex.causal", "true"),
        ("soundex.stateless", "true"),
    ] {
        expect_value(metadata, key, expected)?;
    }

    let sample_rates = parse_numbers(metadata, "soundex.sample_rates")?;
    if sample_rates != [44_100, 48_000] {
        return contract_error(format!(
            "metadata soundex.sample_rates must be 44100,48000, got {sample_rates:?}"
        ));
    }
    if !sample_rates.contains(&config.sample_rate) {
        return contract_error(format!(
            "sample rate {} is unsupported by model {:?}",
            config.sample_rate, sample_rates
        ));
    }
    let fft_size = usize::try_from(parse_number(metadata, "soundex.fft_size")?)
        .map_err(|_| SoundExError::ModelContract("model FFT size exceeds usize".into()))?;
    let hop_size = usize::try_from(parse_number(metadata, "soundex.hop_size")?)
        .map_err(|_| SoundExError::ModelContract("model hop size exceeds usize".into()))?;
    if !SUPPORTED_FRAME_CONTRACTS.contains(&(fft_size, hop_size)) {
        return contract_error(format!(
            "model FFT/hop contract {fft_size}/{hop_size} is unsupported"
        ));
    }
    if artifact_minor < 3 && matches!((fft_size, hop_size), (256, 128) | (512, 256)) {
        return contract_error(
            "small-window contracts require artifact schema 1.3 or newer".into(),
        );
    }
    if artifact_minor < 2 && (fft_size, hop_size) != LEGACY_FRAME_CONTRACT {
        return contract_error(
            "artifact schema 1.1 is limited to the legacy 2048/512 frame contract".into(),
        );
    }
    let frequency_bins = fft_size / 2 + 1;
    let shape = format!("batch,2,1,{frequency_bins}");
    expect_value(metadata, "soundex.input_shape", &shape)?;
    expect_value(metadata, "soundex.output_shape", &shape)?;
    if fft_size != config.fft_size {
        return contract_error(format!(
            "model FFT size {fft_size} does not match processor FFT size {}",
            config.fft_size
        ));
    }
    if hop_size != config.hop_size {
        return contract_error(format!(
            "model hop size {hop_size} does not match processor hop size {}",
            config.hop_size
        ));
    }
    let crossover_width_hz = parse_float(metadata, "soundex.crossover_width_hz")?;
    if !crossover_width_hz.is_finite()
        || (crossover_width_hz - config.crossover_width_hz).abs() > 1e-3
    {
        return contract_error(format!(
            "model crossover width {crossover_width_hz} does not match processor crossover width {}",
            config.crossover_width_hz
        ));
    }
    for key in [
        "soundex.source_checkpoint_sha256",
        "soundex.resolved_config_sha256",
        "soundex.data_recipe_sha256",
        "soundex.manifest_set_sha256",
    ] {
        let value = &metadata[key];
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return contract_error(format!("metadata {key} is not a lowercase SHA-256"));
        }
    }
    Ok(ModelFrameContract { frequency_bins })
}

fn expect_value(metadata: &HashMap<String, String>, key: &str, expected: &str) -> Result<()> {
    let actual = &metadata[key];
    if actual != expected {
        return contract_error(format!(
            "metadata {key} must be {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn expect_major(metadata: &HashMap<String, String>, key: &str, expected: u32) -> Result<()> {
    let value = &metadata[key];
    let parts: Vec<_> = value.split('.').collect();
    let version = match parts.as_slice() {
        [major, minor] => major.parse::<u32>().ok().zip(minor.parse::<u32>().ok()),
        _ => None,
    };
    if !matches!(version, Some((major, _)) if major == expected) {
        return contract_error(format!(
            "metadata {key} has unsupported major version {value:?}"
        ));
    }
    Ok(())
}

fn expect_version_at_least(
    metadata: &HashMap<String, String>,
    key: &str,
    expected_major: u32,
    minimum_minor: u32,
) -> Result<(u32, u32)> {
    let value = &metadata[key];
    let parts: Vec<_> = value.split('.').collect();
    let version = match parts.as_slice() {
        [major, minor] => major.parse::<u32>().ok().zip(minor.parse::<u32>().ok()),
        _ => None,
    };
    let Some((major, minor)) = version else {
        return contract_error(format!(
            "metadata {key} has unsupported version {value:?}; expected {expected_major}.{minimum_minor} or newer within major {expected_major}"
        ));
    };
    if major != expected_major || minor < minimum_minor {
        return contract_error(format!(
            "metadata {key} has unsupported version {value:?}; expected {expected_major}.{minimum_minor} or newer within major {expected_major}"
        ));
    }
    Ok((major, minor))
}

fn contract_error<T>(message: String) -> Result<T> {
    Err(SoundExError::ModelContract(message))
}

#[cfg(test)]
mod tests;
