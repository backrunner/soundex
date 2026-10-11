//! Explicit-state contract, deliberately a new major rather than disguised C9.
use super::*;
use crate::{HighBandGain, PhasePolicy};

pub(super) fn validate(
    session: &ort::session::Session,
    config: &SoundExConfig,
) -> Result<ModelFrameContract> {
    let info = session
        .metadata()
        .map_err(|e| SoundExError::ModelLoad(e.to_string()))?;
    let mut metadata = HashMap::new();
    for &key in REQUIRED_METADATA.iter().chain(
        [
            "soundex.state_size",
            "soundex.state_layout",
            "soundex.state_reset",
            "soundex.temporal_kind",
            "soundex.warmup_frames",
            "soundex.high_band_gain",
            "soundex.phase_policy",
            "soundex.parameter_count",
            "soundex.sample_rate_input_units",
        ]
        .iter(),
    ) {
        metadata.insert(
            key.to_owned(),
            info.custom(key)
                .ok_or_else(|| SoundExError::ModelContract(format!("missing metadata {key}")))?,
        );
    }
    for (key, value) in [
        ("soundex.model_architecture", "soundex-causal-complex"),
        ("soundex.model_architecture_version", "1.0"),
        ("soundex.stateless", "false"),
        ("soundex.context_frames", "1"),
        ("soundex.state_layout", "batch,state"),
        ("soundex.state_reset", "zeros"),
        ("soundex.warmup_frames", "64"),
        ("soundex.high_band_gain", "model"),
        ("soundex.phase_policy", "model"),
        ("soundex.sample_rate_input_units", "Hz"),
        ("soundex.fft_size", "256"),
        ("soundex.hop_size", "128"),
    ] {
        expect_value(&metadata, key, value)?;
    }
    if !matches!(metadata["soundex.temporal_kind"].as_str(), "gru" | "tcn") {
        return contract_error("unknown causal temporal kind".into());
    }
    let size = parse_number(&metadata, "soundex.state_size")? as usize;
    let parameters = parse_number(&metadata, "soundex.parameter_count")?;
    if !(1..=32768).contains(&size) || !(1..=2_000_000).contains(&parameters) {
        return contract_error("state or parameter budget exceeded".into());
    }
    if config.high_band_gain != HighBandGain::Model || config.phase_policy != PhasePolicy::Model {
        return contract_error(
            "causal complex models require model gain and model phase policy".into(),
        );
    }
    // Reuse exact feature-unit, SHA, sample-rate and window validation. This is
    // validation of common fields only; the actual schema/state remains 2.0.
    let mut common = metadata.clone();
    common.insert("soundex.artifact_schema".into(), "1.3".into());
    common.insert(
        "soundex.model_architecture".into(),
        "soundex-generator".into(),
    );
    common.insert("soundex.model_architecture_version".into(), "1.3".into());
    common.insert("soundex.stateless".into(), "true".into());
    let mut contract = validate_metadata(&common, config)?;
    if session.inputs().len() != 3 || session.outputs().len() != 2 {
        return contract_error("stateful model requires three inputs and two outputs".into());
    }
    for (input, name) in session
        .inputs()
        .iter()
        .zip([INPUT_NAME, "input_state", "sample_rate_hz"])
    {
        if input.name() != name {
            return contract_error(format!("expected input {name}"));
        }
    }
    for (output, name) in session.outputs().iter().zip([OUTPUT_NAME, "output_state"]) {
        if output.name() != name {
            return contract_error(format!("expected output {name}"));
        }
    }
    validate_tensor_type(session.inputs()[0].dtype(), "input", 129, 1)?;
    validate_tensor_type(session.outputs()[0].dtype(), "output", 129, 1)?;
    for (value, width) in [
        (session.inputs()[1].dtype(), size),
        (session.outputs()[1].dtype(), size),
        (session.inputs()[2].dtype(), 1),
    ] {
        match value {
            ValueType::Tensor { ty, shape, .. }
                if *ty == TensorElementType::Float32 && shape.as_ref() == [-1, width as i64] => {}
            _ => return contract_error(format!("state/rate tensor must be float32 [B,{width}]")),
        }
    }
    contract.state_size = size;
    contract.warmup_frames = 64;
    Ok(contract)
}
