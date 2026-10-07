//! Contract edge-case regressions.

use super::*;

fn valid_metadata() -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    for &key in REQUIRED_METADATA {
        metadata.insert(key.to_owned(), "a".repeat(64));
    }
    for (key, value) in [
        ("soundex.artifact_schema", "1.2"),
        ("soundex.model_architecture", "soundex-generator"),
        ("soundex.model_architecture_version", "1.0"),
        ("soundex.input_name", INPUT_NAME),
        ("soundex.output_name", OUTPUT_NAME),
        ("soundex.tensor_layout", "BCTF"),
        ("soundex.input_shape", "batch,2,1,513"),
        ("soundex.output_shape", "batch,2,1,513"),
        ("soundex.input_channels", "log_magnitude_db,phase_radians"),
        ("soundex.output_channels", "log_magnitude_db,phase_radians"),
        ("soundex.sample_rates", "44100,48000"),
        ("soundex.fft_size", "1024"),
        ("soundex.hop_size", "512"),
        ("soundex.crossover_width_hz", "1000.0"),
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
        metadata.insert(key.to_owned(), value.to_owned());
    }
    metadata
}

#[test]
fn tensor_contract_rejects_wrong_channel_time_and_frequency() {
    for shape in [
        [1, 2, 1, 1025],
        [2, 2, 1, 1025],
        [-1, 1, 1, 1025],
        [-1, 2, 2, 1025],
        [-1, 2, 1, 257],
    ] {
        assert!(matches!(
            validate_dimensions(&shape, "input", 513),
            Err(SoundExError::ModelContract(_))
        ));
    }
    validate_dimensions(&[-1, 2, 1, 513], "input", 513).unwrap();
}

#[test]
fn metadata_contract_rejects_semantic_and_config_mismatches() {
    let config = SoundExConfig::default().fft_size(1024).hop_size(512);
    validate_metadata(&valid_metadata(), &config).unwrap();

    for (key, value) in [
        ("soundex.artifact_schema", "1.0"),
        ("soundex.input_channels", "phase_radians,log_magnitude_db"),
        ("soundex.input_shape", "batch,2,2,513"),
        ("soundex.output_shape", "batch,2,1,1025"),
        ("soundex.crossover_width_hz", "2000.0"),
    ] {
        let mut metadata = valid_metadata();
        metadata.insert(key.to_owned(), value.to_owned());
        assert!(matches!(
            validate_metadata(&metadata, &config),
            Err(SoundExError::ModelContract(_))
        ));
    }

    let unsupported_rate = SoundExConfig::default()
        .fft_size(1024)
        .hop_size(512)
        .sample_rate(32_000);
    assert!(matches!(
        validate_metadata(&valid_metadata(), &unsupported_rate),
        Err(SoundExError::ModelContract(_))
    ));

    let mut legacy = valid_metadata();
    legacy.insert("soundex.artifact_schema".into(), "1.1".into());
    legacy.insert("soundex.fft_size".into(), "2048".into());
    legacy.insert("soundex.input_shape".into(), "batch,2,1,1025".into());
    legacy.insert("soundex.output_shape".into(), "batch,2,1,1025".into());
    let legacy_config = SoundExConfig::default()
        .fft_size(1024)
        .hop_size(512)
        .fft_size(2048);
    assert_eq!(
        validate_metadata(&legacy, &legacy_config)
            .unwrap()
            .frequency_bins,
        1025
    );

    legacy.insert("soundex.fft_size".into(), "1024".into());
    legacy.insert("soundex.input_shape".into(), "batch,2,1,513".into());
    legacy.insert("soundex.output_shape".into(), "batch,2,1,513".into());
    assert!(matches!(
        validate_metadata(&legacy, &config),
        Err(SoundExError::ModelContract(_))
    ));
}
