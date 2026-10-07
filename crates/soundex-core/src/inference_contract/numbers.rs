//! Metadata numeric parsing.

use super::*;

pub(super) fn parse_numbers(metadata: &HashMap<String, String>, key: &str) -> Result<Vec<u32>> {
    metadata[key]
        .split(',')
        .map(|value| {
            value.parse::<u32>().map_err(|_| {
                SoundExError::ModelContract(format!(
                    "metadata {key} contains invalid integer {value:?}"
                ))
            })
        })
        .collect()
}

pub(super) fn parse_number(metadata: &HashMap<String, String>, key: &str) -> Result<u32> {
    metadata[key].parse::<u32>().map_err(|_| {
        SoundExError::ModelContract(format!(
            "metadata {key} contains invalid integer {:?}",
            metadata[key]
        ))
    })
}

pub(super) fn parse_float(metadata: &HashMap<String, String>, key: &str) -> Result<f32> {
    metadata[key].parse::<f32>().map_err(|_| {
        SoundExError::ModelContract(format!(
            "metadata {key} contains invalid float {:?}",
            metadata[key]
        ))
    })
}
