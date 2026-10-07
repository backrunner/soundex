//! Benchmark reports retain exact traces and earlier completed cases on errors.

#[path = "../benches/realtime_bench/report.rs"]
mod report;

use serde_json::{json, Value};
use std::fs;

#[test]
fn streaming_report_preserves_raw_values_and_cleans_temporary_files() {
    let root = std::env::temp_dir().join(format!("soundex-report-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let target = root.join("report.json");
    let header = json!({"schema_version": 2, "model_path": "quotes\" and\nnewline"});
    let case = report::Case {
        metadata: json!({"sample_rate_hz": 48000, "callback": {"p99_us": 1.0}}),
        callback_samples_ns: vec![1, u64::MAX],
        worker_presentation_missed: vec![false, true],
    };
    let mut reports = report::Report::new(target.clone());
    for count in 1..=2 {
        reports.append(&header, &case).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(&target).unwrap()).unwrap();
        assert_eq!(saved["cases"].as_array().unwrap().len(), count);
        assert_eq!(saved["model_path"], header["model_path"]);
        assert_eq!(
            saved["cases"][count - 1],
            serde_json::to_value(&case).unwrap()
        );
    }
    let previous = fs::read(&target).unwrap();
    assert!(reports.append(&json!({"cases": []}), &case).is_err());
    assert!(reports.append(&json!({}), &case).is_err());
    assert_eq!(fs::read(&target).unwrap(), previous);
    drop(reports);
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}
