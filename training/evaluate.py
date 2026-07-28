"""Run held-out SoundEx evaluation through the deployed Rust stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data.protocol import validate_data_config
from evaluation.gates import load_gate_config
from evaluation.runner import run_manifest_evaluation
from evaluation.visqol import ExternalVisqolAudio
from export_onnx import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Schema-1 checkpoint or exported ONNX artifact")
    parser.add_argument("manifests", nargs="+", type=Path, help="Held-out dataset manifests")
    parser.add_argument("--output-directory", type=Path, default=Path("evaluation-output"))
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path(__file__).parent / "evaluation" / "release_gates.v1.yaml",
    )
    parser.add_argument("--rust-binary", type=Path)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Resolved training profile used to validate recipe and held-out coverage",
    )
    parser.add_argument(
        "--visqol-command",
        help="External command template with {reference} and {degraded} placeholders",
    )
    parser.add_argument("--visqol-sample-rates", default="48000")
    parser.add_argument("--parity-evidence", type=Path)
    parser.add_argument("--listening-record", type=Path)
    parser.add_argument("--audit-files", action="store_true")
    parser.add_argument("--allow-failed-gates", action="store_true")
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    with args.config.open(encoding="utf-8") as handle:
        resolved_config = yaml.safe_load(handle)
    if not isinstance(resolved_config, dict):
        raise ValueError("training config must be a YAML mapping")
    recipe = validate_data_config(resolved_config.get("data"))
    model_path = args.model
    if model_path.suffix.lower() in {".pth", ".pt"}:
        model_path = args.output_directory / "evaluated-model.onnx"
        export_onnx(
            args.model,
            model_path,
            expected_config_path=args.config,
        )
    elif model_path.suffix.lower() != ".onnx":
        raise ValueError("model must be a schema-1.2 checkpoint or artifact-schema-1.2 ONNX file")

    supported_rates = tuple(
        int(item.strip()) for item in args.visqol_sample_rates.split(",") if item.strip()
    )
    visqol = (
        ExternalVisqolAudio(
            args.visqol_command,
            supported_sample_rates=supported_rates,
        )
        if args.visqol_command
        else None
    )
    report = run_manifest_evaluation(
        model_path=model_path,
        manifests=args.manifests,
        output_directory=args.output_directory,
        gate_config=load_gate_config(args.gate_config),
        recipe=recipe,
        rust_binary=args.rust_binary,
        visqol=visqol,
        visqol_supported_rates=supported_rates,
        parity_evidence_path=args.parity_evidence,
        listening_record_path=args.listening_record,
        audit_files=args.audit_files,
    )
    gates = report["release_gates"]
    print(f"JSON report: {args.output_directory / 'evaluation-report.json'}")
    print(f"Markdown report: {args.output_directory / 'evaluation-report.md'}")
    if not gates["passed"]:
        print(f"Failed release gates: {', '.join(gates['failed'])}")
        if not args.allow_failed_gates:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
