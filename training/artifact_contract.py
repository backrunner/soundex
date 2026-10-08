"""Shared constants and validation for the versioned SoundEx ONNX artifact contract."""

from collections.abc import Mapping

ARTIFACT_SCHEMA_VERSION = "1.3"
OPSET_VERSION = 17
INPUT_NAME = "input_features"
OUTPUT_NAME = "output_features"
DEFAULT_FFT_SIZE = 256
DEFAULT_HOP_SIZE = 128
LEGACY_FRAME_CONTRACT = (2048, 512)
SUPPORTED_FRAME_CONTRACTS = frozenset(
    {(DEFAULT_FFT_SIZE, DEFAULT_HOP_SIZE), (512, 256), (1024, 512), LEGACY_FRAME_CONTRACT}
)
MAX_PARAMETERS = 2_000_000
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

REQUIRED_METADATA_KEYS = frozenset(
    {
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
    }
)


class ExportValidationError(RuntimeError):
    """An exported graph failed the deployable artifact contract."""


def tensor_shape(fft_size: int, hop_size: int) -> tuple[int, int, int]:
    """Return the fixed non-batch tensor shape for a supported frame contract."""
    contract = (int(fft_size), int(hop_size))
    if contract not in SUPPORTED_FRAME_CONTRACTS:
        raise ExportValidationError(
            f"unsupported FFT/hop contract {contract}; expected one of "
            f"{sorted(SUPPORTED_FRAME_CONTRACTS)}"
        )
    return (2, 1, contract[0] // 2 + 1)


def frame_contract_from_metadata(metadata: Mapping[str, str]) -> tuple[int, int]:
    """Validate and return the artifact-bound FFT/hop pair."""
    schema = metadata.get("soundex.artifact_schema", "")
    parts = schema.split(".")
    try:
        major, minor = (int(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise ExportValidationError(
            f"unsupported artifact schema {schema!r}; expected 1.1 or newer within major 1"
        ) from error
    if len(parts) != 2 or major != 1 or minor < 1:
        raise ExportValidationError(
            f"unsupported artifact schema {schema!r}; expected 1.1 or newer within major 1"
        )

    try:
        fft_size = int(metadata["soundex.fft_size"])
        hop_size = int(metadata["soundex.hop_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExportValidationError("artifact FFT/hop metadata is missing or malformed") from error
    static_shape = tensor_shape(fft_size, hop_size)
    if minor < 3 and (fft_size, hop_size) in {(256, 128), (512, 256)}:
        raise ExportValidationError("small-window contracts require artifact schema 1.3 or newer")
    if minor < 2 and (fft_size, hop_size) != LEGACY_FRAME_CONTRACT:
        raise ExportValidationError(
            "artifact schema 1.1 is limited to the legacy 2048/512 frame contract"
        )

    expected_shape = f"batch,{','.join(str(value) for value in static_shape)}"
    for key in ("soundex.input_shape", "soundex.output_shape"):
        if metadata.get(key) != expected_shape:
            raise ExportValidationError(
                f"{key} does not match FFT/hop contract {fft_size}/{hop_size}"
            )
    return fft_size, hop_size
