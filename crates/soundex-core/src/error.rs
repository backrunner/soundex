//! Error types for soundex-core.

/// Primary error type for SoundEx operations.
#[derive(Debug, thiserror::Error)]
pub enum SoundExError {
    /// Failed to load the ONNX model.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// The ONNX graph or metadata does not match the SoundEx runtime contract.
    #[error("model contract mismatch: {0}")]
    ModelContract(String),

    /// Inference execution failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// A DSP stage rejected inconsistent spectral data.
    #[error("DSP processing failed: {0}")]
    Dsp(String),

    /// More samples were supplied than the current causal frame can accept.
    #[error("stream buffer overflow: capacity={capacity}, attempted={attempted}")]
    StreamOverflow { capacity: usize, attempted: usize },

    /// The stream was finalized and must be reset before more input is accepted.
    #[error("stream is finalized; call reset before processing more input")]
    StreamFinalized,

    /// A prior processing failure may have partially advanced internal state.
    #[error("processor is poisoned by a prior processing failure; call reset")]
    ProcessorPoisoned,

    /// Fixed-hop and arbitrary-chunk APIs cannot be mixed without a reset.
    #[error("cannot mix fixed-hop and arbitrary-chunk streaming APIs; call reset")]
    StreamModeConflict,

    /// Invalid input provided by caller.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Model file not found.
    #[error("model file not found: {0}")]
    ModelNotFound(String),
}

/// Convenience result type for SoundEx operations.
pub type Result<T> = std::result::Result<T, SoundExError>;
