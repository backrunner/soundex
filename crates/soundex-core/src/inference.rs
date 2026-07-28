//! ONNX model inference engine and deployable artifact validation.
//!
//! SoundEx models exchange fixed-width spectral frames as `[B, 2, 1, F]`, where
//! channel 0 is magnitude in dB and channel 1 is phase in radians.

#[cfg(not(feature = "ort-backend"))]
use ndarray::{Array4, ArrayView4, ArrayViewMut4};

#[cfg(not(feature = "ort-backend"))]
use crate::error::{Result, SoundExError};
#[cfg(not(feature = "ort-backend"))]
use crate::SoundExConfig;

#[cfg(feature = "ort-backend")]
mod inner {
    use std::path::Path;

    use ndarray::{Array4, ArrayView4, ArrayViewMut4};
    use ort::{
        session::{HasSelectedOutputs, OutputSelector, RunOptions},
        value::Tensor,
    };

    use crate::error::{Result, SoundExError};
    use crate::inference_contract::validate_session_contract;
    use crate::SoundExConfig;

    struct InferenceBuffers {
        input: Tensor<f32>,
        run_options: RunOptions<HasSelectedOutputs>,
    }

    impl InferenceBuffers {
        fn new(
            session: &ort::session::Session,
            output_name: &str,
            batch_size: usize,
            frequency_bins: usize,
        ) -> Result<Self> {
            let shape = [batch_size, 2, 1, frequency_bins];
            let input = Tensor::<f32>::new(session.allocator(), shape)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?;
            let output = Tensor::<f32>::new(session.allocator(), shape)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?;
            let run_options = RunOptions::new()
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .with_outputs(OutputSelector::default().preallocate(output_name, output));
            Ok(Self { input, run_options })
        }
    }

    /// ONNX Runtime inference engine for a validated SoundEx model.
    pub struct InferenceEngine {
        session: ort::session::Session,
        output_name: String,
        batch_one: InferenceBuffers,
        batch_two: InferenceBuffers,
        frequency_bins: usize,
        run_count: u64,
        #[cfg(test)]
        fail_after_successes: Option<usize>,
    }

    impl InferenceEngine {
        /// Load an ONNX model and validate its complete SoundEx contract.
        pub fn load(model_path: &Path, config: &SoundExConfig) -> Result<Self> {
            if !model_path.is_file() {
                return Err(SoundExError::ModelNotFound(
                    model_path.display().to_string(),
                ));
            }

            let session = ort::session::Session::builder()
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .with_intra_threads(config.ort_intra_threads)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .with_inter_threads(config.ort_inter_threads)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .with_parallel_execution(config.ort_parallel_execution)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?
                .commit_from_file(model_path)
                .map_err(|error| SoundExError::ModelLoad(error.to_string()))?;
            let frame_contract = validate_session_contract(&session, config)?;

            let output_name = session.outputs()[0].name().to_owned();
            let batch_one =
                InferenceBuffers::new(&session, &output_name, 1, frame_contract.frequency_bins)?;
            let batch_two =
                InferenceBuffers::new(&session, &output_name, 2, frame_contract.frequency_bins)?;

            Ok(Self {
                output_name,
                batch_one,
                batch_two,
                frequency_bins: frame_contract.frequency_bins,
                session,
                run_count: 0,
                #[cfg(test)]
                fail_after_successes: None,
            })
        }

        /// Run inference on one or two artifact-bound `[B, 2, 1, F]` frames.
        pub fn infer(&mut self, input: &Array4<f32>) -> Result<Array4<f32>> {
            let mut output = Array4::zeros(input.raw_dim());
            self.infer_into(input.view(), output.view_mut())?;
            Ok(output)
        }

        /// Run inference from and into caller-owned contiguous tensors.
        pub fn infer_into(
            &mut self,
            input: ArrayView4<'_, f32>,
            mut output: ArrayViewMut4<'_, f32>,
        ) -> Result<()> {
            let expected_shape = input.shape();
            if !matches!(expected_shape[0], 1 | 2)
                || expected_shape[1..] != [2, 1, self.frequency_bins]
            {
                return Err(SoundExError::InvalidInput(format!(
                    "model input must have shape [B, 2, 1, {}] with B in 1..=2, got {expected_shape:?}",
                    self.frequency_bins
                )));
            }
            if output.shape() != expected_shape {
                return Err(SoundExError::InvalidInput(format!(
                    "model output buffer shape {:?} does not match input shape {expected_shape:?}",
                    output.shape()
                )));
            }
            let input_values = input.as_slice().ok_or_else(|| {
                SoundExError::InvalidInput("model input buffer must be contiguous".into())
            })?;
            let output_values = output.as_slice_mut().ok_or_else(|| {
                SoundExError::InvalidInput("model output buffer must be contiguous".into())
            })?;

            #[cfg(test)]
            if let Some(remaining) = self.fail_after_successes.as_mut() {
                if *remaining == 0 {
                    self.fail_after_successes = None;
                    return Err(SoundExError::Inference("injected inference failure".into()));
                }
                *remaining -= 1;
            }

            let buffers = if expected_shape[0] == 1 {
                &mut self.batch_one
            } else {
                &mut self.batch_two
            };
            buffers
                .input
                .extract_tensor_mut()
                .1
                .copy_from_slice(input_values);
            self.run_count += 1;
            let outputs = self
                .session
                .run_with_options(ort::inputs![&buffers.input], &buffers.run_options)
                .map_err(|error| SoundExError::Inference(error.to_string()))?;
            let runtime_output = outputs
                .get(&self.output_name)
                .ok_or_else(|| SoundExError::Inference("model output not found".into()))?;
            let (shape, values) = runtime_output
                .try_extract_tensor::<f32>()
                .map_err(|error| SoundExError::Inference(error.to_string()))?;
            let shape_matches = shape.len() == expected_shape.len()
                && shape
                    .iter()
                    .zip(expected_shape)
                    .all(|(actual, expected)| usize::try_from(*actual) == Ok(*expected));
            if !shape_matches {
                return Err(SoundExError::Inference(format!(
                    "model output shape {shape:?} does not match input shape {expected_shape:?}"
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(SoundExError::Inference(
                    "model output contains non-finite values".into(),
                ));
            }

            output_values.copy_from_slice(values);
            Ok(())
        }

        /// Return the fixed frequency size declared by the SoundEx contract.
        pub fn input_freq_bins(&self) -> Option<usize> {
            Some(self.frequency_bins)
        }

        /// Number of ONNX Runtime session runs issued by this engine.
        pub fn run_count(&self) -> u64 {
            self.run_count
        }

        #[cfg(test)]
        pub(crate) fn fail_after_successes(&mut self, successes: usize) {
            self.fail_after_successes = Some(successes);
        }
    }
}

#[cfg(feature = "ort-backend")]
pub use inner::InferenceEngine;

/// Placeholder engine when the ONNX Runtime backend is disabled.
#[cfg(not(feature = "ort-backend"))]
pub struct InferenceEngine;

#[cfg(not(feature = "ort-backend"))]
impl InferenceEngine {
    /// Return an error because no inference backend is enabled.
    pub fn load(_model_path: &std::path::Path, _config: &SoundExConfig) -> Result<Self> {
        Err(SoundExError::ModelLoad(
            "ort-backend feature not enabled; rebuild with --features ort-backend".into(),
        ))
    }

    /// Return an error because no inference backend is enabled.
    pub fn infer(&mut self, _input: &Array4<f32>) -> Result<Array4<f32>> {
        Err(SoundExError::Inference(
            "ort-backend feature not enabled".into(),
        ))
    }

    /// Return an error because no inference backend is enabled.
    pub fn infer_into(
        &mut self,
        _input: ArrayView4<'_, f32>,
        _output: ArrayViewMut4<'_, f32>,
    ) -> Result<()> {
        Err(SoundExError::Inference(
            "ort-backend feature not enabled".into(),
        ))
    }

    /// No model shape is available without an inference backend.
    pub fn input_freq_bins(&self) -> Option<usize> {
        None
    }

    /// No session runs are possible without an inference backend.
    pub fn run_count(&self) -> u64 {
        0
    }
}
