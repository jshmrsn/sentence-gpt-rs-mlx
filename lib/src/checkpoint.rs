use crate::microgpt::{
    AdamOptimizerConfig, CharacterTokenizer, MicrogptTrainingProgress, OptimizerFeatureConfig,
    TransformerConfig, TransformerFeatureConfig,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const CHECKPOINT_FORMAT_MARKER: &[u8; 8] = b"MGPTCKP1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckpointBackend {
    // The checkpoint stores which implementation produced it because the app can
    // train either with the scalar CPU reference backend or the MLX tensor
    // backend. The two backends intentionally share the same high-level model
    // shape, but they do not share runtime parameter types: CPU parameters are
    // `Value` graph nodes and MLX parameters are `Array` tensor handles. This
    // tag tells import code which runtime objects to rebuild around the saved
    // numeric buffers.
    Cpu,
    Mlx,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointTensor {
    // Shape is saved separately from values so loading can validate structure
    // before rebuilding a model. The values are always flat row-major numbers:
    // a matrix with shape [rows, cols] stores row 0, then row 1, and so on. That
    // convention is simple enough for both CPU Vec<Vec<Value>> parameters and
    // MLX 2-D Array parameters to round-trip through the same format.
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl CheckpointTensor {
    pub fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        // A checkpoint tensor is only meaningful if the number of scalars
        // matches the product of its dimensions. Checking here catches truncated,
        // corrupt, or mismatched state before import code starts assigning values
        // to model fields. Without this check, a bad file could shift every later
        // parameter and produce a model that "loads" but has nonsensical weights.
        let expected_value_count = shape.iter().product::<usize>();
        if values.len() != expected_value_count {
            return Err(format!(
                "tensor shape {:?} expects {expected_value_count} values, got {}",
                shape,
                values.len()
            ));
        }
        Ok(Self { shape, values })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingRunConfig {
    pub validation_step_interval: usize,
    pub training_document_batch_size: usize,
    pub max_document_count: usize,
    pub validation_set_divisor: usize,
    pub validation_set_max_document_count: usize,
    pub context_window_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub embedding_size: usize,
    pub mlp_expansion_factor: usize,
    pub transformer_features: TransformerFeatureConfig,
    pub optimizer_features: OptimizerFeatureConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicrogptCheckpoint {
    // This struct is deliberately a complete training snapshot, not just model
    // weights. Resuming AdamW correctly requires the optimizer moments; resuming
    // progress display requires loss history; reproducing validation requires
    // the same held-out documents. Saving all of it keeps "load checkpoint and
    // continue" semantically close to never having stopped.
    pub backend: CheckpointBackend,
    pub training_run_config: TrainingRunConfig,
    pub config: TransformerConfig,
    pub tokenizer: CharacterTokenizer,
    pub documents: Vec<String>,
    pub validation_documents: Vec<String>,
    pub training_step_count: usize,
    pub optimizer_config: AdamOptimizerConfig,
    pub completed_step_count: usize,
    pub latest_loss: Option<f64>,
    pub latest_validation_loss: Option<f64>,
    pub progress_history: Vec<MicrogptTrainingProgress>,
    pub parameters: Vec<CheckpointTensor>,
    pub first_moment_estimates: Vec<CheckpointTensor>,
    pub second_moment_estimates: Vec<CheckpointTensor>,
}

pub fn save_checkpoint_to_path(
    checkpoint: &MicrogptCheckpoint,
    path: impl AsRef<Path>,
) -> Result<(), String> {
    // `bincode` gives a compact Rust-native binary payload, but raw bincode has
    // no file signature. The marker lets `load_checkpoint_from_path`
    // reject unrelated files before trying to deserialize arbitrary bytes.
    let payload = bincode::serialize(checkpoint).map_err(|error| error.to_string())?;
    let mut bytes = Vec::with_capacity(CHECKPOINT_FORMAT_MARKER.len() + payload.len());
    bytes.extend_from_slice(CHECKPOINT_FORMAT_MARKER);
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn load_checkpoint_from_path(path: impl AsRef<Path>) -> Result<MicrogptCheckpoint, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    // Validate the prefix first so the error points at "wrong file type" instead
    // of a lower-level deserialize failure. The version-like marker also gives us
    // an obvious place to introduce a future incompatible checkpoint format.
    if bytes.len() < CHECKPOINT_FORMAT_MARKER.len()
        || &bytes[..CHECKPOINT_FORMAT_MARKER.len()] != CHECKPOINT_FORMAT_MARKER
    {
        return Err("not a sentence-gpt-rs-mlx checkpoint file".into());
    }
    bincode::deserialize(&bytes[CHECKPOINT_FORMAT_MARKER.len()..])
        .map_err(|error| error.to_string())
}
