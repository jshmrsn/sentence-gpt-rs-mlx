use crate::microgpt::{
    AdamOptimizerConfig, CharacterTokenizer, MicrogptTrainingProgress, TransformerConfig,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const CHECKPOINT_MAGIC: &[u8; 8] = b"MGPTCKP1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckpointBackend {
    Cpu,
    Mlx,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl CheckpointTensor {
    pub fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
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
    pub validation_evaluation_document_count: usize,
    pub context_window_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub embedding_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicrogptCheckpoint {
    pub backend: CheckpointBackend,
    pub training_run_config: Option<TrainingRunConfig>,
    pub config: TransformerConfig,
    pub tokenizer: CharacterTokenizer,
    pub documents: Vec<String>,
    pub validation_documents: Vec<String>,
    pub training_step_count: usize,
    pub validation_evaluation_document_count: usize,
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
    let payload = bincode::serialize(checkpoint).map_err(|error| error.to_string())?;
    let mut bytes = Vec::with_capacity(CHECKPOINT_MAGIC.len() + payload.len());
    bytes.extend_from_slice(CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn load_checkpoint_from_path(path: impl AsRef<Path>) -> Result<MicrogptCheckpoint, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() < CHECKPOINT_MAGIC.len() || &bytes[..CHECKPOINT_MAGIC.len()] != CHECKPOINT_MAGIC
    {
        return Err("not a sentence-gpt-rs-mlx checkpoint file".into());
    }
    bincode::deserialize(&bytes[CHECKPOINT_MAGIC.len()..]).map_err(|error| error.to_string())
}
