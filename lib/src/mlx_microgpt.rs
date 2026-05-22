//! MLX-backed implementation of the same model in `microgpt.rs`.
//!
//! The plain Rust backend is the best place to learn the math because each
//! number is a `Value` with an explicit autodiff graph. This file teaches the
//! production translation: the same operations are batched into MLX `Array`
//! tensors so matrix multiplication, softmax, gradients, and optimizer updates
//! run efficiently on Apple Silicon.
//!
//! The important mental shift is scalar -> tensor:
//!
//! - CPU `Vec<Value>` hidden state becomes one MLX `Array`.
//! - CPU `Vec<Vec<Value>>` weight matrix becomes one 2-D MLX `Array`.
//! - CPU `loss.backward_for(...)` becomes MLX `value_and_grad`.
//! - CPU loops over matrix rows become MLX `matmul`, `sum_axis`, and `softmax`.
//!
//! A practical MLX note for Rust readers: `Array` is more like an owned handle
//! to a tensor expression than a plain `Vec<f32>`. Cloning an `Array` is usually
//! cheap because it clones the handle, not necessarily the underlying values.
//! Many operations are lazy; MLX can build a graph of tensor operations and run
//! it later. Calls such as `eval`, `item`, and `as_slice` force materialization
//! because they need concrete values on the Rust side.

use crate::checkpoint::{CheckpointBackend, CheckpointTensor, MicrogptCheckpoint};
use crate::microgpt::{
    apply_sampling_constraints, document_prediction_count, normalize_training_document,
    random_gaussian, scheduled_learning_rate, shuffled_by, training_batch_token_windows,
    upgrade_legacy_checkpoint_tensors_for_norm_gains, AdamOptimizerConfig, CharacterTokenizer,
    MicrogptTrainingProgress, TrainingTokenWindow, TransformerConfig,
};
// `ops` contains tensor operations such as softmax, reductions, stacking, and
// elementwise math. `transforms` contains autodiff transforms such as
// `value_and_grad`. `IndexOp` provides row/token lookup syntax for embedding
// tables and target-token losses.
use mlx_rs::{error::Result as MlxResult, ops, ops::indexing::IndexOp, transforms, Array};
use rand::Rng;

const MAX_GRADIENT_NORM: f32 = 1.0;
const RESIDUAL_DROPOUT_PROBABILITY: f32 = 0.05;

#[derive(Clone, Debug)]
pub struct MlxAttentionParameters {
    // Shape convention matches the CPU backend:
    //   [output_size, input_size]
    // For attention projections here that is [embedding_size, embedding_size].
    pub query_weights: Array,
    pub query_biases: Array,
    pub key_weights: Array,
    pub key_biases: Array,
    pub value_weights: Array,
    pub value_biases: Array,
    pub output_projection_weights: Array,
    pub output_projection_biases: Array,
}

#[derive(Clone, Debug)]
pub struct MlxFeedForwardParameters {
    // SwiGLU feed-forward shapes:
    //   expansion_weights: [3 * embedding_size, embedding_size]
    //   gate_weights:      [3 * embedding_size, embedding_size]
    //   projection_weights:[embedding_size, 3 * embedding_size]
    pub expansion_weights: Array,
    pub expansion_biases: Array,
    pub gate_weights: Array,
    pub gate_biases: Array,
    pub projection_weights: Array,
    pub projection_biases: Array,
}

#[derive(Clone, Debug)]
pub struct MlxTransformerLayerParameters {
    // Same learned RMSNorm gains as the CPU backend, but stored as MLX tensors.
    // Shape: [embedding_size]. MLX broadcasts this across batch and sequence
    // dimensions when normalizing [batch, sequence, embedding_size] activations.
    pub attention_norm_gain: Array,
    pub attention: MlxAttentionParameters,
    pub feed_forward_norm_gain: Array,
    pub feed_forward: MlxFeedForwardParameters,
}

#[derive(Clone, Debug)]
pub struct MlxTransformerModelParameters {
    // These fields mirror `TransformerModelParameters`, but each matrix is a
    // single MLX tensor. Keeping the shapes and names aligned makes it easier to
    // compare CPU and MLX behavior.
    pub token_embedding: Array,
    // Retained for checkpoint/UI compatibility. The active MLX forward pass is
    // RoPE-only and does not add this learned absolute position table.
    pub position_embedding: Array,
    // Retained for checkpoint/UI compatibility. Active output logits use tied
    // token embeddings instead of this independent head matrix.
    pub language_model_head: Array,
    pub language_model_head_biases: Array,
    // Shape: [embedding_size]. Applied before tied output logits.
    pub final_norm_gain: Array,
    pub layers: Vec<MlxTransformerLayerParameters>,
}

impl MlxTransformerModelParameters {
    pub fn initialize(
        vocabulary_size: usize,
        context_window_size: usize,
        embedding_size: usize,
        layer_count: usize,
        random_number_generator: &mut impl Rng,
    ) -> Self {
        // The initialization scheme intentionally matches the CPU backend. If
        // CPU and MLX start from the same random seed and data order, differences
        // are mostly from numeric precision and backend execution, not from
        // different model definitions.
        let embedding_std = 0.02;
        let feed_forward_size = 3 * embedding_size;
        let projection_std = (1.0 / embedding_size as f64).sqrt();
        let residual_projection_std = projection_std / (2.0 * layer_count as f64).sqrt();

        Self {
            token_embedding: mlx_matrix(
                vocabulary_size,
                embedding_size,
                random_number_generator,
                embedding_std,
            ),
            position_embedding: mlx_matrix(
                context_window_size,
                embedding_size,
                random_number_generator,
                embedding_std,
            ),
            language_model_head: mlx_matrix(
                vocabulary_size,
                embedding_size,
                random_number_generator,
                projection_std,
            ),
            language_model_head_biases: mlx_zero_vector(vocabulary_size),
            final_norm_gain: mlx_one_vector(embedding_size),
            layers: (0..layer_count)
                .map(|_| MlxTransformerLayerParameters {
                    attention_norm_gain: mlx_one_vector(embedding_size),
                    attention: MlxAttentionParameters {
                        query_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        query_biases: mlx_zero_vector(embedding_size),
                        key_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        key_biases: mlx_zero_vector(embedding_size),
                        value_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        value_biases: mlx_zero_vector(embedding_size),
                        output_projection_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            residual_projection_std,
                        ),
                        output_projection_biases: mlx_zero_vector(embedding_size),
                    },
                    feed_forward_norm_gain: mlx_one_vector(embedding_size),
                    feed_forward: MlxFeedForwardParameters {
                        expansion_weights: mlx_matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        expansion_biases: mlx_zero_vector(feed_forward_size),
                        gate_weights: mlx_matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        gate_biases: mlx_zero_vector(feed_forward_size),
                        projection_weights: mlx_matrix(
                            embedding_size,
                            feed_forward_size,
                            random_number_generator,
                            residual_projection_std,
                        ),
                        projection_biases: mlx_zero_vector(embedding_size),
                    },
                })
                .collect(),
        }
    }

    fn zero_initialized(
        vocabulary_size: usize,
        context_window_size: usize,
        embedding_size: usize,
        layer_count: usize,
    ) -> Self {
        let feed_forward_size = 3 * embedding_size;
        Self {
            token_embedding: mlx_zero_matrix(vocabulary_size, embedding_size),
            position_embedding: mlx_zero_matrix(context_window_size, embedding_size),
            language_model_head: mlx_zero_matrix(vocabulary_size, embedding_size),
            language_model_head_biases: mlx_zero_vector(vocabulary_size),
            final_norm_gain: mlx_one_vector(embedding_size),
            layers: (0..layer_count)
                .map(|_| MlxTransformerLayerParameters {
                    attention_norm_gain: mlx_one_vector(embedding_size),
                    attention: MlxAttentionParameters {
                        query_weights: mlx_zero_matrix(embedding_size, embedding_size),
                        query_biases: mlx_zero_vector(embedding_size),
                        key_weights: mlx_zero_matrix(embedding_size, embedding_size),
                        key_biases: mlx_zero_vector(embedding_size),
                        value_weights: mlx_zero_matrix(embedding_size, embedding_size),
                        value_biases: mlx_zero_vector(embedding_size),
                        output_projection_weights: mlx_zero_matrix(embedding_size, embedding_size),
                        output_projection_biases: mlx_zero_vector(embedding_size),
                    },
                    feed_forward_norm_gain: mlx_one_vector(embedding_size),
                    feed_forward: MlxFeedForwardParameters {
                        expansion_weights: mlx_zero_matrix(feed_forward_size, embedding_size),
                        expansion_biases: mlx_zero_vector(feed_forward_size),
                        gate_weights: mlx_zero_matrix(feed_forward_size, embedding_size),
                        gate_biases: mlx_zero_vector(feed_forward_size),
                        projection_weights: mlx_zero_matrix(embedding_size, feed_forward_size),
                        projection_biases: mlx_zero_vector(embedding_size),
                    },
                })
                .collect(),
        }
    }

    pub fn values(&self) -> Vec<Array> {
        // MLX's gradient transform works over a flat slice of Array arguments.
        // This order must match `with_values` and `params_from_arrays`.
        let mut values = vec![
            self.token_embedding.clone(),
            self.position_embedding.clone(),
            self.language_model_head.clone(),
            self.language_model_head_biases.clone(),
            self.final_norm_gain.clone(),
        ];
        for layer in &self.layers {
            values.push(layer.attention_norm_gain.clone());
            values.push(layer.attention.query_weights.clone());
            values.push(layer.attention.query_biases.clone());
            values.push(layer.attention.key_weights.clone());
            values.push(layer.attention.key_biases.clone());
            values.push(layer.attention.value_weights.clone());
            values.push(layer.attention.value_biases.clone());
            values.push(layer.attention.output_projection_weights.clone());
            values.push(layer.attention.output_projection_biases.clone());
            values.push(layer.feed_forward_norm_gain.clone());
            values.push(layer.feed_forward.expansion_weights.clone());
            values.push(layer.feed_forward.expansion_biases.clone());
            values.push(layer.feed_forward.gate_weights.clone());
            values.push(layer.feed_forward.gate_biases.clone());
            values.push(layer.feed_forward.projection_weights.clone());
            values.push(layer.feed_forward.projection_biases.clone());
        }
        values
    }

    pub fn with_values(&self, values: &[Array]) -> Self {
        // Reconstruct the structured model after AdamW has produced replacement
        // tensors. No tensor data is copied here unless MLX needs to materialize
        // it; these are cheap Array handles.
        let mut index = 0;
        let mut next = || {
            let value = values[index].clone();
            index += 1;
            value
        };

        Self {
            token_embedding: next(),
            position_embedding: next(),
            language_model_head: next(),
            language_model_head_biases: next(),
            final_norm_gain: next(),
            layers: self
                .layers
                .iter()
                .map(|_| MlxTransformerLayerParameters {
                    attention_norm_gain: next(),
                    attention: MlxAttentionParameters {
                        query_weights: next(),
                        query_biases: next(),
                        key_weights: next(),
                        key_biases: next(),
                        value_weights: next(),
                        value_biases: next(),
                        output_projection_weights: next(),
                        output_projection_biases: next(),
                    },
                    feed_forward_norm_gain: next(),
                    feed_forward: MlxFeedForwardParameters {
                        expansion_weights: next(),
                        expansion_biases: next(),
                        gate_weights: next(),
                        gate_biases: next(),
                        projection_weights: next(),
                        projection_biases: next(),
                    },
                })
                .collect(),
        }
    }

    fn checkpoint_tensor_shapes(&self) -> Vec<CheckpointTensor> {
        self.values()
            .iter()
            .map(|array| {
                let shape = array
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect::<Vec<_>>();
                let value_count = shape.iter().product::<usize>();
                CheckpointTensor {
                    shape,
                    values: vec![0.0; value_count],
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct MlxTrainedMicrogpt {
    pub model: MlxTransformerModelParameters,
    pub config: TransformerConfig,
    pub tokenizer: CharacterTokenizer,
}

#[derive(Clone, Debug)]
pub struct MlxAdamOptimizerState {
    // These vectors have exactly one tensor per parameter tensor. If
    // `model.values()[i]` has shape [64, 64], both moment tensors at index i
    // have shape [64, 64] too.
    pub first_moment_estimates: Vec<Array>,
    pub second_moment_estimates: Vec<Array>,
}

#[derive(Clone, Debug)]
pub struct MlxMicrogptTrainingSession {
    // A session is intentionally CPU-owned Rust state containing MLX tensor
    // handles. Training workers clone and return whole sessions so the UI never
    // mutates a model while MLX is computing with it.
    pub trained_microgpt: MlxTrainedMicrogpt,
    pub documents: Vec<String>,
    pub validation_documents: Vec<String>,
    pub training_step_count: usize,
    pub validation_evaluation_document_count: usize,
    pub optimizer_config: AdamOptimizerConfig,
    pub optimizer_state: MlxAdamOptimizerState,
    pub completed_step_count: usize,
    pub latest_loss: Option<f64>,
    pub latest_validation_loss: Option<f64>,
    pub progress_history: Vec<MicrogptTrainingProgress>,
}

impl MlxMicrogptTrainingSession {
    pub fn is_complete(&self) -> bool {
        self.completed_step_count >= self.training_step_count
    }

    pub fn with_initial_progress(mut self) -> MlxResult<Self> {
        // `calculate_*_loss` eventually calls `item::<f32>()`, which forces MLX
        // to execute enough of the graph to return a scalar Rust value. Until
        // then, most tensor operations are just scheduled expressions.
        let train_loss = calculate_training_loss_baseline(&self)?;
        let validation_loss = calculate_validation_loss(&self, 0, 50)?;
        let loss = train_loss.or(validation_loss).unwrap_or(0.0);
        let progress = MicrogptTrainingProgress {
            completed_step_count: 0,
            training_step_count: self.training_step_count,
            loss,
            validation_loss,
        };
        self.latest_loss = train_loss;
        self.latest_validation_loss = validation_loss;
        self.progress_history = vec![progress];
        Ok(self)
    }
}

pub fn export_training_session_checkpoint(
    session: &MlxMicrogptTrainingSession,
) -> Result<MicrogptCheckpoint, String> {
    let parameter_arrays = session.trained_microgpt.model.values();
    Ok(MicrogptCheckpoint {
        backend: CheckpointBackend::Mlx,
        training_run_config: None,
        config: session.trained_microgpt.config.clone(),
        tokenizer: session.trained_microgpt.tokenizer.clone(),
        documents: session.documents.clone(),
        validation_documents: session.validation_documents.clone(),
        training_step_count: session.training_step_count,
        validation_evaluation_document_count: session.validation_evaluation_document_count,
        optimizer_config: session.optimizer_config.clone(),
        completed_step_count: session.completed_step_count,
        latest_loss: session.latest_loss,
        latest_validation_loss: session.latest_validation_loss,
        progress_history: session.progress_history.clone(),
        parameters: arrays_to_checkpoint_tensors(&parameter_arrays)?,
        first_moment_estimates: arrays_to_checkpoint_tensors(
            &session.optimizer_state.first_moment_estimates,
        )?,
        second_moment_estimates: arrays_to_checkpoint_tensors(
            &session.optimizer_state.second_moment_estimates,
        )?,
    })
}

pub fn import_training_session_checkpoint(
    checkpoint: &MicrogptCheckpoint,
) -> Result<MlxMicrogptTrainingSession, String> {
    let tokenizer = CharacterTokenizer::new(
        checkpoint.tokenizer.unique_characters.clone(),
        checkpoint.tokenizer.sequence_boundary_token_id,
    );
    let model_template = MlxTransformerModelParameters::zero_initialized(
        tokenizer.vocabulary_size(),
        checkpoint.config.context_window_size,
        checkpoint.config.embedding_size,
        checkpoint.config.layer_count,
    );
    let expected_tensors = model_template.checkpoint_tensor_shapes();
    let parameter_tensors = upgrade_legacy_checkpoint_tensors_for_norm_gains(
        &checkpoint.parameters,
        &expected_tensors,
        checkpoint.config.layer_count,
        1.0,
    )?;
    let first_moment_tensors = upgrade_legacy_checkpoint_tensors_for_norm_gains(
        &checkpoint.first_moment_estimates,
        &expected_tensors,
        checkpoint.config.layer_count,
        0.0,
    )?;
    let second_moment_tensors = upgrade_legacy_checkpoint_tensors_for_norm_gains(
        &checkpoint.second_moment_estimates,
        &expected_tensors,
        checkpoint.config.layer_count,
        0.0,
    )?;
    let parameter_arrays = checkpoint_tensors_to_arrays(&parameter_tensors, &expected_tensors)?;
    let first_moment_estimates =
        checkpoint_tensors_to_arrays(&first_moment_tensors, &expected_tensors)?;
    let second_moment_estimates =
        checkpoint_tensors_to_arrays(&second_moment_tensors, &expected_tensors)?;
    let model = model_template.with_values(&parameter_arrays);

    Ok(MlxMicrogptTrainingSession {
        trained_microgpt: MlxTrainedMicrogpt {
            model,
            config: checkpoint.config.clone(),
            tokenizer,
        },
        documents: checkpoint.documents.clone(),
        validation_documents: checkpoint.validation_documents.clone(),
        training_step_count: checkpoint.training_step_count,
        validation_evaluation_document_count: checkpoint.validation_evaluation_document_count,
        optimizer_config: checkpoint.optimizer_config.clone(),
        optimizer_state: MlxAdamOptimizerState {
            first_moment_estimates,
            second_moment_estimates,
        },
        completed_step_count: checkpoint.completed_step_count,
        latest_loss: checkpoint.latest_loss,
        latest_validation_loss: checkpoint.latest_validation_loss,
        progress_history: checkpoint.progress_history.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct MlxMicrogptTrainingStepResult {
    pub session: MlxMicrogptTrainingSession,
    pub progress: MicrogptTrainingProgress,
    pub progress_history: Vec<MicrogptTrainingProgress>,
}

struct MlxParamView<'a> {
    // Borrowed view used only inside a differentiable MLX closure. It lets us
    // treat the flat `inputs: &[Array]` from `value_and_grad` as a named model
    // without cloning every tensor.
    token_embedding: &'a Array,
    position_embedding: &'a Array,
    language_model_head: &'a Array,
    language_model_head_biases: &'a Array,
    final_norm_gain: &'a Array,
    rotary_position_matrices: &'a [Array],
    layers: Vec<MlxLayerParamView<'a>>,
}

struct MlxLayerParamView<'a> {
    // These borrowed views are not long-lived model objects. They only give
    // names to entries in MLX's flat input slice while the loss closure runs.
    attention_norm_gain: &'a Array,
    attention: MlxAttentionParamView<'a>,
    feed_forward_norm_gain: &'a Array,
    feed_forward: MlxFeedForwardParamView<'a>,
}

struct MlxAttentionParamView<'a> {
    query_weights: &'a Array,
    query_biases: &'a Array,
    key_weights: &'a Array,
    key_biases: &'a Array,
    value_weights: &'a Array,
    value_biases: &'a Array,
    output_projection_weights: &'a Array,
    output_projection_biases: &'a Array,
}

struct MlxFeedForwardParamView<'a> {
    expansion_weights: &'a Array,
    expansion_biases: &'a Array,
    gate_weights: &'a Array,
    gate_biases: &'a Array,
    projection_weights: &'a Array,
    projection_biases: &'a Array,
}

struct BatchDropoutMasks {
    attention_residual_masks: Vec<Array>,
    feed_forward_residual_masks: Vec<Array>,
}

pub fn create_mlx_microgpt_training_session(
    input_documents: Vec<String>,
    random_number_generator: &mut impl Rng,
    training_step_count: usize,
    validation_set_divisor: usize,
    validation_evaluation_document_count: usize,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
) -> MlxMicrogptTrainingSession {
    // Data prep is deliberately identical to the CPU backend: same normalized
    // text, same train/validation split, same character vocabulary. That makes
    // the backend toggle a performance choice rather than a modeling choice.
    let trimmed_documents: Vec<_> = input_documents
        .into_iter()
        .map(|document| normalize_training_document(&document))
        .filter(|document| !document.is_empty())
        .collect();
    let shuffled_documents = shuffled_by(&trimmed_documents, random_number_generator);
    let validation_document_count = shuffled_documents.len() / validation_set_divisor;
    let validation_documents = shuffled_documents[..validation_document_count].to_vec();
    let documents = shuffled_documents[validation_document_count..].to_vec();

    create_mlx_microgpt_training_session_from_splits(
        documents,
        validation_documents,
        random_number_generator,
        training_step_count,
        validation_evaluation_document_count,
        transformer_config,
        optimizer_config,
    )
}

pub fn create_mlx_microgpt_training_session_from_splits(
    input_documents: Vec<String>,
    input_validation_documents: Vec<String>,
    random_number_generator: &mut impl Rng,
    training_step_count: usize,
    validation_evaluation_document_count: usize,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
) -> MlxMicrogptTrainingSession {
    let documents: Vec<_> = input_documents
        .into_iter()
        .map(|document| normalize_training_document(&document))
        .filter(|document| !document.is_empty())
        .collect();
    let validation_documents: Vec<_> = input_validation_documents
        .into_iter()
        .map(|document| normalize_training_document(&document))
        .filter(|document| !document.is_empty())
        .collect();

    let mut unique_characters: Vec<char> = documents
        .iter()
        .chain(validation_documents.iter())
        .flat_map(|document| document.chars())
        .collect();
    unique_characters.sort_unstable();
    unique_characters.dedup();

    let sequence_boundary_token_id = unique_characters.len();
    let tokenizer = CharacterTokenizer::new(unique_characters, sequence_boundary_token_id);
    let model = MlxTransformerModelParameters::initialize(
        tokenizer.vocabulary_size(),
        transformer_config.context_window_size,
        transformer_config.embedding_size,
        transformer_config.layer_count,
        random_number_generator,
    );
    let parameters = model.values();
    // Adam stores two state tensors per parameter. They start as zeros with the
    // exact same shape as each parameter tensor.
    //
    // `ops::zeros_like` returns an MLX result because allocation or shape
    // handling can fail. This constructor unwraps because all shapes came from
    // freshly created model tensors; failure here would indicate a programming
    // error rather than user input.
    let zeros = parameters
        .iter()
        .map(ops::zeros_like)
        .collect::<MlxResult<Vec<_>>>()
        .unwrap();
    let zeros_2 = parameters
        .iter()
        .map(ops::zeros_like)
        .collect::<MlxResult<Vec<_>>>()
        .unwrap();

    MlxMicrogptTrainingSession {
        trained_microgpt: MlxTrainedMicrogpt {
            model,
            config: transformer_config,
            tokenizer,
        },
        documents,
        validation_documents,
        training_step_count,
        validation_evaluation_document_count,
        optimizer_config,
        optimizer_state: MlxAdamOptimizerState {
            first_moment_estimates: zeros,
            second_moment_estimates: zeros_2,
        },
        completed_step_count: 0,
        latest_loss: None,
        latest_validation_loss: None,
        progress_history: Vec::new(),
    }
}

pub fn train_mlx_microgpt_step(
    session: MlxMicrogptTrainingSession,
    batch_document_count: usize,
) -> MlxResult<Option<MlxMicrogptTrainingStepResult>> {
    // MLX training is the tensor equivalent of `train_microgpt_step`.
    // Instead of manually building a `Value` graph and calling backward, we give
    // MLX a closure from parameters -> loss and ask it for both the loss value
    // and gradients with respect to every parameter tensor.
    if session.is_complete() {
        return Ok(None);
    }
    assert!(
        batch_document_count > 0,
        "batch_document_count must be positive"
    );

    let step = session.completed_step_count;
    let batch_windows = training_batch_token_windows(
        &session.documents,
        &session.trained_microgpt.tokenizer,
        step,
        batch_document_count,
        session.trained_microgpt.config.context_window_size,
    );
    let mut parameters = session.trained_microgpt.model.values();
    // `argnums` says "differentiate with respect to every parameter array in
    // this input slice." The returned gradient list has the same order.
    let argnums = (0..parameters.len() as i32).collect::<Vec<_>>();
    let config = session.trained_microgpt.config.clone();
    let rotary_position_matrices = rotary_position_matrices(&config);
    // Capturing `layer_count` separately lets `params_from_arrays` reconstruct a
    // borrowed view without needing to capture the whole model inside the MLX
    // autodiff closure.
    let layer_count = session.trained_microgpt.model.layers.len();
    let dropout_masks = config.features.use_residual_dropout.then(|| {
        batch_dropout_masks(
            step,
            batch_document_count,
            session.trained_microgpt.config.context_window_size,
            session.trained_microgpt.config.embedding_size,
            layer_count,
            RESIDUAL_DROPOUT_PROBABILITY,
        )
    });

    let loss_fn = move |inputs: &[Array]| -> MlxResult<Vec<Array>> {
        // MLX closures receive a flat parameter list. Convert it back into a
        // named view, compute scalar batch loss, and return it as a one-element
        // vector because the transform API is vector-valued.
        let params = params_from_arrays(inputs, layer_count, &rotary_position_matrices);
        let loss = batch_loss(&params, &config, &batch_windows, dropout_masks.as_ref())?;
        Ok(vec![loss])
    };

    let mut value_and_grad = transforms::value_and_grad_with_argnums(loss_fn, argnums.as_slice());
    // Calling the transformed function does two things:
    //
    // - evaluates the forward loss,
    // - builds and runs the reverse pass for every `argnums` input.
    //
    // `loss_values` mirrors the closure return vector; `gradients[i]` is the
    // derivative of the scalar loss with respect to `parameters[i]`.
    let (loss_values, gradients) = value_and_grad(&parameters)?;
    // `item::<f32>()` copies a scalar tensor from MLX into a Rust number. This is
    // fine for logging loss, but do not do this inside large inner loops unless
    // you intentionally need host-side data.
    let loss = loss_values[0].item::<f32>() as f64;

    let (updated_parameters, optimizer_state) = apply_adam_update(
        parameters.as_slice(),
        gradients.as_slice(),
        &session.optimizer_state,
        &session.optimizer_config,
        step,
        session.training_step_count,
    )?;
    parameters = updated_parameters;
    // MLX is lazy: operations build a computation graph and may not execute
    // immediately. `eval` materializes the updated tensors before storing them
    // in the session, making UI reads and later steps predictable.
    transforms::eval(parameters.iter())?;

    let updated_model = session.trained_microgpt.model.with_values(&parameters);
    let updated_microgpt = MlxTrainedMicrogpt {
        model: updated_model,
        ..session.trained_microgpt.clone()
    };
    let progress = MicrogptTrainingProgress {
        completed_step_count: session.completed_step_count + 1,
        training_step_count: session.training_step_count,
        loss,
        validation_loss: None,
    };
    let mut progress_history = session.progress_history.clone();
    progress_history.push(progress.clone());

    let updated_session = MlxMicrogptTrainingSession {
        trained_microgpt: updated_microgpt,
        optimizer_state,
        completed_step_count: progress.completed_step_count,
        latest_loss: Some(progress.loss),
        progress_history: progress_history.clone(),
        ..session
    };

    Ok(Some(MlxMicrogptTrainingStepResult {
        session: updated_session,
        progress,
        progress_history,
    }))
}

pub fn attach_validation_loss(
    mut result: MlxMicrogptTrainingStepResult,
    validation_loss: Option<f64>,
) -> MlxMicrogptTrainingStepResult {
    let Some(validation_loss) = validation_loss else {
        return result;
    };

    result.progress.validation_loss = Some(validation_loss);
    if let Some(last_progress) = result.progress_history.last_mut() {
        *last_progress = result.progress.clone();
    }
    result.session.latest_validation_loss = Some(validation_loss);
    result.session.progress_history = result.progress_history.clone();
    result
}

pub fn calculate_training_loss_baseline(
    session: &MlxMicrogptTrainingSession,
) -> MlxResult<Option<f64>> {
    session
        .documents
        .first()
        .map(|document| {
            calculate_document_loss(
                &session.trained_microgpt.model,
                &session.trained_microgpt.config,
                &session.trained_microgpt.tokenizer,
                document,
            )
        })
        .transpose()
}

pub fn calculate_validation_loss(
    session: &MlxMicrogptTrainingSession,
    completed_step_count: usize,
    validation_step_interval: usize,
) -> MlxResult<Option<f64>> {
    if session.validation_documents.is_empty() {
        return Ok(None);
    }
    let validation_document_count = session
        .validation_evaluation_document_count
        .min(session.validation_documents.len());
    let validation_batch_index = completed_step_count / validation_step_interval;
    let mut weighted_loss_sum = 0.0;
    let mut token_count = 0usize;
    for validation_offset in 0..validation_document_count {
        let validation_index = (validation_batch_index * validation_document_count
            + validation_offset)
            % session.validation_documents.len();
        let document = &session.validation_documents[validation_index];
        let document_token_count = document_prediction_count(
            &session.trained_microgpt.tokenizer,
            document,
            session.trained_microgpt.config.context_window_size,
        );
        let document_loss = calculate_document_loss(
            &session.trained_microgpt.model,
            &session.trained_microgpt.config,
            &session.trained_microgpt.tokenizer,
            document,
        )?;
        weighted_loss_sum += document_loss * document_token_count as f64;
        token_count += document_token_count;
    }
    Ok((token_count > 0).then(|| weighted_loss_sum / token_count as f64))
}

pub fn calculate_document_loss(
    model: &MlxTransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    document: &str,
) -> MlxResult<f64> {
    // Validation and display code do not need gradients, but they use the same
    // forward loss function so the reported number means the same thing as the
    // training objective.
    let token_window = training_batch_token_windows(
        &[document.to_string()],
        tokenizer,
        0,
        1,
        config.context_window_size,
    )
    .into_iter()
    .next()
    .unwrap_or_else(|| TrainingTokenWindow {
        input_tokens: vec![tokenizer.sequence_boundary_token_id; config.context_window_size],
        target_tokens: vec![tokenizer.sequence_boundary_token_id; config.context_window_size],
        loss_mask: vec![0.0; config.context_window_size],
        batch_offset: 0,
    });
    let model_values = model.values();
    let rotary_position_matrices = rotary_position_matrices(config);
    let params = params_from_arrays(&model_values, model.layers.len(), &rotary_position_matrices);
    Ok(batch_loss(&params, config, &[token_window], None)?.item::<f32>() as f64)
}

pub fn generate_samples(
    model: &MlxTransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    prefix: &str,
    sample_count: usize,
    temperature: f64,
    random_number_generator: &mut impl Rng,
) -> MlxResult<Vec<String>> {
    (0..sample_count)
        .map(|_| {
            generate_sample(
                model,
                config,
                tokenizer,
                prefix,
                temperature,
                random_number_generator,
            )
        })
        .collect()
}

pub fn generate_sample(
    model: &MlxTransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    prefix: &str,
    temperature: f64,
    random_number_generator: &mut impl Rng,
) -> MlxResult<String> {
    // Sampling uses MLX for the forward pass and softmax, then copies the small
    // probability vector back to Rust so the CPU-side sampling constraints and
    // RNG behavior stay shared with the reference backend.
    let model_values = model.values();
    let rotary_position_matrices = rotary_position_matrices(config);
    let params = params_from_arrays(&model_values, model.layers.len(), &rotary_position_matrices);
    let mut keys = create_key_value_cache(config.layer_count);
    let mut values = create_key_value_cache(config.layer_count);
    let mut token_id = tokenizer.sequence_boundary_token_id;
    let normalized_prefix = prefix
        .trim()
        .chars()
        .filter(|character| tokenizer.character_to_token_id.contains_key(character))
        .take(config.context_window_size - 1)
        .collect::<String>();
    let prefix_characters = normalized_prefix.chars().collect::<Vec<_>>();
    let mut sample = normalized_prefix.clone();

    for position_id in 0..config.context_window_size {
        let run = run_transformer_model(&params, config, token_id, position_id, keys, values)?;
        keys = run.keys;
        values = run.values;

        if let Some(prefix_character) = prefix_characters.get(position_id) {
            token_id = tokenizer.character_to_token_id[prefix_character];
            continue;
        }

        let scaled_logits = &run.logits / Array::from_f32(temperature as f32);
        let probabilities = ops::softmax_axis(&scaled_logits, 0, None)?;
        probabilities.eval()?;
        // The vocabulary is tiny, so copying probabilities to Rust is cheap. For
        // a production-size vocabulary, sampling would usually stay on device or
        // use specialized top-k/top-p tensor kernels.
        let mut weights = probabilities
            .as_slice::<f32>()
            .iter()
            .map(|probability| *probability as f64)
            .collect::<Vec<_>>();
        apply_sampling_constraints(&mut weights, tokenizer, &sample, prefix_characters.len());
        token_id = weighted_choice(&weights, random_number_generator);
        if token_id == tokenizer.sequence_boundary_token_id {
            break;
        }
        sample.push(tokenizer.unique_characters[token_id]);
    }

    Ok(sample)
}

struct TransformerRun {
    logits: Array,
    keys: Vec<Vec<Array>>,
    values: Vec<Vec<Array>>,
}

struct TransformerLayerRun {
    hidden_state: Array,
    keys: Vec<Vec<Array>>,
    values: Vec<Vec<Array>>,
}

fn params_from_arrays<'a>(
    arrays: &'a [Array],
    layer_count: usize,
    rotary_position_matrices: &'a [Array],
) -> MlxParamView<'a> {
    // This must stay in lockstep with `MlxTransformerModelParameters::values`.
    // A wrong order would train the right shapes under the wrong names, which is
    // the kind of bug that compiles but ruins learning.
    let mut index = 0;
    let mut next = || {
        // Borrow each Array rather than cloning it. The closure should describe
        // math on the exact input tensors that MLX is differentiating.
        let value = &arrays[index];
        index += 1;
        value
    };

    let token_embedding = next();
    let position_embedding = next();
    let language_model_head = next();
    let language_model_head_biases = next();
    let final_norm_gain = next();

    MlxParamView {
        token_embedding,
        position_embedding,
        language_model_head,
        language_model_head_biases,
        final_norm_gain,
        rotary_position_matrices,
        layers: (0..layer_count)
            .map(|_| MlxLayerParamView {
                attention_norm_gain: next(),
                attention: MlxAttentionParamView {
                    query_weights: next(),
                    query_biases: next(),
                    key_weights: next(),
                    key_biases: next(),
                    value_weights: next(),
                    value_biases: next(),
                    output_projection_weights: next(),
                    output_projection_biases: next(),
                },
                feed_forward_norm_gain: next(),
                feed_forward: MlxFeedForwardParamView {
                    expansion_weights: next(),
                    expansion_biases: next(),
                    gate_weights: next(),
                    gate_biases: next(),
                    projection_weights: next(),
                    projection_biases: next(),
                },
            })
            .collect(),
    }
}

fn batch_loss(
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
    batch_windows: &[TrainingTokenWindow],
    dropout_masks: Option<&BatchDropoutMasks>,
) -> MlxResult<Array> {
    // Full-sequence training: instead of looping Rust-side over every character
    // and growing a KV cache, build [batch, sequence] token tensors, run all
    // positions through every layer, and compute cross-entropy for the entire
    // mini-batch in one MLX graph. This is the path that lets Apple Silicon do
    // large matrix work instead of thousands of tiny scalar-shaped operations.
    let batch_size = batch_windows.len();
    let sequence_len = config.context_window_size;
    // Convert the Rust-side training windows into compact device tensors:
    //
    // input_tokens:  [batch, sequence]
    // target_tokens: [batch * sequence, 1] after flattening, for gather
    // loss_mask:     [batch * sequence], 1 for real targets and 0 for padding
    //
    // The token ids are small integers; MLX uses them to gather rows from the
    // embedding table. The loss mask is what lets short sentences share the same
    // fixed tensor shape as long sentences without teaching on padding.
    let flat_input_tokens = batch_windows
        .iter()
        .flat_map(|window| window.input_tokens.iter().map(|token| *token as i32))
        .collect::<Vec<_>>();
    let flat_target_tokens = batch_windows
        .iter()
        .flat_map(|window| window.target_tokens.iter().map(|token| *token as i32))
        .collect::<Vec<_>>();
    let flat_loss_mask = batch_windows
        .iter()
        .flat_map(|window| window.loss_mask.iter().map(|mask| *mask as f32))
        .collect::<Vec<_>>();

    let input_tokens = Array::from_slice(
        &flat_input_tokens,
        &[batch_size as i32, sequence_len as i32],
    );
    let target_tokens = Array::from_slice(
        &flat_target_tokens,
        &[(batch_size * sequence_len) as i32, 1],
    );
    let loss_mask = Array::from_slice(&flat_loss_mask, &[(batch_size * sequence_len) as i32]);

    let logits = run_transformer_batch(params, config, &input_tokens, dropout_masks)?;
    let vocabulary_size = params.token_embedding.shape()[0];
    // `logits` is [batch, sequence, vocab]. Flatten to [tokens, vocab] so each
    // row is one next-character prediction. Cross-entropy is:
    //
    // log(sum(exp(all logits))) - logit(correct target)
    //
    // `logsumexp_axis` computes the first term stably without overflow.
    let flat_logits = logits.reshape(&[(batch_size * sequence_len) as i32, vocabulary_size])?;
    let log_normalizer = ops::logsumexp_axis(&flat_logits, 1, true)?;
    let target_logits = flat_logits
        .take_along_axis(&target_tokens, 1)?
        .reshape(&[-1])?;
    let token_losses = (log_normalizer.reshape(&[-1])? - target_logits) * &loss_mask;
    ops::sum(&token_losses, None)
        .and_then(|loss_sum| Ok(loss_sum / (ops::sum(&loss_mask, None)? + Array::from_f32(1e-6))))
}

fn run_transformer_batch(
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
    input_tokens: &Array,
    dropout_masks: Option<&BatchDropoutMasks>,
) -> MlxResult<Array> {
    // Gather token vectors for every batch item and every sequence position at
    // once. Shape changes from token ids [batch, sequence] to hidden states
    // [batch, sequence, embedding_size]. This single gather replaces the
    // token-by-token embedding lookup loop used by the reference backend.
    let mut hidden_state = params.token_embedding.take_axis(input_tokens, 0)?;
    if config.features.use_learned_absolute_position_encoding {
        let position_ids = Array::from_slice(
            &(0..config.context_window_size as i32).collect::<Vec<_>>(),
            &[config.context_window_size as i32],
        );
        hidden_state += &params
            .position_embedding
            .take_axis(&position_ids, 0)?
            .expand_dims(0)?;
    }

    for (layer_index, layer) in params.layers.iter().enumerate() {
        hidden_state = run_transformer_layer_batch(
            &hidden_state,
            layer,
            layer_index,
            params.rotary_position_matrices,
            config,
            dropout_masks,
        )?;
    }

    // Final norm makes the scale of the hidden states predictable before the
    // output dot products. With tied embeddings, this is especially helpful:
    // logits are literal dot products against token embedding rows.
    if config.features.use_final_rmsnorm {
        hidden_state = rmsnorm_last_axis(&hidden_state, params.final_norm_gain, config)?;
    }
    language_model_logits_batch(&hidden_state, params, config)
}

fn run_transformer_layer_batch(
    hidden_state: &Array,
    layer: &MlxLayerParamView<'_>,
    layer_index: usize,
    rotary_position_matrices: &[Array],
    config: &TransformerConfig,
    dropout_masks: Option<&BatchDropoutMasks>,
) -> MlxResult<Array> {
    let residual_state = hidden_state.clone();
    // Pre-norm Transformer layout: normalize before each sub-block, then add the
    // sub-block output back to the residual stream. Pre-norm is generally easier
    // to train because gradients can flow through the residual path even when a
    // sub-block is poorly initialized.
    let normalized_state = rmsnorm_last_axis(hidden_state, layer.attention_norm_gain, config)?;

    let mut query = linear_last_axis(
        &normalized_state,
        layer.attention.query_weights,
        layer.attention.query_biases,
        config,
    )?;
    let mut key = linear_last_axis(
        &normalized_state,
        layer.attention.key_weights,
        layer.attention.key_biases,
        config,
    )?;
    if config.features.use_rope_position_encoding {
        query = apply_rotary_position_embedding_batch(&query, rotary_position_matrices, config)?;
        key = apply_rotary_position_embedding_batch(&key, rotary_position_matrices, config)?;
    }
    let value = linear_last_axis(
        &normalized_state,
        layer.attention.value_weights,
        layer.attention.value_biases,
        config,
    )?;

    let attention_output = run_multi_head_attention_batch(&query, &key, &value, config)?;
    let mut block_output = linear_last_axis(
        &attention_output,
        layer.attention.output_projection_weights,
        layer.attention.output_projection_biases,
        config,
    )?;
    if let Some(dropout_masks) = dropout_masks {
        // Drop only the residual update, not the residual state itself. The
        // model still has a clean path for information to move forward while
        // the proposed attention edit is regularized during training.
        block_output *= &dropout_masks.attention_residual_masks[layer_index];
    }
    let updated_hidden_state = block_output + residual_state;

    let residual_state = updated_hidden_state.clone();
    let normalized_state =
        rmsnorm_last_axis(&updated_hidden_state, layer.feed_forward_norm_gain, config)?;
    let expanded_output = linear_last_axis(
        &normalized_state,
        layer.feed_forward.expansion_weights,
        layer.feed_forward.expansion_biases,
        config,
    )?;
    let gated_output = linear_last_axis(
        &normalized_state,
        layer.feed_forward.gate_weights,
        layer.feed_forward.gate_biases,
        config,
    )?;
    let block_output = if config.features.use_swiglu_feed_forward {
        silu(&expanded_output)? * gated_output
    } else {
        relu(&expanded_output)?
    };
    let mut block_output = linear_last_axis(
        &block_output,
        layer.feed_forward.projection_weights,
        layer.feed_forward.projection_biases,
        config,
    )?;
    if let Some(dropout_masks) = dropout_masks {
        // Same idea for the feed-forward update. During inference this mask is
        // absent, and inverted-dropout scaling makes the expected train-time
        // magnitude match inference-time magnitude.
        block_output *= &dropout_masks.feed_forward_residual_masks[layer_index];
    }
    Ok(block_output + residual_state)
}

fn run_multi_head_attention_batch(
    query: &Array,
    key: &Array,
    value: &Array,
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // Batched attention shape walkthrough:
    //
    // query/key/value start as [batch, sequence, embedding].
    // Reshape splits the embedding into heads:
    //   [batch, sequence, head, head_size]
    // Transpose puts heads before sequence:
    //   [batch, head, sequence, head_size]
    //
    // Then `query.matmul(key_transposed)` computes every query-position versus
    // every key-position score for every batch item and head in one operation:
    //   [batch, head, sequence, sequence]
    //
    // That is the core win over the earlier Rust loop: MLX sees one large tensor
    // problem instead of many tiny per-character dot products.
    let shape = query.shape();
    let batch_size = shape[0];
    let sequence_len = shape[1];
    let head_count = config.attention_head_count as i32;
    let head_size = config.attention_head_size as i32;

    let query = query
        .reshape(&[batch_size, sequence_len, head_count, head_size])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let key = key
        .reshape(&[batch_size, sequence_len, head_count, head_size])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let value = value
        .reshape(&[batch_size, sequence_len, head_count, head_size])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let key_transposed = key.transpose_axes(&[0, 1, 3, 2])?;
    let mut attention_scores = query.matmul(&key_transposed)?
        / Array::from_f32((config.attention_head_size as f32).sqrt());
    attention_scores += causal_attention_mask(sequence_len as usize);
    let attention_weights = ops::softmax_axis(&attention_scores, -1, None)?;
    attention_weights
        .matmul(&value)?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch_size, sequence_len, config.embedding_size as i32])
}

fn causal_attention_mask(sequence_len: usize) -> Array {
    // A language model must not look at future characters while predicting the
    // next one. The mask adds a huge negative number above the diagonal:
    //
    // row 0 can attend to column 0
    // row 1 can attend to columns 0..1
    // row 2 can attend to columns 0..2
    //
    // After softmax, those huge negative future scores become probability ~0.
    let values = (0..sequence_len)
        .flat_map(|query_position| {
            (0..sequence_len).map(move |key_position| {
                if key_position > query_position {
                    -1.0e9_f32
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Array::from_slice(&values, &[1, 1, sequence_len as i32, sequence_len as i32])
}

fn apply_rotary_position_embedding_batch(
    vectors: &Array,
    rotary_position_matrices: &[Array],
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // RoPE rotates query/key channels by a position-dependent angle. This
    // batched version stacks one rotation matrix per sequence position, then
    // multiplies every [batch, position] vector by that position's matrix.
    //
    // We rotate queries and keys, not values, because attention scores are dot
    // products between query/key. Applying the same rotation rule to both makes
    // those dot products sensitive to relative distance between characters.
    let shape = vectors.shape();
    let batch_size = shape[0];
    let sequence_len = shape[1];
    let rotary_stack = ops::stack_axis(rotary_position_matrices, 0)?
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[
            1,
            sequence_len,
            config.embedding_size as i32,
            config.embedding_size as i32,
        ])?;
    vectors
        .reshape(&[batch_size, sequence_len, 1, config.embedding_size as i32])?
        .matmul(&rotary_stack)?
        .reshape(&[batch_size, sequence_len, config.embedding_size as i32])
}

fn linear_last_axis(
    input: &Array,
    weights: &Array,
    biases: &Array,
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // Apply the same dense layer independently at every [batch, sequence]
    // location. `input` is [..., input_size], weights are [output_size,
    // input_size], so transposing weights lets MLX produce [..., output_size].
    let output = input.matmul(&weights.transpose()?)?;
    if config.features.use_learned_biases {
        Ok(output + biases)
    } else {
        Ok(output)
    }
}

fn language_model_logits_batch(
    hidden_state: &Array,
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // Tied output head: every final hidden vector is dotted against every token
    // embedding row. Result shape is [batch, sequence, vocabulary_size].
    let weights = if config.features.use_tied_output_embeddings {
        params.token_embedding
    } else {
        params.language_model_head
    };
    let output = hidden_state.matmul(&weights.transpose()?)?;
    if config.features.use_learned_biases {
        Ok(output + params.language_model_head_biases)
    } else {
        Ok(output)
    }
}

fn rmsnorm_last_axis(input: &Array, gain: &Array, config: &TransformerConfig) -> MlxResult<Array> {
    // Normalize only the feature/channel axis, not batch or time. For a hidden
    // vector [x1, x2, ...], RMSNorm divides by sqrt(mean(x_i^2) + epsilon), then
    // multiplies by the learned gain vector. The gain is the trainable "how loud
    // should this channel be after normalization?" parameter.
    let mean_square = ops::mean_axis(&ops::square(input)?, -1, true)?;
    let scale = (mean_square + Array::from_f32(1e-5)).sqrt()?;
    let output = input / scale;
    if config.features.use_learned_rmsnorm_gain {
        Ok(output * gain)
    } else {
        Ok(output)
    }
}

fn batch_dropout_masks(
    step: usize,
    batch_size: usize,
    sequence_len: usize,
    embedding_size: usize,
    layer_count: usize,
    dropout_probability: f32,
) -> BatchDropoutMasks {
    // Build deterministic dropout masks on the Rust side and hand them to MLX as
    // tensors. Deterministic means "recomputable from step/layer/index", not
    // "same every step". This keeps checkpoint resume simple: if training
    // resumes at step N, the same masks for step N are regenerated.
    let attention_residual_masks = (0..layer_count)
        .map(|layer_index| {
            dropout_mask_array(
                step,
                layer_index,
                0,
                batch_size,
                sequence_len,
                embedding_size,
                dropout_probability,
            )
        })
        .collect();
    let feed_forward_residual_masks = (0..layer_count)
        .map(|layer_index| {
            dropout_mask_array(
                step,
                layer_index,
                1,
                batch_size,
                sequence_len,
                embedding_size,
                dropout_probability,
            )
        })
        .collect();
    BatchDropoutMasks {
        attention_residual_masks,
        feed_forward_residual_masks,
    }
}

fn dropout_mask_array(
    step: usize,
    layer_index: usize,
    stream: usize,
    batch_size: usize,
    sequence_len: usize,
    embedding_size: usize,
    dropout_probability: f32,
) -> Array {
    // Inverted dropout mask:
    //
    // - dropped channels get 0
    // - kept channels get 1 / keep_probability
    //
    // The scale-up means average activation magnitude during training matches
    // inference, where dropout is disabled and every channel is kept.
    if dropout_probability <= 0.0 {
        return Array::from_slice(
            &vec![1.0_f32; batch_size * sequence_len * embedding_size],
            &[
                batch_size as i32,
                sequence_len as i32,
                embedding_size as i32,
            ],
        );
    }
    let keep_probability = 1.0 - dropout_probability;
    let scale = 1.0 / keep_probability;
    let values = (0..batch_size * sequence_len * embedding_size)
        .map(|index| {
            let random = deterministic_unit_float(step, layer_index, stream, index);
            if random < keep_probability {
                scale
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    Array::from_slice(
        &values,
        &[
            batch_size as i32,
            sequence_len as i32,
            embedding_size as i32,
        ],
    )
}

fn deterministic_unit_float(step: usize, layer_index: usize, stream: usize, index: usize) -> f32 {
    // Fast deterministic pseudo-random float in [0, 1). `stream` separates the
    // attention residual mask from the feed-forward residual mask so both blocks
    // do not drop the exact same channels.
    let mixed = splitmix64(
        (step as u64)
            .wrapping_mul(0xa076_1d64_78bd_642f)
            .wrapping_add((layer_index as u64).wrapping_mul(0xe703_7ed1_a0b4_28db))
            .wrapping_add((stream as u64).wrapping_mul(0x8ebc_6af0_9c88_c6e3))
            .wrapping_add(index as u64),
    );
    ((mixed >> 40) as f32) / ((1_u64 << 24) as f32)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn run_transformer_model(
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
    token_id: usize,
    position_id: usize,
    keys: Vec<Vec<Array>>,
    values: Vec<Vec<Array>>,
) -> MlxResult<TransformerRun> {
    // One-token forward pass. `index` gathers a row from the embedding tables;
    // the rest of the network is tensor math.
    //
    // Shapes:
    //   token_embedding table:    [vocab_size, embedding_size]
    //   token_embedding row:      [embedding_size]
    //   position_embedding table: [context_window_size, embedding_size]
    //   hidden_state:             [embedding_size]
    let mut hidden_state = params.token_embedding.index(token_id as i32);
    if config.features.use_learned_absolute_position_encoding {
        hidden_state += params.position_embedding.index(position_id as i32);
    }
    let mut current_keys = keys;
    let mut current_values = values;

    for layer_index in 0..config.layer_count {
        // Each layer receives the current KV cache and returns the updated cache.
        // The cache is stored as Rust Vecs of MLX Arrays because sequence length
        // grows one step at a time in this simple implementation.
        let layer_run = run_transformer_layer(
            &hidden_state,
            &params.layers[layer_index],
            layer_index,
            &params.rotary_position_matrices[position_id],
            config,
            current_keys,
            current_values,
        )?;
        hidden_state = layer_run.hidden_state;
        current_keys = layer_run.keys;
        current_values = layer_run.values;
    }
    if config.features.use_final_rmsnorm {
        hidden_state = rmsnorm(&hidden_state, params.final_norm_gain, config)?;
    }

    Ok(TransformerRun {
        logits: language_model_logits(&hidden_state, params, config)?,
        keys: current_keys,
        values: current_values,
    })
}

fn run_transformer_layer(
    hidden_state: &Array,
    layer: &MlxLayerParamView<'_>,
    layer_index: usize,
    rotary_position_matrix: &Array,
    config: &TransformerConfig,
    mut keys: Vec<Vec<Array>>,
    mut values: Vec<Vec<Array>>,
) -> MlxResult<TransformerLayerRun> {
    // Same pre-norm residual block as the CPU implementation, now expressed as
    // tensor operations. The residual stream is an Array of shape [embedding].
    let residual_state = hidden_state.clone();
    let normalized_state = rmsnorm(hidden_state, layer.attention_norm_gain, config)?;

    // `linear` returns [embedding_size]. The precomputed RoPE matrix keeps the
    // same shape while rotating pairs of values inside each attention head.
    let mut query = linear(
        &normalized_state,
        layer.attention.query_weights,
        layer.attention.query_biases,
        config,
    )?;
    let mut key = linear(
        &normalized_state,
        layer.attention.key_weights,
        layer.attention.key_biases,
        config,
    )?;
    if config.features.use_rope_position_encoding {
        query = apply_rotary_position_embedding(&query, rotary_position_matrix)?;
        key = apply_rotary_position_embedding(&key, rotary_position_matrix)?;
    }
    let value = linear(
        &normalized_state,
        layer.attention.value_weights,
        layer.attention.value_biases,
        config,
    )?;

    // Cache entries are per layer. At time step t, `keys[layer_index]` contains
    // t+1 tensors, each shaped [embedding_size].
    keys[layer_index].push(key);
    values[layer_index].push(value);

    let attention_output =
        run_multi_head_attention(&query, &keys[layer_index], &values[layer_index], config)?;
    let block_output = linear(
        &attention_output,
        layer.attention.output_projection_weights,
        layer.attention.output_projection_biases,
        config,
    )?;
    let mut updated_hidden_state = block_output + residual_state;

    let residual_state = updated_hidden_state.clone();
    let normalized_state = rmsnorm(&updated_hidden_state, layer.feed_forward_norm_gain, config)?;
    // Feed-forward tensor shapes:
    //   normalized_state: [embedding_size]
    //   expanded_output:  [3 * embedding_size]
    //   gated_output:     [3 * embedding_size]
    //   block_output:     [embedding_size] after projection
    let expanded_output = linear(
        &normalized_state,
        layer.feed_forward.expansion_weights,
        layer.feed_forward.expansion_biases,
        config,
    )?;
    let gated_output = linear(
        &normalized_state,
        layer.feed_forward.gate_weights,
        layer.feed_forward.gate_biases,
        config,
    )?;
    let block_output = if config.features.use_swiglu_feed_forward {
        silu(&expanded_output)? * gated_output
    } else {
        relu(&expanded_output)?
    };
    let block_output = linear(
        &block_output,
        layer.feed_forward.projection_weights,
        layer.feed_forward.projection_biases,
        config,
    )?;
    updated_hidden_state = block_output + residual_state;

    Ok(TransformerLayerRun {
        hidden_state: updated_hidden_state,
        keys,
        values,
    })
}

fn run_multi_head_attention(
    query: &Array,
    keys: &[Array],
    values: &[Array],
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // Arrange cached keys/values as:
    //   [head, time, head_size]
    // and the current query as:
    //   [head, 1, head_size]
    // Broadcasting then computes all head dot products at once.
    let head_count = config.attention_head_count as i32;
    let head_size = config.attention_head_size as i32;
    let time_steps = keys.len() as i32;

    // Before reshape, `keys` is a Rust Vec of `time_steps` arrays, each
    // [embedding_size]. Stacking on axis 0 creates [time, embedding]. Reshaping
    // splits embedding into heads: [time, head, head_size]. Transposing gives
    // [head, time, head_size], which makes the later reduction operate
    // independently for every head.
    let key_stack = ops::stack_axis(keys, 0)?
        .reshape(&[time_steps, head_count, head_size])?
        .transpose_axes(&[1, 0, 2])?;
    let value_stack = ops::stack_axis(values, 0)?
        .reshape(&[time_steps, head_count, head_size])?
        .transpose_axes(&[1, 0, 2])?;
    // Query is one vector for the current position. Reshape to [head, 1,
    // head_size] so MLX broadcasts the "1" time dimension across all cached
    // key time steps.
    let query = query.reshape(&[head_count, 1, head_size])?;

    // Elementwise multiply followed by `sum_axis(..., -1)` is batched dot
    // product over the head_size dimension. Result shape is [head, time].
    let scaled_dot_products = ops::sum_axis(&(query * &key_stack), -1, None)?
        / Array::from_f32((config.attention_head_size as f32).sqrt());
    // Softmax over axis 1 means "for each head, normalize across time steps".
    let attention_weights = ops::softmax_axis(&scaled_dot_products, 1, None)?;
    // Expand attention weights to [head, time, 1], multiply each value vector,
    // sum over time, then flatten [head, head_size] back to [embedding_size].
    let weighted_values = attention_weights.expand_dims(-1)? * value_stack;
    ops::sum_axis(&weighted_values, 1, None)?.reshape(&[config.embedding_size as i32])
}

fn linear(
    input_vector: &Array,
    weights: &Array,
    biases: &Array,
    config: &TransformerConfig,
) -> MlxResult<Array> {
    // MLX matmul is the high-performance version of the CPU backend's row-wise
    // dot-product loop. Adding the bias vector turns pure matrix multiplication
    // into the affine projection used by ordinary dense neural-network layers.
    let output = weights.matmul(input_vector)?;
    if config.features.use_learned_biases {
        Ok(output + biases)
    } else {
        Ok(output)
    }
}

fn language_model_logits(
    hidden_state: &Array,
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
) -> MlxResult<Array> {
    let weights = if config.features.use_tied_output_embeddings {
        params.token_embedding
    } else {
        params.language_model_head
    };
    let output = weights.matmul(hidden_state)?;
    if config.features.use_learned_biases {
        Ok(output + params.language_model_head_biases)
    } else {
        Ok(output)
    }
}

fn apply_rotary_position_embedding(
    vector: &Array,
    rotary_position_matrix: &Array,
) -> MlxResult<Array> {
    // A previous version built RoPE with scalar MLX indexing. That can panic
    // inside MLX's autodiff path for some shapes. The dense rotation matrix is a
    // little more work but keeps the operation as a plain differentiable matmul.
    // Matrix shape is [embedding_size, embedding_size]. Multiplying by the
    // [embedding_size] vector returns another [embedding_size] vector.
    rotary_position_matrix.matmul(vector)
}

fn rotary_position_matrices(config: &TransformerConfig) -> Vec<Array> {
    // RoPE matrices depend only on model shape and position, not on trainable
    // parameters. Build them once per loss/generation call instead of allocating
    // the same matrix for every layer at every time step.
    (0..config.context_window_size)
        .map(|position_id| rotary_position_matrix(position_id, config))
        .collect()
}

fn rotary_position_matrix(position_id: usize, config: &TransformerConfig) -> Array {
    // Build a block-diagonal matrix of 2-D rotations. Each pair [x_even, x_odd]
    // becomes:
    //
    // [ cos  -sin ] [x_even]
    // [ sin   cos ] [x_odd ]
    //
    // Different pair indexes use different frequencies, so some dimensions
    // track short distances while others change slowly across longer distances.
    let mut matrix = vec![0.0; config.embedding_size * config.embedding_size];
    let pair_count = config.attention_head_size / 2;
    for head_index in 0..config.attention_head_count {
        let head_start_index = head_index * config.attention_head_size;
        for pair_index in 0..pair_count {
            let even_index = head_start_index + 2 * pair_index;
            let odd_index = even_index + 1;
            let frequency =
                10_000.0_f32.powf(-((2 * pair_index) as f32) / config.attention_head_size as f32);
            let angle = position_id as f32 * frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            matrix[even_index * config.embedding_size + even_index] = cosine;
            matrix[even_index * config.embedding_size + odd_index] = -sine;
            matrix[odd_index * config.embedding_size + even_index] = sine;
            matrix[odd_index * config.embedding_size + odd_index] = cosine;
        }
        if config.attention_head_size % 2 == 1 {
            let last_index = head_start_index + config.attention_head_size - 1;
            matrix[last_index * config.embedding_size + last_index] = 1.0;
        }
    }
    Array::from_slice(
        &matrix,
        &[config.embedding_size as i32, config.embedding_size as i32],
    )
}

fn rmsnorm(input_vector: &Array, gain: &Array, config: &TransformerConfig) -> MlxResult<Array> {
    // Tensor RMSNorm. `None` means reduce over every element in this 1-D vector.
    let mean_square = ops::mean(&ops::square(input_vector)?, None)?;
    let scale = (mean_square + Array::from_f32(1e-5)).sqrt()?;
    let output = input_vector / scale;
    if config.features.use_learned_rmsnorm_gain {
        Ok(output * gain)
    } else {
        Ok(output)
    }
}

fn silu(input: &Array) -> MlxResult<Array> {
    // Tensor SiLU: x / (1 + exp(-x)).
    Ok(input / (Array::from_f32(1.0) + ops::exp(&(input * Array::from_f32(-1.0)))?))
}

fn relu(input: &Array) -> MlxResult<Array> {
    ops::maximum(input, &Array::from_f32(0.0))
}

fn apply_adam_update(
    parameters: &[Array],
    gradients: &[Array],
    optimizer_state: &MlxAdamOptimizerState,
    optimizer_config: &AdamOptimizerConfig,
    step: usize,
    training_step_count: usize,
) -> MlxResult<(Vec<Array>, MlxAdamOptimizerState)> {
    // Tensor AdamW mirrors the CPU scalar optimizer. Each parameter Array gets a
    // same-shaped first-moment tensor, second-moment tensor, gradient tensor,
    // and update tensor.
    let step_learning_rate = scheduled_learning_rate(optimizer_config, step, training_step_count);
    let beta1 = optimizer_config.first_moment_decay as f32;
    let beta2 = optimizer_config.second_moment_decay as f32;
    let learning_rate = Array::from_f32(step_learning_rate as f32);
    let epsilon = Array::from_f32(optimizer_config.epsilon as f32);
    let first_bias_correction = Array::from_f32(1.0 - beta1.powf(step as f32 + 1.0));
    let second_bias_correction = Array::from_f32(1.0 - beta2.powf(step as f32 + 1.0));
    // `gradient_scale` is a scalar Array, not a Rust f32. Keeping it as an Array
    // lets MLX apply it to every gradient tensor inside the same lazy graph.
    let gradient_scale = if optimizer_config.features.use_gradient_clipping {
        gradient_clip_scale(gradients, MAX_GRADIENT_NORM)?
    } else {
        Array::from_f32(1.0)
    };

    let mut updated_parameters = Vec::with_capacity(parameters.len());
    let mut first_moment_estimates = Vec::with_capacity(parameters.len());
    let mut second_moment_estimates = Vec::with_capacity(parameters.len());

    for (((parameter, gradient), first_moment), second_moment) in parameters
        .iter()
        .zip(gradients.iter())
        .zip(optimizer_state.first_moment_estimates.iter())
        .zip(optimizer_state.second_moment_estimates.iter())
    {
        // All arithmetic here is elementwise tensor arithmetic. If `parameter`
        // is [64, 64], every expression below is [64, 64] except scalar Arrays
        // such as `learning_rate`, which MLX broadcasts.
        let gradient = gradient * &gradient_scale;
        let new_first_moment =
            first_moment * Array::from_f32(beta1) + &gradient * Array::from_f32(1.0 - beta1);
        let new_second_moment = second_moment * Array::from_f32(beta2)
            + ops::square(&gradient)? * Array::from_f32(1.0 - beta2);
        let bias_corrected_first_moment = &new_first_moment / &first_bias_correction;
        let bias_corrected_second_moment = &new_second_moment / &second_bias_correction;
        let adam_update =
            bias_corrected_first_moment / (bias_corrected_second_moment.sqrt()? + &epsilon);
        let decay_update = if optimizer_config.features.use_weight_decay {
            parameter * Array::from_f32(optimizer_config.weight_decay as f32)
        } else {
            Array::from_f32(0.0)
        };
        let update = &learning_rate * (adam_update + decay_update);

        // `parameter - update` creates the next parameter tensor. It does not
        // mutate the old tensor in place; keeping old and new values separate is
        // simpler and works well with MLX's functional transform style.
        updated_parameters.push(parameter - update);
        first_moment_estimates.push(new_first_moment);
        second_moment_estimates.push(new_second_moment);
    }

    // Force all new optimizer tensors to be realized before returning. This
    // prevents a long chain of lazy updates from accumulating across training
    // steps and gives clearer failure points if MLX reports an error.
    transforms::eval(updated_parameters.iter())?;
    transforms::eval(first_moment_estimates.iter())?;
    transforms::eval(second_moment_estimates.iter())?;

    Ok((
        updated_parameters,
        MlxAdamOptimizerState {
            first_moment_estimates,
            second_moment_estimates,
        },
    ))
}

fn gradient_clip_scale(gradients: &[Array], max_norm: f32) -> MlxResult<Array> {
    // Compute one global norm across all parameter tensors. Returning a scalar
    // scale tensor lets MLX multiply every gradient by the same factor.
    let squared_norms = gradients
        .iter()
        .map(|gradient| ops::sum(&ops::square(gradient)?, None))
        .collect::<MlxResult<Vec<_>>>()?;
    // `squared_norms` is a Rust Vec of scalar Arrays. Stack them into a 1-D
    // tensor, sum again to get total squared length, then compute
    // max_norm / norm. `ops::clip(scale, ((), 1.0))` means "no lower bound,
    // upper bound 1.0", so small gradients keep scale 1 and large gradients
    // shrink.
    let total_squared_norm = ops::sum(&ops::stack_axis(&squared_norms, 0)?, None)?;
    let scale = Array::from_f32(max_norm) / (total_squared_norm.sqrt()? + Array::from_f32(1e-12));
    ops::clip(scale, ((), 1.0))
}

fn create_key_value_cache(layer_count: usize) -> Vec<Vec<Array>> {
    // The outer Vec is one entry per layer. The inner Vec grows over time steps.
    // Each stored Array is a key or value vector shaped [embedding_size].
    vec![Vec::new(); layer_count]
}

fn mlx_matrix(
    output_size: usize,
    input_size: usize,
    random_number_generator: &mut impl Rng,
    standard_deviation: f64,
) -> Array {
    // Initialize on CPU, then hand the contiguous f32 buffer to MLX. Keeping
    // initialization deterministic in Rust makes CPU/MLX comparisons easier.
    let data = (0..output_size * input_size)
        .map(|_| random_gaussian(random_number_generator, 0.0, standard_deviation) as f32)
        .collect::<Vec<_>>();
    Array::from_slice(&data, &[output_size as i32, input_size as i32])
}

fn mlx_zero_matrix(output_size: usize, input_size: usize) -> Array {
    Array::from_slice(
        &vec![0.0_f32; output_size * input_size],
        &[output_size as i32, input_size as i32],
    )
}

fn mlx_zero_vector(size: usize) -> Array {
    Array::from_slice(&vec![0.0_f32; size], &[size as i32])
}

fn mlx_one_vector(size: usize) -> Array {
    Array::from_slice(&vec![1.0_f32; size], &[size as i32])
}

fn arrays_to_checkpoint_tensors(arrays: &[Array]) -> Result<Vec<CheckpointTensor>, String> {
    arrays
        .iter()
        .map(array_to_checkpoint_tensor)
        .collect::<Result<Vec<_>, _>>()
}

fn array_to_checkpoint_tensor(array: &Array) -> Result<CheckpointTensor, String> {
    array.eval().map_err(|error| error.to_string())?;
    let shape = array
        .shape()
        .iter()
        .map(|dimension| *dimension as usize)
        .collect::<Vec<_>>();
    let values = array
        .as_slice::<f32>()
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    CheckpointTensor::new(shape, values)
}

fn checkpoint_tensors_to_arrays(
    tensors: &[CheckpointTensor],
    expected_tensors: &[CheckpointTensor],
) -> Result<Vec<Array>, String> {
    if tensors.len() != expected_tensors.len() {
        return Err(format!(
            "checkpoint has {} tensors, expected {}",
            tensors.len(),
            expected_tensors.len()
        ));
    }

    tensors
        .iter()
        .zip(expected_tensors.iter())
        .enumerate()
        .map(|(tensor_index, (tensor, expected))| {
            if tensor.shape != expected.shape {
                return Err(format!(
                    "checkpoint tensor {tensor_index} has shape {:?}, expected {:?}",
                    tensor.shape, expected.shape
                ));
            }
            let expected_value_count = tensor.shape.iter().product::<usize>();
            if tensor.values.len() != expected_value_count {
                return Err(format!(
                    "checkpoint tensor {tensor_index} has {} values, expected {expected_value_count}",
                    tensor.values.len()
                ));
            }
            let values = tensor
                .values
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let shape = tensor
                .shape
                .iter()
                .map(|dimension| *dimension as i32)
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&values, shape.as_slice()))
        })
        .collect()
}

fn weighted_choice(weights: &[f64], random_number_generator: &mut impl Rng) -> usize {
    // Sampling is deliberately kept in Rust, not MLX, because the vocabulary is
    // tiny and the CPU backend uses the same helper. That makes generated output
    // behavior consistent across backends.
    let total: f64 = weights.iter().sum();
    let mut random_threshold = random_number_generator.gen::<f64>() * total;
    for (weight_index, weight) in weights.iter().enumerate() {
        random_threshold -= weight;
        if random_threshold <= 0.0 {
            return weight_index;
        }
    }
    weights.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microgpt::OptimizerFeatureConfig;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn mlx_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("MLX test lock should not be poisoned")
    }

    #[test]
    fn mlx_can_train_one_tiny_step_and_generate() {
        let _guard = mlx_test_lock();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let config = TransformerConfig::new(1, 8, 12, 2).unwrap();
        let optimizer = AdamOptimizerConfig {
            learning_rate: 0.006,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            warmup_step_count: 0,
            minimum_learning_rate_ratio: 0.0,
            features: OptimizerFeatureConfig::optimized_defaults(),
        };
        let session = create_mlx_microgpt_training_session(
            vec!["anna".into(), "anne".into(), "emma".into(), "ella".into()],
            &mut rng,
            2,
            4,
            1,
            config,
            optimizer,
        )
        .with_initial_progress()
        .unwrap();
        let result = train_mlx_microgpt_step(session, 2)
            .unwrap()
            .expect("step should run");
        let sample = generate_sample(
            &result.session.trained_microgpt.model,
            &result.session.trained_microgpt.config,
            &result.session.trained_microgpt.tokenizer,
            "a",
            0.8,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.progress.completed_step_count, 1);
        assert!(!sample.is_empty());
    }

    #[test]
    fn mlx_rope_handles_app_sized_attention_inside_training() {
        let _guard = mlx_test_lock();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let config = TransformerConfig::new(2, 64, 20, 16).unwrap();
        let optimizer = AdamOptimizerConfig {
            learning_rate: 0.001,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            warmup_step_count: 0,
            minimum_learning_rate_ratio: 0.0,
            features: OptimizerFeatureConfig::optimized_defaults(),
        };
        let session = create_mlx_microgpt_training_session(
            vec![
                "a small cat sat".into(),
                "a red hen ran".into(),
                "the sun is up".into(),
                "we can see it".into(),
            ],
            &mut rng,
            1,
            4,
            1,
            config,
            optimizer,
        )
        .with_initial_progress()
        .unwrap();
        let result = train_mlx_microgpt_step(session, 1)
            .unwrap()
            .expect("step should run without indexing panic");

        assert_eq!(result.progress.completed_step_count, 1);
    }

    #[test]
    fn mlx_checkpoint_restores_training_progress_and_parameters() {
        let _guard = mlx_test_lock();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let config = TransformerConfig::new(1, 8, 12, 2).unwrap();
        let optimizer = AdamOptimizerConfig {
            learning_rate: 0.006,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            warmup_step_count: 0,
            minimum_learning_rate_ratio: 0.0,
            features: OptimizerFeatureConfig::optimized_defaults(),
        };
        let session = create_mlx_microgpt_training_session(
            vec!["anna".into(), "anne".into(), "emma".into(), "ella".into()],
            &mut rng,
            3,
            4,
            1,
            config,
            optimizer,
        )
        .with_initial_progress()
        .unwrap();
        let session = train_mlx_microgpt_step(session, 2)
            .unwrap()
            .expect("training step should run")
            .session;

        let checkpoint = export_training_session_checkpoint(&session)
            .expect("checkpoint export should read MLX tensors");
        let restored =
            import_training_session_checkpoint(&checkpoint).expect("checkpoint should restore");

        assert_eq!(restored.completed_step_count, session.completed_step_count);
        assert_eq!(
            restored.progress_history.len(),
            session.progress_history.len()
        );
        assert_eq!(
            restored.trained_microgpt.model.values().len(),
            session.trained_microgpt.model.values().len()
        );
        assert_eq!(
            restored.optimizer_state.first_moment_estimates.len(),
            session.optimizer_state.first_moment_estimates.len()
        );
    }
}
