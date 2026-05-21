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

use crate::microgpt::{
    apply_sampling_constraints, normalize_training_document, random_gaussian,
    scheduled_learning_rate, shuffled_by, AdamOptimizerConfig, CharacterTokenizer,
    MicrogptTrainingProgress, TransformerConfig,
};
// `ops` contains tensor operations such as softmax, reductions, stacking, and
// elementwise math. `transforms` contains autodiff transforms such as
// `value_and_grad`. `IndexOp` provides row/token lookup syntax for embedding
// tables and target-token losses.
use mlx_rs::{error::Result as MlxResult, ops, ops::indexing::IndexOp, transforms, Array};
use rand::Rng;

const MAX_GRADIENT_NORM: f32 = 1.0;

#[derive(Clone, Debug)]
pub struct MlxAttentionParameters {
    // Shape convention matches the CPU backend:
    //   [output_size, input_size]
    // For attention projections here that is [embedding_size, embedding_size].
    pub query_weights: Array,
    pub key_weights: Array,
    pub value_weights: Array,
    pub output_projection_weights: Array,
}

#[derive(Clone, Debug)]
pub struct MlxFeedForwardParameters {
    // SwiGLU feed-forward shapes:
    //   expansion_weights: [3 * embedding_size, embedding_size]
    //   gate_weights:      [3 * embedding_size, embedding_size]
    //   projection_weights:[embedding_size, 3 * embedding_size]
    pub expansion_weights: Array,
    pub gate_weights: Array,
    pub projection_weights: Array,
}

#[derive(Clone, Debug)]
pub struct MlxTransformerLayerParameters {
    pub attention: MlxAttentionParameters,
    pub feed_forward: MlxFeedForwardParameters,
}

#[derive(Clone, Debug)]
pub struct MlxTransformerModelParameters {
    // These fields mirror `TransformerModelParameters`, but each matrix is a
    // single MLX tensor. Keeping the shapes and names aligned makes it easier to
    // compare CPU and MLX behavior.
    pub token_embedding: Array,
    pub position_embedding: Array,
    pub language_model_head: Array,
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
            layers: (0..layer_count)
                .map(|_| MlxTransformerLayerParameters {
                    attention: MlxAttentionParameters {
                        query_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        key_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        value_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        output_projection_weights: mlx_matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            residual_projection_std,
                        ),
                    },
                    feed_forward: MlxFeedForwardParameters {
                        expansion_weights: mlx_matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        gate_weights: mlx_matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        projection_weights: mlx_matrix(
                            embedding_size,
                            feed_forward_size,
                            random_number_generator,
                            residual_projection_std,
                        ),
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
        ];
        for layer in &self.layers {
            values.push(layer.attention.query_weights.clone());
            values.push(layer.attention.key_weights.clone());
            values.push(layer.attention.value_weights.clone());
            values.push(layer.attention.output_projection_weights.clone());
            values.push(layer.feed_forward.expansion_weights.clone());
            values.push(layer.feed_forward.gate_weights.clone());
            values.push(layer.feed_forward.projection_weights.clone());
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
            layers: self
                .layers
                .iter()
                .map(|_| MlxTransformerLayerParameters {
                    attention: MlxAttentionParameters {
                        query_weights: next(),
                        key_weights: next(),
                        value_weights: next(),
                        output_projection_weights: next(),
                    },
                    feed_forward: MlxFeedForwardParameters {
                        expansion_weights: next(),
                        gate_weights: next(),
                        projection_weights: next(),
                    },
                })
                .collect(),
        }
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

#[derive(Clone, Debug)]
pub struct MlxMicrogptTrainingStepResult {
    pub session: MlxMicrogptTrainingSession,
    pub progress: MicrogptTrainingProgress,
    pub progress_history: Vec<MicrogptTrainingProgress>,
}

#[derive(Clone, Debug)]
pub struct MlxMatrixSummary {
    pub label: String,
    pub rows: usize,
    pub columns: usize,
    pub min: f32,
    pub max: f32,
    pub mean_abs: f32,
}

#[derive(Clone, Debug)]
pub struct MlxMatrixHeatmap {
    pub label: String,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
    pub min: f32,
    pub max: f32,
    pub mean_abs: f32,
}

struct MlxParamView<'a> {
    // Borrowed view used only inside a differentiable MLX closure. It lets us
    // treat the flat `inputs: &[Array]` from `value_and_grad` as a named model
    // without cloning every tensor.
    token_embedding: &'a Array,
    position_embedding: &'a Array,
    language_model_head: &'a Array,
    rotary_position_matrices: &'a [Array],
    layers: Vec<MlxLayerParamView<'a>>,
}

struct MlxLayerParamView<'a> {
    // These borrowed views are not long-lived model objects. They only give
    // names to entries in MLX's flat input slice while the loss closure runs.
    attention: MlxAttentionParamView<'a>,
    feed_forward: MlxFeedForwardParamView<'a>,
}

struct MlxAttentionParamView<'a> {
    query_weights: &'a Array,
    key_weights: &'a Array,
    value_weights: &'a Array,
    output_projection_weights: &'a Array,
}

struct MlxFeedForwardParamView<'a> {
    expansion_weights: &'a Array,
    gate_weights: &'a Array,
    projection_weights: &'a Array,
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

    let mut unique_characters: Vec<char> = shuffled_documents
        .iter()
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
    let batch_documents = training_batch_documents(&session.documents, step, batch_document_count);
    // Tokens remain normal Rust `usize` values. We only convert model parameters
    // and activations into MLX tensors; the training text itself is small and
    // easier to keep in ordinary Rust collections.
    let batch_tokens = batch_documents
        .iter()
        .map(|document| session.trained_microgpt.tokenizer.encode_document(document))
        .collect::<Vec<_>>();
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

    let loss_fn = move |inputs: &[Array]| -> MlxResult<Vec<Array>> {
        // MLX closures receive a flat parameter list. Convert it back into a
        // named view, compute scalar batch loss, and return it as a one-element
        // vector because the transform API is vector-valued.
        let params = params_from_arrays(inputs, layer_count, &rotary_position_matrices);
        let loss = batch_loss(
            &params,
            &config,
            &batch_tokens,
            session
                .trained_microgpt
                .tokenizer
                .sequence_boundary_token_id,
        )?;
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
    let mut losses = Vec::with_capacity(validation_document_count);
    for validation_offset in 0..validation_document_count {
        let validation_index = (validation_batch_index * validation_document_count
            + validation_offset)
            % session.validation_documents.len();
        losses.push(calculate_document_loss(
            &session.trained_microgpt.model,
            &session.trained_microgpt.config,
            &session.trained_microgpt.tokenizer,
            &session.validation_documents[validation_index],
        )?);
    }
    Ok(Some(losses.iter().sum::<f64>() / losses.len() as f64))
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
    let tokens = tokenizer.encode_document(document);
    let model_values = model.values();
    let rotary_position_matrices = rotary_position_matrices(config);
    let params = params_from_arrays(&model_values, model.layers.len(), &rotary_position_matrices);
    Ok(document_loss(
        &params,
        config,
        &tokens,
        tokenizer.sequence_boundary_token_id,
    )?
    .item::<f32>() as f64)
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
        .to_lowercase()
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

pub fn matrix_summaries(model: &MlxTransformerModelParameters) -> Vec<MlxMatrixSummary> {
    // Summaries are cheap-ish scalar diagnostics for the TUI: shape plus rough
    // value scale. They help spot exploding weights without drawing every cell.
    let mut summaries = vec![
        matrix_summary("Token embedding", &model.token_embedding),
        matrix_summary("Position embedding", &model.position_embedding),
        matrix_summary("Language head", &model.language_model_head),
    ];
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let prefix = format!("Layer {}", layer_index + 1);
        summaries.push(matrix_summary(
            &format!("{prefix} Q"),
            &layer.attention.query_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} K"),
            &layer.attention.key_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} V"),
            &layer.attention.value_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} Attn out"),
            &layer.attention.output_projection_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF expand"),
            &layer.feed_forward.expansion_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF gate"),
            &layer.feed_forward.gate_weights,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF project"),
            &layer.feed_forward.projection_weights,
        ));
    }
    summaries
}

pub fn matrix_heatmaps(model: &MlxTransformerModelParameters) -> Vec<MlxMatrixHeatmap> {
    // Heatmaps copy full parameter arrays back to CPU for visualization. That is
    // useful for learning but expensive during training, so the UI keeps this
    // hidden by default.
    let mut heatmaps = vec![
        matrix_heatmap("Token embedding", &model.token_embedding),
        matrix_heatmap("Position embedding", &model.position_embedding),
        matrix_heatmap("Language head", &model.language_model_head),
    ];
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let prefix = format!("Layer {}", layer_index + 1);
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} Q"),
            &layer.attention.query_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} K"),
            &layer.attention.key_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} V"),
            &layer.attention.value_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} Attn out"),
            &layer.attention.output_projection_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} FF expand"),
            &layer.feed_forward.expansion_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} FF gate"),
            &layer.feed_forward.gate_weights,
        ));
        heatmaps.push(matrix_heatmap(
            &format!("{prefix} FF project"),
            &layer.feed_forward.projection_weights,
        ));
    }
    heatmaps
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

    MlxParamView {
        token_embedding: next(),
        position_embedding: next(),
        language_model_head: next(),
        rotary_position_matrices,
        layers: (0..layer_count)
            .map(|_| MlxLayerParamView {
                attention: MlxAttentionParamView {
                    query_weights: next(),
                    key_weights: next(),
                    value_weights: next(),
                    output_projection_weights: next(),
                },
                feed_forward: MlxFeedForwardParamView {
                    expansion_weights: next(),
                    gate_weights: next(),
                    projection_weights: next(),
                },
            })
            .collect(),
    }
}

fn batch_loss(
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
    batch_tokens: &[Vec<usize>],
    sequence_boundary_token_id: usize,
) -> MlxResult<Array> {
    // Stack one scalar loss per document, then average. The result is a scalar
    // tensor so MLX can differentiate it with respect to parameter tensors.
    let mut losses = Vec::with_capacity(batch_tokens.len());
    for tokens in batch_tokens {
        // Each document loss is scalar-shaped. `stack_axis(&losses, 0)` turns
        // N scalar losses into a length-N tensor, then `mean` makes one scalar
        // batch loss for autodiff.
        losses.push(document_loss(
            params,
            config,
            tokens,
            sequence_boundary_token_id,
        )?);
    }
    ops::mean(&ops::stack_axis(&losses, 0)?, None)
}

fn document_loss(
    params: &MlxParamView<'_>,
    config: &TransformerConfig,
    tokens: &[usize],
    _sequence_boundary_token_id: usize,
) -> MlxResult<Array> {
    // Same teacher-forcing objective as the CPU backend. `logsumexp` implements
    // stable cross-entropy: log(sum(exp(logits))) - target_logit.
    let prediction_step_count = config
        .context_window_size
        .min(tokens.len().saturating_sub(1));
    let mut keys = create_key_value_cache(config.layer_count);
    let mut values = create_key_value_cache(config.layer_count);
    let mut losses = Vec::with_capacity(prediction_step_count);

    for position_id in 0..prediction_step_count {
        let token_id = tokens[position_id];
        let target_token_id = tokens[position_id + 1];
        let run = run_transformer_model(params, config, token_id, position_id, keys, values)?;
        keys = run.keys;
        values = run.values;
        // `ops::logsumexp(&run.logits, None)` reduces all vocabulary logits to
        // one scalar. `None` means "all axes"; logits are 1-D here. Indexing the
        // target log-prob gives the loss for the correct next character.
        let log_probabilities = &run.logits - ops::logsumexp(&run.logits, None)?;
        losses.push(-log_probabilities.index(target_token_id as i32));
    }

    ops::mean(&ops::stack_axis(&losses, 0)?, None)
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
    let token_embedding = params.token_embedding.index(token_id as i32);
    let position_embedding = params.position_embedding.index(position_id as i32);
    let mut hidden_state = token_embedding + position_embedding;
    hidden_state = rmsnorm(&hidden_state)?;
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

    Ok(TransformerRun {
        logits: linear(&hidden_state, params.language_model_head)?,
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
    let normalized_state = rmsnorm(hidden_state)?;

    // `linear` returns [embedding_size]. The precomputed RoPE matrix keeps the
    // same shape while rotating pairs of values inside each attention head.
    let query = apply_rotary_position_embedding(
        &linear(&normalized_state, layer.attention.query_weights)?,
        rotary_position_matrix,
    )?;
    let key = apply_rotary_position_embedding(
        &linear(&normalized_state, layer.attention.key_weights)?,
        rotary_position_matrix,
    )?;
    let value = linear(&normalized_state, layer.attention.value_weights)?;

    // Cache entries are per layer. At time step t, `keys[layer_index]` contains
    // t+1 tensors, each shaped [embedding_size].
    keys[layer_index].push(key);
    values[layer_index].push(value);

    let attention_output =
        run_multi_head_attention(&query, &keys[layer_index], &values[layer_index], config)?;
    let block_output = linear(&attention_output, layer.attention.output_projection_weights)?;
    let mut updated_hidden_state = block_output + residual_state;

    let residual_state = updated_hidden_state.clone();
    let normalized_state = rmsnorm(&updated_hidden_state)?;
    // Feed-forward tensor shapes:
    //   normalized_state: [embedding_size]
    //   expanded_output:  [3 * embedding_size]
    //   gated_output:     [3 * embedding_size]
    //   block_output:     [embedding_size] after projection
    let expanded_output = linear(&normalized_state, layer.feed_forward.expansion_weights)?;
    let gated_output = linear(&normalized_state, layer.feed_forward.gate_weights)?;
    let block_output = silu(&expanded_output)? * gated_output;
    let block_output = linear(&block_output, layer.feed_forward.projection_weights)?;
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

fn linear(input_vector: &Array, weights: &Array) -> MlxResult<Array> {
    // MLX matmul is the high-performance version of the CPU backend's row-wise
    // dot-product loop.
    weights.matmul(input_vector)
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

fn rmsnorm(input_vector: &Array) -> MlxResult<Array> {
    // Tensor RMSNorm. `None` means reduce over every element in this 1-D vector.
    let mean_square = ops::mean(&ops::square(input_vector)?, None)?;
    let scale = (mean_square + Array::from_f32(1e-5)).sqrt()?;
    Ok(input_vector / scale)
}

fn silu(input: &Array) -> MlxResult<Array> {
    // Tensor SiLU: x / (1 + exp(-x)).
    Ok(input / (Array::from_f32(1.0) + ops::exp(&(input * Array::from_f32(-1.0)))?))
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
    let gradient_scale = gradient_clip_scale(gradients, MAX_GRADIENT_NORM)?;

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
        let decay_update = parameter * Array::from_f32(optimizer_config.weight_decay as f32);
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

fn training_batch_documents(
    documents: &[String],
    step: usize,
    batch_document_count: usize,
) -> Vec<String> {
    let batch_start_index = (step * batch_document_count) % documents.len();
    (0..batch_document_count)
        .map(|batch_offset| documents[(batch_start_index + batch_offset) % documents.len()].clone())
        .collect()
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

fn matrix_summary(label: &str, matrix: &Array) -> MlxMatrixSummary {
    // `as_slice` needs concrete host-visible values, so call `eval` first. This
    // is exactly why visualizing values has a runtime cost: it synchronizes MLX
    // work and copies/reads tensor data on the CPU side.
    matrix.eval().unwrap();
    let shape = matrix.shape();
    let data = matrix.as_slice::<f32>();
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs = data.iter().map(|value| value.abs()).sum::<f32>() / data.len().max(1) as f32;
    MlxMatrixSummary {
        label: label.into(),
        rows: shape.first().copied().unwrap_or(0) as usize,
        columns: shape.get(1).copied().unwrap_or(0) as usize,
        min,
        max,
        mean_abs,
    }
}

fn matrix_heatmap(label: &str, matrix: &Array) -> MlxMatrixHeatmap {
    // Heatmaps need every value, not just aggregates. For large production
    // models this kind of full tensor readback would be prohibitive; here it is
    // acceptable only because the model is tiny and visualization is opt-in.
    matrix.eval().unwrap();
    let shape = matrix.shape();
    let values = matrix.as_slice::<f32>().to_vec();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f32>() / values.len().max(1) as f32;
    MlxMatrixHeatmap {
        label: label.into(),
        rows: shape.first().copied().unwrap_or(0) as usize,
        columns: shape.get(1).copied().unwrap_or(0) as usize,
        values,
        min,
        max,
        mean_abs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn mlx_can_train_one_tiny_step_and_generate() {
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
}
