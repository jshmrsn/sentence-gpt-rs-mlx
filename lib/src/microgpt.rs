//! Plain Rust reference implementation of a tiny character-level Transformer.
//!
//! This file intentionally keeps the math direct instead of hiding it behind a
//! neural-network framework. A training step is:
//!
//! 1. Convert each sentence into token ids.
//! 2. Run the Transformer forward one character at a time.
//! 3. Measure how surprised the model was with cross-entropy loss.
//! 4. Use reverse-mode autodiff to get one gradient per parameter.
//! 5. Use AdamW to move parameters in the direction that lowers loss.
//!
//! The model is small enough to read end-to-end, but it uses real production
//! ideas: embeddings, residual connections, RMSNorm, multi-head attention,
//! rotary position embeddings, SwiGLU feed-forward blocks, gradient clipping,
//! weight decay, warmup, cosine learning-rate decay, validation loss, and
//! constrained sampling.

use crate::value::Value;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::f64::consts::PI;

// A matrix is stored as rows of scalar autodiff Values. The shape convention in
// this file is `[output_size][input_size]`, so `linear(input, weights)` computes
// one dot product per row.
pub type Matrix = Vec<Vec<Value>>;
// During autoregressive generation/training, each layer remembers all previous
// keys and values. This is the same idea as a KV cache in production LLM
// inference, just stored in plain Rust vectors.
pub type KeyValueCache = Vec<Vec<Vec<Value>>>;

// Gradient clipping limits one bad batch from producing a huge update. Top-k
// sampling and the minimum length rule make short demo samples easier to read.
const MAX_GRADIENT_NORM: f64 = 1.0;
const SAMPLING_TOP_K: usize = 8;
const MIN_GENERATED_CHARACTER_COUNT: usize = 8;

#[derive(Clone, Debug)]
pub struct AttentionParameters {
    // Query, key, and value are three learned projections of the same hidden
    // state. Attention compares query to keys, then mixes values.
    pub query_weights: Matrix,
    pub key_weights: Matrix,
    pub value_weights: Matrix,
    // After all heads are concatenated, this projects the result back into the
    // normal embedding size so the residual stream shape stays constant.
    pub output_projection_weights: Matrix,
}

#[derive(Clone, Debug)]
pub struct FeedForwardParameters {
    // SwiGLU uses two parallel projections. `expansion_weights` creates candidate
    // features; `gate_weights` decides which candidate features should pass.
    pub expansion_weights: Matrix,
    pub gate_weights: Matrix,
    // Projection returns the wider feed-forward vector to the embedding size.
    pub projection_weights: Matrix,
}

#[derive(Clone, Debug)]
pub struct TransformerLayerParameters {
    pub attention: AttentionParameters,
    pub feed_forward: FeedForwardParameters,
}

#[derive(Clone, Debug)]
pub struct TransformerModelParameters {
    // Token embeddings answer: "what character is this?"
    pub token_embedding: Matrix,
    // Learned absolute position embeddings answer: "where are we in the
    // sequence?" RoPE below also injects position into attention comparisons.
    pub position_embedding: Matrix,
    // The language-model head maps the final hidden state to one score per next
    // character. Those scores are logits, not probabilities yet.
    pub language_model_head: Matrix,
    pub layers: Vec<TransformerLayerParameters>,
}

impl TransformerModelParameters {
    pub fn initialize(
        vocabulary_size: usize,
        context_window_size: usize,
        embedding_size: usize,
        layer_count: usize,
        random_number_generator: &mut impl Rng,
    ) -> Self {
        // Good initialization keeps early activations from exploding or
        // collapsing. Small embeddings keep early token vectors gentle; projection
        // std scales with hidden size; residual projections shrink as layers grow
        // so many residual additions do not swamp the signal.
        let embedding_std = 0.02;
        let feed_forward_size = 3 * embedding_size;
        let projection_std = (1.0 / embedding_size as f64).sqrt();
        let residual_projection_std = projection_std / (2.0 * layer_count as f64).sqrt();

        Self {
            token_embedding: matrix(
                vocabulary_size,
                embedding_size,
                random_number_generator,
                embedding_std,
            ),
            position_embedding: matrix(
                context_window_size,
                embedding_size,
                random_number_generator,
                embedding_std,
            ),
            language_model_head: matrix(
                vocabulary_size,
                embedding_size,
                random_number_generator,
                projection_std,
            ),
            layers: (0..layer_count)
                .map(|_| TransformerLayerParameters {
                    attention: AttentionParameters {
                        query_weights: matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        key_weights: matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        value_weights: matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        output_projection_weights: matrix(
                            embedding_size,
                            embedding_size,
                            random_number_generator,
                            residual_projection_std,
                        ),
                    },
                    feed_forward: FeedForwardParameters {
                        expansion_weights: matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        gate_weights: matrix(
                            feed_forward_size,
                            embedding_size,
                            random_number_generator,
                            projection_std,
                        ),
                        projection_weights: matrix(
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

    // Flatten every trainable scalar into a stable order. Optimizers and
    // autodiff gradients operate on flat lists; `with_values` below rebuilds the
    // structured model afterward.
    pub fn values(&self) -> Vec<Value> {
        let mut values = Vec::new();
        push_matrix_values(&mut values, &self.token_embedding);
        push_matrix_values(&mut values, &self.position_embedding);
        push_matrix_values(&mut values, &self.language_model_head);
        for layer in &self.layers {
            push_matrix_values(&mut values, &layer.attention.query_weights);
            push_matrix_values(&mut values, &layer.attention.key_weights);
            push_matrix_values(&mut values, &layer.attention.value_weights);
            push_matrix_values(&mut values, &layer.attention.output_projection_weights);
            push_matrix_values(&mut values, &layer.feed_forward.expansion_weights);
            push_matrix_values(&mut values, &layer.feed_forward.gate_weights);
            push_matrix_values(&mut values, &layer.feed_forward.projection_weights);
        }
        values
    }

    // Rebuild the nested model with replacement parameter Values in the exact
    // same order produced by `values`.
    fn with_values(&self, values: Vec<Value>) -> Self {
        let mut value_index = 0;
        let mut next_matrix = |matrix: &Matrix| {
            matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|_| {
                            let value = values[value_index].clone();
                            value_index += 1;
                            value
                        })
                        .collect()
                })
                .collect()
        };

        Self {
            token_embedding: next_matrix(&self.token_embedding),
            position_embedding: next_matrix(&self.position_embedding),
            language_model_head: next_matrix(&self.language_model_head),
            layers: self
                .layers
                .iter()
                .map(|layer| TransformerLayerParameters {
                    attention: AttentionParameters {
                        query_weights: next_matrix(&layer.attention.query_weights),
                        key_weights: next_matrix(&layer.attention.key_weights),
                        value_weights: next_matrix(&layer.attention.value_weights),
                        output_projection_weights: next_matrix(
                            &layer.attention.output_projection_weights,
                        ),
                    },
                    feed_forward: FeedForwardParameters {
                        expansion_weights: next_matrix(&layer.feed_forward.expansion_weights),
                        gate_weights: next_matrix(&layer.feed_forward.gate_weights),
                        projection_weights: next_matrix(&layer.feed_forward.projection_weights),
                    },
                })
                .collect(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.values().len()
    }
}

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    // Number of repeated Transformer blocks.
    pub layer_count: usize,
    // Width of every token/hidden vector. Larger width increases capacity and
    // compute.
    pub embedding_size: usize,
    // Maximum number of characters the model can condition on in one pass.
    pub context_window_size: usize,
    // Attention splits the embedding into heads. Each head can learn a different
    // kind of relationship, such as "previous word starts here" or "copy this
    // vowel pattern".
    pub attention_head_count: usize,
    pub attention_head_size: usize,
}

impl TransformerConfig {
    pub fn new(
        layer_count: usize,
        embedding_size: usize,
        context_window_size: usize,
        attention_head_count: usize,
    ) -> Result<Self, String> {
        if layer_count == 0 {
            return Err("layer_count must be positive".into());
        }
        if embedding_size == 0 {
            return Err("embedding_size must be positive".into());
        }
        if context_window_size == 0 {
            return Err("context_window_size must be positive".into());
        }
        if attention_head_count == 0 {
            return Err("attention_head_count must be positive".into());
        }
        if embedding_size % attention_head_count != 0 {
            return Err("embedding_size must be divisible by attention_head_count".into());
        }
        Ok(Self {
            layer_count,
            embedding_size,
            context_window_size,
            attention_head_count,
            attention_head_size: embedding_size / attention_head_count,
        })
    }
}

#[derive(Clone, Debug)]
pub struct CharacterTokenizer {
    // This toy model deliberately uses characters, not subwords. That makes the
    // vocabulary tiny and easy to inspect, but it makes long-range language
    // harder because words span many time steps.
    pub unique_characters: Vec<char>,
    // A single special token marks both start-of-sequence and end-of-sequence.
    pub sequence_boundary_token_id: usize,
    pub character_to_token_id: HashMap<char, usize>,
}

impl CharacterTokenizer {
    pub fn new(unique_characters: Vec<char>, sequence_boundary_token_id: usize) -> Self {
        let character_to_token_id = unique_characters
            .iter()
            .copied()
            .enumerate()
            .map(|(index, character)| (character, index))
            .collect();
        Self {
            unique_characters,
            sequence_boundary_token_id,
            character_to_token_id,
        }
    }

    pub fn vocabulary_size(&self) -> usize {
        self.unique_characters.len() + 1
    }

    pub fn encode_document(&self, document: &str) -> Vec<usize> {
        // The boundary token at the start teaches the model how sentences begin.
        // The boundary token at the end teaches it when to stop generating.
        let mut encoded = Vec::with_capacity(document.chars().count() + 2);
        encoded.push(self.sequence_boundary_token_id);
        encoded.extend(
            document
                .chars()
                .filter_map(|character| self.character_to_token_id.get(&character).copied()),
        );
        encoded.push(self.sequence_boundary_token_id);
        encoded
    }
}

#[derive(Clone, Debug)]
pub struct AdamOptimizerState {
    // Adam keeps an exponential moving average of gradients. This is momentum:
    // repeated pushes in the same direction reinforce each other.
    pub first_moment_estimates: Vec<f64>,
    // Adam also tracks squared gradients, so parameters with consistently large
    // gradients get smaller effective steps.
    pub second_moment_estimates: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AdamOptimizerConfig {
    // The base step size before warmup/cosine scheduling.
    pub learning_rate: f64,
    // Usually called beta1.
    pub first_moment_decay: f64,
    // Usually called beta2.
    pub second_moment_decay: f64,
    // Prevents division by zero when squared-gradient estimates are tiny.
    pub epsilon: f64,
    // AdamW-style decoupled weight decay gently pulls parameters toward zero.
    // This can improve generalization and reduce runaway weights.
    pub weight_decay: f64,
    // Warmup starts with smaller updates, useful while optimizer moments are
    // still uncalibrated.
    pub warmup_step_count: usize,
    // Cosine decay never drops below this fraction of the base learning rate.
    pub minimum_learning_rate_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct TrainedMicrogpt {
    pub model: TransformerModelParameters,
    pub config: TransformerConfig,
    pub tokenizer: CharacterTokenizer,
}

#[derive(Clone, Debug)]
pub struct MicrogptTrainingProgress {
    pub completed_step_count: usize,
    pub training_step_count: usize,
    pub loss: f64,
    pub validation_loss: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct MicrogptTrainingSession {
    pub trained_microgpt: TrainedMicrogpt,
    pub documents: Vec<String>,
    pub validation_documents: Vec<String>,
    pub training_step_count: usize,
    pub validation_evaluation_document_count: usize,
    pub optimizer_config: AdamOptimizerConfig,
    pub optimizer_state: AdamOptimizerState,
    pub completed_step_count: usize,
    pub latest_loss: Option<f64>,
    pub latest_validation_loss: Option<f64>,
    pub progress_history: Vec<MicrogptTrainingProgress>,
}

impl MicrogptTrainingSession {
    pub fn is_complete(&self) -> bool {
        self.completed_step_count >= self.training_step_count
    }

    pub fn with_initial_progress(
        mut self,
        train_loss: Option<f64>,
        validation_loss: Option<f64>,
    ) -> Self {
        if train_loss.is_none() && validation_loss.is_none() {
            return self;
        }

        let progress = MicrogptTrainingProgress {
            completed_step_count: 0,
            training_step_count: self.training_step_count,
            loss: train_loss.or(validation_loss).unwrap_or(0.0),
            validation_loss,
        };
        self.latest_loss = train_loss;
        self.latest_validation_loss = validation_loss;
        self.progress_history = vec![progress];
        self
    }
}

#[derive(Clone, Debug)]
pub struct MicrogptTrainingStepResult {
    pub session: MicrogptTrainingSession,
    pub progress: MicrogptTrainingProgress,
    pub progress_history: Vec<MicrogptTrainingProgress>,
}

#[derive(Clone, Debug)]
pub struct AdamUpdateResult {
    pub model: TransformerModelParameters,
    pub optimizer_state: AdamOptimizerState,
}

#[derive(Clone, Debug)]
pub struct TransformerRun {
    // One raw score per vocabulary item for "the next token is ...".
    pub logits: Vec<Value>,
    pub keys: KeyValueCache,
    pub values: KeyValueCache,
}

#[derive(Clone, Debug)]
pub struct TransformerLayerRun {
    pub hidden_state: Vec<Value>,
    pub keys: KeyValueCache,
    pub values: KeyValueCache,
}

struct DocumentTrainingResult {
    loss: f64,
    parameter_gradients: Vec<f64>,
}

pub struct ParameterUpdate {
    value: Value,
    first_moment_estimate: f64,
    second_moment_estimate: f64,
}

fn push_matrix_values(values: &mut Vec<Value>, matrix: &Matrix) {
    for row in matrix {
        values.extend(row.iter().cloned());
    }
}

pub fn random_gaussian(
    random_number_generator: &mut impl Rng,
    mean: f64,
    standard_deviation: f64,
) -> f64 {
    // Box-Muller turns two uniform random numbers in [0, 1) into one bell-curve
    // random number. Neural weights usually start with a zero-centered normal
    // distribution so positive and negative signals are balanced.
    let mut first_uniform_sample = 0.0;
    while first_uniform_sample == 0.0 {
        first_uniform_sample = random_number_generator.gen::<f64>();
    }
    let second_uniform_sample = random_number_generator.gen::<f64>();
    let standard_normal_sample =
        (-2.0 * first_uniform_sample.ln()).sqrt() * (2.0 * PI * second_uniform_sample).cos();
    mean + standard_deviation * standard_normal_sample
}

pub fn shuffled_by<T: Clone>(items: &[T], random_number_generator: &mut impl Rng) -> Vec<T> {
    let mut shuffled = items.to_vec();
    for current_index in (1..shuffled.len()).rev() {
        let swap_index = random_number_generator.gen_range(0..=current_index);
        shuffled.swap(current_index, swap_index);
    }
    shuffled
}

pub fn matrix(
    output_size: usize,
    input_size: usize,
    random_number_generator: &mut impl Rng,
    standard_deviation: f64,
) -> Matrix {
    (0..output_size)
        .map(|_| {
            (0..input_size)
                .map(|_| {
                    Value::new(random_gaussian(
                        random_number_generator,
                        0.0,
                        standard_deviation,
                    ))
                })
                .collect()
        })
        .collect()
}

pub fn linear(input_vector: &[Value], weights: &[Vec<Value>]) -> Vec<Value> {
    // A linear layer is just many weighted sums. If the input has values
    // [x1, x2, x3], one output row computes:
    //
    // y = w1*x1 + w2*x2 + w3*x3
    //
    // No activation happens here; this is pure matrix-vector multiplication.
    weights
        .iter()
        .map(|row| {
            row.iter()
                .zip(input_vector.iter())
                .fold(Value::new(0.0), |output_value, (weight, input)| {
                    output_value.add(&weight.mul(input))
                })
        })
        .collect()
}

pub fn softmax(logits: &[Value]) -> Vec<Value> {
    // Logits are arbitrary scores. Softmax converts them into probabilities:
    // exp(score_i) / sum(exp(all scores)). Subtracting the max score first keeps
    // `exp` from overflowing; it does not change the final probabilities because
    // the same constant is subtracted from every score.
    let max_logit_value = logits
        .iter()
        .map(Value::data)
        .fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<_> = logits
        .iter()
        .map(|logit| logit.add_f64(-max_logit_value).exp())
        .collect();
    let total = exponentials
        .iter()
        .fold(Value::new(0.0), |sum, exponential| sum.add(exponential));
    exponentials
        .iter()
        .map(|exponential| exponential.div(&total))
        .collect()
}

pub fn cross_entropy_loss(logits: &[Value], target_token_id: usize) -> Value {
    // Cross-entropy is "how surprised was the model by the correct answer?"
    // If the model assigned probability 1.0 to the target, loss is 0. If it
    // assigned probability near 0, loss is large. This log-sum-exp form is the
    // numerically stable equivalent of `-log(softmax(logits)[target])`.
    let max_logit_value = logits
        .iter()
        .map(Value::data)
        .fold(f64::NEG_INFINITY, f64::max);
    let exponential_sum = logits.iter().fold(Value::new(0.0), |sum, logit| {
        sum.add(&logit.add_f64(-max_logit_value).exp())
    });
    exponential_sum
        .log()
        .add_f64(max_logit_value)
        .sub(&logits[target_token_id])
}

pub fn rmsnorm(input_vector: &[Value]) -> Vec<Value> {
    // RMSNorm rescales a vector so its root-mean-square magnitude is near 1.
    // It does not change direction much; it mainly keeps layer inputs in a
    // predictable numeric range, which makes deep residual networks easier to
    // train.
    let mean_square = input_vector
        .iter()
        .fold(Value::new(0.0), |sum, value| sum.add(&value.mul(value)))
        .div_f64(input_vector.len() as f64);
    let scale = mean_square.add_f64(1e-5).powf(-0.5);
    input_vector.iter().map(|value| value.mul(&scale)).collect()
}

pub fn weighted_choice(weights: &[f64], random_number_generator: &mut impl Rng) -> usize {
    // Sampling chooses an index with probability proportional to its weight.
    // Example: weights [0.1, 0.7, 0.2] will choose index 1 most often, but not
    // always. That controlled randomness is why generated text varies.
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

pub fn create_key_value_cache(layer_count: usize) -> KeyValueCache {
    vec![Vec::new(); layer_count]
}

pub fn run_transformer_model(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    token_id: usize,
    position_id: usize,
    keys: KeyValueCache,
    values: KeyValueCache,
) -> TransformerRun {
    // Forward pass for one time step. The model sees the current token and the
    // current position, updates the KV cache, then returns logits for the next
    // token. Training calls this repeatedly over a sentence.
    let token_embedding = &model.token_embedding[token_id];
    let position_embedding = &model.position_embedding[position_id];
    let mut hidden_state: Vec<_> = token_embedding
        .iter()
        .zip(position_embedding.iter())
        .map(|(token_value, position_value)| token_value.add(position_value))
        .collect();

    hidden_state = rmsnorm(&hidden_state);
    let mut current_keys = keys;
    let mut current_values = values;

    for layer_index in 0..config.layer_count {
        let layer_run = run_transformer_layer(
            &hidden_state,
            &model.layers[layer_index],
            layer_index,
            position_id,
            config,
            current_keys,
            current_values,
        );
        hidden_state = layer_run.hidden_state;
        current_keys = layer_run.keys;
        current_values = layer_run.values;
    }

    TransformerRun {
        logits: linear(&hidden_state, &model.language_model_head),
        keys: current_keys,
        values: current_values,
    }
}

pub fn run_transformer_layer(
    hidden_state: &[Value],
    layer: &TransformerLayerParameters,
    layer_index: usize,
    position_id: usize,
    config: &TransformerConfig,
    mut keys: KeyValueCache,
    mut values: KeyValueCache,
) -> TransformerLayerRun {
    // A pre-norm Transformer block:
    //
    // residual -> RMSNorm -> attention -> add residual
    //          -> RMSNorm -> feed-forward -> add residual
    //
    // Residual additions let each block make an incremental edit instead of
    // relearning the whole representation from scratch.
    let residual_state = hidden_state.to_vec();
    let normalized_state = rmsnorm(hidden_state);

    let query = apply_rotary_position_embedding(
        &linear(&normalized_state, &layer.attention.query_weights),
        position_id,
        config,
    );
    let key = apply_rotary_position_embedding(
        &linear(&normalized_state, &layer.attention.key_weights),
        position_id,
        config,
    );
    let value = linear(&normalized_state, &layer.attention.value_weights);

    // Store this step's key/value so future positions can attend to it.
    keys[layer_index].push(key);
    values[layer_index].push(value);

    let attention_output =
        run_multi_head_attention(&query, &keys[layer_index], &values[layer_index], config);
    let block_output = linear(
        &attention_output,
        &layer.attention.output_projection_weights,
    );
    let mut updated_hidden_state: Vec<_> = block_output
        .iter()
        .zip(residual_state.iter())
        .map(|(attention_value, residual_value)| attention_value.add(residual_value))
        .collect();

    let residual_state = updated_hidden_state.clone();
    let normalized_state = rmsnorm(&updated_hidden_state);
    let expanded_output = linear(&normalized_state, &layer.feed_forward.expansion_weights);
    let gated_output = linear(&normalized_state, &layer.feed_forward.gate_weights);
    // SwiGLU: silu(candidate) * gate. The multiplication lets the model suppress
    // or emphasize features depending on context, which is more expressive than
    // a plain ReLU MLP at similar compute.
    let block_output = expanded_output
        .iter()
        .zip(gated_output.iter())
        .map(|(expanded_value, gate_value)| silu(expanded_value).mul(gate_value))
        .collect::<Vec<_>>();
    let block_output = linear(&block_output, &layer.feed_forward.projection_weights);

    updated_hidden_state = block_output
        .iter()
        .zip(residual_state.iter())
        .map(|(feed_forward_value, residual_value)| feed_forward_value.add(residual_value))
        .collect();

    TransformerLayerRun {
        hidden_state: updated_hidden_state,
        keys,
        values,
    }
}

pub fn run_multi_head_attention(
    query: &[Value],
    keys: &[Vec<Value>],
    values: &[Vec<Value>],
    config: &TransformerConfig,
) -> Vec<Value> {
    // Multi-head attention is the same attention operation repeated over slices
    // of the vector. With embedding 64 and 16 heads, each head sees 4 numbers.
    // Concatenating the head outputs restores the full embedding width.
    (0..config.attention_head_count)
        .flat_map(|head_index| {
            let head_start_index = head_index * config.attention_head_size;
            let head_query =
                &query[head_start_index..head_start_index + config.attention_head_size];
            let head_keys: Vec<_> = keys
                .iter()
                .map(|key| {
                    key[head_start_index..head_start_index + config.attention_head_size].to_vec()
                })
                .collect();
            let head_values: Vec<_> = values
                .iter()
                .map(|value| {
                    value[head_start_index..head_start_index + config.attention_head_size].to_vec()
                })
                .collect();
            let attention_weights = softmax(&attention_logits(
                head_query,
                &head_keys,
                config.attention_head_size,
            ));

            (0..config.attention_head_size)
                .map(|head_value_index| {
                    weighted_head_value_sum(&attention_weights, &head_values, head_value_index)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn attention_logits(
    head_query: &[Value],
    head_keys: &[Vec<Value>],
    attention_head_size: usize,
) -> Vec<Value> {
    // A dot product is large when two vectors point in similar directions. Here
    // query dot key asks: "how relevant is that previous position to this one?"
    // Dividing by sqrt(head_size) keeps logits from growing just because vectors
    // have more dimensions.
    head_keys
        .iter()
        .map(|previous_key| {
            let dot_product =
                (0..attention_head_size).fold(Value::new(0.0), |sum, head_value_index| {
                    sum.add(&head_query[head_value_index].mul(&previous_key[head_value_index]))
                });
            dot_product.div_f64((attention_head_size as f64).sqrt())
        })
        .collect()
}

pub fn weighted_head_value_sum(
    attention_weights: &[Value],
    head_values: &[Vec<Value>],
    head_value_index: usize,
) -> Value {
    // Attention weights are probabilities over previous time steps. The output
    // is the weighted average of value vectors, so the model can copy or blend
    // information from earlier characters.
    head_values.iter().enumerate().fold(
        Value::new(0.0),
        |weighted_value_sum, (time_index, head_value)| {
            weighted_value_sum
                .add(&attention_weights[time_index].mul(&head_value[head_value_index]))
        },
    )
}

pub fn apply_rotary_position_embedding(
    vector: &[Value],
    position_id: usize,
    config: &TransformerConfig,
) -> Vec<Value> {
    // RoPE encodes position by rotating pairs of numbers in each attention head.
    // Nearby positions get similar rotations; farther positions get different
    // rotations. Because the same rotation rule is applied to queries and keys,
    // their dot product can express relative distance between characters.
    let mut rotated = vector.to_vec();
    let pair_count = config.attention_head_size / 2;
    for head_index in 0..config.attention_head_count {
        let head_start_index = head_index * config.attention_head_size;
        for pair_index in 0..pair_count {
            let even_index = head_start_index + 2 * pair_index;
            let odd_index = even_index + 1;
            let frequency =
                10_000.0_f64.powf(-((2 * pair_index) as f64) / config.attention_head_size as f64);
            let angle = position_id as f64 * frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            rotated[even_index] = vector[even_index]
                .mul_f64(cosine)
                .sub(&vector[odd_index].mul_f64(sine));
            rotated[odd_index] = vector[even_index]
                .mul_f64(sine)
                .add(&vector[odd_index].mul_f64(cosine));
        }
    }
    rotated
}

pub fn silu(value: &Value) -> Value {
    // SiLU(x) = x * sigmoid(x). It is a smooth activation: negative values are
    // damped, positive values mostly pass through, and gradients do not abruptly
    // jump at zero like ReLU.
    let sigmoid = value.neg().exp().add_f64(1.0).powf(-1.0);
    value.mul(&sigmoid)
}

fn training_batch_documents(
    documents: &[String],
    step: usize,
    batch_document_count: usize,
) -> Vec<String> {
    // This deterministic batching scheme cycles through the shuffled document
    // list. It avoids extra randomness during a run, which makes learning curves
    // easier to reproduce while still exposing all documents over time.
    let batch_start_index = (step * batch_document_count) % documents.len();
    (0..batch_document_count)
        .map(|batch_offset| documents[(batch_start_index + batch_offset) % documents.len()].clone())
        .collect()
}

fn train_on_document_with_gradients(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    document: &str,
    parameter_index_by_value: &HashMap<usize, usize>,
    parameter_count: usize,
) -> DocumentTrainingResult {
    // One document produces one scalar loss. Calling backward on that loss tells
    // us how every parameter should change to reduce surprise on this document.
    let loss = train_on_document(model, config, tokenizer, document);
    DocumentTrainingResult {
        loss: loss.data(),
        parameter_gradients: loss.backward_for(parameter_index_by_value, parameter_count),
    }
}

pub fn train_microgpt_step(
    session: MicrogptTrainingSession,
    batch_document_count: usize,
) -> Option<MicrogptTrainingStepResult> {
    // A training step is mini-batch gradient descent:
    //
    // - pick several documents,
    // - compute each document's gradients,
    // - average those gradients,
    // - update the model once.
    //
    // Averaging makes the update less noisy than learning from one example at a
    // time, while still being much cheaper than using the whole dataset.
    if session.is_complete() {
        return None;
    }
    assert!(
        batch_document_count > 0,
        "batch_document_count must be positive"
    );

    let step = session.completed_step_count;
    let parameters = session.trained_microgpt.model.values();
    // `Value::backward_for` returns gradients by node id. This map tells it which
    // graph nodes are trainable parameters and where they live in the flat
    // parameter vector.
    let parameter_index_by_value: HashMap<_, _> = parameters
        .iter()
        .enumerate()
        .map(|(parameter_index, parameter)| (parameter.id(), parameter_index))
        .collect();
    let batch_documents = training_batch_documents(&session.documents, step, batch_document_count);
    let parameter_count = parameters.len();

    // Each document builds an independent scalar computation graph, so CPU
    // training can parallelize documents safely with Rayon.
    let document_results: Vec<_> = batch_documents
        .par_iter()
        .map(|document| {
            train_on_document_with_gradients(
                &session.trained_microgpt.model,
                &session.trained_microgpt.config,
                &session.trained_microgpt.tokenizer,
                document,
                &parameter_index_by_value,
                parameter_count,
            )
        })
        .collect();

    let average_loss = document_results
        .iter()
        .map(|result| result.loss)
        .sum::<f64>()
        / document_results.len() as f64;
    let mut accumulated_parameter_gradients = vec![0.0; parameter_count];

    // Sum gradients for matching parameters across documents, then divide by
    // batch size to get the mean gradient.
    for document_result in &document_results {
        for (parameter_index, gradient) in document_result.parameter_gradients.iter().enumerate() {
            accumulated_parameter_gradients[parameter_index] += gradient;
        }
    }

    let inverse_document_count = 1.0 / batch_documents.len() as f64;
    for gradient in &mut accumulated_parameter_gradients {
        *gradient *= inverse_document_count;
    }

    let update = apply_adam_update(
        &session.trained_microgpt.model,
        &accumulated_parameter_gradients,
        &session.optimizer_state,
        &session.optimizer_config,
        step,
        session.training_step_count,
    );

    let updated_microgpt = TrainedMicrogpt {
        model: update.model,
        ..session.trained_microgpt.clone()
    };

    let progress = MicrogptTrainingProgress {
        completed_step_count: session.completed_step_count + 1,
        training_step_count: session.training_step_count,
        loss: average_loss,
        validation_loss: None,
    };
    let mut progress_history = session.progress_history.clone();
    progress_history.push(progress.clone());

    let updated_session = MicrogptTrainingSession {
        trained_microgpt: updated_microgpt,
        optimizer_state: update.optimizer_state,
        completed_step_count: progress.completed_step_count,
        latest_loss: Some(progress.loss),
        progress_history: progress_history.clone(),
        ..session
    };

    Some(MicrogptTrainingStepResult {
        session: updated_session,
        progress,
        progress_history,
    })
}

pub fn train_on_document(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    document: &str,
) -> Value {
    // Teacher forcing: at position t, feed the true token from the document and
    // ask the model to predict token t+1. This is how language models learn from
    // plain text without hand-written labels.
    let tokens = tokenizer.encode_document(document);
    let prediction_step_count = config
        .context_window_size
        .min(tokens.len().saturating_sub(1));
    let mut keys = create_key_value_cache(config.layer_count);
    let mut values = create_key_value_cache(config.layer_count);
    let mut loss = Value::new(0.0);

    for position_id in 0..prediction_step_count {
        let token_id = tokens[position_id];
        let target_token_id = tokens[position_id + 1];
        let model_run = run_transformer_model(model, config, token_id, position_id, keys, values);
        keys = model_run.keys;
        values = model_run.values;
        let position_loss = cross_entropy_loss(&model_run.logits, target_token_id);
        // Total document loss is the average surprise across positions.
        loss = loss.add(&position_loss);
    }

    loss.mul_f64(1.0 / prediction_step_count as f64)
}

pub fn calculate_document_loss(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    document: &str,
) -> f64 {
    train_on_document(model, config, tokenizer, document).data()
}

pub fn calculate_validation_loss(
    session: &MicrogptTrainingSession,
    completed_step_count: usize,
    validation_step_interval: usize,
) -> Option<f64> {
    // Validation documents are held out from training. If training loss improves
    // but validation loss worsens, the model may be memorizing the training set
    // instead of learning general patterns.
    if session.validation_documents.is_empty() {
        return None;
    }
    let validation_document_count = session
        .validation_evaluation_document_count
        .min(session.validation_documents.len());
    let validation_batch_index = completed_step_count / validation_step_interval;
    let loss = (0..validation_document_count)
        .map(|validation_offset| {
            let validation_index = (validation_batch_index * validation_document_count
                + validation_offset)
                % session.validation_documents.len();
            calculate_document_loss(
                &session.trained_microgpt.model,
                &session.trained_microgpt.config,
                &session.trained_microgpt.tokenizer,
                &session.validation_documents[validation_index],
            )
        })
        .sum::<f64>()
        / validation_document_count as f64;
    Some(loss)
}

pub fn calculate_training_loss_baseline(session: &MicrogptTrainingSession) -> Option<f64> {
    session.documents.first().map(|document| {
        calculate_document_loss(
            &session.trained_microgpt.model,
            &session.trained_microgpt.config,
            &session.trained_microgpt.tokenizer,
            document,
        )
    })
}

pub fn attach_validation_loss(
    mut result: MicrogptTrainingStepResult,
    validation_loss: Option<f64>,
) -> MicrogptTrainingStepResult {
    // Training workers can compute validation less often than every step. This
    // helper attaches a validation result to the most recent progress entry.
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

pub fn apply_adam_update(
    model: &TransformerModelParameters,
    gradients: &[f64],
    optimizer_state: &AdamOptimizerState,
    optimizer_config: &AdamOptimizerConfig,
    step: usize,
    training_step_count: usize,
) -> AdamUpdateResult {
    // AdamW update, in words:
    //
    // 1. Clip the whole gradient vector if it is too large.
    // 2. Update moving averages of gradient and squared gradient.
    // 3. Bias-correct those averages because they start at zero.
    // 4. Divide by the square-root squared-gradient average so noisy parameters
    //    get smaller steps.
    // 5. Add decoupled weight decay.
    // 6. Subtract the update from each parameter.
    let parameters = model.values();
    let clipped_gradients = clipped_gradients(gradients, MAX_GRADIENT_NORM);
    let step_learning_rate = scheduled_learning_rate(optimizer_config, step, training_step_count);

    let parameter_updates: Vec<_> = parameters
        .iter()
        .enumerate()
        .map(|(parameter_index, parameter)| {
            let gradient = clipped_gradients[parameter_index];
            // Exponential moving average: new_average = mostly_old + little_new.
            let first_moment_estimate = optimizer_config.first_moment_decay
                * optimizer_state.first_moment_estimates[parameter_index]
                + (1.0 - optimizer_config.first_moment_decay) * gradient;
            let second_moment_estimate = optimizer_config.second_moment_decay
                * optimizer_state.second_moment_estimates[parameter_index]
                + (1.0 - optimizer_config.second_moment_decay) * gradient.powi(2);

            // Bias correction matters early in training because both moving
            // averages start at zero, which would otherwise make them too small.
            let bias_corrected_first_moment = first_moment_estimate
                / (1.0 - optimizer_config.first_moment_decay.powf(step as f64 + 1.0));
            let bias_corrected_second_moment = second_moment_estimate
                / (1.0 - optimizer_config.second_moment_decay.powf(step as f64 + 1.0));
            let adam_update = bias_corrected_first_moment
                / (bias_corrected_second_moment.sqrt() + optimizer_config.epsilon);
            let decay_update = optimizer_config.weight_decay * parameter.data();
            let parameter_update = step_learning_rate * (adam_update + decay_update);

            ParameterUpdate {
                value: Value::new(parameter.data() - parameter_update),
                first_moment_estimate,
                second_moment_estimate,
            }
        })
        .collect();

    AdamUpdateResult {
        model: model.with_values(
            parameter_updates
                .iter()
                .map(|update| update.value.clone())
                .collect(),
        ),
        optimizer_state: AdamOptimizerState {
            first_moment_estimates: parameter_updates
                .iter()
                .map(|update| update.first_moment_estimate)
                .collect(),
            second_moment_estimates: parameter_updates
                .iter()
                .map(|update| update.second_moment_estimate)
                .collect(),
        },
    }
}

pub fn scheduled_learning_rate(
    optimizer_config: &AdamOptimizerConfig,
    step: usize,
    training_step_count: usize,
) -> f64 {
    // Warmup ramps linearly from small steps to the base learning rate. After
    // warmup, cosine decay smoothly lowers the rate. This gives fast early
    // learning and gentler late fine-tuning.
    let warmup_step_count = optimizer_config.warmup_step_count.min(training_step_count);
    if warmup_step_count > 0 && step < warmup_step_count {
        return optimizer_config.learning_rate * (step + 1) as f64 / warmup_step_count as f64;
    }

    let scheduled_step = step.saturating_sub(warmup_step_count);
    let scheduled_step_count = training_step_count.saturating_sub(warmup_step_count).max(1);
    let progress = (scheduled_step as f64 / scheduled_step_count as f64).clamp(0.0, 1.0);
    let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
    optimizer_config.learning_rate
        * (optimizer_config.minimum_learning_rate_ratio
            + (1.0 - optimizer_config.minimum_learning_rate_ratio) * cosine_decay)
}

fn clipped_gradients(gradients: &[f64], max_norm: f64) -> Vec<f64> {
    // Treat all parameter gradients as one long vector and measure its length.
    // If the length is above `max_norm`, scale the entire vector down while
    // preserving direction. This is like saying "take the same step direction,
    // just not that far."
    let norm = gradients
        .iter()
        .map(|gradient| gradient * gradient)
        .sum::<f64>()
        .sqrt();
    if norm <= max_norm || norm == 0.0 {
        return gradients.to_vec();
    }
    let scale = max_norm / (norm + 1e-12);
    gradients.iter().map(|gradient| gradient * scale).collect()
}

pub fn generate_samples(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    prefix: &str,
    sample_count: usize,
    temperature: f64,
    random_number_generator: &mut impl Rng,
) -> Vec<String> {
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
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    prefix: &str,
    temperature: f64,
    random_number_generator: &mut impl Rng,
) -> String {
    // Generation is autoregressive: start from a boundary token, repeatedly ask
    // for next-token probabilities, sample one token, append it, and feed it
    // back in. The same KV cache idea used during training avoids recomputing
    // all previous keys and values at each position.
    let mut keys = create_key_value_cache(config.layer_count);
    let mut values = create_key_value_cache(config.layer_count);
    let mut token_id = tokenizer.sequence_boundary_token_id;
    let normalized_prefix: String = prefix
        .trim()
        .to_lowercase()
        .chars()
        .filter(|character| tokenizer.character_to_token_id.contains_key(character))
        .take(config.context_window_size - 1)
        .collect();
    let prefix_characters = normalized_prefix.chars().collect::<Vec<_>>();
    let mut sample = normalized_prefix.clone();

    for position_id in 0..config.context_window_size {
        let model_run = run_transformer_model(model, config, token_id, position_id, keys, values);
        keys = model_run.keys;
        values = model_run.values;

        if let Some(prefix_character) = prefix_characters.get(position_id) {
            // A prefix is forced into the context. The model consumes those
            // characters before it is allowed to sample freely.
            token_id = tokenizer.character_to_token_id[prefix_character];
            continue;
        }

        // Temperature changes confidence. Lower than 1 sharpens probabilities
        // and makes output more predictable; higher than 1 flattens them and
        // makes output more random.
        let scaled_logits: Vec<_> = model_run
            .logits
            .iter()
            .map(|logit| logit.div_f64(temperature))
            .collect();
        let probabilities = softmax(&scaled_logits);
        let mut probability_data: Vec<_> = probabilities.iter().map(Value::data).collect();
        apply_sampling_constraints(
            &mut probability_data,
            tokenizer,
            &sample,
            prefix_characters.len(),
        );

        token_id = weighted_choice(&probability_data, random_number_generator);
        if token_id == tokenizer.sequence_boundary_token_id {
            break;
        }

        sample.push(tokenizer.unique_characters[token_id]);
    }

    sample
}

pub fn apply_sampling_constraints(
    probabilities: &mut [f64],
    tokenizer: &CharacterTokenizer,
    sample: &str,
    prefix_character_count: usize,
) {
    // These constraints are not learned by the model; they are demo-time guard
    // rails. Production systems often use similar decoding rules for format,
    // safety, or style requirements.
    if probabilities.is_empty() {
        return;
    }

    if sample
        .chars()
        .count()
        .saturating_sub(prefix_character_count)
        < MIN_GENERATED_CHARACTER_COUNT
    {
        probabilities[tokenizer.sequence_boundary_token_id] = 0.0;
    }

    if sample.is_empty() || sample.ends_with(' ') {
        if let Some(space_token_id) = tokenizer.character_to_token_id.get(&' ') {
            probabilities[*space_token_id] = 0.0;
        }
    }

    keep_top_k(probabilities, SAMPLING_TOP_K);

    if probabilities.iter().all(|probability| *probability <= 0.0) {
        for probability in probabilities {
            *probability = 1.0;
        }
    }
}

fn keep_top_k(probabilities: &mut [f64], top_k: usize) {
    // Top-k sampling discards every token except the k most likely. It is a
    // simple way to prevent very unlikely characters from creating unreadable
    // samples while keeping some randomness.
    if top_k == 0 || probabilities.len() <= top_k {
        return;
    }

    let mut sorted_probabilities = probabilities.to_vec();
    sorted_probabilities.sort_by(|left, right| right.total_cmp(left));
    let threshold = sorted_probabilities[top_k - 1];
    for probability in probabilities {
        if *probability < threshold {
            *probability = 0.0;
        }
    }
}

pub fn normalize_training_document(document: &str) -> String {
    // Keep this dataset intentionally simple: lowercase a-z plus spaces. A
    // production tokenizer would preserve far more information, but this tiny
    // character vocabulary makes every token and probability easy to inspect.
    let mut normalized = String::with_capacity(document.len());
    let mut previous_was_space = true;

    for character in document.chars().flat_map(char::to_lowercase) {
        if character.is_ascii_lowercase() {
            normalized.push(character);
            previous_was_space = false;
        } else if character.is_whitespace() && !previous_was_space {
            normalized.push(' ');
            previous_was_space = true;
        }
    }

    normalized.trim().to_string()
}

pub fn create_microgpt_training_session(
    input_documents: Vec<String>,
    random_number_generator: &mut impl Rng,
    training_step_count: usize,
    validation_set_divisor: usize,
    validation_evaluation_document_count: usize,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
) -> MicrogptTrainingSession {
    // Session creation owns data preparation: normalize text, shuffle once,
    // split held-out validation examples, build the character vocabulary, then
    // initialize model parameters and optimizer state.
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
    let vocabulary_size = tokenizer.vocabulary_size();
    let model = TransformerModelParameters::initialize(
        vocabulary_size,
        transformer_config.context_window_size,
        transformer_config.embedding_size,
        transformer_config.layer_count,
        random_number_generator,
    );
    let parameter_count = model.parameter_count();

    MicrogptTrainingSession {
        trained_microgpt: TrainedMicrogpt {
            model,
            config: transformer_config,
            tokenizer,
        },
        documents,
        validation_documents,
        training_step_count,
        validation_evaluation_document_count,
        optimizer_config,
        optimizer_state: AdamOptimizerState {
            first_moment_estimates: vec![0.0; parameter_count],
            second_moment_estimates: vec![0.0; parameter_count],
        },
        completed_step_count: 0,
        latest_loss: None,
        latest_validation_loss: None,
        progress_history: Vec::new(),
    }
}
