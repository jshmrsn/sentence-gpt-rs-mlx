# CPU Backend And Overall Architecture

This document walks through the plain Rust CPU backend, the surrounding application architecture, and the machine-learning ideas used by the project. It is written as a companion to `docs/mlx-backend.md`.

The CPU backend is the reference implementation. It avoids a tensor framework so the math remains visible in ordinary Rust data structures:

- A scalar number is a `Value`.
- A vector is `Vec<Value>`.
- A matrix is `Vec<Vec<Value>>`.
- The computation graph is built implicitly by calling methods such as `add`, `mul`, `exp`, and `log`.
- Reverse-mode autodiff walks that graph to compute gradients.

CPU training is much slower than MLX training. In exchange, the computation graph, derivatives, and optimizer updates can be read directly from the code.

## Source Map

The main files are:

- `lib/src/value.rs`: scalar reverse-mode automatic differentiation.
- `lib/src/microgpt.rs`: CPU model, training loop, validation loss, generation, optimizer, checkpoint import/export helpers.
- `lib/src/mlx_microgpt.rs`: tensorized MLX backend with the same model family.
- `lib/src/checkpoint.rs`: shared checkpoint format and training-run config serialization.
- `config/src/lib.rs`: story loading, train/validation splitting, backend selection, frame-budgeted training orchestration, formatting helpers.
- `app/src/main.rs`: Dioxus desktop GUI and background worker scheduling.

The central CPU functions are:

- `create_microgpt_training_session_from_splits`
- `train_microgpt_step`
- `training_batch_token_windows`
- `train_on_token_window_with_dropout`
- `run_transformer_model_with_dropout`
- `run_transformer_layer`
- `run_multi_head_attention`
- `apply_rotary_position_embedding`
- `cross_entropy_loss`
- `apply_adam_update`
- `calculate_validation_loss`
- `generate_sample`

The central shared architecture functions are:

- `create_training_session`
- `train_session_until_budget`
- `train_cpu_until_budget`
- `train_mlx_until_budget`
- `TrainingSession::export_checkpoint`
- `TrainingSession::import_checkpoint`

## Overview

The project trains a small decoder-only Transformer as a character-level language model. The model reads a short sequence of characters and predicts the next character at every position.

Example training pair:

```text
input:   <BOS> T h e   c a t
target:  T     h e   c a t <EOS>
```

The model is "decoder-only" because it predicts future text from previous text. Sequence-to-sequence translation models have a separate encoder side; this model has only the decoder-style prediction path.

The model is "causal" because each position can only attend to earlier positions and itself. During CPU training this causality comes naturally from processing one position at a time and appending keys/values to the KV cache. There is no future information in the cache.

The CPU backend follows this training loop:

```text
documents
  -> deterministic token windows
  -> forward pass through Transformer
  -> cross-entropy loss
  -> scalar reverse-mode autodiff
  -> averaged mini-batch gradients
  -> clipped AdamW update
  -> progress history and periodic validation
```

The GUI wraps that loop in short background chunks so the app remains responsive while training continues.

## Crate-Level Architecture

The workspace is split into three active crates:

```text
app/
  Dioxus desktop GUI
  user actions, buttons, charts, samples, checkpoints

config/
  app-facing orchestration
  data loading and filtering
  backend selection
  train/validation split
  frame-budgeted training loop

lib/
  model implementations
  CPU backend
  MLX backend
  checkpoint format
  scalar autodiff
```

The `lib` crate contains the model code for training and generation. The `config` crate chooses the backend, prepares data, and schedules training chunks. The `app` crate holds the interactive state and sends expensive work to blocking worker tasks.

The boundary is plain: the GUI can work without knowing the details of attention or AdamW, and the model code can work without knowing about buttons, file dialogs, or charts.

## Backend Abstraction

The app has one active backend at a time:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Mlx,
    Cpu,
}
```

The active training session is stored behind an enum:

```rust
#[derive(Clone)]
pub enum TrainingSession {
    Mlx(MlxMicrogptTrainingSession),
    Cpu(MicrogptTrainingSession),
}
```

Most app code talks to methods on `TrainingSession`, not directly to the CPU or MLX structs:

```rust
impl TrainingSession {
    pub fn latest_loss(&self) -> Option<f64> {
        match self {
            TrainingSession::Mlx(session) => session.latest_loss,
            TrainingSession::Cpu(session) => session.latest_loss,
        }
    }
}
```

This gives the UI one path for:

- completed step count
- current loss
- validation loss
- tokenizer vocabulary size
- parameter count
- progress history
- checkpoint export/import
- trained snapshot for generation

The enum wrapper is a direct fit for the current code. It keeps each backend's concrete type available where needed without introducing a trait layer.

## Default Configs

The CPU backend has a smaller default configuration than MLX:

```rust
pub const CPU_DEFAULT_TRAINING_RUN_CONFIG: TrainingRunConfig = TrainingRunConfig {
    validation_step_interval: 25,
    training_document_batch_size: 8,
    max_document_count: 100,
    validation_set_divisor: 20,
    validation_evaluation_document_count: 4,
    context_window_size: 32,
    layer_count: 2,
    attention_heads: 4,
    embedding_size: 16,
};
```

The CPU path is scalar and graph-heavy, so the default is sized for interactive training. The MLX default can use larger batches and model dimensions because MLX executes batched tensor operations on Apple Silicon.

The shared optimizer config is:

```rust
pub fn get_optimizer_config() -> AdamOptimizerConfig {
    AdamOptimizerConfig {
        learning_rate: 0.003,
        first_moment_decay: 0.85,
        second_moment_decay: 0.99,
        epsilon: 1e-8,
        weight_decay: 0.01,
        warmup_step_count: 200,
        minimum_learning_rate_ratio: 0.1,
    }
}
```

Those values define AdamW behavior and the learning-rate schedule for both backends.

## Data Pipeline

The app loads TinyStories-derived JSON from `data/input-stories-00.json`, then filters it into short sentence-like training examples.

The high-level data path is:

```text
JSON stories
  -> keep source == "GPT-4"
  -> split on '.', '?', '!'
  -> keep boundary punctuation
  -> remove sentences with disallowed characters
  -> normalize newlines to spaces
  -> require length, spaces, and context-window fit
  -> remove duplicate sentences globally
  -> group sentences by original story
  -> shuffle stories
  -> split stories into validation and training groups
  -> flatten selected story groups into sentence documents
```

One design choice is especially relevant for validation: sentences stay grouped by original source story until after the train/validation split.

```rust
let shuffled_stories = shuffled_by(&input_stories, rng);
let validation_story_count =
    shuffled_stories.len() / training_run_config.validation_set_divisor;
let validation_documents = flatten_story_sentences(&shuffled_stories[..validation_story_count]);
let documents = flatten_story_sentences(&shuffled_stories[validation_story_count..]);
```

This reduces leakage. If one story contributes two very similar sentences, keeping the story together prevents those near-duplicates from landing on opposite sides of the split.

Duplicate sentence filtering is global:

```rust
let mut seen_sentences = HashSet::new();

// ...
.filter(|sentence| seen_sentences.insert(sentence.clone()))
```

That prevents repeated text from overweighting the objective and also reduces the chance that validation examples duplicate training examples.

## Character Tokenization

The tokenizer uses one token id for every distinct character, plus one sequence-boundary token:

```rust
pub struct CharacterTokenizer {
    pub unique_characters: Vec<char>,
    pub sequence_boundary_token_id: usize,
    pub character_to_token_id: HashMap<char, usize>,
}
```

Encoding adds the boundary token to both ends:

```rust
pub fn encode_document(&self, document: &str) -> Vec<usize> {
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
```

The first boundary token teaches how text starts. The final boundary token teaches when generation should stop.

Most larger language models use BPE, Unigram, WordPiece, or another subword tokenizer. Character tokenization produces longer sequences, but the examples stay close to the original text and the vocabulary is easy to inspect.

## CPU Data Structures

The CPU model stores all trainable numbers as scalar `Value` objects:

```rust
pub type Matrix = Vec<Vec<Value>>;
pub type Vector = Vec<Value>;
pub type KeyValueCache = Vec<Vec<Vec<Value>>>;
```

Matrix shape convention is:

```text
[output_size][input_size]
```

So a linear layer computes one dot product per row:

```rust
pub fn linear(input_vector: &[Value], weights: &[Vec<Value>], biases: &[Value]) -> Vec<Value> {
    weights
        .iter()
        .zip(biases.iter())
        .map(|(row, bias)| {
            row.iter()
                .zip(input_vector.iter())
                .fold(bias.clone(), |output_value, (weight, input)| {
                    output_value.add(&weight.mul(input))
                })
        })
        .collect()
}
```

Mathematically, for row `j`:

```text
y_j = b_j + sum_i W[j, i] * x_i
```

Every multiplication and addition creates a new `Value` node in the autodiff graph.

## Parameter Layout

The model has these top-level parameters:

```rust
pub struct TransformerModelParameters {
    pub token_embedding: Matrix,
    pub position_embedding: Matrix,
    pub language_model_head: Matrix,
    pub language_model_head_biases: Vector,
    pub final_norm_gain: Vector,
    pub layers: Vec<TransformerLayerParameters>,
}
```

The active forward pass uses:

- `token_embedding` as the input embedding table.
- `token_embedding` again as the tied language-model output head.
- `language_model_head_biases` as output logits bias.
- `final_norm_gain` for the final RMSNorm scale.
- each layer's RMSNorm gains, attention projections, and feed-forward projections.

Two fields are retained mostly for compatibility:

- `position_embedding`: active position information comes from RoPE.
- `language_model_head`: active logits are tied to `token_embedding`.

Layer parameters are:

```rust
pub struct TransformerLayerParameters {
    pub attention_norm_gain: Vector,
    pub attention: AttentionParameters,
    pub feed_forward_norm_gain: Vector,
    pub feed_forward: FeedForwardParameters,
}
```

Attention parameters:

```rust
pub struct AttentionParameters {
    pub query_weights: Matrix,
    pub query_biases: Vector,
    pub key_weights: Matrix,
    pub key_biases: Vector,
    pub value_weights: Matrix,
    pub value_biases: Vector,
    pub output_projection_weights: Matrix,
    pub output_projection_biases: Vector,
}
```

Feed-forward parameters:

```rust
pub struct FeedForwardParameters {
    pub expansion_weights: Matrix,
    pub expansion_biases: Vector,
    pub gate_weights: Matrix,
    pub gate_biases: Vector,
    pub projection_weights: Matrix,
    pub projection_biases: Vector,
}
```

The feed-forward hidden width is `3 * embedding_size`, matching the SwiGLU expansion used by the backend.

## Initialization

Parameters are initialized with small Gaussian random values:

```rust
let embedding_std = 0.02;
let feed_forward_size = 3 * embedding_size;
let projection_std = (1.0 / embedding_size as f64).sqrt();
let residual_projection_std = projection_std / (2.0 * layer_count as f64).sqrt();
```

The main ideas are:

- Embeddings start small, so initial logits stay near zero.
- Projection matrices scale roughly with hidden size.
- Residual projection weights are smaller when there are more layers, so repeated residual additions start at a controlled scale.
- RMSNorm gains start at one.
- Biases start at zero.

The Gaussian sampler uses Box-Muller:

```rust
let standard_normal_sample =
    (-2.0 * first_uniform_sample.ln()).sqrt()
        * (2.0 * PI * second_uniform_sample).cos();
```

This converts uniform random samples into normally distributed samples.

## Scalar Autodiff

The CPU backend's key teaching feature is `Value`.

Each `Value` is one scalar plus graph metadata:

```rust
struct Node {
    data: f64,
    children: Vec<Value>,
    local_gradients: Vec<f64>,
}

pub struct Value(Arc<Node>);
```

If the code computes:

```rust
let c = a.mul(&b);
```

then `c` stores:

```text
data = a * b
children = [a, b]
local_gradients = [b, a]
```

Those local gradients are the partial derivatives:

```text
d(a*b)/da = b
d(a*b)/db = a
```

Addition is similar:

```rust
pub fn add(&self, other: &Value) -> Value {
    Value::with_children(
        self.data() + other.data(),
        vec![self.clone(), other.clone()],
        vec![1.0, 1.0],
    )
}
```

The local gradients for addition are both one:

```text
d(a+b)/da = 1
d(a+b)/db = 1
```

More operations are enough to build the model:

- `powf`
- `log`
- `exp`
- `relu`
- `neg`
- `sub`
- `div`

The graph is immutable. Every operation returns a new `Value`. Model updates create a new model containing new leaf `Value`s rather than mutating old graph nodes.

## Reverse-Mode Backpropagation

Training needs gradients of one scalar loss with respect to many parameters. Reverse-mode autodiff is the efficient direction for that case.

The implementation first topologically sorts the graph:

```rust
fn topological_order(&self) -> Vec<Value> {
    fn visit(node: &Value, visited: &mut HashSet<usize>, order: &mut Vec<Value>) {
        if !visited.insert(node.id()) {
            return;
        }
        for child in &node.0.children {
            visit(child, visited, order);
        }
        order.push(node.clone());
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    visit(self, &mut visited, &mut order);
    order
}
```

Then it walks backward:

```rust
pub fn backward(&self) -> HashMap<usize, f64> {
    let order = self.topological_order();
    let mut gradients = HashMap::from([(self.id(), 1.0)]);

    for node in order.iter().rev() {
        let node_gradient = *gradients.get(&node.id()).unwrap_or(&0.0);
        for (child, local_gradient) in node.0.children.iter().zip(node.0.local_gradients.iter()) {
            *gradients.entry(child.id()).or_insert(0.0) += local_gradient * node_gradient;
        }
    }

    gradients
}
```

The key formula is the chain rule:

```text
child_gradient += parent_gradient * local_gradient
```

If a parameter affects the loss through several paths, the `+=` accumulates all those paths.

The training loop only needs gradients for trainable parameters, so it calls:

```rust
loss.backward_for(&parameter_index_by_value, parameter_count)
```

The result is a dense vector aligned with `model.values()`.

## Flattened Parameters

The model is nested, while the optimizer works over a flat vector of parameters and gradients. The CPU model provides a stable flattened parameter order:

```rust
pub fn values(&self) -> Vec<Value> {
    let mut values = Vec::new();
    push_matrix_values(&mut values, &self.token_embedding);
    push_matrix_values(&mut values, &self.position_embedding);
    push_matrix_values(&mut values, &self.language_model_head);
    push_vector_values(&mut values, &self.language_model_head_biases);
    push_vector_values(&mut values, &self.final_norm_gain);
    // then every layer...
    values
}
```

After AdamW computes replacement scalar values, `with_values` rebuilds the nested model in exactly the same order:

```rust
AdamUpdateResult {
    model: model.with_values(
        parameter_updates
            .iter()
            .map(|update| update.value.clone())
            .collect(),
    ),
    optimizer_state: AdamOptimizerState {
        first_moment_estimates,
        second_moment_estimates,
    },
}
```

This stable order is also used by checkpoints and optimizer state.

## Training Token Windows

Each training step samples deterministic fixed-size windows:

```rust
pub fn training_batch_token_windows(
    documents: &[String],
    tokenizer: &CharacterTokenizer,
    step: usize,
    batch_document_count: usize,
    context_window_size: usize,
) -> Vec<TrainingTokenWindow>
```

A window contains:

```rust
pub struct TrainingTokenWindow {
    pub input_tokens: Vec<usize>,
    pub target_tokens: Vec<usize>,
    pub loss_mask: Vec<f64>,
    pub(crate) batch_offset: usize,
}
```

Targets are shifted by one:

```text
input_tokens[t]  = tokens[start + t]
target_tokens[t] = tokens[start + t + 1]
```

The `loss_mask` is `1.0` for real predictions and `0.0` for padding. Short documents are padded with the boundary token, but padding positions do not contribute to the objective.

Window selection is deterministic:

```rust
let document_index = deterministic_index(step, batch_offset, 0x9e37, documents.len());
```

The same `(step, batch_offset)` picks the same document/window after restoring a checkpoint. The checkpoint therefore does not need to store mutable batch-sampling RNG state.

## Forward Pass

The CPU backend processes one position at a time:

```rust
fn run_transformer_model_with_dropout(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    token_id: usize,
    position_id: usize,
    keys: KeyValueCache,
    values: KeyValueCache,
    dropout_context: Option<CpuDropoutContext>,
) -> TransformerRun
```

The forward pass:

1. Looks up the token embedding.
2. Runs each Transformer layer.
3. Updates each layer's KV cache.
4. Applies final RMSNorm.
5. Computes tied language-model logits.

In code:

```rust
let mut hidden_state = model.token_embedding[token_id].clone();

for layer_index in 0..config.layer_count {
    let layer_run = run_transformer_layer(...);
    hidden_state = layer_run.hidden_state;
    current_keys = layer_run.keys;
    current_values = layer_run.values;
}

hidden_state = rmsnorm(&hidden_state, &model.final_norm_gain);
let logits = tied_language_model_logits(
    &hidden_state,
    &model.token_embedding,
    &model.language_model_head_biases,
);
```

The final logits are raw scores over the vocabulary. They become probabilities only when passed through softmax.

## Pre-Norm Transformer Block

Each layer is a pre-norm Transformer block:

```text
x
  -> RMSNorm
  -> causal self-attention
  -> output projection
  -> residual dropout
  -> add residual
  -> RMSNorm
  -> SwiGLU feed-forward
  -> projection
  -> residual dropout
  -> add residual
```

The residual form is:

```text
x = x + attention_block(norm(x))
x = x + feed_forward_block(norm(x))
```

Pre-norm means normalization happens before each sub-block. This is common in modern decoder-only Transformers because it usually trains more reliably than putting normalization only after residual additions.

## RMSNorm

RMSNorm rescales a vector by its root mean square:

```rust
pub fn rmsnorm(input_vector: &[Value], gain: &[Value]) -> Vec<Value> {
    let mean_square = input_vector
        .iter()
        .fold(Value::new(0.0), |sum, value| sum.add(&value.mul(value)))
        .div_f64(input_vector.len() as f64);
    let scale = mean_square.add_f64(1e-5).powf(-0.5);
    input_vector
        .iter()
        .zip(gain.iter())
        .map(|(value, gain)| value.mul(&scale).mul(gain))
        .collect()
}
```

Math:

```text
rms(x) = sqrt(mean(x_i^2) + eps)
y_i = gain_i * x_i / rms(x)
```

RMSNorm leaves the mean alone and controls vector magnitude. The learned `gain` vector lets the model choose which channels should be louder or quieter after normalization.

## Causal Self-Attention

For each layer and position, the model projects the normalized hidden state into query, key, and value vectors:

```rust
let query = apply_rotary_position_embedding(
    &linear(&normalized_state, &layer.attention.query_weights, &layer.attention.query_biases),
    position_id,
    config,
);
let key = apply_rotary_position_embedding(
    &linear(&normalized_state, &layer.attention.key_weights, &layer.attention.key_biases),
    position_id,
    config,
);
let value = linear(
    &normalized_state,
    &layer.attention.value_weights,
    &layer.attention.value_biases,
);
```

The new key/value are appended to the layer cache:

```rust
keys[layer_context.layer_index].push(key);
values[layer_context.layer_index].push(value);
```

Because the cache only contains positions already processed, attention is causal by construction.

Attention scores are scaled dot products:

```rust
let dot_product =
    (0..attention_head_size).fold(Value::new(0.0), |sum, head_value_index| {
        sum.add(&head_query[head_value_index].mul(&previous_key[head_value_index]))
    });
dot_product.div_f64((attention_head_size as f64).sqrt())
```

Math:

```text
score_t = dot(q, k_t) / sqrt(head_size)
weights = softmax(scores)
output = sum_t weights_t * v_t
```

Dividing by `sqrt(head_size)` keeps scores from growing too large as head dimension increases.

## Multi-Head Attention

The embedding vector is split into attention heads:

```rust
(0..config.attention_head_count)
    .flat_map(|head_index| {
        let head_start_index = head_index * config.attention_head_size;
        let head_query =
            &query[head_start_index..head_start_index + config.attention_head_size];
        // ...
    })
    .collect()
```

Each head attends independently over a slice of the embedding. The head outputs are concatenated back into one embedding-sized vector.

The reason for multiple heads is representational diversity. One head can learn local character patterns, another can learn punctuation or word-boundary behavior, and another can learn longer dependencies.

## RoPE Position Encoding

The active model uses rotary position embeddings, or RoPE. Instead of adding a learned position vector to the hidden state, RoPE rotates pairs of query/key channels by a position-dependent angle:

```rust
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
```

For a pair `(x0, x1)`, the rotation is:

```text
y0 = x0 * cos(angle) - x1 * sin(angle)
y1 = x0 * sin(angle) + x1 * cos(angle)
```

RoPE is applied to queries and keys, not values. After both vectors have been rotated by their positions, their dot product can carry information about relative distance. Attention can then learn both which previous token is relevant and how far back it is.

## SwiGLU Feed-Forward Block

After attention, each layer applies a feed-forward network with SwiGLU gating:

```rust
let expanded_output = linear(
    &normalized_state,
    &layer.feed_forward.expansion_weights,
    &layer.feed_forward.expansion_biases,
);
let gated_output = linear(
    &normalized_state,
    &layer.feed_forward.gate_weights,
    &layer.feed_forward.gate_biases,
);

let block_output = expanded_output
    .iter()
    .zip(gated_output.iter())
    .map(|(expanded_value, gate_value)| silu(expanded_value).mul(gate_value))
    .collect::<Vec<_>>();
```

The SiLU activation is:

```rust
pub fn silu(value: &Value) -> Value {
    let sigmoid = value.neg().exp().add_f64(1.0).powf(-1.0);
    value.mul(&sigmoid)
}
```

Math:

```text
sigmoid(x) = 1 / (1 + exp(-x))
silu(x) = x * sigmoid(x)
swiglu(x) = silu(W_expand x + b_expand) * (W_gate x + b_gate)
```

The gate lets the model suppress or emphasize candidate features based on context.

## Residual Dropout

The CPU backend applies dropout to residual-branch updates during training:

```rust
let keep_probability = 1.0 - RESIDUAL_DROPOUT_PROBABILITY;
let multiplier = if random < keep_probability {
    1.0 / keep_probability
} else {
    0.0
};
*value = value.mul_f64(multiplier);
```

This is inverted dropout. Kept values are scaled by `1 / keep_probability` so the expected activation scale is unchanged.

Dropout is deterministic:

```rust
deterministic_unit_float(
    dropout_context.step,
    dropout_context.batch_offset,
    layer_index,
    position_id,
    stream,
    channel_index,
)
```

The pseudo-random value is a pure function of training location. This makes checkpoint restore reproducible without serializing a dropout RNG stream.

Dropout is only active during training windows. Generation and validation call forward functions without a dropout context.

## Softmax And Cross-Entropy

Softmax converts logits into probabilities:

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

The implementation subtracts the max logit before exponentiating for numerical stability:

```rust
let max_logit_value = logits
    .iter()
    .map(Value::data)
    .fold(f64::NEG_INFINITY, f64::max);
```

Cross-entropy measures surprise at the correct target token:

```rust
pub fn cross_entropy_loss(logits: &[Value], target_token_id: usize) -> Value {
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
```

Math:

```text
loss = -log softmax(logits)[target]
     = log(sum_j exp(logit_j)) - logit_target
```

With the max-logit trick:

```text
loss = log(sum_j exp(logit_j - m)) + m - logit_target
```

where `m = max(logits)`.

## Teacher Forcing

Training uses teacher forcing. At every position, the model receives the true previous token rather than its own sampled output:

```rust
let token_id = token_window.input_tokens[position_id];
let target_token_id = token_window.target_tokens[position_id];
let model_run = run_transformer_model_with_dropout(...);
let position_loss = cross_entropy_loss(&model_run.logits, target_token_id);
```

Teacher forcing makes text training straightforward: every plain sentence provides many labeled next-token predictions.

## Document Loss

The loss for one window is the mean of active position losses:

```rust
let mut loss = Value::new(0.0);
let mut active_position_count = 0.0;

for position_id in 0..config.context_window_size {
    let mask = token_window.loss_mask[position_id];
    // ...
    loss = loss.add(&position_loss.mul_f64(mask));
    active_position_count += mask;
}

loss.mul_f64(1.0 / active_position_count.max(1.0))
```

The mask excludes padding. Averaging by active positions keeps short and long windows on a comparable loss scale.

## Mini-Batch Training

One CPU training step computes gradients for several windows and averages them:

```rust
let document_results: Vec<_> = batch_windows
    .par_iter()
    .map(|token_window| {
        train_on_document_with_gradients(
            &session.trained_microgpt.model,
            &session.trained_microgpt.config,
            step,
            token_window,
            &parameter_index_by_value,
            parameter_count,
        )
    })
    .collect();
```

Rayon parallelism works here because every document/window builds an independent computation graph. The shared model is read-only. Each worker returns:

```rust
struct DocumentTrainingResult {
    loss: f64,
    parameter_gradients: Vec<f64>,
}
```

Then gradients are averaged:

```rust
for document_result in &document_results {
    for (parameter_index, gradient) in document_result.parameter_gradients.iter().enumerate() {
        accumulated_parameter_gradients[parameter_index] += gradient;
    }
}

let inverse_document_count = 1.0 / batch_windows.len() as f64;
for gradient in &mut accumulated_parameter_gradients {
    *gradient *= inverse_document_count;
}
```

This is mini-batch gradient descent. It is noisier than full-batch training, but much cheaper and usually steadier than one example at a time.

## Gradient Clipping

Before AdamW, gradients are globally clipped:

```rust
fn clipped_gradients(gradients: &[f64], max_norm: f64) -> Vec<f64> {
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
```

Math:

```text
norm = sqrt(sum_i g_i^2)
if norm > max_norm:
    g_i = g_i * max_norm / norm
```

This preserves gradient direction while limiting step size. It protects training from rare batches that produce very large gradients.

## AdamW Optimizer

The CPU backend uses AdamW-style updates. Optimizer state contains two vectors aligned with the flattened model parameters:

```rust
pub struct AdamOptimizerState {
    pub first_moment_estimates: Vec<f64>,
    pub second_moment_estimates: Vec<f64>,
}
```

For each parameter:

```rust
let first_moment_estimate = beta1 * old_m + (1.0 - beta1) * gradient;
let second_moment_estimate = beta2 * old_v + (1.0 - beta2) * gradient.powi(2);

let m_hat = first_moment_estimate / (1.0 - beta1.powf(step as f64 + 1.0));
let v_hat = second_moment_estimate / (1.0 - beta2.powf(step as f64 + 1.0));

let adam_update = m_hat / (v_hat.sqrt() + epsilon);
let decay_update = weight_decay * parameter.data();
let parameter_update = learning_rate * (adam_update + decay_update);
```

Math:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2

m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
```

The weight-decay term is decoupled in the AdamW style: it directly pulls parameters toward zero instead of being mixed into the raw gradient before moment estimation.

## Learning-Rate Schedule

The project uses linear warmup followed by cosine decay:

```rust
pub fn scheduled_learning_rate(
    optimizer_config: &AdamOptimizerConfig,
    step: usize,
    training_step_count: usize,
) -> f64
```

Warmup:

```text
lr = base_lr * (step + 1) / warmup_steps
```

After warmup:

```text
progress = scheduled_step / scheduled_step_count
cosine_decay = 0.5 * (1 + cos(pi * progress))
lr = base_lr * (min_ratio + (1 - min_ratio) * cosine_decay)
```

Warmup prevents very large early updates while Adam's moving averages are still poorly calibrated. Cosine decay gives fast early learning and smaller late updates.

## Validation Loss

Validation loss uses held-out documents. `config/src/lib.rs` computes it periodically:

```rust
if result.session.completed_step_count >= *next_validation_step {
    let validation_loss = calculate_cpu_validation_loss(
        &result.session,
        result.session.completed_step_count,
        training_run_config.validation_step_interval,
    );
    result = attach_cpu_validation_loss(result, validation_loss);
    *next_validation_step += training_run_config.validation_step_interval;
}
```

The CPU validation function rotates through validation documents in batches:

```rust
let validation_batch_index = completed_step_count / validation_step_interval;
let validation_index = (validation_batch_index * validation_document_count
    + validation_offset)
    % session.validation_documents.len();
```

Validation loss is weighted by prediction count:

```rust
weighted_loss_sum += document_loss * document_token_count as f64;
token_count += document_token_count;
```

Final value:

```text
validation_loss = sum(document_loss * document_prediction_count)
                  / sum(document_prediction_count)
```

This avoids giving a short sentence and a long sentence equal weight when the long sentence contributed many more next-token predictions.

## Perplexity

The app can display perplexity from loss:

```rust
pub fn format_perplexity(loss: f64) -> String {
    let perplexity = loss.exp();
    // formatting...
}
```

Perplexity is:

```text
perplexity = exp(cross_entropy_loss)
```

Interpretation: if loss is `log(N)`, perplexity is `N`, roughly meaning the model is as uncertain as choosing among `N` equally likely tokens. Lower is better.

For this model, perplexity is measured per character. It is a different scale from word-level or subword-level LLM perplexity.

## Progress History

Each step appends a `MicrogptTrainingProgress`:

```rust
pub struct MicrogptTrainingProgress {
    pub completed_step_count: usize,
    pub training_step_count: usize,
    pub loss: f64,
    pub validation_loss: Option<f64>,
}
```

The app also computes a smoothed running mean:

```rust
previous_loss * (1.0 - RUNNING_MEAN_LOSS_RECENT_WEIGHT)
    + progress.loss * RUNNING_MEAN_LOSS_RECENT_WEIGHT
```

This gives the UI a steadier curve than the raw mini-batch loss alone.

## Frame-Budgeted Training

The GUI should remain interactive, so training happens in short chunks. The shared config crate defines:

```rust
pub const TRAINING_FRAME_BUDGET: Duration = Duration::from_millis(500);
```

Training continues until one of these happens:

- the session is complete
- validation was attached on this chunk
- the next validation step is reached
- the frame budget expires

CPU path:

```rust
fn train_cpu_until_budget(
    mut session: MicrogptTrainingSession,
    next_validation_step: &mut usize,
    frame_start: Instant,
    training_run_config: TrainingRunConfig,
) -> MicrogptTrainingSession
```

The app runs this work in `spawn_blocking`:

```rust
tokio::task::spawn_blocking(move || {
    train_session_until_budget(session, next_validation_step, training_run_config)
})
```

Scalar graph construction and backpropagation run away from the UI runtime.

## Pausing And Queued Work

The app tracks whether training is active, busy, or queued:

```rust
is_training_active: bool,
is_training_busy: bool,
manual_training_chunk_requested: bool,
```

The worker takes training work only when appropriate:

```rust
if self.is_training_busy {
    return None;
}
let session = self.session.as_ref()?;
if session.is_complete() {
    self.is_training_active = false;
    self.manual_training_chunk_requested = false;
    return None;
}
if !self.is_training_active && !self.manual_training_chunk_requested {
    return None;
}

self.is_training_busy = true;
self.manual_training_chunk_requested = false;
Some((session.clone(), self.next_validation_step))
```

The UI keeps the current session until the worker returns a complete updated session. This avoids partial state updates during a training step.

## Training Config State

The UI stores separate staged config state for each backend:

```rust
training_run_config: TrainingRunConfig,
mlx_training_run_config: TrainingRunConfig,
cpu_training_run_config: TrainingRunConfig,
```

`training_run_config` is the config applied to the current session and checkpoints. The per-backend configs are editable staged values.

The selected staged config depends on the active backend:

```rust
fn selected_training_run_config(&self) -> TrainingRunConfig {
    match self.backend {
        Backend::Mlx => self.mlx_training_run_config,
        Backend::Cpu => self.cpu_training_run_config,
    }
}
```

When training starts, the staged config is applied if needed:

```rust
fn apply_selected_training_config(&mut self) -> bool {
    if !self.can_configure_training_run() {
        return self.session.is_some();
    }
    let selected_training_run_config = self.selected_training_run_config();
    if self.session.is_some() && self.training_run_config == selected_training_run_config {
        return true;
    }
    self.recreate_session_with_config(selected_training_run_config);
    self.session.is_some()
}
```

This prevents every UI config edit from immediately destroying and recreating the current training session. There is also an explicit default-restoration path for the selected backend.

## Checkpoints

Checkpoints use a shared serialized format:

```rust
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
```

CPU export flattens parameters into tensors:

```rust
let parameter_tensors = session.trained_microgpt.model.checkpoint_tensors();
```

Optimizer moments are stored with shapes matching the parameter tensors. Import checks tensor counts, shapes, and value counts, then reconstructs a CPU model with `Value::new` leaves.

Checkpoint files start with a magic header:

```rust
const CHECKPOINT_MAGIC: &[u8; 8] = b"MGPTCKP1";
```

That lets the loader reject unrelated files before trying to deserialize them.

## Generation

Generation is autoregressive:

```text
start with <BOS>
run model
sample next token
append sampled character
feed sampled token back in
repeat until <EOS> or context window ends
```

The CPU function:

```rust
pub fn generate_sample(
    model: &TransformerModelParameters,
    config: &TransformerConfig,
    tokenizer: &CharacterTokenizer,
    prefix: &str,
    temperature: f64,
    random_number_generator: &mut impl Rng,
) -> String
```

A prefix is forced into context before free sampling:

```rust
if let Some(prefix_character) = prefix_characters.get(position_id) {
    token_id = tokenizer.character_to_token_id[prefix_character];
    continue;
}
```

Temperature rescales logits:

```rust
let scaled_logits: Vec<_> = model_run
    .logits
    .iter()
    .map(|logit| logit.div_f64(temperature))
    .collect();
```

Math:

```text
probabilities = softmax(logits / temperature)
```

Lower temperature sharpens the distribution. Higher temperature flattens it.

The backend applies sampling constraints:

- do not end before a minimum generated length
- do not sample a leading or repeated space in certain positions
- keep only the top `k` probabilities
- fall back to uniform probabilities if all constraints zero out the distribution

Top-k sampling:

```rust
fn keep_top_k(probabilities: &mut [f64], top_k: usize) {
    let mut sorted_probabilities = probabilities.to_vec();
    sorted_probabilities.sort_by(|left, right| right.total_cmp(left));
    let threshold = sorted_probabilities[top_k - 1];
    for probability in probabilities {
        if *probability < threshold {
            *probability = 0.0;
        }
    }
}
```

Sampling itself is weighted random choice:

```rust
let mut random_threshold = random_number_generator.gen::<f64>() * total;
for (weight_index, weight) in weights.iter().enumerate() {
    random_threshold -= weight;
    if random_threshold <= 0.0 {
        return weight_index;
    }
}
```

## CPU Versus MLX

Both backends implement the same model family and training objective, but they differ in execution style.

CPU backend:

```text
many scalar Value nodes
explicit computation graph
manual reverse-mode autodiff
one position at a time
document-level Rayon parallelism
scalar and inspectable, with low throughput
```

MLX backend:

```text
batched tensors
framework autodiff
full-sequence tensor operations
Apple Silicon acceleration
batched and much faster, with more work handled inside MLX
```

The CPU backend's role is clarity rather than throughput. It shows the scalar operations that the tensor backend collects into larger array computations.

## Training Features And Techniques

The project combines several modern LLM training ideas in miniature:

- Character-level language modeling for inspectability.
- Decoder-only Transformer architecture.
- Causal self-attention.
- KV cache during sequential CPU forward passes and generation.
- Multi-head attention.
- RoPE positional encoding.
- Pre-norm residual blocks.
- RMSNorm with learned gains.
- SwiGLU feed-forward blocks.
- Tied input embeddings and output classifier weights.
- Cross-entropy next-token objective.
- Teacher forcing.
- Deterministic random training windows.
- Mini-batch gradient averaging.
- Rayon document-level CPU parallelism.
- Residual dropout during training.
- Global gradient norm clipping.
- AdamW optimization.
- Linear warmup and cosine learning-rate decay.
- Story-level train/validation split.
- Token-weighted validation loss.
- Validation scheduling independent of training steps.
- Checkpoint save/load with optimizer state.
- Top-k sampling with temperature and decoding constraints.
- Background worker training for UI responsiveness.

## Simplifications

Several parts are kept small enough to read in one sitting:

- Tokenization is character-based rather than subword-based.
- There is no fused kernel or optimized tensor math in the CPU backend.
- CPU attention runs one position at a time.
- Context windows are short.
- The dataset is filtered heavily for short, simple examples.
- Sampling constraints are handwritten decoding rules.
- There is no distributed training, mixed precision, or gradient accumulation across many device steps.

The result is still a GPT-style training loop, but with the surrounding machinery kept small enough to inspect.

## CPU Training Step As A Pipeline

A CPU training step can be read as:

```text
1. Flatten current model parameters.
2. Build a map from Value node id to parameter index.
3. Choose deterministic token windows for this step.
4. For each window, in parallel:
   a. Run the Transformer one position at a time.
   b. Accumulate masked cross-entropy loss.
   c. Backpropagate from scalar loss to parameter gradients.
5. Average gradients across windows.
6. Clip the global gradient vector.
7. Compute scheduled learning rate.
8. Apply AdamW to every parameter.
9. Rebuild the nested model with new leaf Values.
10. Append progress and return the updated session.
```

The essential equation is still the language-model objective:

```text
minimize average_t -log P(token[t + 1] | token[0..t])
```

Everything else in the backend exists to make that objective trainable, stable, reproducible, inspectable, and usable from the desktop app.
