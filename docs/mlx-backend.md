# MLX Backend Overview

This document walks through the MLX backend in `lib/src/mlx_microgpt.rs`: the shape of the code, the model it trains, and the machine-learning ideas used along the way. The backend trains the same small character-level GPT-style model as the CPU implementation, with the arithmetic written as MLX tensor operations so Apple Silicon can do the larger matrix work efficiently.

The main difference from the CPU backend is the unit of computation:

- The CPU backend builds many scalar `Value` objects and runs explicit reverse-mode autodiff over that graph.
- The MLX backend stores parameters and activations as MLX `Array` tensors and asks MLX to compute gradients with `value_and_grad`.

The learning problem is the same in both cases. MLX changes how the arithmetic is represented and scheduled.

## Source Map

The main implementation lives in:

- `lib/src/mlx_microgpt.rs`: MLX model, training, validation, generation, optimizer, tensor math.
- `lib/src/microgpt.rs`: shared tokenizer, config types, sampling helpers, data windows, CPU reference behavior.
- `config/src/lib.rs`: app-level session creation, story loading, train/validation split orchestration, training chunk scheduling.

The central MLX functions are:

- `create_mlx_microgpt_training_session_from_splits`
- `train_mlx_microgpt_step`
- `batch_loss`
- `run_transformer_batch`
- `run_transformer_layer_batch`
- `run_multi_head_attention_batch`
- `apply_rotary_position_embedding_batch`
- `rmsnorm_last_axis`
- `apply_adam_update`
- `generate_sample`

## Model Overview

The model is a small decoder-only Transformer trained as a character-level language model.

Given a sequence of characters, it learns:

```text
input:   <BOS> T h e   c a t
target:  T     h e   c a t <EOS>
```

At each position, the model sees the true previous characters and predicts the next character. This is teacher forcing.

The model has:

- Character token embeddings.
- RoPE positional encoding on attention query/key vectors.
- Stacked causal self-attention blocks.
- SwiGLU feed-forward blocks.
- RMSNorm before attention, before feed-forward, and before logits.
- Residual connections.
- Tied input/output embeddings.
- AdamW optimization with warmup/cosine scheduling and global gradient clipping.
- Dropout on residual branch updates during training.

## Why Character-Level?

The tokenizer uses one token per character in the dataset, plus a boundary token. The boundary token is used both as a "start of sequence" and "end of sequence" marker.

That choice keeps the model close to the text:

- No BPE or SentencePiece vocabulary.
- No subword merge rules.
- The target at each position is the next character.

The cost is sequence length. A word is several training steps, not one token. The model has to learn spelling, word boundaries, punctuation, and sentence structure from character transitions.

## Parameters And Tensor Shapes

The MLX parameter structs mirror the CPU backend, but each matrix/vector is an MLX `Array`.

```rust
pub struct MlxTransformerModelParameters {
    pub token_embedding: Array,
    pub position_embedding: Array,
    pub language_model_head: Array,
    pub language_model_head_biases: Array,
    pub final_norm_gain: Array,
    pub layers: Vec<MlxTransformerLayerParameters>,
}
```

Active shapes:

```text
token_embedding:              [vocab_size, embedding_size]
language_model_head_biases:   [vocab_size]
final_norm_gain:              [embedding_size]

attention projection weights: [embedding_size, embedding_size]
attention projection biases:  [embedding_size]

SwiGLU expansion weights:     [3 * embedding_size, embedding_size]
SwiGLU gate weights:          [3 * embedding_size, embedding_size]
SwiGLU projection weights:    [embedding_size, 3 * embedding_size]
```

Two fields are retained mostly for compatibility:

- `position_embedding`: the active MLX forward pass uses RoPE instead.
- `language_model_head`: the active output head uses tied token embeddings instead.

## Session Creation

The app now splits data at story granularity before flattening sentences. The MLX backend receives explicit train and validation sentence lists:

```rust
pub fn create_mlx_microgpt_training_session_from_splits(
    input_documents: Vec<String>,
    input_validation_documents: Vec<String>,
    random_number_generator: &mut impl Rng,
    training_step_count: usize,
    validation_evaluation_document_count: usize,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
) -> MlxMicrogptTrainingSession
```

Inside the constructor:

1. Empty documents are filtered out.
2. A character vocabulary is built from train plus validation documents.
3. The boundary token id is assigned after all real characters.
4. The model parameters are initialized.
5. Adam first/second moment tensors are initialized as zeros with the same shape as every parameter tensor.

The optimizer state stores tensors with the same shapes as the parameters:

```rust
let parameters = model.values();
let zeros = parameters
    .iter()
    .map(ops::zeros_like)
    .collect::<MlxResult<Vec<_>>>()
    .unwrap();
```

Every trainable parameter has:

- One first-moment estimate, `m`.
- One second-moment estimate, `v`.

Both have the same shape as the parameter itself.

## Training Windows

The MLX backend uses the shared `training_batch_token_windows` helper. Each mini-batch contains fixed-length windows:

```text
input_tokens:  [context_window_size]
target_tokens: [context_window_size]
loss_mask:     [context_window_size]
```

Short sentences are padded with the boundary token. Padding positions get `loss_mask = 0`, so the backend can build rectangular tensors while excluding those positions from the loss.

For a batch of `B` windows and sequence length `T`:

```text
input_tokens:  [B, T]
target_tokens: [B * T, 1]
loss_mask:     [B * T]
```

The fixed shape is what makes the MLX implementation efficient: one batch becomes one tensor graph.

## Training Step

`train_mlx_microgpt_step` is the core training update.

The high-level flow is:

1. Select deterministic training windows.
2. Flatten model parameters into `Vec<Array>`.
3. Build RoPE matrices for the configured context length.
4. Build deterministic dropout masks for this step.
5. Define a closure from parameters to scalar loss.
6. Ask MLX for both loss and gradients.
7. Apply AdamW update.
8. Materialize the new tensors with `eval`.
9. Store a new model/session.

The key autodiff section:

```rust
let loss_fn = move |inputs: &[Array]| -> MlxResult<Vec<Array>> {
    let params = params_from_arrays(inputs, layer_count, &rotary_position_matrices);
    let loss = batch_loss(&params, &config, &batch_windows, Some(&dropout_masks))?;
    Ok(vec![loss])
};

let mut value_and_grad = transforms::value_and_grad_with_argnums(
    loss_fn,
    argnums.as_slice(),
);

let (loss_values, gradients) = value_and_grad(&parameters)?;
```

`argnums` tells MLX which entries in the input slice should receive gradients. Here it is every trainable parameter.

MLX returns:

```text
loss_values[0]: scalar loss tensor
gradients[i]:   d(loss) / d(parameters[i])
```

That replaces the CPU backend's scalar graph traversal.

## Lazy Evaluation In MLX

MLX operations are lazy. Many calls build a computation graph rather than running immediately. This gives MLX room to fuse and schedule work, and it means the Rust code has to mark the points where values are needed on the host.

The backend forces evaluation when it needs host-side values:

```rust
let loss = loss_values[0].item::<f32>() as f64;
```

It also evaluates updated tensors after optimizer steps:

```rust
transforms::eval(parameters.iter())?;
```

That avoids accumulating a long chain of lazy update expressions across many training steps.

## Batched Cross-Entropy Loss

`batch_loss` computes the training objective for a full mini-batch.

The model returns logits with shape:

```text
[batch, sequence, vocabulary]
```

The code flattens this to:

```text
[batch * sequence, vocabulary]
```

Then it gathers the logit assigned to each correct target token and computes cross-entropy:

```text
loss_i = log(sum_j exp(logit_ij)) - logit_i,target
```

The implementation uses `logsumexp` for numerical stability:

```rust
let flat_logits = logits.reshape(&[(batch_size * sequence_len) as i32, vocabulary_size])?;
let log_normalizer = ops::logsumexp_axis(&flat_logits, 1, true)?;
let target_logits = flat_logits
    .take_along_axis(&target_tokens, 1)?
    .reshape(&[-1])?;
let token_losses = (log_normalizer.reshape(&[-1])? - target_logits) * &loss_mask;
```

The final loss is the average over unmasked prediction positions:

```rust
ops::sum(&token_losses, None)
    .and_then(|loss_sum| {
        Ok(loss_sum / (ops::sum(&loss_mask, None)? + Array::from_f32(1e-6)))
    })
```

The small `1e-6` prevents division by zero in degenerate all-padding cases.

## Forward Pass

`run_transformer_batch` maps token ids to logits:

```rust
let mut hidden_state = params.token_embedding.take_axis(input_tokens, 0)?;

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

hidden_state = rmsnorm_last_axis(&hidden_state, params.final_norm_gain)?;
tied_language_model_logits_batch(
    &hidden_state,
    params.token_embedding,
    params.language_model_head_biases,
)
```

Shape flow:

```text
input token ids: [B, T]
embedding lookup: [B, T, C]
transformer layers: [B, T, C]
final norm: [B, T, C]
logits: [B, T, V]
```

Where:

- `B` is batch size.
- `T` is context window size.
- `C` is embedding size.
- `V` is vocabulary size.

## Tied Embeddings

The output head is tied to the input embedding table:

```rust
fn tied_language_model_logits_batch(
    hidden_state: &Array,
    token_embedding: &Array,
    biases: &Array,
) -> MlxResult<Array> {
    Ok(hidden_state.matmul(&token_embedding.transpose()?)? + biases)
}
```

Instead of learning a separate `[vocab, embedding]` output matrix, the model reuses `token_embedding`. For each final hidden state, logits are dot products against every token embedding row.

Benefits:

- Fewer trainable parameters.
- The input and output token spaces are forced to share geometry.
- Common language-modeling trick for small and large models.

## Transformer Layer Layout

Each layer is pre-norm:

```text
x1 = x + Attention(RMSNorm(x))
x2 = x1 + FeedForward(RMSNorm(x1))
```

The code follows that structure:

```rust
let residual_state = hidden_state.clone();
let normalized_state = rmsnorm_last_axis(hidden_state, layer.attention_norm_gain)?;
...
let updated_hidden_state = block_output + residual_state;

let residual_state = updated_hidden_state.clone();
let normalized_state = rmsnorm_last_axis(&updated_hidden_state, layer.feed_forward_norm_gain)?;
...
Ok(block_output + residual_state)
```

Pre-norm helps training stability because the residual stream gives gradients a direct path through the network.

## RMSNorm

RMSNorm normalizes by root mean square rather than subtracting a mean like LayerNorm.

For a hidden vector `x`:

```text
rms = sqrt(mean(x_i^2) + epsilon)
y_i = (x_i / rms) * gain_i
```

The batched implementation normalizes only the last axis:

```rust
let mean_square = ops::mean_axis(&ops::square(input)?, -1, true)?;
let scale = (mean_square + Array::from_f32(1e-5)).sqrt()?;
Ok((input / scale) * gain)
```

For activations shaped `[B, T, C]`, RMSNorm reduces over `C`. Batch and time positions are normalized independently.

## Multi-Head Causal Self-Attention

Attention lets every position mix information from earlier positions. Because this is a language model, future positions must be hidden.

Query, key, and value tensors start as:

```text
[B, T, C]
```

They are reshaped into heads:

```text
[B, T, H, D]
```

Then transposed so each head can perform attention independently:

```text
[B, H, T, D]
```

The score matrix is:

```text
scores = Q K^T / sqrt(D)
```

Shape:

```text
[B, H, T, T]
```

The code:

```rust
let key_transposed = key.transpose_axes(&[0, 1, 3, 2])?;
let mut attention_scores = query.matmul(&key_transposed)?
    / Array::from_f32((config.attention_head_size as f32).sqrt());
attention_scores += causal_attention_mask(sequence_len as usize);
let attention_weights = ops::softmax_axis(&attention_scores, -1, None)?;
```

The division by `sqrt(D)` prevents dot products from growing too large as head size grows. Without scaling, softmax can become too sharp early in training.

## Causal Mask

The causal mask prevents tokens from seeing the future:

```text
[ 0   -inf -inf -inf ]
[ 0    0   -inf -inf ]
[ 0    0    0   -inf ]
[ 0    0    0    0   ]
```

The implementation uses `-1.0e9` instead of actual negative infinity:

```rust
if key_position > query_position {
    -1.0e9_f32
} else {
    0.0
}
```

After softmax, masked positions receive probability very close to zero.

## RoPE Positional Encoding

The active MLX forward pass uses Rotary Position Embeddings, or RoPE. RoPE rotates query and key channels by an angle determined by token position.

For each pair of channels:

```text
[x_even']   [ cos(theta)  -sin(theta) ] [x_even]
[x_odd' ] = [ sin(theta)   cos(theta) ] [x_odd ]
```

The angle is:

```text
theta = position * frequency
frequency = 10000 ^ (-(2 * pair_index) / head_size)
```

The code builds dense rotation matrices:

```rust
let frequency =
    10_000.0_f32.powf(-((2 * pair_index) as f32) / config.attention_head_size as f32);
let angle = position_id as f32 * frequency;
let cosine = angle.cos();
let sine = angle.sin();
```

RoPE is applied to queries and keys. Values are left as content vectors:

```rust
let query = apply_rotary_position_embedding_batch(...)?;
let key = apply_rotary_position_embedding_batch(...)?;
let value = linear_last_axis(...)?;
```

Why query/key only? Attention scores are query-key dot products. Rotating queries and keys makes those dot products position-aware, especially relative-position-aware. Values are the content being mixed, so they are left unrotated.

## SwiGLU Feed-Forward Block

After attention, each position runs through a feed-forward block independently.

This backend uses SwiGLU:

```text
expanded = W_exp x + b_exp
gate = W_gate x + b_gate
hidden = SiLU(expanded) * gate
output = W_proj hidden + b_proj
```

Code:

```rust
let expanded_output = linear_last_axis(...)?;
let gated_output = linear_last_axis(...)?;
let block_output = silu(&expanded_output)? * gated_output;
let block_output = linear_last_axis(
    &block_output,
    layer.feed_forward.projection_weights,
    layer.feed_forward.projection_biases,
)?;
```

SiLU is:

```text
SiLU(x) = x / (1 + exp(-x))
```

SwiGLU gives the network a learned multiplicative gate. This is more expressive than a plain MLP with a single activation because one projection controls how much of another projection passes through.

## Residual Dropout

During training, dropout masks are applied to residual branch updates:

```rust
block_output *= &dropout_masks.attention_residual_masks[layer_index];
let updated_hidden_state = block_output + residual_state;
```

And similarly for feed-forward updates.

This drops parts of the block's proposed update while leaving the residual path intact.

The backend uses inverted dropout:

```text
kept value = 1 / keep_probability
dropped value = 0
```

So expected activation scale at training time matches inference time.

The masks are deterministic functions of:

- Training step.
- Layer index.
- Stream id, separating attention masks from feed-forward masks.
- Element index.

Checkpoint resume can rebuild the same mask from the step and shape, so there is no separate dropout RNG state to serialize.

## AdamW Optimizer

The optimizer is AdamW with:

- First moment, `m`, an exponential moving average of gradients.
- Second moment, `v`, an exponential moving average of squared gradients.
- Bias correction.
- Decoupled weight decay.
- Scheduled learning rate.
- Global gradient clipping.

For each parameter tensor:

```text
g = clipped_gradient
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
adam_update = m_hat / (sqrt(v_hat) + epsilon)
decay_update = weight_decay * parameter
parameter = parameter - learning_rate * (adam_update + decay_update)
```

The code uses tensor arithmetic for all of this:

```rust
let new_first_moment =
    first_moment * Array::from_f32(beta1) + &gradient * Array::from_f32(1.0 - beta1);
let new_second_moment = second_moment * Array::from_f32(beta2)
    + ops::square(&gradient)? * Array::from_f32(1.0 - beta2);
let adam_update =
    bias_corrected_first_moment / (bias_corrected_second_moment.sqrt()? + &epsilon);
let decay_update = parameter * Array::from_f32(optimizer_config.weight_decay as f32);
let update = &learning_rate * (adam_update + decay_update);
updated_parameters.push(parameter - update);
```

AdamW differs from classic Adam with L2 regularization because weight decay is decoupled from the gradient moment estimates. Here it appears as a direct parameter shrink term added to the update.

## Global Gradient Clipping

Before AdamW updates, gradients are globally clipped.

The global norm is:

```text
global_norm = sqrt(sum over all parameters and elements of g_i^2)
```

The scale is:

```text
scale = min(1, max_norm / (global_norm + epsilon))
```

The implementation keeps the scale as an MLX scalar tensor:

```rust
let squared_norms = gradients
    .iter()
    .map(|gradient| ops::sum(&ops::square(gradient)?, None))
    .collect::<MlxResult<Vec<_>>>()?;
let total_squared_norm = ops::sum(&ops::stack_axis(&squared_norms, 0)?, None)?;
let scale = Array::from_f32(max_norm) / (total_squared_norm.sqrt()? + Array::from_f32(1e-12));
ops::clip(scale, ((), 1.0))
```

This prevents rare large gradients from producing unstable parameter updates.

## Validation Loss

Validation uses the same forward loss as training, with dropout disabled:

```rust
Ok(batch_loss(&params, config, &[token_window], None)?.item::<f32>() as f64)
```

Important details:

- Validation documents are held out from training.
- The validation subset rotates across validation checks.
- Document losses are weighted by prediction-token count.
- The reported value is still cross-entropy in nats per predicted character.

The validation average is weighted by prediction count. A three-character sentence carries fewer prediction targets than a thirty-character sentence, and the average reflects that. The implementation uses:

```rust
weighted_loss_sum += document_loss * document_token_count as f64;
token_count += document_token_count;
```

Then:

```rust
weighted_loss_sum / token_count
```

## Generation Path

Training uses a full-sequence batched forward pass. Generation uses a one-token-at-a-time forward pass with a key/value cache.

Generation loop:

1. Encode the prefix.
2. Feed tokens one at a time.
3. At each step, sample from the next-token distribution.
4. Append the sampled character.
5. Stop at the boundary token or max length.

The one-token path stores per-layer keys and values:

```rust
let mut keys = create_key_value_cache(config.layer_count);
let mut values = create_key_value_cache(config.layer_count);
```

At each generated position, each layer appends the new key and value:

```rust
keys[layer_index].push(key);
values[layer_index].push(value);
```

Then attention for the current token attends over the cached time steps. This avoids recomputing previous key/value projections from scratch on every generated token.

This generation path uses ordinary Rust vectors of MLX arrays for the cache. Larger inference systems usually use preallocated tensor caches. For this model size, the vector form keeps the data structure easy to follow.

## CPU Backend Versus MLX Backend

The CPU backend is the more explicit reference implementation:

```text
scalar Value graph -> explicit backward pass
```

The MLX backend carries the same computation in batched tensor form:

```text
batched Array graph -> MLX autodiff
```

Conceptual mapping:

```text
CPU Vec<Value> hidden vector       -> MLX Array [embedding]
CPU Vec<Vec<Value>> matrix         -> MLX Array [rows, columns]
CPU loop over positions/documents  -> MLX batch tensor [B, T, C]
CPU backward_for(...)              -> MLX value_and_grad(...)
CPU scalar Adam update             -> MLX tensor AdamW update
```

Both backends share the same high-level model choices and tokenizer behavior. The MLX backend changes how the math is executed; the training objective remains the same.

## Numerical Notes

Several implementation choices are specifically about numerical stability:

- Cross-entropy uses `logsumexp` instead of `log(sum(exp(...)))` directly.
- Attention scores are divided by `sqrt(head_size)`.
- RMSNorm adds `1e-5` before square root.
- Validation loss divides by `sum(mask) + 1e-6`.
- Adam uses `epsilon` in the denominator.
- Gradient clipping limits update spikes.

These details are modest in code, but they keep the numerical scale well behaved, especially around attention softmax and adaptive optimizer denominators.

## Training Step As A Pipeline

A single MLX training step can be read as:

```text
documents
  -> deterministic fixed-length token windows
  -> [B, T] input token tensor
  -> embedding lookup [B, T, C]
  -> repeated Transformer layers
       -> RMSNorm
       -> Q/K/V projections
       -> RoPE on Q/K
       -> causal multi-head attention
       -> residual add
       -> RMSNorm
       -> SwiGLU MLP
       -> residual add
  -> final RMSNorm
  -> tied output logits [B, T, V]
  -> masked next-character cross-entropy
  -> MLX value_and_grad
  -> global gradient clipping
  -> AdamW update
  -> new immutable session/model state
```

That pipeline is the backend: a batch of token windows is turned into a scalar loss, MLX differentiates the loss with respect to the parameters, and AdamW produces the next model state.
