# microgpt Rust Visualized

This repository now contains a Rust port of the original Kotlin Multiplatform Compose microgpt demo.

- `lib/` contains the Rust LLM training library, including an MLX-backed tensor implementation optimized for Apple Silicon.
- `app/` contains the Dioxus desktop UI.
- `terminal/` contains the Ratatui terminal UI.
- `shared/src/commonMain/composeResources/files/` is still used as the source for the bundled training data.

The original scalar reverse-mode autograd code remains in `microgpt_lib::microgpt`. The default app and TUI now train through `microgpt_lib::mlx_microgpt`, which stores parameters as MLX arrays and runs the Transformer matrix math, gradients, Adam updates, validation loss, and autoregressive sampling through `mlx-rs`.

Both backends keep character-level tokens. Input documents are normalized to lowercase `a-z` plus spaces before tokenization, which keeps the vocabulary focused on simple sentence generation.

## Prerequisites

The MLX backend needs the native Apple build tools:

```sh
brew install cmake
xcodebuild -downloadComponent MetalToolchain
```

`mlx-rs` is enabled by default for `microgpt-lib`.

## Run

Dioxus desktop app:

```sh
cargo run --release -p microgpt-app
```

Ratatui terminal app:

```sh
cargo run --release -p microgpt-tui
```

## Optimized Run And Build

Use `--release` for optimized training performance:

```sh
cargo run --release -p microgpt-app
cargo run --release -p microgpt-tui
```

Build optimized binaries without running them:

```sh
cargo build --release -p microgpt-app
cargo build --release -p microgpt-tui
```

The binaries land under `target/release/`.

## Check And Test

```sh
cargo fmt --all -- --check
cargo check -p microgpt-app
cargo check -p microgpt-tui
cargo test --workspace
```

## Project Shape

The scalar library mirrors the original Kotlin `shared/src/commonMain/kotlin/org/jshmrsn/microgpt/lib` logic:

- `Value` is the scalar autograd node.
- `TransformerModelParameters` owns token embeddings, position embeddings, attention weights, feed-forward weights, and the language-model head.
- `train_microgpt_step` batches documents, accumulates gradients, and applies Adam on CPU.
- `generate_sample` and `generate_samples` run autoregressive character sampling.
- Training uses log-sum-exp cross-entropy, residual-scaled initialization, RoPE attention, SwiGLU feed-forward blocks, global gradient clipping, AdamW-style decay, and warmup plus cosine learning-rate decay.
- Sampling uses top-k filtering plus simple sentence constraints to avoid immediate end tokens, leading spaces, and repeated spaces.

The Dioxus app mirrors the Compose UI flow:

- Start, pause, reset, and chunk training controls.
- Runtime backend switching between MLX and the dry Rust CPU backend. Switching resets the current session.
- Training and validation loss metrics.
- Loss history chart.
- Prefix and temperature sample generation.
- Parameter heatmaps for embeddings, language head, and each Transformer layer.

The Ratatui app provides the same training and sampling loop in the terminal:

- `s` starts or pauses background training.
- `c` runs one background training chunk.
- `g` or `Enter` generates samples.
- `b` toggles between the MLX and dry Rust CPU backends, resetting the current session.
- Typed characters edit the prefix; `Backspace` deletes.
- `+` and `-` adjust temperature.
- `r` resets and `q` or `Esc` quits.
