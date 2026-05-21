# microgpt Rust Visualized

This repository now contains a Rust port of the original Kotlin Multiplatform Compose microgpt demo.

- `lib/` contains the CPU-only LLM training library.
- `app/` contains the Dioxus desktop UI.
- `terminal/` contains the Ratatui terminal UI.
- `shared/src/commonMain/composeResources/files/` is still used as the source for the bundled training data.

The training implementation is deliberately dry Rust: scalar reverse-mode autograd, character tokenization, a small Transformer, cross-entropy loss, Adam updates, validation loss, and autoregressive sampling. It does not use tensor libraries or GPU acceleration.

## Run

Dioxus desktop app:

```sh
cargo run -p microgpt-app
```

Ratatui terminal app:

```sh
cargo run -p microgpt-tui
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

The library mirrors the original Kotlin `shared/src/commonMain/kotlin/org/jshmrsn/microgpt/lib` logic:

- `Value` is the scalar autograd node.
- `TransformerModelParameters` owns token embeddings, position embeddings, attention weights, feed-forward weights, and the language-model head.
- `train_microgpt_step` batches documents, accumulates gradients, and applies Adam on CPU.
- `generate_sample` and `generate_samples` run autoregressive character sampling.

The Dioxus app mirrors the Compose UI flow:

- Start, pause, reset, and chunk training controls.
- Training and validation loss metrics.
- Loss history chart.
- Prefix and temperature sample generation.
- Weight heatmaps for embeddings, language head, and each Transformer layer.

The Ratatui app provides the same training and sampling loop in the terminal:

- `s` starts or pauses background training.
- `c` runs one background training chunk.
- `g` or `Enter` generates samples.
- Typed characters edit the prefix; `Backspace` deletes.
- `+` and `-` adjust temperature.
- `r` resets and `q` or `Esc` quits.
