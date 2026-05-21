# sentence-gpt-rs-mlx

GPT training demo based on microgpt, but targeting full simple sentences, accelerated on Apple Silicon with MLX (via mlx-rs) and additional optimizations relative to microgpt, written in Rust, with both Dioxus GUI and ratatui TUI frontends.
The alternative CPU backend is more closely derived from the original microgpt (refactored to be functional/immutable and use structures over dictionaries), but using it requires manually reducing model size parameters.
The repo is largely written by Codex GPT 5.5, but engineered over many feedback iterations.
Includes many LLM-generated inline comments to describe machine learning techniques and math, which enables this repo to be a helpful learning resource for people like myself who have limited background in ML.

![sentence-gpt-rs-mlx GUI screenshot](screenshot.png)

## Project Layout

- `lib/` contains the core character-level Transformer training code, checkpoint format, dry Rust CPU backend, and MLX backend.
- `config/` contains shared app/TUI configuration, story loading, backend/session orchestration, training chunk scheduling, and display formatting.
- `app/` contains the Dioxus desktop GUI.
- `terminal/` contains the ratatui terminal UI.
- `data/input-stories-00.json` is the required training corpus.

## Model And Data

The project trains a tiny GPT-style character model on short story sentences. It keeps per-character tokens instead of using BPE or sentencepiece-style tokenization, so model behavior remains easy to inspect while still training toward full simple-sentence generation.
Stories are originally sourced from https://huggingface.co/datasets/roneneldan/TinyStories
Stories are split into sentence-like examples on `.`, `?`, and `!` while keeping the punctuation, and then builds train/validation splits during session creation.

The default backend is MLX on Apple Silicon. The dry Rust CPU backend is kept for comparison, learning, and debugging.

Current architecture features include:

- batched full-sequence MLX training
- causal self-attention with RoPE
- SwiGLU feed-forward blocks
- RMSNorm with learned gains and final norm
- tied token embeddings / LM head
- learned biases
- AdamW-style optimization with warmup and cosine decay
- visual validation loss tracking graph
- checkpoint export/import
- sample generation available during training
- sample allows specifying a prefix
- GUI allows for most similar training example for a selected generated sample

Currently, the GUI frontend is more feature-rich and tested than the TUI frontend.
There are some model visualization features in the GUI, but they are not well thought through yet.
The project started as a Kotlin port of Andrej Karpathy's microgpt, but then I ported a second time to Rust so I could leverage MLX via mlx-rs.
https://github.com/oxiglade/mlx-rs

## Prerequisites

The MLX backend needs native Apple build tooling:

```sh
brew install cmake
xcodebuild -downloadComponent MetalToolchain
```

`mlx-rs` is enabled by default for `sentence-gpt-rs-mlx-lib`.

## Run

Dioxus desktop GUI:

```sh
cargo run --release -p sentence-gpt-rs-mlx-app
```

ratatui terminal UI:

```sh
cargo run --release -p sentence-gpt-rs-mlx-tui
```

## Optimized Build

Use `--release` for training performance:

```sh
cargo run --release -p sentence-gpt-rs-mlx-app
cargo run --release -p sentence-gpt-rs-mlx-tui
```

Build optimized binaries without running them:

```sh
cargo build --release -p sentence-gpt-rs-mlx-app
cargo build --release -p sentence-gpt-rs-mlx-tui
```

The binaries land under `target/release/`.

## Check And Test

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## GUI Features

- Start, pause, reset, and one-chunk training controls.
- Runtime backend switching between MLX and dry Rust CPU backends.
- Training and validation loss metrics.
- Responsive running-mean loss trend line.
- Prefix and temperature sample generation.
- Training/validation document browser with cached search results.
- Generated-sample-to-training-search shortcuts.
- Optional parameter heatmaps for embeddings, LM head, and Transformer layers.
- Checkpoint import/export plus automatic validation-step snapshots.

## TUI Features

- `s` starts or pauses background training.
- `c` runs one background training chunk.
- `g` or `Enter` generates samples.
- `b` toggles between MLX and dry Rust CPU backends, resetting the current session.
- `v` toggles parameter-value visualization.
- `F5` exports a checkpoint and `F6` imports it.
- Typed characters edit the prefix; `Backspace` deletes.
- `+` and `-` adjust temperature.
- `r` resets and `q` or `Esc` quits.
