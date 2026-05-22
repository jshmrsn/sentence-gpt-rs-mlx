# sentence-gpt-rs-mlx

GPT training demo based on Andrej Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/), but targeting full simple sentences, accelerated on Apple Silicon with MLX (via [mlx-rs](https://github.com/oxiglade/mlx-rs)) and additional optimizations relative to microgpt, written in Rust with a [Dioxus](https://dioxuslabs.com/) desktop GUI.

The alternative CPU backend is more closely derived from the original microgpt, but refactored to a functional/immutable style, and to use structures over dictionaries. Switching to the CPU backend will automatically use a smaller default model size, and therefore the output quality will be reduced. 

The project is largely written by Codex GPT 5.5, but engineered over many feedback iterations.

The model code includes many LLM-generated inline comments to describe the machine learning techniques and math, which makes this project a helpful learning resource for people like myself who have limited background in ML.

![sentence-gpt-rs-mlx GUI screenshot](screenshot.png)

## More info

The project trains a tiny GPT-style character model on short story sentences. It keeps per-character tokens instead of using BPE or sentencepiece-style tokenization, so model behavior remains easy to inspect while still training toward full simple-sentence generation.

Stories are originally sourced from [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

Stories are grouped by source story, split into sentence-like examples on `.`, `?`, and `!` while keeping the punctuation, and then separated into train/validation sets at story granularity during session creation.

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

## Prerequisites

The MLX backend needs native Apple build tooling:

```sh
brew install cmake
xcodebuild -downloadComponent MetalToolchain
```

`mlx-rs` is enabled by default for `sentence-gpt-rs-mlx-lib`.

## Build & run

#### Run Desktop GUI app:

```sh
cargo run --release -p sentence-gpt-rs-mlx-app
```

#### Build standalone macOS app (optional, for distribution):

Install the Dioxus CLI (dx) according to the instructions here: https://dioxuslabs.com/learn/0.7/getting_started/#install-the-dioxus-cli

```sh
dx bundle --release --platform desktop --package-types macos --package sentence-gpt-rs-mlx-app
```

The macOS app will be located at `app/dist/SentenceGptRsMlxApp.app`.

## Project Layout

- `lib/` contains the core character-level Transformer training code, checkpoint format, dry Rust CPU backend, and MLX backend.
- `config/` contains shared app configuration, story loading, backend/session orchestration, training chunk scheduling, and display formatting.
- `app/` contains the Dioxus desktop GUI.
- `data/input-stories-00.json` contains the training data.
