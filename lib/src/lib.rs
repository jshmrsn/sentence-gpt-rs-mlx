//! Teaching-oriented microGPT library.
//!
//! `microgpt` is the plain Rust reference implementation. It uses scalar
//! reverse-mode automatic differentiation so every piece of the math is visible.
//! `mlx_microgpt` keeps the same model and training loop, but stores numbers in
//! MLX tensors so Apple Silicon can do the matrix math much faster. When learning
//! the model, read `value.rs` first, then `microgpt.rs`, then compare the tensor
//! version in `mlx_microgpt.rs`.

pub mod microgpt;
#[cfg(feature = "mlx")]
pub mod mlx_microgpt;
pub mod value;
