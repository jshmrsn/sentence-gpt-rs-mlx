use std::time::Duration;

pub use microgpt_lib::microgpt::AdamOptimizerConfig;

pub const TRAINING_FRAME_BUDGET: Duration = Duration::from_millis(500);
pub const VALIDATION_STEP_INTERVAL: usize = 25;
pub const TRAINING_DOCUMENT_BATCH_SIZE: usize = 20;
pub const MAX_DOCUMENT_COUNT: usize = 20000;
pub const MAX_TRAINING_STEP_COUNT: usize = 8_000;
pub const VALIDATION_SET_DIVISOR: usize = 20;
pub const VALIDATION_EVALUATION_DOCUMENT_COUNT: usize = 12;
pub const CONTEXT_WINDOW_SIZE: usize = 50;
pub const LAYER_COUNT: usize = 4;
pub const ATTENTION_HEADS: usize = 8;
pub const EMBEDDING_SIZE: usize = 64;

pub fn get_optimizer_config() -> AdamOptimizerConfig {
    AdamOptimizerConfig {
        learning_rate: 0.01,
        first_moment_decay: 0.85,
        second_moment_decay: 0.99,
        epsilon: 1e-8,
        weight_decay: 0.01,
        warmup_step_count: 100,
        minimum_learning_rate_ratio: 0.1,
    }
}
