use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

use rand_chacha::ChaCha8Rng;
use sentence_gpt_rs_mlx_lib::{
    checkpoint::{CheckpointBackend, MicrogptCheckpoint, TrainingRunConfig},
    microgpt::{
        attach_validation_loss as attach_cpu_validation_loss,
        calculate_training_loss_baseline as calculate_cpu_training_loss_baseline,
        calculate_validation_loss as calculate_cpu_validation_loss,
        create_microgpt_training_session_from_splits,
        export_training_session_checkpoint as export_cpu_training_session_checkpoint,
        import_training_session_checkpoint as import_cpu_training_session_checkpoint,
        scheduled_learning_rate, shuffled_by, train_microgpt_step, CharacterTokenizer,
        MicrogptTrainingProgress, MicrogptTrainingSession, TrainedMicrogpt, TransformerConfig,
    },
    mlx_microgpt::{
        attach_validation_loss as attach_mlx_validation_loss,
        calculate_validation_loss as calculate_mlx_validation_loss,
        create_mlx_microgpt_training_session_from_splits,
        export_training_session_checkpoint as export_mlx_training_session_checkpoint,
        import_training_session_checkpoint as import_mlx_training_session_checkpoint,
        train_mlx_microgpt_step, MlxMicrogptTrainingSession, MlxTrainedMicrogpt,
    },
};
use serde::Deserialize;

pub use sentence_gpt_rs_mlx_lib::microgpt::AdamOptimizerConfig;

pub const TRAINING_FRAME_BUDGET: Duration = Duration::from_millis(500);
pub const MAX_TRAINING_STEP_COUNT: usize = 1_000_000;
pub const RUNNING_MEAN_LOSS_RECENT_WEIGHT: f64 = 0.35;
pub const MLX_DEFAULT_TRAINING_RUN_CONFIG: TrainingRunConfig = TrainingRunConfig {
    validation_step_interval: 25,
    training_document_batch_size: 32,
    max_document_count: 5_000_000,
    validation_set_divisor: 50,
    validation_evaluation_document_count: 12,
    context_window_size: 128,
    layer_count: 6,
    attention_heads: 8,
    embedding_size: 128,
};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Mlx,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Mlx => "MLX",
            Backend::Cpu => "CPU",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Backend::Mlx => Backend::Cpu,
            Backend::Cpu => Backend::Mlx,
        }
    }

    pub fn default_training_run_config(self) -> TrainingRunConfig {
        match self {
            Backend::Mlx => MLX_DEFAULT_TRAINING_RUN_CONFIG,
            Backend::Cpu => CPU_DEFAULT_TRAINING_RUN_CONFIG,
        }
    }

    pub fn from_checkpoint_backend(backend: CheckpointBackend) -> Self {
        match backend {
            CheckpointBackend::Mlx => Backend::Mlx,
            CheckpointBackend::Cpu => Backend::Cpu,
        }
    }
}

#[derive(Clone)]
pub enum TrainingSession {
    Mlx(MlxMicrogptTrainingSession),
    Cpu(MicrogptTrainingSession),
}

impl TrainingSession {
    pub fn backend(&self) -> Backend {
        match self {
            TrainingSession::Mlx(_) => Backend::Mlx,
            TrainingSession::Cpu(_) => Backend::Cpu,
        }
    }

    pub fn is_complete(&self) -> bool {
        match self {
            TrainingSession::Mlx(session) => session.is_complete(),
            TrainingSession::Cpu(session) => session.is_complete(),
        }
    }

    pub fn completed_step_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.completed_step_count,
            TrainingSession::Cpu(session) => session.completed_step_count,
        }
    }

    pub fn training_step_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.training_step_count,
            TrainingSession::Cpu(session) => session.training_step_count,
        }
    }

    pub fn latest_loss(&self) -> Option<f64> {
        match self {
            TrainingSession::Mlx(session) => session.latest_loss,
            TrainingSession::Cpu(session) => session.latest_loss,
        }
    }

    pub fn latest_validation_loss(&self) -> Option<f64> {
        match self {
            TrainingSession::Mlx(session) => session.latest_validation_loss,
            TrainingSession::Cpu(session) => session.latest_validation_loss,
        }
    }

    pub fn training_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.documents.len(),
            TrainingSession::Cpu(session) => session.documents.len(),
        }
    }

    pub fn training_documents(&self) -> &[String] {
        match self {
            TrainingSession::Mlx(session) => session.documents.as_slice(),
            TrainingSession::Cpu(session) => session.documents.as_slice(),
        }
    }

    pub fn validation_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.validation_documents.len(),
            TrainingSession::Cpu(session) => session.validation_documents.len(),
        }
    }

    pub fn validation_documents(&self) -> &[String] {
        match self {
            TrainingSession::Mlx(session) => session.validation_documents.as_slice(),
            TrainingSession::Cpu(session) => session.validation_documents.as_slice(),
        }
    }

    pub fn validation_evaluation_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session
                .validation_evaluation_document_count
                .min(session.validation_documents.len()),
            TrainingSession::Cpu(session) => session
                .validation_evaluation_document_count
                .min(session.validation_documents.len()),
        }
    }

    pub fn tokenizer(&self) -> &CharacterTokenizer {
        match self {
            TrainingSession::Mlx(session) => &session.trained_microgpt.tokenizer,
            TrainingSession::Cpu(session) => &session.trained_microgpt.tokenizer,
        }
    }

    pub fn tokenizer_vocabulary_size(&self) -> usize {
        self.tokenizer().vocabulary_size()
    }

    pub fn config(&self) -> &TransformerConfig {
        match self {
            TrainingSession::Mlx(session) => &session.trained_microgpt.config,
            TrainingSession::Cpu(session) => &session.trained_microgpt.config,
        }
    }

    pub fn parameter_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session
                .trained_microgpt
                .model
                .values()
                .iter()
                .map(|array| {
                    array
                        .shape()
                        .iter()
                        .map(|dimension| *dimension as usize)
                        .product::<usize>()
                })
                .sum(),
            TrainingSession::Cpu(session) => session.trained_microgpt.model.parameter_count(),
        }
    }

    pub fn current_learning_rate(&self) -> f64 {
        match self {
            TrainingSession::Mlx(session) => scheduled_learning_rate(
                &session.optimizer_config,
                session.completed_step_count,
                session.training_step_count,
            ),
            TrainingSession::Cpu(session) => scheduled_learning_rate(
                &session.optimizer_config,
                session.completed_step_count,
                session.training_step_count,
            ),
        }
    }

    pub fn progress_history(&self) -> &[MicrogptTrainingProgress] {
        match self {
            TrainingSession::Mlx(session) => session.progress_history.as_slice(),
            TrainingSession::Cpu(session) => session.progress_history.as_slice(),
        }
    }

    pub fn trained_snapshot(&self) -> TrainedSnapshot {
        match self {
            TrainingSession::Mlx(session) => TrainedSnapshot::Mlx(session.trained_microgpt.clone()),
            TrainingSession::Cpu(session) => TrainedSnapshot::Cpu(session.trained_microgpt.clone()),
        }
    }

    pub fn export_checkpoint(
        &self,
        training_run_config: TrainingRunConfig,
    ) -> Result<MicrogptCheckpoint, String> {
        let mut checkpoint = match self {
            TrainingSession::Mlx(session) => export_mlx_training_session_checkpoint(session),
            TrainingSession::Cpu(session) => Ok(export_cpu_training_session_checkpoint(session)),
        }?;
        checkpoint.training_run_config = Some(training_run_config);
        Ok(checkpoint)
    }

    pub fn import_checkpoint(checkpoint: &MicrogptCheckpoint) -> Result<Self, String> {
        match checkpoint.backend {
            CheckpointBackend::Mlx => {
                import_mlx_training_session_checkpoint(checkpoint).map(TrainingSession::Mlx)
            }
            CheckpointBackend::Cpu => Ok(TrainingSession::Cpu(
                import_cpu_training_session_checkpoint(checkpoint)?,
            )),
        }
    }
}

#[derive(Clone)]
pub enum TrainedSnapshot {
    Mlx(MlxTrainedMicrogpt),
    Cpu(TrainedMicrogpt),
}

pub fn create_training_session(
    input_stories: Vec<TrainingStoryDocuments>,
    rng: &mut ChaCha8Rng,
    backend: Backend,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
    training_run_config: TrainingRunConfig,
) -> Result<TrainingSession, String> {
    let shuffled_stories = shuffled_by(&input_stories, rng);
    let validation_story_count =
        shuffled_stories.len() / training_run_config.validation_set_divisor;
    let validation_documents = flatten_story_sentences(&shuffled_stories[..validation_story_count]);
    let documents = flatten_story_sentences(&shuffled_stories[validation_story_count..]);

    match backend {
        Backend::Mlx => create_mlx_microgpt_training_session_from_splits(
            documents,
            validation_documents,
            rng,
            MAX_TRAINING_STEP_COUNT,
            training_run_config.validation_evaluation_document_count,
            transformer_config,
            optimizer_config,
        )
        .with_initial_progress()
        .map(TrainingSession::Mlx)
        .map_err(|error| error.to_string()),
        Backend::Cpu => {
            let mut session = create_microgpt_training_session_from_splits(
                documents,
                validation_documents,
                rng,
                MAX_TRAINING_STEP_COUNT,
                training_run_config.validation_evaluation_document_count,
                transformer_config,
                optimizer_config,
            );
            let train_loss = calculate_cpu_training_loss_baseline(&session);
            let validation_loss = calculate_cpu_validation_loss(
                &session,
                0,
                training_run_config.validation_step_interval,
            );
            session = session.with_initial_progress(train_loss, validation_loss);
            Ok(TrainingSession::Cpu(session))
        }
    }
}

pub struct TrainingBudgetResult {
    pub session: TrainingSession,
    pub next_validation_step: usize,
    pub elapsed_millis: u128,
}

pub fn train_session_until_budget(
    session: TrainingSession,
    mut next_validation_step: usize,
    training_run_config: TrainingRunConfig,
) -> Result<TrainingBudgetResult, String> {
    let chunk_start = Instant::now();
    let frame_start = Instant::now();
    let session = match session {
        TrainingSession::Mlx(session) => TrainingSession::Mlx(train_mlx_until_budget(
            session,
            &mut next_validation_step,
            frame_start,
            training_run_config,
        )?),
        TrainingSession::Cpu(session) => TrainingSession::Cpu(train_cpu_until_budget(
            session,
            &mut next_validation_step,
            frame_start,
            training_run_config,
        )),
    };

    Ok(TrainingBudgetResult {
        session,
        next_validation_step,
        elapsed_millis: chunk_start.elapsed().as_millis(),
    })
}

fn train_mlx_until_budget(
    mut session: MlxMicrogptTrainingSession,
    next_validation_step: &mut usize,
    frame_start: Instant,
    training_run_config: TrainingRunConfig,
) -> Result<MlxMicrogptTrainingSession, String> {
    loop {
        if session.is_complete() {
            break;
        }

        let mut result =
            train_mlx_microgpt_step(session, training_run_config.training_document_batch_size)
                .map_err(|error| error.to_string())?
                .expect("incomplete MLX session should produce a training step");

        let mut validation_was_attached = false;
        if result.session.completed_step_count >= *next_validation_step {
            let validation_loss = calculate_mlx_validation_loss(
                &result.session,
                result.session.completed_step_count,
                training_run_config.validation_step_interval,
            )
            .map_err(|error| error.to_string())?;
            result = attach_mlx_validation_loss(result, validation_loss);
            *next_validation_step += training_run_config.validation_step_interval;
            validation_was_attached = true;
        }

        let should_stop = result.session.is_complete()
            || validation_was_attached
            || result.session.completed_step_count >= *next_validation_step
            || frame_start.elapsed() >= TRAINING_FRAME_BUDGET;
        session = result.session;

        if should_stop {
            break;
        }
    }

    Ok(session)
}

fn train_cpu_until_budget(
    mut session: MicrogptTrainingSession,
    next_validation_step: &mut usize,
    frame_start: Instant,
    training_run_config: TrainingRunConfig,
) -> MicrogptTrainingSession {
    loop {
        if session.is_complete() {
            break;
        }

        let mut result =
            train_microgpt_step(session, training_run_config.training_document_batch_size)
                .expect("incomplete CPU session should produce a training step");

        let mut validation_was_attached = false;
        if result.session.completed_step_count >= *next_validation_step {
            let validation_loss = calculate_cpu_validation_loss(
                &result.session,
                result.session.completed_step_count,
                training_run_config.validation_step_interval,
            );
            result = attach_cpu_validation_loss(result, validation_loss);
            *next_validation_step += training_run_config.validation_step_interval;
            validation_was_attached = true;
        }

        let should_stop = result.session.is_complete()
            || validation_was_attached
            || result.session.completed_step_count >= *next_validation_step
            || frame_start.elapsed() >= TRAINING_FRAME_BUDGET;
        session = result.session;

        if should_stop {
            break;
        }
    }

    session
}

pub fn next_validation_step_after(
    completed_step_count: usize,
    validation_step_interval: usize,
) -> usize {
    ((completed_step_count / validation_step_interval) + 1) * validation_step_interval
}

pub fn running_mean_loss(progress_history: &[MicrogptTrainingProgress]) -> Option<f64> {
    running_mean_loss_values(progress_history).last().copied()
}

pub fn running_mean_loss_values(progress_history: &[MicrogptTrainingProgress]) -> Vec<f64> {
    let mut smoothed_losses = Vec::new();
    let mut smoothed_loss = None;
    for progress in progress_history {
        if progress.completed_step_count == 0 && progress_history.len() > 1 {
            continue;
        }
        smoothed_loss = Some(match smoothed_loss {
            Some(previous_loss) => {
                previous_loss * (1.0 - RUNNING_MEAN_LOSS_RECENT_WEIGHT)
                    + progress.loss * RUNNING_MEAN_LOSS_RECENT_WEIGHT
            }
            None => progress.loss,
        });
        smoothed_losses.push(smoothed_loss.expect("smoothed loss was just initialized"));
    }
    smoothed_losses
}

pub fn format_loss(loss: f64) -> String {
    format!("{loss:.4}")
}

pub fn format_perplexity(loss: f64) -> String {
    let perplexity = loss.exp();
    if perplexity >= 1_000.0 {
        format!("{perplexity:.0}")
    } else if perplexity >= 100.0 {
        format!("{perplexity:.1}")
    } else {
        format!("{perplexity:.2}")
    }
}

pub fn format_learning_rate(learning_rate: f64) -> String {
    format!("{learning_rate:.6}")
}

pub fn format_percent(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

pub fn format_percent_style(value: f64) -> String {
    format!("{:.3}%", (value * 100.0).clamp(0.0, 100.0))
}

pub fn format_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

pub fn format_compact(value: f64) -> String {
    format!("{value:.3}")
}

#[derive(Deserialize)]
struct Story {
    story: String,
    source: String,
}

#[derive(Clone, Debug)]
pub struct TrainingStoryDocuments {
    sentences: Vec<String>,
}

pub fn load_input_documents(
    training_run_config: TrainingRunConfig,
) -> Result<Vec<TrainingStoryDocuments>, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let stories_path = root.join("data/input-stories-00.json");
    let stories_json = std::fs::read_to_string(&stories_path).map_err(|error| {
        format!(
            "could not read required {}: {error}",
            stories_path.display()
        )
    })?;
    let stories: Vec<Story> = serde_json::from_str(&stories_json)
        .map_err(|error| format!("could not parse {}: {error}", stories_path.display()))?;
    let story_documents = stories_to_sentence_groups(stories, training_run_config);
    if story_documents.is_empty() {
        Err(format!(
            "no training documents survived filtering in required {}",
            stories_path.display()
        ))
    } else {
        Ok(story_documents)
    }
}

fn flatten_story_sentences(stories: &[TrainingStoryDocuments]) -> Vec<String> {
    stories
        .iter()
        .flat_map(|story| story.sentences.iter().cloned())
        .collect()
}

fn stories_to_sentence_groups(
    stories: Vec<Story>,
    training_run_config: TrainingRunConfig,
) -> Vec<TrainingStoryDocuments> {
    const SENTENCE_DISQUALIFYING_CHARACTERS: &[char] = &[
        '$', '&', '"', '“', '”', '(', ')', '*', '\'', '_', '-', '–', '…', '%', '~', '`', '[', ']',
        '{', '}', '\\', ';', '|', '—', 'é', '/', '’', '‘', ':', '0', '1', '2', '3', '4', '5', '6',
        '7', '8', '9',
    ];

    let mut total_document_count = 0usize;
    let mut seen_sentences = HashSet::new();
    let mut story_documents = Vec::new();
    for story in stories.into_iter().filter(|story| story.source == "GPT-4") {
        if total_document_count >= training_run_config.max_document_count {
            break;
        }
        let remaining_document_count =
            training_run_config.max_document_count - total_document_count;
        let sentences = split_sentences_keep_punctuation(&story.story)
            .into_iter()
            .filter(|sentence| {
                !SENTENCE_DISQUALIFYING_CHARACTERS
                    .iter()
                    .any(|excluded| sentence.contains(*excluded))
            })
            .map(|sentence| sentence.replace(['\n'], " ").trim().to_string())
            .filter(|sentence| {
                sentence.len() > 10
                    && sentence.contains(' ')
                    && sentence.chars().count() < training_run_config.context_window_size
            })
            .filter(|sentence| seen_sentences.insert(sentence.clone()))
            .take(remaining_document_count)
            .collect::<Vec<_>>();
        if sentences.is_empty() {
            continue;
        }
        total_document_count += sentences.len();
        story_documents.push(TrainingStoryDocuments { sentences });
    }
    story_documents
}

fn split_sentences_keep_punctuation(story: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut sentence_start = 0;
    for (index, character) in story.char_indices() {
        if matches!(character, '.' | '?' | '!') {
            let sentence_end = index + character.len_utf8();
            sentences.push(story[sentence_start..sentence_end].to_string());
            sentence_start = sentence_end;
        }
    }
    if sentence_start < story.len() {
        sentences.push(story[sentence_start..].to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::{
        create_training_session, get_optimizer_config, split_sentences_keep_punctuation,
        stories_to_sentence_groups, Backend, Story, TrainingRunConfig, TransformerConfig,
    };
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn sentence_splitter_keeps_boundary_punctuation() {
        assert_eq!(
            split_sentences_keep_punctuation("Are you there? Yes! I am."),
            vec!["Are you there?", " Yes!", " I am."]
        );
    }

    #[test]
    fn stories_stay_grouped_before_train_validation_split() {
        let training_run_config = TrainingRunConfig {
            validation_step_interval: 25,
            training_document_batch_size: 2,
            max_document_count: 10,
            validation_set_divisor: 2,
            validation_evaluation_document_count: 2,
            context_window_size: 64,
            layer_count: 1,
            attention_heads: 2,
            embedding_size: 8,
        };
        let stories = stories_to_sentence_groups(
            vec![
                Story {
                    story: "Alpha went home. Alpha ate cake.".into(),
                    source: "GPT-4".into(),
                },
                Story {
                    story: "Ignored went home. Ignored ate cake.".into(),
                    source: "GPT-3.5".into(),
                },
                Story {
                    story: "Beta saw stars. Beta went home.".into(),
                    source: "GPT-4".into(),
                },
            ],
            training_run_config,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let session = create_training_session(
            stories,
            &mut rng,
            Backend::Cpu,
            TransformerConfig::new(1, 8, 64, 2).unwrap(),
            get_optimizer_config(),
            training_run_config,
        )
        .unwrap();

        assert_eq!(session.training_document_count(), 2);
        assert_eq!(session.validation_document_count(), 2);
        assert!(
            session
                .training_documents()
                .iter()
                .all(|document| document.starts_with("Alpha"))
                || session
                    .training_documents()
                    .iter()
                    .all(|document| document.starts_with("Beta"))
        );
        assert!(
            session
                .validation_documents()
                .iter()
                .all(|document| document.starts_with("Alpha"))
                || session
                    .validation_documents()
                    .iter()
                    .all(|document| document.starts_with("Beta"))
        );
        assert_ne!(
            session.training_documents()[0].chars().next(),
            session.validation_documents()[0].chars().next()
        );
    }

    #[test]
    fn duplicate_sentences_are_filtered_globally() {
        let training_run_config = TrainingRunConfig {
            validation_step_interval: 25,
            training_document_batch_size: 2,
            max_document_count: 10,
            validation_set_divisor: 2,
            validation_evaluation_document_count: 2,
            context_window_size: 64,
            layer_count: 1,
            attention_heads: 2,
            embedding_size: 8,
        };
        let stories = stories_to_sentence_groups(
            vec![
                Story {
                    story: "Shared sentence here. Unique alpha sentence.".into(),
                    source: "GPT-4".into(),
                },
                Story {
                    story: "Shared sentence here. Unique beta sentence.".into(),
                    source: "GPT-4".into(),
                },
            ],
            training_run_config,
        );
        let sentences = stories
            .iter()
            .flat_map(|story| story.sentences.iter())
            .collect::<Vec<_>>();

        assert_eq!(sentences.len(), 3);
        assert_eq!(
            sentences
                .iter()
                .filter(|sentence| sentence.as_str() == "Shared sentence here.")
                .count(),
            1
        );
    }
}
