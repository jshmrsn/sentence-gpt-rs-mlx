// Dioxus desktop app for watching the tiny Transformer train.
//
// The ML concepts live in `sentence-gpt-rs-mlx-lib`; this file is mainly about turning
// training into an interactive learning tool. The important production lesson
// here is scheduling: training and sample generation are CPU/GPU-heavy work, so
// they run in blocking worker tasks while the UI stays responsive.

use chrono::Local;
use dioxus::desktop::{tao::dpi::LogicalSize, Config, WindowBuilder};
use dioxus::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rfd::FileDialog;
use sentence_gpt_rs_mlx_config::{
    create_training_session, format_count, format_learning_rate, format_loss, format_percent_style,
    load_input_documents, next_validation_step_after, optimizer_config_for_training_run,
    running_mean_loss, running_mean_loss_values,
    train_session_until_budget as train_shared_session_until_budget, Backend, TrainedSnapshot,
    TrainingSession,
};
use sentence_gpt_rs_mlx_lib::checkpoint::{
    load_checkpoint_from_path, save_checkpoint_to_path, TrainingRunConfig,
};
use sentence_gpt_rs_mlx_lib::microgpt::{
    generate_sample_trace as generate_cpu_sample_trace, generate_samples as generate_cpu_samples,
    CharacterTokenizer, Matrix, MicrogptTrainingProgress, SampleGenerationTrace, TransformerConfig,
    Vector,
};
use sentence_gpt_rs_mlx_lib::mlx_microgpt::{
    generate_sample_trace as generate_mlx_sample_trace, generate_samples as generate_mlx_samples,
};
use std::path::PathBuf;
use std::time::Duration;

mod styles;

use styles::CSS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DocumentBrowserDataset {
    Training,
    Validation,
}

#[derive(Clone, Copy)]
enum TrainingRunConfigField {
    ValidationStepInterval,
    TrainingDocumentBatchSize,
    MaxDocumentCount,
    ValidationSetDivisor,
    ValidationSetMaxDocumentCount,
    ContextWindowSize,
    LayerCount,
    AttentionHeads,
    EmbeddingSize,
    MlpExpansionFactor,
}

#[derive(Clone, Copy)]
enum TrainingRunConfigToggleField {
    LearnedBiases,
    RopePositionEncoding,
    LearnedAbsolutePositionEncoding,
    ResidualDropout,
    LearnedRmsNormGain,
    FinalRmsNorm,
    SwigluFeedForward,
    GeluFeedForward,
    TiedOutputEmbeddings,
    GradientClipping,
    WeightDecay,
    WarmupCosineLearningRate,
}

impl DocumentBrowserDataset {
    fn label(self) -> &'static str {
        match self {
            DocumentBrowserDataset::Training => "Training",
            DocumentBrowserDataset::Validation => "Validation",
        }
    }

    fn toggled(self) -> Self {
        match self {
            DocumentBrowserDataset::Training => DocumentBrowserDataset::Validation,
            DocumentBrowserDataset::Validation => DocumentBrowserDataset::Training,
        }
    }
}

impl DocumentBrowserDataset {
    fn documents_from(self, session: &TrainingSession) -> &[String] {
        match self {
            DocumentBrowserDataset::Training => session.training_documents(),
            DocumentBrowserDataset::Validation => session.validation_documents(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    backend: Backend,
    // Applied to the current session and checkpoints. UI edits go into the
    // per-backend staged configs below and are applied when training starts.
    training_run_config: TrainingRunConfig,
    mlx_training_run_config: TrainingRunConfig,
    cpu_training_run_config: TrainingRunConfig,
    session: Option<TrainingSession>,
    is_training_active: bool,
    is_training_busy: bool,
    manual_training_chunk_requested: bool,
    // Generation is queued behind training so MLX is not asked to train and
    // sample from the same model concurrently.
    generation_requested: bool,
    inspect_generation_requested: bool,
    is_generating_samples: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    throughput_start_step: usize,
    prefix: String,
    document_browser_dataset: DocumentBrowserDataset,
    training_document_search: String,
    cached_browser_search_matches: Vec<(usize, String)>,
    training_document_page: usize,
    system_overview_expanded: bool,
    training_config_expanded: bool,
    temperature: f64,
    samples: Vec<String>,
    inspected_generation: Option<SampleGenerationTrace>,
    selected_inspection_token_index: usize,
    initialization_error: Option<String>,
    checkpoint_message: Option<String>,
    snapshot_export_directory: Option<PathBuf>,
    sample_rng: ChaCha8Rng,
    training_document_page_rng: ChaCha8Rng,
}

struct TrainingChunkResult {
    session: TrainingSession,
    next_validation_step: usize,
    elapsed_millis: u128,
}

struct GenerationWork {
    trained_microgpt: TrainedSnapshot,
    prefix: String,
    temperature: f64,
    sample_rng: ChaCha8Rng,
    inspect: bool,
}

struct GenerationResult {
    samples: Vec<String>,
    inspected_generation: Option<SampleGenerationTrace>,
    sample_rng: ChaCha8Rng,
}

struct TokenEmbeddingSnapshot {
    rows: Vec<TokenEmbeddingRow>,
    embedding_size: usize,
    min_value: f64,
    max_value: f64,
    max_abs_value: f64,
}

struct TokenEmbeddingRow {
    label: String,
    title: String,
    values: Vec<f64>,
    l2_norm: f64,
}

struct SystemOverviewStep {
    title: String,
    status: String,
    details: Vec<String>,
    info_key: Option<&'static str>,
}

#[derive(Clone, Copy, Default)]
struct ParameterStats {
    parameter_count: usize,
    sum_abs_value: f64,
    max_abs_value: f64,
}

struct LayerInspectionSnapshot {
    layers: Vec<LayerInspection>,
    max_mean_abs_value: f64,
}

struct LayerInspection {
    norm: ParameterStats,
    attention: ParameterStats,
    mlp_expansion: ParameterStats,
    mlp_gate: ParameterStats,
    mlp_projection: ParameterStats,
}

fn main() {
    dioxus::LaunchBuilder::desktop()
        .with_cfg(
            Config::new().with_window(
                WindowBuilder::new()
                    .with_title("sentence-gpt-rs-mlx")
                    .with_inner_size(LogicalSize::new(1280.0, 900.0)),
            ),
        )
        .launch(App);
}

#[component]
fn App() -> Element {
    let mut state = use_signal(AppState::initialize);

    use_future(move || async move {
        loop {
            // Generation has priority once the current training chunk finishes.
            // This gives the user fast feedback without interrupting an in-flight
            // optimizer update.
            let generation_work = {
                let mut current = state.write();
                current.take_generation_work()
            };

            if let Some(generation_work) = generation_work {
                match tokio::task::spawn_blocking(move || {
                    generate_samples_from_work(generation_work)
                })
                .await
                {
                    Ok(Ok(generation_result)) => {
                        state.write().apply_generation_result(generation_result);
                    }
                    Ok(Err(error)) => {
                        state
                            .write()
                            .apply_generation_error(format!("generation failed: {error}"));
                    }
                    Err(error) => {
                        state
                            .write()
                            .apply_generation_error(format!("generation worker failed: {error}"));
                    }
                }
                tokio::time::sleep(Duration::from_millis(16)).await;
                continue;
            }

            let training_work = {
                let mut current = state.write();
                current.take_training_work()
            };

            if let Some((session, next_validation_step)) = training_work {
                let training_run_config = {
                    let current = state.read();
                    current.training_run_config
                };
                // `spawn_blocking` keeps the Dioxus/Tokio UI runtime from being
                // monopolized by MLX or CPU matrix math.
                match tokio::task::spawn_blocking(move || {
                    train_session_until_budget(session, next_validation_step, training_run_config)
                })
                .await
                {
                    Ok(Ok(chunk_result)) => state.write().apply_training_chunk(chunk_result),
                    Ok(Err(error)) => {
                        let mut current = state.write();
                        current.is_training_active = false;
                        current.is_training_busy = false;
                        current.initialization_error = Some(format!("training failed: {error}"));
                    }
                    Err(error) => {
                        let mut current = state.write();
                        current.is_training_active = false;
                        current.is_training_busy = false;
                        current.initialization_error =
                            Some(format!("training worker failed: {error}"));
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(16)).await;
        }
    });

    let snapshot = state.read().clone();
    let status = snapshot.status_label();
    let progress = snapshot.progress_fraction();
    let completed_document_train_count = snapshot.completed_document_train_count();
    let total_document_train_count = snapshot.total_document_train_count();
    let document_trains_per_minute = snapshot.document_trains_per_minute();
    let backend_label = snapshot.backend.label();
    let training_example_count = snapshot
        .session
        .as_ref()
        .map(TrainingSession::training_document_count)
        .unwrap_or(0);
    let validation_example_count = snapshot
        .session
        .as_ref()
        .map(TrainingSession::validation_document_count)
        .unwrap_or(0);
    let latest_loss = snapshot
        .session
        .as_ref()
        .and_then(TrainingSession::latest_loss);
    let latest_validation_loss = snapshot
        .session
        .as_ref()
        .and_then(TrainingSession::latest_validation_loss);
    let vocabulary_size = snapshot
        .session
        .as_ref()
        .map(|session| session.tokenizer().vocabulary_size())
        .unwrap_or(0);
    let selected_training_run_config = snapshot.selected_training_run_config();
    let training_config_arrow = if snapshot.training_config_expanded {
        "▾"
    } else {
        "▸"
    };
    let system_overview_arrow = if snapshot.system_overview_expanded {
        "▾"
    } else {
        "▸"
    };
    let is_complete = snapshot
        .session
        .as_ref()
        .is_some_and(TrainingSession::is_complete);
    let can_configure_training_run = snapshot.can_configure_training_run();
    let visible_documents = snapshot.visible_browser_documents();
    let is_document_search_empty = snapshot.training_document_search.trim().is_empty();
    let browser_document_count = snapshot.browser_document_count();
    let browser_dataset_label = snapshot.document_browser_dataset.label();
    let browser_dataset_label_lowercase = browser_dataset_label.to_lowercase();
    let document_page_count = snapshot.document_page_count();
    let current_document_page = snapshot
        .training_document_page
        .min(document_page_count.saturating_sub(1));
    let snapshot_export_directory_label = snapshot
        .snapshot_export_directory
        .as_ref()
        .map(|path| path.display().to_string());
    let token_embedding_snapshot = snapshot
        .system_overview_expanded
        .then(|| {
            snapshot
                .session
                .as_ref()
                .map(token_embedding_snapshot_for_session)
        })
        .flatten();

    rsx! {
        style { "{CSS}" }
        main { class: "app",
            div { class: "shell",
                div { class: "topbar",
                    div {
                        h1 { class: "title", "sentence-gpt-rs-mlx" }
                    }
                    div { class: "actions",
                        div { class: "primary-actions",
                            button {
                                class: "button",
                                disabled: !snapshot.is_training_active
                                    && (snapshot.is_training_busy
                                        || snapshot.is_generating_samples
                                        || is_complete),
                                onclick: move |_| state.write().toggle_training(),
                                if snapshot.is_training_active { "Pause" } else { "Start" }
                            }
                            button {
                                class: "button secondary",
                                disabled: snapshot.is_training_busy || snapshot.is_generating_samples,
                                onclick: move |_| state.write().toggle_backend(),
                                "Backend: {backend_label}"
                            }
                            button {
                                class: "button secondary",
                                disabled: snapshot.is_training_busy || snapshot.is_generating_samples,
                                onclick: move |_| state.write().reset_training(),
                                "Reset"
                            }
                        }
                        div { class: "snapshot-actions",
                            span {
                                class: "action-label",
                                title: "Export or import full training snapshots, including model parameters, optimizer state, data split, and progress.",
                                "Snapshot"
                            }
                            button {
                                class: "button secondary",
                                disabled: snapshot.session.is_none() || snapshot.is_training_busy || snapshot.is_generating_samples,
                                onclick: move |_| {
                                    let suggested_file_name = {
                                        state
                                            .read()
                                            .session
                                            .as_ref()
                                            .map(snapshot_checkpoint_file_name)
                                    };
                                    let Some(suggested_file_name) = suggested_file_name else {
                                        return;
                                    };
                                    if let Some(path) = FileDialog::new()
                                        .set_title("Export sentence-gpt-rs-mlx snapshot")
                                        .add_filter("sentence-gpt-rs-mlx checkpoint", &["bin"])
                                        .set_file_name(&suggested_file_name)
                                        .save_file()
                                    {
                                        state.write().export_checkpoint_to_path(path);
                                    }
                                },
                                "Export"
                            }
                            button {
                                class: "button secondary",
                                disabled: snapshot.is_training_busy || snapshot.is_generating_samples,
                                onclick: move |_| {
                                    // Open the native file dialog before borrowing state. On macOS
                                    // the dialog runs a nested event loop, so holding a Dioxus
                                    // signal borrow here can conflict with background task wakeups.
                                    if let Some(path) = FileDialog::new()
                                        .set_title("Import sentence-gpt-rs-mlx checkpoint")
                                        .add_filter("sentence-gpt-rs-mlx checkpoint", &["bin"])
                                        .pick_file()
                                    {
                                        state.write().import_checkpoint_from_path(path);
                                    }
                                },
                                "Import"
                            }
                            button {
                                class: "button secondary",
                                disabled: snapshot.is_training_busy || snapshot.is_generating_samples,
                                onclick: move |_| {
                                    // The native macOS dialog runs a nested event loop. Do not hold
                                    // a Dioxus signal borrow while it is open, or the background
                                    // `use_future` task can wake up and hit `AlreadyBorrowed`.
                                    if let Some(directory) = FileDialog::new()
                                        .set_title("Select snapshot export directory")
                                        .pick_folder()
                                    {
                                        state.write().set_snapshot_export_directory(directory);
                                    }
                                },
                                "Set Auto-Export Directory"
                            }
                            if let Some(directory_label) = &snapshot_export_directory_label {
                                span {
                                    class: "directory-label",
                                    title: "Selected directory for automatic validation snapshots: {directory_label}",
                                    "{directory_label}"
                                }
                            }
                        }
                    }
                }

                if let Some(error) = &snapshot.initialization_error {
                    div { class: "panel", "Error: {error}" }
                }
                if let Some(message) = &snapshot.checkpoint_message {
                    div { class: "panel", "{message}" }
                }

                section { class: "panel",
                    div { class: "status-grid",
                        {metric("Status", "Current worker state for training or sample generation.", status)}
                        {metric("Backend", "Execution backend used for training and inference.", backend_label.into())}
                        {metric("Model params", "Number of trainable model parameters.", format_count(snapshot.session.as_ref().map(TrainingSession::parameter_count).unwrap_or(0)))}
                        {metric("Learning rate", "Current optimizer learning rate after scheduling.", snapshot.session.as_ref().map(|session| format_learning_rate(session.current_learning_rate())).unwrap_or_else(|| "pending".into()))}
                        {metric("Training step", "Completed training steps out of the configured training budget.", format!("{} / {}", snapshot.completed_step_count(), snapshot.training_step_count()))}
                        {metric("Train loss", "Latest cross-entropy loss measured on training data.", latest_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
                        {metric("Validation loss", "Latest cross-entropy loss measured on the full fixed validation set.", latest_validation_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
                    }
                    div { class: "progress-track",
                        div { class: "progress-fill", style: "width: {format_percent_style(progress)};" }
                    }
                    div { class: "model-summary",
                        "document trains {completed_document_train_count} / {total_document_train_count} | running avg {format_rate(document_trains_per_minute)}/min | elapsed {format_elapsed_training_time(snapshot.accumulated_training_millis)}"
                    }
                    div { class: "model-summary",
                        "training docs {training_example_count} | validation docs {validation_example_count} | vocab size {vocabulary_size}"
                    }
                }

                section { class: "panel",
                    div { class: "model-header",
                        button {
                            class: "disclosure-button",
                            onclick: move |_| state.write().toggle_training_config_expanded(),
                            title: "Show or hide training configuration controls.",
                            span { class: "disclosure-arrow", "{training_config_arrow}" }
                            span { class: "section-title", "Training configuration" }
                        }
                        div { class: "model-summary",
                            if can_configure_training_run {
                                "Staged until training starts"
                            } else {
                                "Locked after training starts"
                            }
                        }
                        button {
                            class: "button secondary",
                            disabled: !can_configure_training_run,
                            onclick: move |_| state.write().restore_default_training_config(),
                            "Restore defaults"
                        }
                    }
                    if snapshot.training_config_expanded {
                        div { class: "config-grid",
                            {config_number_input("Validation interval steps", selected_training_run_config.validation_step_interval, can_configure_training_run, TrainingRunConfigField::ValidationStepInterval, state)}
                            {config_number_input("Docs per batch", selected_training_run_config.training_document_batch_size, can_configure_training_run, TrainingRunConfigField::TrainingDocumentBatchSize, state)}
                            {config_number_input("Max total docs", selected_training_run_config.max_document_count, can_configure_training_run, TrainingRunConfigField::MaxDocumentCount, state)}
                            {config_number_input("Validation docs divisor", selected_training_run_config.validation_set_divisor, can_configure_training_run, TrainingRunConfigField::ValidationSetDivisor, state)}
                            {config_number_input("Validation docs cap", selected_training_run_config.validation_set_max_document_count, can_configure_training_run, TrainingRunConfigField::ValidationSetMaxDocumentCount, state)}
                            {config_number_input("Context size", selected_training_run_config.context_window_size, can_configure_training_run, TrainingRunConfigField::ContextWindowSize, state)}
                            {config_number_input("Layers", selected_training_run_config.layer_count, can_configure_training_run, TrainingRunConfigField::LayerCount, state)}
                            {config_number_input("Attention heads", selected_training_run_config.attention_heads, can_configure_training_run, TrainingRunConfigField::AttentionHeads, state)}
                            {config_number_input("Embedding size", selected_training_run_config.embedding_size, can_configure_training_run, TrainingRunConfigField::EmbeddingSize, state)}
                            {config_number_input("MLP expansion factor", selected_training_run_config.mlp_expansion_factor, can_configure_training_run, TrainingRunConfigField::MlpExpansionFactor, state)}
                            {config_checkbox_input("Learned biases", selected_training_run_config.transformer_features.use_learned_biases, can_configure_training_run, TrainingRunConfigToggleField::LearnedBiases, state)}
                            {config_checkbox_input("RoPE positions", selected_training_run_config.transformer_features.use_rope_position_encoding, can_configure_training_run, TrainingRunConfigToggleField::RopePositionEncoding, state)}
                            {config_checkbox_input("Absolute positions", selected_training_run_config.transformer_features.use_learned_absolute_position_encoding, can_configure_training_run, TrainingRunConfigToggleField::LearnedAbsolutePositionEncoding, state)}
                            {config_checkbox_input("Residual dropout", selected_training_run_config.transformer_features.use_residual_dropout, can_configure_training_run, TrainingRunConfigToggleField::ResidualDropout, state)}
                            {config_checkbox_input("Learned RMSNorm gain", selected_training_run_config.transformer_features.use_learned_rmsnorm_gain, can_configure_training_run, TrainingRunConfigToggleField::LearnedRmsNormGain, state)}
                            {config_checkbox_input("Final RMSNorm", selected_training_run_config.transformer_features.use_final_rmsnorm, can_configure_training_run, TrainingRunConfigToggleField::FinalRmsNorm, state)}
                            {config_checkbox_input("SwiGLU MLP", selected_training_run_config.transformer_features.use_swiglu_feed_forward, can_configure_training_run, TrainingRunConfigToggleField::SwigluFeedForward, state)}
                            {config_checkbox_input("GELU MLP activation", selected_training_run_config.transformer_features.use_gelu_feed_forward, can_configure_training_run, TrainingRunConfigToggleField::GeluFeedForward, state)}
                            {config_checkbox_input("Tied output embeddings", selected_training_run_config.transformer_features.use_tied_output_embeddings, can_configure_training_run, TrainingRunConfigToggleField::TiedOutputEmbeddings, state)}
                            {config_checkbox_input("Gradient clipping", selected_training_run_config.optimizer_features.use_gradient_clipping, can_configure_training_run, TrainingRunConfigToggleField::GradientClipping, state)}
                            {config_checkbox_input("Weight decay", selected_training_run_config.optimizer_features.use_weight_decay, can_configure_training_run, TrainingRunConfigToggleField::WeightDecay, state)}
                            {config_checkbox_input("Warmup/cosine LR", selected_training_run_config.optimizer_features.use_warmup_cosine_learning_rate, can_configure_training_run, TrainingRunConfigToggleField::WarmupCosineLearningRate, state)}
                        }
                    }
                }

                section { class: "panel",
                    h2 { class: "section-title", "Loss over steps" }
                    {loss_history_chart(snapshot.progress_history())}
                }

                section { class: "panel",
                    div { class: "model-header",
                        h2 { class: "section-title", "{browser_dataset_label} documents" }
                        button {
                            class: "button secondary",
                            disabled: snapshot.session.is_none(),
                            onclick: move |_| state.write().toggle_document_browser_dataset(),
                            "Showing: {browser_dataset_label}"
                        }
                    }
                    div { class: "field",
                        label {
                            title: "Filter and rank visible {browser_dataset_label_lowercase} documents by matching text.",
                            "Search {browser_dataset_label_lowercase} examples"
                        }
                        div { class: "search-row",
                            input {
                                class: "text-input",
                                value: "{snapshot.training_document_search}",
                                placeholder: "Use pages below, or type to rank matches",
                                oninput: move |event| state.write().set_training_document_search(event.value()),
                                autocapitalize: false,
                                autocomplete: false,
                                autocorrect: false,
                                spellcheck: false
                            }
                            button {
                                class: "button secondary",
                                disabled: is_document_search_empty,
                                onclick: move |_| state.write().clear_training_document_search(),
                                "Clear"
                            }
                        }
                    }
                    if is_document_search_empty && document_page_count > 1 {
                        div { class: "document-controls",
                            button {
                                class: "page-button",
                                disabled: current_document_page == 0,
                                onclick: move |_| state.write().first_training_document_page(),
                                "First"
                            }
                            button {
                                class: "page-button",
                                disabled: current_document_page == 0,
                                onclick: move |_| state.write().previous_training_document_page(),
                                "Prev"
                            }
                            button {
                                class: "page-button",
                                disabled: current_document_page + 1 >= document_page_count,
                                onclick: move |_| state.write().next_training_document_page(),
                                "Next"
                            }
                            button {
                                class: "page-button",
                                onclick: move |_| state.write().random_training_document_page(),
                                "Random"
                            }
                        }
                    }
                    div { class: "model-summary",
                        if is_document_search_empty {
                            "Showing page {current_document_page + 1} of {document_page_count.max(1)} | {browser_document_count} {browser_dataset_label_lowercase} examples"
                        } else {
                            "Showing {visible_documents.len()} best matches from {browser_document_count} {browser_dataset_label_lowercase} examples"
                        }
                    }
                    div { class: "document-list",
                        for (document_index, document) in visible_documents.iter() {
                            div { class: "document-item",
                                div { class: "document-index", "#{document_index + 1}" }
                                div { class: "document-text", "{document}" }
                            }
                        }
                    }
                }

                section { class: "panel",
                    h2 { class: "section-title", "Generate samples" }
                    div { class: "controls",
                        div { class: "field",
                            label {
                                title: "Text to seed generation before the model continues the sample.",
                                "Prefix"
                            }
                            input {
                                class: "text-input",
                                value: "{snapshot.prefix}",
                                oninput: move |event| state.write().prefix = event.value(),
                                autocapitalize: false,
                                autocomplete: false,
                                autocorrect: false,
                                spellcheck: false
                            }
                        }
                        div { class: "field",
                            label {
                                title: "Sampling randomness. Lower values are more conservative; higher values are more varied.",
                                "Temperature {format_rate(snapshot.temperature)}"
                            }
                            input {
                                class: "range",
                                r#type: "range",
                                min: "0.1",
                                max: "2.0",
                                step: "0.1",
                                value: "{snapshot.temperature}",
                                oninput: move |event| {
                                    if let Ok(value) = event.value().parse::<f64>() {
                                        state.write().temperature = value;
                                    }
                                }
                            }
                        }
                        button {
                            class: "button",
                            disabled: snapshot.session.is_none() || snapshot.is_generating_samples,
                            onclick: move |_| state.write().generate(),
                            if snapshot.is_generating_samples { "Generating" } else { "Generate" }
                        }
                        button {
                            class: "button secondary",
                            disabled: snapshot.session.is_none() || snapshot.is_generating_samples,
                            onclick: move |_| state.write().generate_one_and_inspect(),
                            if snapshot.is_generating_samples { "Generating" } else { "Generate one and inspect" }
                        }
                    }
                    div { class: "samples",
                        for sample in snapshot.samples.iter().cloned() {
                            div { class: "sample",
                                div { class: "sample-text", "{sample}" }
                                button {
                                    class: "sample-search-button",
                                    onclick: {
                                        let sample = sample.clone();
                                        move |_| state.write().search_training_documents_for_sample(sample.clone())
                                    },
                                    "Find matches"
                                }
                            }
                        }
                    }
                    if let (Some(trace), Some(session)) = (
                        snapshot.inspected_generation.as_ref(),
                        snapshot.session.as_ref(),
                    ) {
                        {generation_inspection_panel(
                            trace,
                            session.tokenizer(),
                            snapshot.selected_inspection_token_index,
                            state,
                        )}
                    }
                }

                section { class: "panel",
                    div { class: "model-header",
                        button {
                            class: "disclosure-button",
                            onclick: move |_| state.write().toggle_system_overview_expanded(),
                            title: "Show or hide the high-level training and inference system overview.",
                            span { class: "disclosure-arrow", "{system_overview_arrow}" }
                            span { class: "section-title", "System overview" }
                        }
                        div { class: "model-summary",
                            "Aggregated by subsystem"
                        }
                    }
                    if snapshot.system_overview_expanded {
                        {system_overview_panel(&snapshot, token_embedding_snapshot.as_ref())}
                    }
                }

            }
        }
    }
}

impl AppState {
    fn initialize() -> Self {
        Self::initialize_with_config_state(
            Backend::Mlx,
            Backend::Mlx.default_training_run_config(),
            Backend::Cpu.default_training_run_config(),
        )
    }

    fn initialize_with_config_state(
        backend: Backend,
        mlx_training_run_config: TrainingRunConfig,
        cpu_training_run_config: TrainingRunConfig,
    ) -> Self {
        let training_run_config = match backend {
            Backend::Mlx => mlx_training_run_config,
            Backend::Cpu => cpu_training_run_config,
        };
        Self::initialize_with_config(
            backend,
            training_run_config,
            mlx_training_run_config,
            cpu_training_run_config,
        )
    }

    fn initialize_with_config(
        backend: Backend,
        training_run_config: TrainingRunConfig,
        mlx_training_run_config: TrainingRunConfig,
        cpu_training_run_config: TrainingRunConfig,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let transformer_config = TransformerConfig::new_with_features_and_mlp_expansion_factor(
            training_run_config.layer_count,
            training_run_config.embedding_size,
            training_run_config.context_window_size,
            training_run_config.attention_heads,
            training_run_config.mlp_expansion_factor,
            training_run_config.transformer_features,
        );
        let optimizer_config = optimizer_config_for_training_run(training_run_config);

        match transformer_config {
            Ok(transformer_config) => {
                let documents = load_input_documents(training_run_config);
                match documents {
                    Ok(input_documents) => {
                        match create_training_session(
                            input_documents,
                            &mut rng,
                            backend,
                            transformer_config,
                            optimizer_config,
                            training_run_config,
                        ) {
                            Ok(session) => Self {
                                backend,
                                training_run_config,
                                mlx_training_run_config,
                                cpu_training_run_config,
                                session: Some(session),
                                is_training_active: false,
                                is_training_busy: false,
                                manual_training_chunk_requested: false,
                                generation_requested: false,
                                inspect_generation_requested: false,
                                is_generating_samples: false,
                                next_validation_step: training_run_config.validation_step_interval,
                                accumulated_training_millis: 0,
                                throughput_start_step: 0,
                                prefix: String::new(),
                                document_browser_dataset: DocumentBrowserDataset::Training,
                                training_document_search: String::new(),
                                cached_browser_search_matches: Vec::new(),
                                training_document_page: 0,
                                system_overview_expanded: false,
                                training_config_expanded: true,
                                temperature: 0.5,
                                samples: Vec::new(),
                                inspected_generation: None,
                                selected_inspection_token_index: 0,
                                initialization_error: None,
                                checkpoint_message: None,
                                snapshot_export_directory: None,
                                sample_rng: ChaCha8Rng::seed_from_u64(1),
                                training_document_page_rng: ChaCha8Rng::seed_from_u64(2),
                            },
                            Err(error) => Self::failed_initialization(
                                backend,
                                training_run_config,
                                mlx_training_run_config,
                                cpu_training_run_config,
                                error,
                            ),
                        }
                    }
                    Err(error) => Self::failed_initialization(
                        backend,
                        training_run_config,
                        mlx_training_run_config,
                        cpu_training_run_config,
                        error,
                    ),
                }
            }
            Err(error) => Self::failed_initialization(
                backend,
                training_run_config,
                mlx_training_run_config,
                cpu_training_run_config,
                error,
            ),
        }
    }

    fn failed_initialization(
        backend: Backend,
        training_run_config: TrainingRunConfig,
        mlx_training_run_config: TrainingRunConfig,
        cpu_training_run_config: TrainingRunConfig,
        error: String,
    ) -> Self {
        Self {
            backend,
            training_run_config,
            mlx_training_run_config,
            cpu_training_run_config,
            session: None,
            is_training_active: false,
            is_training_busy: false,
            manual_training_chunk_requested: false,
            generation_requested: false,
            inspect_generation_requested: false,
            is_generating_samples: false,
            next_validation_step: training_run_config.validation_step_interval,
            accumulated_training_millis: 0,
            throughput_start_step: 0,
            prefix: String::new(),
            document_browser_dataset: DocumentBrowserDataset::Training,
            training_document_search: String::new(),
            cached_browser_search_matches: Vec::new(),
            training_document_page: 0,
            system_overview_expanded: false,
            training_config_expanded: true,
            temperature: 0.5,
            samples: Vec::new(),
            inspected_generation: None,
            selected_inspection_token_index: 0,
            initialization_error: Some(error),
            checkpoint_message: None,
            snapshot_export_directory: None,
            sample_rng: ChaCha8Rng::seed_from_u64(1),
            training_document_page_rng: ChaCha8Rng::seed_from_u64(2),
        }
    }

    fn set_training_document_search(&mut self, search: String) {
        self.training_document_search = search;
        self.training_document_page = 0;
        self.refresh_cached_browser_search_matches();
    }

    fn selected_training_run_config(&self) -> TrainingRunConfig {
        match self.backend {
            Backend::Mlx => self.mlx_training_run_config,
            Backend::Cpu => self.cpu_training_run_config,
        }
    }

    fn selected_training_run_config_mut(&mut self) -> &mut TrainingRunConfig {
        match self.backend {
            Backend::Mlx => &mut self.mlx_training_run_config,
            Backend::Cpu => &mut self.cpu_training_run_config,
        }
    }

    fn can_configure_training_run(&self) -> bool {
        !self.is_training_busy
            && !self.is_generating_samples
            && !self.is_training_active
            && self
                .session
                .as_ref()
                .is_none_or(|session| session.completed_step_count() == 0)
    }

    fn set_training_run_config_value(&mut self, field: TrainingRunConfigField, value: String) {
        if !self.can_configure_training_run() {
            return;
        }
        let Ok(value) = value.parse::<usize>() else {
            return;
        };
        if value == 0 {
            return;
        }

        let training_run_config = self.selected_training_run_config_mut();
        match field {
            TrainingRunConfigField::ValidationStepInterval => {
                training_run_config.validation_step_interval = value;
            }
            TrainingRunConfigField::TrainingDocumentBatchSize => {
                training_run_config.training_document_batch_size = value;
            }
            TrainingRunConfigField::MaxDocumentCount => {
                training_run_config.max_document_count = value;
            }
            TrainingRunConfigField::ValidationSetDivisor => {
                training_run_config.validation_set_divisor = value;
            }
            TrainingRunConfigField::ValidationSetMaxDocumentCount => {
                training_run_config.validation_set_max_document_count = value;
            }
            TrainingRunConfigField::ContextWindowSize => {
                training_run_config.context_window_size = value;
            }
            TrainingRunConfigField::LayerCount => {
                training_run_config.layer_count = value;
            }
            TrainingRunConfigField::AttentionHeads => {
                training_run_config.attention_heads = value;
            }
            TrainingRunConfigField::EmbeddingSize => {
                training_run_config.embedding_size = value;
            }
            TrainingRunConfigField::MlpExpansionFactor => {
                training_run_config.mlp_expansion_factor = value;
            }
        }
        self.initialization_error = None;
    }

    fn set_training_run_config_toggle(&mut self, field: TrainingRunConfigToggleField, value: bool) {
        if !self.can_configure_training_run() {
            return;
        }

        let training_run_config = self.selected_training_run_config_mut();
        match field {
            TrainingRunConfigToggleField::LearnedBiases => {
                training_run_config.transformer_features.use_learned_biases = value;
            }
            TrainingRunConfigToggleField::RopePositionEncoding => {
                training_run_config
                    .transformer_features
                    .use_rope_position_encoding = value;
            }
            TrainingRunConfigToggleField::LearnedAbsolutePositionEncoding => {
                training_run_config
                    .transformer_features
                    .use_learned_absolute_position_encoding = value;
            }
            TrainingRunConfigToggleField::ResidualDropout => {
                training_run_config
                    .transformer_features
                    .use_residual_dropout = value;
            }
            TrainingRunConfigToggleField::LearnedRmsNormGain => {
                training_run_config
                    .transformer_features
                    .use_learned_rmsnorm_gain = value;
            }
            TrainingRunConfigToggleField::FinalRmsNorm => {
                training_run_config.transformer_features.use_final_rmsnorm = value;
            }
            TrainingRunConfigToggleField::SwigluFeedForward => {
                training_run_config
                    .transformer_features
                    .use_swiglu_feed_forward = value;
            }
            TrainingRunConfigToggleField::GeluFeedForward => {
                training_run_config
                    .transformer_features
                    .use_gelu_feed_forward = value;
            }
            TrainingRunConfigToggleField::TiedOutputEmbeddings => {
                training_run_config
                    .transformer_features
                    .use_tied_output_embeddings = value;
            }
            TrainingRunConfigToggleField::GradientClipping => {
                training_run_config.optimizer_features.use_gradient_clipping = value;
            }
            TrainingRunConfigToggleField::WeightDecay => {
                training_run_config.optimizer_features.use_weight_decay = value;
            }
            TrainingRunConfigToggleField::WarmupCosineLearningRate => {
                training_run_config
                    .optimizer_features
                    .use_warmup_cosine_learning_rate = value;
            }
        }
        self.initialization_error = None;
    }

    fn restore_default_training_config(&mut self) {
        if !self.can_configure_training_run() {
            return;
        }
        *self.selected_training_run_config_mut() = self.backend.default_training_run_config();
        self.initialization_error = None;
    }

    fn toggle_training_config_expanded(&mut self) {
        self.training_config_expanded = !self.training_config_expanded;
    }

    fn toggle_system_overview_expanded(&mut self) {
        self.system_overview_expanded = !self.system_overview_expanded;
    }

    fn reset_training(&mut self) {
        if self.is_training_busy || self.is_generating_samples {
            return;
        }
        self.recreate_session_with_config(self.training_run_config);
    }

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

    fn recreate_session_with_config(&mut self, training_run_config: TrainingRunConfig) {
        let mut next = Self::initialize_with_config(
            self.backend,
            training_run_config,
            self.mlx_training_run_config,
            self.cpu_training_run_config,
        );
        next.prefix = self.prefix.clone();
        next.temperature = self.temperature;
        next.snapshot_export_directory = self.snapshot_export_directory.clone();
        next.sample_rng = self.sample_rng.clone();
        next.training_document_page_rng = self.training_document_page_rng.clone();
        next.document_browser_dataset = self.document_browser_dataset;
        next.training_document_search = self.training_document_search.clone();
        next.system_overview_expanded = self.system_overview_expanded;
        next.training_config_expanded = self.training_config_expanded;
        next.refresh_cached_browser_search_matches();
        *self = next;
    }

    fn clear_training_document_search(&mut self) {
        self.set_training_document_search(String::new());
    }

    fn search_training_documents_for_sample(&mut self, sample: String) {
        self.document_browser_dataset = DocumentBrowserDataset::Training;
        self.set_training_document_search(sample);
    }

    fn toggle_document_browser_dataset(&mut self) {
        self.document_browser_dataset = self.document_browser_dataset.toggled();
        self.training_document_page = 0;
        self.refresh_cached_browser_search_matches();
    }

    fn random_training_document_page(&mut self) {
        let page_count = self.document_page_count();
        if page_count > 0 {
            self.training_document_page = self.training_document_page_rng.gen_range(0..page_count);
        }
    }

    fn first_training_document_page(&mut self) {
        self.training_document_page = 0;
    }

    fn previous_training_document_page(&mut self) {
        self.training_document_page = self.training_document_page.saturating_sub(1);
    }

    fn next_training_document_page(&mut self) {
        let page_count = self.document_page_count();
        if page_count > 0 {
            self.training_document_page =
                (self.training_document_page + 1).min(page_count.saturating_sub(1));
        }
    }

    fn browser_document_count(&self) -> usize {
        self.session
            .as_ref()
            .map(|session| self.document_browser_dataset.documents_from(session).len())
            .unwrap_or(0)
    }

    fn document_page_count(&self) -> usize {
        self.browser_document_count().div_ceil(10)
    }

    fn browser_documents(&self) -> &[String] {
        self.session
            .as_ref()
            .map(|session| self.document_browser_dataset.documents_from(session))
            .unwrap_or(&[])
    }

    fn visible_browser_documents(&self) -> Vec<(usize, String)> {
        if !self.training_document_search.trim().is_empty() {
            return self.cached_browser_search_matches.clone();
        }

        let page_start = self
            .training_document_page
            .min(self.document_page_count().saturating_sub(1))
            * 10;
        self.browser_documents()
            .iter()
            .enumerate()
            .skip(page_start)
            .take(10)
            .map(|(index, document)| (index, document.clone()))
            .collect()
    }

    fn refresh_cached_browser_search_matches(&mut self) {
        let query = self.training_document_search.trim().to_lowercase();
        if query.is_empty() {
            self.cached_browser_search_matches.clear();
            return;
        }

        let terms = query
            .split_whitespace()
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        let mut scored_documents = self
            .browser_documents()
            .iter()
            .enumerate()
            .map(|(index, document)| {
                (
                    document_match_score(&document.to_lowercase(), &query, &terms),
                    index,
                    document.clone(),
                )
            })
            .collect::<Vec<_>>();
        scored_documents
            .sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        self.cached_browser_search_matches = scored_documents
            .into_iter()
            .take(10)
            .map(|(_, index, document)| (index, document))
            .collect();
    }

    fn toggle_backend(&mut self) {
        if self.is_training_busy || self.is_generating_samples {
            return;
        }
        self.backend = self.backend.toggled();
        self.recreate_session_with_config(self.selected_training_run_config());
    }

    fn export_checkpoint_to_path(&mut self, path: PathBuf) {
        if self.is_training_busy || self.is_generating_samples {
            return;
        }
        let Some(session) = &self.session else {
            return;
        };
        match session
            .export_checkpoint(self.training_run_config)
            .and_then(|checkpoint| save_checkpoint_to_path(&checkpoint, &path))
        {
            Ok(()) => {
                self.initialization_error = None;
                self.checkpoint_message =
                    Some(format!("Exported checkpoint to {}", path.display()));
            }
            Err(error) => {
                self.initialization_error = Some(format!("Export failed: {error}"));
                self.checkpoint_message = None;
            }
        }
    }

    fn import_checkpoint_from_path(&mut self, path: PathBuf) {
        if self.is_training_busy || self.is_generating_samples {
            return;
        }
        match load_checkpoint_from_path(&path).and_then(|checkpoint| {
            let training_run_config = checkpoint.training_run_config;
            TrainingSession::import_checkpoint(&checkpoint)
                .map(|session| (session, training_run_config))
        }) {
            Ok((session, training_run_config)) => {
                self.backend = session.backend();
                self.training_run_config = training_run_config;
                match self.backend {
                    Backend::Mlx => self.mlx_training_run_config = training_run_config,
                    Backend::Cpu => self.cpu_training_run_config = training_run_config,
                }
                self.next_validation_step = next_validation_step_after(
                    session.completed_step_count(),
                    self.training_run_config.validation_step_interval,
                );
                self.accumulated_training_millis = 0;
                self.throughput_start_step = session.completed_step_count();
                self.is_training_active = false;
                self.manual_training_chunk_requested = false;
                self.generation_requested = false;
                self.inspect_generation_requested = false;
                self.is_generating_samples = false;
                self.training_document_page = 0;
                self.initialization_error = None;
                self.checkpoint_message =
                    Some(format!("Imported checkpoint from {}", path.display()));
                self.session = Some(session);
                self.refresh_cached_browser_search_matches();
            }
            Err(error) => {
                self.initialization_error = Some(format!("Import failed: {error}"));
                self.checkpoint_message = None;
            }
        }
    }

    fn set_snapshot_export_directory(&mut self, directory: PathBuf) {
        self.snapshot_export_directory = Some(directory.clone());
        self.initialization_error = None;
        self.checkpoint_message = Some(format!(
            "Automatic validation snapshots will export to {}",
            directory.display()
        ));
    }

    fn toggle_training(&mut self) {
        if self
            .session
            .as_ref()
            .is_some_and(TrainingSession::is_complete)
        {
            self.is_training_active = false;
            return;
        }
        if !self.is_training_active && !self.apply_selected_training_config() {
            return;
        }
        self.is_training_active = !self.is_training_active;
    }

    fn take_training_work(&mut self) -> Option<(TrainingSession, usize)> {
        // Clone the session for the worker. The UI keeps ownership of state and
        // replaces it only when the worker returns a complete updated session.
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
    }

    fn apply_training_chunk(&mut self, chunk_result: TrainingChunkResult) {
        let is_complete = chunk_result.session.is_complete();
        let completed_validation_step =
            chunk_result.next_validation_step != self.next_validation_step;
        self.accumulated_training_millis += chunk_result.elapsed_millis;
        self.next_validation_step = chunk_result.next_validation_step;
        self.is_training_active = !is_complete && self.is_training_active;
        self.is_training_busy = false;
        if completed_validation_step {
            self.export_validation_snapshot(&chunk_result.session);
        }
        self.session = Some(chunk_result.session);
    }

    fn export_validation_snapshot(&mut self, session: &TrainingSession) {
        let Some(directory) = self.snapshot_export_directory.clone() else {
            return;
        };
        let path = directory.join(snapshot_checkpoint_file_name(session));
        match session
            .export_checkpoint(self.training_run_config)
            .and_then(|checkpoint| save_checkpoint_to_path(&checkpoint, &path))
        {
            Ok(()) => {
                self.initialization_error = None;
                self.checkpoint_message = Some(format!(
                    "Exported validation snapshot to {}",
                    path.display()
                ));
            }
            Err(error) => {
                self.initialization_error = Some(format!("Snapshot export failed: {error}"));
                self.checkpoint_message = None;
            }
        }
    }

    fn generate(&mut self) {
        if self.session.is_none() {
            return;
        }
        self.generation_requested = true;
        self.inspect_generation_requested = false;
        self.inspected_generation = None;
        self.is_generating_samples = true;
        self.initialization_error = None;
    }

    fn generate_one_and_inspect(&mut self) {
        if self.session.is_none() {
            return;
        }
        self.generation_requested = true;
        self.inspect_generation_requested = true;
        self.selected_inspection_token_index = 0;
        self.is_generating_samples = true;
        self.initialization_error = None;
    }

    fn select_inspection_token(&mut self, token_index: usize) {
        self.selected_inspection_token_index = token_index;
    }

    fn take_generation_work(&mut self) -> Option<GenerationWork> {
        if self.is_training_busy || !self.generation_requested {
            return None;
        }
        let session = self.session.as_ref()?;
        self.generation_requested = false;
        let inspect = self.inspect_generation_requested;
        self.inspect_generation_requested = false;
        Some(GenerationWork {
            trained_microgpt: session.trained_snapshot(),
            prefix: self.prefix.clone(),
            temperature: self.temperature,
            sample_rng: self.sample_rng.clone(),
            inspect,
        })
    }

    fn apply_generation_result(&mut self, generation_result: GenerationResult) {
        self.samples = generation_result.samples;
        self.inspected_generation = generation_result.inspected_generation;
        if let Some(trace) = &self.inspected_generation {
            self.selected_inspection_token_index = self
                .selected_inspection_token_index
                .min(trace.tokens.len().saturating_sub(1));
        }
        self.sample_rng = generation_result.sample_rng;
        self.is_generating_samples = false;
        self.initialization_error = None;
    }

    fn apply_generation_error(&mut self, error: String) {
        self.is_generating_samples = false;
        self.initialization_error = Some(error);
    }

    fn status_label(&self) -> String {
        match &self.session {
            None => "Initializing".into(),
            Some(session) if session.is_complete() => "Ready".into(),
            Some(_) if self.is_generating_samples => "Generating".into(),
            Some(_) if self.is_training_busy => "Training".into(),
            Some(_) if self.is_training_active => "Training queued".into(),
            Some(_) => "Paused".into(),
        }
    }

    fn completed_step_count(&self) -> usize {
        self.session
            .as_ref()
            .map(TrainingSession::completed_step_count)
            .unwrap_or(0)
    }

    fn training_step_count(&self) -> usize {
        self.session
            .as_ref()
            .map(TrainingSession::training_step_count)
            .unwrap_or(1)
    }

    fn progress_fraction(&self) -> f64 {
        self.completed_step_count() as f64 / self.training_step_count().max(1) as f64
    }

    fn completed_document_train_count(&self) -> usize {
        self.completed_step_count() * self.training_run_config.training_document_batch_size
    }

    fn total_document_train_count(&self) -> usize {
        self.training_step_count() * self.training_run_config.training_document_batch_size
    }

    fn document_trains_per_minute(&self) -> f64 {
        if self.accumulated_training_millis == 0 {
            return 0.0;
        }
        let completed_steps_since_rate_start = self
            .completed_step_count()
            .saturating_sub(self.throughput_start_step);
        completed_steps_since_rate_start as f64
            * self.training_run_config.training_document_batch_size as f64
            * 60_000.0
            / self.accumulated_training_millis as f64
    }

    fn progress_history(&self) -> &[MicrogptTrainingProgress] {
        self.session
            .as_ref()
            .map(TrainingSession::progress_history)
            .unwrap_or(&[])
    }
}

fn train_session_until_budget(
    session: TrainingSession,
    next_validation_step: usize,
    training_run_config: TrainingRunConfig,
) -> Result<TrainingChunkResult, String> {
    // One background chunk trains until the frame budget expires. The app then
    // yields back to the UI, updates metrics, and queues another chunk if
    // continuous training is active.
    let training_result =
        train_shared_session_until_budget(session, next_validation_step, training_run_config)?;

    Ok(TrainingChunkResult {
        session: training_result.session,
        next_validation_step: training_result.next_validation_step,
        elapsed_millis: training_result.elapsed_millis,
    })
}

fn generate_samples_from_work(mut work: GenerationWork) -> Result<GenerationResult, String> {
    if work.inspect {
        let inspected_generation = match &work.trained_microgpt {
            TrainedSnapshot::Mlx(trained_microgpt) => generate_mlx_sample_trace(
                &trained_microgpt.model,
                &trained_microgpt.config,
                &trained_microgpt.tokenizer,
                &work.prefix,
                work.temperature,
                &mut work.sample_rng,
            )
            .map_err(|error| error.to_string())?,
            TrainedSnapshot::Cpu(trained_microgpt) => generate_cpu_sample_trace(
                &trained_microgpt.model,
                &trained_microgpt.config,
                &trained_microgpt.tokenizer,
                &work.prefix,
                work.temperature,
                &mut work.sample_rng,
            ),
        };
        return Ok(GenerationResult {
            samples: vec![inspected_generation.sample.clone()],
            inspected_generation: Some(inspected_generation),
            sample_rng: work.sample_rng,
        });
    }

    let samples = match &work.trained_microgpt {
        TrainedSnapshot::Mlx(trained_microgpt) => generate_mlx_samples(
            &trained_microgpt.model,
            &trained_microgpt.config,
            &trained_microgpt.tokenizer,
            &work.prefix,
            10,
            work.temperature,
            &mut work.sample_rng,
        )
        .map_err(|error| error.to_string())?,
        TrainedSnapshot::Cpu(trained_microgpt) => generate_cpu_samples(
            &trained_microgpt.model,
            &trained_microgpt.config,
            &trained_microgpt.tokenizer,
            &work.prefix,
            10,
            work.temperature,
            &mut work.sample_rng,
        ),
    };

    Ok(GenerationResult {
        samples,
        inspected_generation: None,
        sample_rng: work.sample_rng,
    })
}

fn snapshot_checkpoint_file_name(session: &TrainingSession) -> String {
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let backend = session.backend().label().to_ascii_lowercase();
    let step = session.completed_step_count();
    let loss = session
        .latest_loss()
        .map(|loss| format!("{loss:.4}"))
        .unwrap_or_else(|| "pending".into());
    format!("sentence-gpt-rs-mlx-{backend}-{timestamp}-step-{step:06}-train-loss-{loss}.bin")
}

fn system_overview_panel(
    snapshot: &AppState,
    token_embedding_snapshot: Option<&Result<TokenEmbeddingSnapshot, String>>,
) -> Element {
    let Some(session) = snapshot.session.as_ref() else {
        return rsx! { div { class: "model-summary", "Waiting for a training session" } };
    };

    let config = session.config();
    let features = config.features;
    let optimizer_features = snapshot.training_run_config.optimizer_features;
    let overview_training_step = session.completed_step_count();
    let layer_inspection_snapshot = layer_inspection_snapshot_for_session(session);
    let latest_train_loss = session
        .latest_loss()
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let latest_validation_loss = session
        .latest_validation_loss()
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let position_status = match (
        features.use_rope_position_encoding,
        features.use_learned_absolute_position_encoding,
    ) {
        (true, true) => "RoPE + absolute",
        (true, false) => "RoPE",
        (false, true) => "absolute",
        (false, false) => "none",
    };
    let mlp_activation = if features.use_swiglu_feed_forward {
        "SwiGLU"
    } else if features.use_gelu_feed_forward {
        "GELU"
    } else {
        "ReLU"
    };
    let data_steps = vec![
        SystemOverviewStep {
            title: "Corpus".into(),
            status: format!(
                "{} train / {} val",
                session.training_document_count(),
                session.validation_document_count()
            ),
            details: vec![
                format!(
                    "batch {}",
                    snapshot.training_run_config.training_document_batch_size
                ),
                format!(
                    "validation every {} steps",
                    snapshot.training_run_config.validation_step_interval
                ),
                format!(
                    "max docs {}",
                    format_count(snapshot.training_run_config.max_document_count)
                ),
            ],
            info_key: Some("data-corpus"),
        },
        SystemOverviewStep {
            title: "Batch windows".into(),
            status: format!(
                "{} docs/step",
                snapshot.training_run_config.training_document_batch_size
            ),
            details: vec![
                "deterministic sampling".into(),
                format!("context {}", config.context_window_size),
                format!(
                    "document trains {}",
                    format_count(snapshot.completed_document_train_count())
                ),
            ],
            info_key: Some("data-batch-windows"),
        },
        SystemOverviewStep {
            title: "Validation".into(),
            status: latest_validation_loss.clone(),
            details: vec![
                format!(
                    "every {} steps",
                    snapshot.training_run_config.validation_step_interval
                ),
                format!("next step {}", snapshot.next_validation_step),
                "held-out documents".into(),
            ],
            info_key: Some("data-validation"),
        },
    ];

    let token_steps = vec![
        SystemOverviewStep {
            title: "Vocabulary".into(),
            status: format!("{} tokens", session.tokenizer_vocabulary_size()),
            details: vec![
                "character ids".into(),
                format!(
                    "boundary id {}",
                    session.tokenizer().sequence_boundary_token_id
                ),
                format!("context {}", config.context_window_size),
            ],
            info_key: Some("tokens-vocabulary"),
        },
        SystemOverviewStep {
            title: "Positions".into(),
            status: position_status.into(),
            details: vec![
                format!("context {}", config.context_window_size),
                format!(
                    "RoPE {}",
                    format_enabled(features.use_rope_position_encoding)
                ),
                format!(
                    "absolute {}",
                    format_enabled(features.use_learned_absolute_position_encoding)
                ),
            ],
            info_key: Some("tokens-positions"),
        },
        SystemOverviewStep {
            title: "Input vectors".into(),
            status: format!(
                "{} x {}",
                session.tokenizer_vocabulary_size(),
                config.embedding_size
            ),
            details: vec![
                format!("token table {}", format_enabled(true)),
                format!("positions {position_status}"),
                format!(
                    "tied output {}",
                    format_enabled(features.use_tied_output_embeddings)
                ),
            ],
            info_key: Some("tokens-input-vectors"),
        },
    ];

    let optimizer_steps = vec![
        SystemOverviewStep {
            title: "Loss".into(),
            status: latest_train_loss.clone(),
            details: vec![
                format!(
                    "validation {}",
                    session
                        .latest_validation_loss()
                        .map(format_loss)
                        .unwrap_or_else(|| "pending".into())
                ),
                "cross entropy".into(),
                "masked padding".into(),
            ],
            info_key: None,
        },
        SystemOverviewStep {
            title: "Gradients".into(),
            status: format!(
                "clip {}",
                format_enabled(optimizer_features.use_gradient_clipping)
            ),
            details: vec![
                "reverse-mode autodiff".into(),
                format!("params {}", format_count(session.parameter_count())),
                format!(
                    "batch {}",
                    snapshot.training_run_config.training_document_batch_size
                ),
            ],
            info_key: None,
        },
        SystemOverviewStep {
            title: "AdamW".into(),
            status: format_learning_rate(session.current_learning_rate()),
            details: vec![
                format!(
                    "decay {}",
                    format_enabled(optimizer_features.use_weight_decay)
                ),
                format!(
                    "schedule {}",
                    format_enabled(optimizer_features.use_warmup_cosine_learning_rate)
                ),
                format!(
                    "step {} / {}",
                    session.completed_step_count(),
                    session.training_step_count()
                ),
            ],
            info_key: None,
        },
    ];

    rsx! {
        div { class: "overview-sections",
            {model_circuit_overview(snapshot, config, token_embedding_snapshot, &latest_train_loss, &latest_validation_loss, &layer_inspection_snapshot)}
            {overview_flow_section("Data", "Corpus selection, batching, and validation split.", &data_steps, false)}
            {overview_flow_section("Tokens", "Character ids become fixed-width vectors with position information.", &token_steps, false)}
            div { class: "overview-section",
                div { class: "layer-overview-header",
                    div { class: "overview-step-title", "Embedding table" }
                    div { class: "model-summary",
                        "Step {overview_training_step} | one row per token, one column per embedding dimension"
                    }
                }
                {token_embedding_table(token_embedding_snapshot)}
            }
            {layer_stack_section(config, mlp_activation, &latest_train_loss, &latest_validation_loss, &layer_inspection_snapshot)}
            {overview_flow_section("Optimizer", "Loss, gradients, and AdamW update state.", &optimizer_steps, true)}
        }
    }
}

fn model_circuit_overview(
    snapshot: &AppState,
    config: &TransformerConfig,
    token_embedding_snapshot: Option<&Result<TokenEmbeddingSnapshot, String>>,
    latest_train_loss: &str,
    latest_validation_loss: &str,
    layer_inspection_snapshot: &Result<LayerInspectionSnapshot, String>,
) -> Element {
    let layer_energy = layer_inspection_snapshot
        .as_ref()
        .ok()
        .map(network_energy)
        .unwrap_or(0.0);
    let active_token_slots = snapshot.prefix.chars().count().min(8);
    let embedding_probe_styles = embedding_probe_styles(token_embedding_snapshot);

    rsx! {
        div { class: "overview-circuit",
            div { class: "circuit-mainline",
                div { class: "circuit-node token-node",
                    {overview_info_button("circuit-tokens", "tokens")}
                    div { class: "circuit-label", "tokens" }
                    div { class: "token-matrix",
                        for row in 0..8 {
                            span {
                                class: "{token_slot_class(row, active_token_slots)}",
                                title: "Prefix token slot {row + 1}. Lit slots reflect the current prefix length, capped at eight visible slots."
                            }
                        }
                    }
                    div { class: "circuit-dim", "{config.context_window_size} x ids" }
                }
                div { class: "circuit-arrow", "→" }
                div { class: "circuit-node embed-node",
                    {overview_info_button("circuit-embed", "embed")}
                    div { class: "circuit-label", "embed" }
                    div { class: "embedding-probe",
                        for style in &embedding_probe_styles {
                            span {
                                class: "embedding-probe-cell",
                                style: "{style}",
                                title: "Sampled token embedding weight from the current table."
                            }
                        }
                    }
                    div { class: "circuit-dim", "{config.embedding_size} channels" }
                }
                div { class: "circuit-arrow", "→" }
                div { class: "circuit-node attention-node",
                    {overview_info_button("circuit-attention", "attention")}
                    div { class: "circuit-label", "attention" }
                    div { class: "attention-bank",
                        for _head_index in 0..config.attention_head_count.min(4) {
                            div { class: "attention-head",
                                div { class: "attention-head-graph",
                                    span {}
                                    span {}
                                    span {}
                                    span {}
                                }
                                div { class: "attention-projections",
                                    span { "Q" }
                                    span { "K" }
                                    span { "V" }
                                    span { "O" }
                                }
                            }
                        }
                    }
                    div { class: "circuit-dim", "{config.attention_head_count} heads x {config.attention_head_size}" }
                }
                div { class: "circuit-arrow", "→" }
                div { class: "circuit-node mlp-node",
                    {overview_info_button("circuit-mlp", "mlp")}
                    div { class: "circuit-label", "mlp" }
                    svg { class: "mlp-mini-network", view_box: "0 0 160 120",
                        for y in [22, 48, 72, 98] {
                            line { x1: "26", y1: "{y}", x2: "80", y2: "22", stroke: "#6f6a5d", stroke_width: "1" }
                            line { x1: "26", y1: "{y}", x2: "80", y2: "60", stroke: "#6f6a5d", stroke_width: "1" }
                            line { x1: "26", y1: "{y}", x2: "80", y2: "98", stroke: "#6f6a5d", stroke_width: "1" }
                        }
                        for y in [22, 60, 98] {
                            line { x1: "80", y1: "{y}", x2: "134", y2: "34", stroke: "#6f6a5d", stroke_width: "1" }
                            line { x1: "80", y1: "{y}", x2: "134", y2: "86", stroke: "#6f6a5d", stroke_width: "1" }
                        }
                        for y in [22, 48, 72, 98] {
                            circle { cx: "26", cy: "{y}", r: "6", fill: "#06090d", stroke: "#f0e8d0", stroke_width: "2" }
                        }
                        for y in [22, 60, 98] {
                            circle { cx: "80", cy: "{y}", r: "7", fill: "#06090d", stroke: "#4fd8ff", stroke_width: "2" }
                        }
                        for y in [34, 86] {
                            circle { cx: "134", cy: "{y}", r: "7", fill: "#06090d", stroke: "#f6d365", stroke_width: "2" }
                        }
                    }
                    div { class: "circuit-dim", "{config.mlp_expansion_factor}x hidden expansion" }
                }
                div { class: "circuit-arrow", "→" }
                div { class: "circuit-node output-node",
                    {overview_info_button("circuit-logits", "logits")}
                    div { class: "circuit-label", "logits" }
                    div { class: "logit-strip",
                        for row in 0..10 {
                            span { class: "{logit_probe_class(row)}" }
                        }
                    }
                    div { class: "circuit-dim", "{snapshot.session.as_ref().map(TrainingSession::tokenizer_vocabulary_size).unwrap_or(0)} scores" }
                }
            }
            div { class: "circuit-footer",
                div { "step {snapshot.completed_step_count()} / {snapshot.training_step_count()}" }
                div { "train {latest_train_loss}" }
                div { "validation {latest_validation_loss}" }
                div { "parameter field {format_percent_style(layer_energy)}" }
            }
        }
    }
}

fn layer_stack_section(
    config: &TransformerConfig,
    mlp_activation: &str,
    latest_train_loss: &str,
    latest_validation_loss: &str,
    layer_inspection_snapshot: &Result<LayerInspectionSnapshot, String>,
) -> Element {
    rsx! {
        div { class: "overview-section",
            div { class: "layer-overview-header",
                div { class: "overview-step-title", "Layer stack" }
                div { class: "model-summary",
                    "train {latest_train_loss} | val {latest_validation_loss} | tint follows mean absolute trained weight"
                }
            }
            match layer_inspection_snapshot {
                Ok(layer_inspection_snapshot) => rsx! {
                    div { class: "layer-stack",
                        for (layer_index, layer) in layer_inspection_snapshot.layers.iter().enumerate() {
                            {layer_visualization_row(
                                layer_index,
                                config,
                                mlp_activation,
                                layer,
                                layer_inspection_snapshot.max_mean_abs_value,
                            )}
                        }
                    }
                },
                Err(error) => rsx! {
                    div { class: "model-summary", "Could not inspect layer parameters: {error}" }
                },
            }
        }
    }
}

fn layer_visualization_row(
    layer_index: usize,
    config: &TransformerConfig,
    mlp_activation: &str,
    layer: &LayerInspection,
    max_mean_abs_value: f64,
) -> Element {
    let feed_forward_size = config.feed_forward_size();
    let gate_label = if config.features.use_swiglu_feed_forward {
        "Gate"
    } else {
        "Gate off"
    };
    let gate_class = if config.features.use_swiglu_feed_forward {
        "mlp-gate"
    } else {
        "mlp-gate-off"
    };
    let gate_detail = if config.features.use_swiglu_feed_forward {
        format!(
            "{} → {} | {} params",
            config.embedding_size,
            feed_forward_size,
            format_count(layer.mlp_gate.parameter_count)
        )
    } else {
        "unused when SwiGLU is off".into()
    };
    let mixer_label = if config.features.use_swiglu_feed_forward {
        "SiLU x gate"
    } else {
        mlp_activation
    };
    let mixer_detail = if config.features.use_swiglu_feed_forward {
        "candidate features are multiplied by gate features"
    } else {
        "activation has no learned weights"
    };

    rsx! {
        div {
            class: "layer-visual-row",
            style: "{layer_row_tint_style(layer, max_mean_abs_value)}",
            div { class: "layer-visual-label",
                div { class: "layer-label", "L{layer_index + 1}" }
                div { class: "layer-mini-stat",
                    "avg |w| {format_embedding_value(layer_mean_abs_value(layer))}"
                }
            }
            div { class: "layer-pipeline",
                {layer_stage_card(
                    "norm",
                    "Norm",
                    "RMSNorm x2".into(),
                    format!("2 x {} channel gains", config.embedding_size),
                    layer.norm,
                    max_mean_abs_value,
                )}
                div { class: "layer-arrow", "→" }
                {layer_stage_card(
                    "attention",
                    "Attention",
                    format!("{} heads x {}", config.attention_head_count, config.attention_head_size),
                    format!(
                        "Q/K/V/O | {} params",
                        format_count(layer.attention.parameter_count)
                    ),
                    layer.attention,
                    max_mean_abs_value,
                )}
                div { class: "layer-arrow", "→" }
                div { class: "mlp-visual-group",
                    div { class: "mlp-group-title", "MLP" }
                    div { class: "mlp-subpipeline",
                        {layer_stage_card(
                            "mlp-expand",
                            "Expand",
                            format!("{} → {}", config.embedding_size, feed_forward_size),
                            format!("{} params", format_count(layer.mlp_expansion.parameter_count)),
                            layer.mlp_expansion,
                            max_mean_abs_value,
                        )}
                        div { class: "layer-arrow small", "→" }
                        {layer_stage_card(
                            gate_class,
                            gate_label,
                            format!("{} → {}", config.embedding_size, feed_forward_size),
                            gate_detail,
                            layer.mlp_gate,
                            max_mean_abs_value,
                        )}
                        div { class: "layer-arrow small", "→" }
                        {non_parameter_stage_card("mlp-mix", "Mix", mixer_label, mixer_detail)}
                        div { class: "layer-arrow small", "→" }
                        {layer_stage_card(
                            "mlp-project",
                            "Project",
                            format!("{} → {}", feed_forward_size, config.embedding_size),
                            format!("{} params", format_count(layer.mlp_projection.parameter_count)),
                            layer.mlp_projection,
                            max_mean_abs_value,
                        )}
                    }
                }
            }
        }
    }
}

fn layer_stage_card(
    class_suffix: &str,
    title: &str,
    main: String,
    detail: String,
    stats: ParameterStats,
    max_mean_abs_value: f64,
) -> Element {
    rsx! {
        div {
            class: "layer-stage {class_suffix}",
            style: "{layer_stage_tint_style(class_suffix, stats, max_mean_abs_value)}",
            {stage_info_button(class_suffix, title)}
            div { class: "stage-copy",
                div { class: "layer-chunk-title", "{title}" }
                div { class: "layer-chunk-main", "{main}" }
                div { class: "layer-chunk-detail", "{detail}" }
                div { class: "layer-chunk-detail",
                    "mean |w| {format_embedding_value(stats.mean_abs_value())} | max {format_embedding_value(stats.max_abs_value)}"
                }
                div { class: "stage-meter",
                    div {
                        class: "stage-meter-fill",
                        style: "width: {format_percent_style(normalized_parameter_stat(stats.mean_abs_value(), max_mean_abs_value))};"
                    }
                }
            }
        }
    }
}

fn non_parameter_stage_card(class_suffix: &str, title: &str, main: &str, detail: &str) -> Element {
    rsx! {
        div {
            class: "layer-stage {class_suffix} no-params",
            style: "{non_parameter_stage_tint_style(class_suffix)}",
            {stage_info_button(class_suffix, title)}
            div { class: "stage-copy",
                div { class: "layer-chunk-title", "{title}" }
                div { class: "layer-chunk-main", "{main}" }
                div { class: "layer-chunk-detail", "{detail}" }
            }
        }
    }
}

fn stage_info_button(class_suffix: &str, title: &str) -> Element {
    info_button(title, layer_stage_description(class_suffix))
}

fn overview_info_button(info_key: &str, title: &str) -> Element {
    info_button(title, overview_chunk_description(info_key))
}

fn info_button(title: &str, paragraphs: &'static [&'static str]) -> Element {
    rsx! {
        div { class: "stage-info-control",
            button {
                class: "stage-info-button",
                r#type: "button",
                title: "Explain {title}",
                "i"
            }
            div { class: "stage-info-popover", role: "tooltip",
                div { class: "stage-info-title", "{title}" }
                for paragraph in paragraphs {
                    p { "{paragraph}" }
                }
            }
        }
    }
}

fn overview_chunk_description(info_key: &str) -> &'static [&'static str] {
    match info_key {
        "circuit-tokens" => &[
            "This chunk represents the integer token ids fed into the model. In this app each character maps to a vocabulary id, and a boundary token marks document or sequence edges.",
            "The visible dots are a compact stand-in for the current context window, not one UI element per possible position. Lit slots reflect how much of the current prefix is present, capped so the overview stays readable as context length grows.",
            "At this stage the numbers do not yet contain learned meaning. They are discrete lookup keys that will be converted into dense vectors by the embedding table.",
        ],
        "circuit-embed" => &[
            "The embedding table converts each token id into a learned vector with the model's embedding width. Tokens that the model learns to use similarly can develop related vector patterns over training.",
            "The colored mini-grid samples real values from the current embedding table. Blue and warm cells indicate opposite signs, while stronger color means larger magnitude.",
            "Position information is combined with these token vectors before the transformer layers read them, so the model can distinguish the same character appearing at different places in the context.",
        ],
        "circuit-attention" => &[
            "Attention is the part of the transformer that lets each token position read from earlier positions. Queries ask what to look for, keys advertise what each position contains, and values carry the information to copy or blend.",
            "Multiple heads split the embedding channels into smaller views. Each head can specialize in a different pattern, while the output projection recombines those head results back into the residual stream.",
            "The overview draws only a few heads and projection labels because the useful high-level structure is stable even when the model grows to many heads and millions of parameters.",
        ],
        "circuit-mlp" => &[
            "The MLP works independently at each token position after attention has mixed information across time. It expands the vector into a wider hidden space, applies a nonlinearity or gate, and projects back to the embedding width.",
            "This block is where many per-token feature transformations happen: sharpening, suppressing, combining, or rewriting features that attention placed in the residual stream.",
            "The diagram shows representative neuron columns instead of every hidden channel, so it scales with model size while still showing the expand, mix, and project shape.",
        ],
        "circuit-logits" => &[
            "The logits chunk represents the final score assigned to every possible next token. A softmax turns these raw scores into probabilities for sampling or inspection.",
            "When output embeddings are tied, the same learned token vectors used at input are reused to score output tokens. When untied, a separate output table performs that scoring.",
            "The vertical strip is a compact vocabulary summary. It does not draw every vocabulary entry, but it shows that the model ends by ranking candidate next tokens.",
        ],
        "data-corpus" => &[
            "The corpus is the source text after sentence extraction, deduplication, and the fixed train/validation split. Training documents are used for gradient updates; validation documents are held out to measure generalization.",
            "The train and validation counts matter because a model can reduce training loss by memorizing. A separate validation set gives a better signal for whether the learned patterns transfer to unseen examples.",
            "The max document setting caps how much source material enters the run, which keeps experiments small and repeatable while changing model or optimizer settings.",
        ],
        "data-batch-windows" => &[
            "Each optimizer step samples a batch of documents and cuts them into fixed-length token windows. The context length controls how many preceding tokens the model can use when predicting the next token.",
            "Deterministic sampling means the run can be reproduced from the same seed and configuration. That makes UI comparisons and checkpoint debugging easier because data order is not an extra hidden variable.",
            "The document-train count is larger than the step count because one step can include multiple documents. It is a throughput-oriented measure of how many document examples have contributed to updates.",
        ],
        "data-validation" => &[
            "Validation periodically runs the current model on held-out documents without applying optimizer updates. The resulting loss is a checkpoint on model quality, not another training signal.",
            "Running validation every step would be expensive, so the interval controls how often the app pauses to measure it. The next-step value shows when the next validation pass will happen.",
            "A falling validation loss usually means the model is learning reusable structure. If training loss falls while validation loss stalls or rises, the model may be overfitting the training documents.",
        ],
        "tokens-vocabulary" => &[
            "The vocabulary is the complete set of token ids the model can receive or generate. This app uses character-level tokens plus a boundary token, so the vocabulary is small and easy to inspect.",
            "A character tokenizer avoids complex subword tokenization machinery. The tradeoff is that longer text requires more token positions because words are represented one character at a time.",
            "The boundary id gives the model an explicit marker for sequence edges, helping it learn when one training example ends and another begins.",
        ],
        "tokens-positions" => &[
            "Position encoding tells the model where each token sits in the context. Without it, attention would see a bag of token vectors and would not know their order.",
            "RoPE rotates query and key channels by position, which makes relative distance information available inside attention scores. Learned absolute positions instead add a trained vector for each slot.",
            "The configuration can enable either or both mechanisms. This chunk reports the active choice because position handling changes what the attention layers can easily learn.",
        ],
        "tokens-input-vectors" => &[
            "Input vectors are the dense per-position representations that enter the transformer stack. Their shape is context length by embedding width, though this summary reports vocabulary size by embedding width for the token table itself.",
            "The token table is always active because token ids need learned vectors before the network can process them. Position information is added or applied alongside those token vectors.",
            "Tied output embeddings reuse the token table at the output side, reducing parameter count and encouraging input and output token geometry to stay aligned.",
        ],
        _ => &["This chunk summarizes one concrete part of the training or inference pipeline without expanding it into every token, neuron, or parameter."],
    }
}

fn layer_stage_description(class_suffix: &str) -> &'static [&'static str] {
    match class_suffix {
        "norm" => &[
            "RMSNorm rescales the current residual vector before a sub-block reads it. It divides by the vector's root-mean-square size, then applies learned per-channel gain values when that feature is enabled.",
            "This model uses pre-norm transformer blocks: normalize first, run attention or the MLP, then add the result back to the residual stream. Pre-norm keeps signal magnitudes steadier, which makes deeper stacks easier to train.",
            "The two norm passes shown here are the attention norm and the feed-forward norm. They do not mix tokens or channels by themselves; they prepare the vector so the following learned projections operate in a predictable numeric range.",
        ],
        "attention" => &[
            "Attention is the token-mixing part of the layer. It turns each position's vector into queries, keys, and values. Query-key dot products decide which earlier positions this token should read from, and values carry the information being read.",
            "The heads split the embedding width into smaller subspaces. Each head can learn a different relation, such as nearby punctuation, repeated characters, or boundary tokens. The output projection then recombines all heads into one residual update.",
            "Causal masking prevents a position from looking into the future during training, so the same block can be used for autoregressive generation one token at a time.",
        ],
        "mlp-expand" => &[
            "The expansion projection is the first learned linear map in the feed-forward block. It maps each token's vector from the model width into a wider hidden space, usually exposing more candidate features than can fit in the residual stream.",
            "Unlike attention, this projection acts independently at each token position. It does not move information across time; it transforms the features already present at the current position.",
            "The hidden width is the embedding width multiplied by the configured MLP expansion factor, so the MLP can form a wider set of candidate feature directions before reducing back down.",
        ],
        "mlp-gate" => &[
            "The gate projection is the second parallel linear map used by SwiGLU. It produces context-dependent gate values for the same hidden-width channels produced by the expansion projection.",
            "During the mix step, the activated expansion stream is multiplied by the gate stream. This lets the model suppress or emphasize candidate features instead of passing every activated feature forward uniformly.",
            "The gate has learned weights, so training can discover which features should be conditionally opened or closed for different token contexts.",
        ],
        "mlp-gate-off" => &[
            "This model instance is not using the SwiGLU gate path for the active feed-forward computation. The gate parameters still exist in the parameter layout, but the forward pass ignores the gate output when SwiGLU is disabled.",
            "With the gate off, the MLP behaves like a standard expand, activate, project block. The expansion stream is transformed by the selected activation and then projected back to the model width.",
        ],
        "mlp-mix" => &[
            "The mix step is the nonlinear part of the MLP. For SwiGLU, it applies SiLU to the expansion stream and multiplies that by the gate stream. For non-SwiGLU modes, it applies the selected activation directly to the expansion stream.",
            "This step has no learned weights, but it matters because without a nonlinearity, stacked linear projections would collapse into one bigger linear projection. The nonlinearity lets the MLP represent conditional feature interactions.",
        ],
        "mlp-project" => &[
            "The output projection maps the wide MLP hidden vector back down to the embedding width so it can be added to the residual stream.",
            "This is where the feed-forward block converts many candidate hidden features into a single update vector for the token position. In residual form, the projected update edits the current state rather than replacing it outright.",
            "Because this projection feeds back into the main residual stream, its scale strongly affects how much the MLP changes the layer's output.",
        ],
        _ => &["This node summarizes one bounded component of the transformer layer without rendering individual weights or neurons."],
    }
}

fn overview_flow_section(
    title: &str,
    summary: &str,
    steps: &[SystemOverviewStep],
    show_flow_arrows: bool,
) -> Element {
    rsx! {
        div { class: "overview-section",
            div { class: "layer-overview-header",
                div { class: "overview-step-title", "{title}" }
                div { class: "model-summary", "{summary}" }
            }
            div { class: "overview-flow",
                for (index, step) in steps.iter().enumerate() {
                    div { class: "overview-step",
                        if let Some(info_key) = step.info_key {
                            {overview_info_button(info_key, &step.title)}
                        }
                        div { class: "overview-step-title", "{step.title}" }
                        div { class: "overview-step-status", "{step.status}" }
                        div { class: "overview-step-details",
                            for detail in &step.details {
                                div { "{detail}" }
                            }
                        }
                    }
                    if show_flow_arrows && index + 1 < steps.len() {
                        div { class: "overview-arrow", "→" }
                    }
                }
            }
        }
    }
}

impl ParameterStats {
    fn add_value(&mut self, value: f64) {
        let abs_value = value.abs();
        self.parameter_count += 1;
        self.sum_abs_value += abs_value;
        self.max_abs_value = self.max_abs_value.max(abs_value);
    }

    fn merge(&mut self, other: ParameterStats) {
        self.parameter_count += other.parameter_count;
        self.sum_abs_value += other.sum_abs_value;
        self.max_abs_value = self.max_abs_value.max(other.max_abs_value);
    }

    fn mean_abs_value(self) -> f64 {
        if self.parameter_count == 0 {
            0.0
        } else {
            self.sum_abs_value / self.parameter_count as f64
        }
    }
}

fn layer_inspection_snapshot_for_session(
    session: &TrainingSession,
) -> Result<LayerInspectionSnapshot, String> {
    let layers = match session {
        TrainingSession::Cpu(session) => session
            .trained_microgpt
            .model
            .layers
            .iter()
            .map(|layer| {
                let mut norm = ParameterStats::default();
                add_vector_stats(&mut norm, &layer.attention_norm_gain);
                add_vector_stats(&mut norm, &layer.feed_forward_norm_gain);

                let mut attention = ParameterStats::default();
                add_matrix_stats(&mut attention, &layer.attention.query_weights);
                add_vector_stats(&mut attention, &layer.attention.query_biases);
                add_matrix_stats(&mut attention, &layer.attention.key_weights);
                add_vector_stats(&mut attention, &layer.attention.key_biases);
                add_matrix_stats(&mut attention, &layer.attention.value_weights);
                add_vector_stats(&mut attention, &layer.attention.value_biases);
                add_matrix_stats(&mut attention, &layer.attention.output_projection_weights);
                add_vector_stats(&mut attention, &layer.attention.output_projection_biases);

                let mut mlp_expansion = ParameterStats::default();
                add_matrix_stats(&mut mlp_expansion, &layer.feed_forward.expansion_weights);
                add_vector_stats(&mut mlp_expansion, &layer.feed_forward.expansion_biases);

                let mut mlp_gate = ParameterStats::default();
                add_matrix_stats(&mut mlp_gate, &layer.feed_forward.gate_weights);
                add_vector_stats(&mut mlp_gate, &layer.feed_forward.gate_biases);

                let mut mlp_projection = ParameterStats::default();
                add_matrix_stats(&mut mlp_projection, &layer.feed_forward.projection_weights);
                add_vector_stats(&mut mlp_projection, &layer.feed_forward.projection_biases);

                LayerInspection {
                    norm,
                    attention,
                    mlp_expansion,
                    mlp_gate,
                    mlp_projection,
                }
            })
            .collect::<Vec<_>>(),
        TrainingSession::Mlx(session) => {
            let mut layers = Vec::new();
            for layer in &session.trained_microgpt.model.layers {
                macro_rules! add_array_stats {
                    ($stats:expr, $array:expr) => {{
                        $array.eval().map_err(|error| error.to_string())?;
                        for value in $array.as_slice::<f32>() {
                            $stats.add_value(*value as f64);
                        }
                        Ok::<(), String>(())
                    }};
                }

                let mut norm = ParameterStats::default();
                add_array_stats!(&mut norm, &layer.attention_norm_gain)?;
                add_array_stats!(&mut norm, &layer.feed_forward_norm_gain)?;

                let mut attention = ParameterStats::default();
                add_array_stats!(&mut attention, &layer.attention.query_weights)?;
                add_array_stats!(&mut attention, &layer.attention.query_biases)?;
                add_array_stats!(&mut attention, &layer.attention.key_weights)?;
                add_array_stats!(&mut attention, &layer.attention.key_biases)?;
                add_array_stats!(&mut attention, &layer.attention.value_weights)?;
                add_array_stats!(&mut attention, &layer.attention.value_biases)?;
                add_array_stats!(&mut attention, &layer.attention.output_projection_weights)?;
                add_array_stats!(&mut attention, &layer.attention.output_projection_biases)?;

                let mut mlp_expansion = ParameterStats::default();
                add_array_stats!(&mut mlp_expansion, &layer.feed_forward.expansion_weights)?;
                add_array_stats!(&mut mlp_expansion, &layer.feed_forward.expansion_biases)?;

                let mut mlp_gate = ParameterStats::default();
                add_array_stats!(&mut mlp_gate, &layer.feed_forward.gate_weights)?;
                add_array_stats!(&mut mlp_gate, &layer.feed_forward.gate_biases)?;

                let mut mlp_projection = ParameterStats::default();
                add_array_stats!(&mut mlp_projection, &layer.feed_forward.projection_weights)?;
                add_array_stats!(&mut mlp_projection, &layer.feed_forward.projection_biases)?;

                layers.push(LayerInspection {
                    norm,
                    attention,
                    mlp_expansion,
                    mlp_gate,
                    mlp_projection,
                });
            }
            layers
        }
    };

    let max_mean_abs_value = layers
        .iter()
        .flat_map(|layer| {
            [
                layer.norm.mean_abs_value(),
                layer.attention.mean_abs_value(),
                layer.mlp_expansion.mean_abs_value(),
                layer.mlp_gate.mean_abs_value(),
                layer.mlp_projection.mean_abs_value(),
            ]
        })
        .fold(1e-12_f64, f64::max);

    Ok(LayerInspectionSnapshot {
        layers,
        max_mean_abs_value,
    })
}

fn add_matrix_stats(stats: &mut ParameterStats, matrix: &Matrix) {
    for row in matrix {
        add_vector_stats(stats, row);
    }
}

fn add_vector_stats(stats: &mut ParameterStats, vector: &Vector) {
    for value in vector {
        stats.add_value(value.data());
    }
}

fn layer_mean_abs_value(layer: &LayerInspection) -> f64 {
    let mut combined = ParameterStats::default();
    combined.merge(layer.norm);
    combined.merge(layer.attention);
    combined.merge(layer.mlp_expansion);
    combined.merge(layer.mlp_gate);
    combined.merge(layer.mlp_projection);
    combined.mean_abs_value()
}

fn normalized_parameter_stat(value: f64, max_value: f64) -> f64 {
    if max_value <= 0.0 {
        0.0
    } else {
        (value / max_value).clamp(0.0, 1.0)
    }
}

fn network_energy(snapshot: &LayerInspectionSnapshot) -> f64 {
    if snapshot.layers.is_empty() {
        return 0.0;
    }
    let total = snapshot
        .layers
        .iter()
        .map(|layer| {
            normalized_parameter_stat(layer_mean_abs_value(layer), snapshot.max_mean_abs_value)
        })
        .sum::<f64>();
    (total / snapshot.layers.len() as f64).clamp(0.0, 1.0)
}

fn token_slot_class(row: usize, active_token_slots: usize) -> &'static str {
    if row < active_token_slots {
        "token-dot active"
    } else {
        "token-dot"
    }
}

fn embedding_probe_styles(
    snapshot: Option<&Result<TokenEmbeddingSnapshot, String>>,
) -> Vec<String> {
    let Some(Ok(snapshot)) = snapshot else {
        return (0..8)
            .map(|_| "background: rgba(79, 216, 255, 0.12);".into())
            .collect();
    };
    if snapshot.rows.is_empty() || snapshot.embedding_size == 0 {
        return (0..8)
            .map(|_| "background: rgba(79, 216, 255, 0.12);".into())
            .collect();
    }

    (0..8)
        .map(|index| {
            let row_index = index * snapshot.rows.len() / 8;
            let column_index = index * snapshot.embedding_size / 8;
            let value = snapshot.rows[row_index]
                .values
                .get(column_index)
                .copied()
                .unwrap_or(0.0);
            embedding_cell_style(value, snapshot.max_abs_value)
        })
        .collect()
}

fn logit_probe_class(row: usize) -> &'static str {
    match row {
        9 => "logit-dot boundary",
        _ => "logit-dot",
    }
}

fn layer_row_tint_style(layer: &LayerInspection, max_mean_abs_value: f64) -> String {
    let intensity = normalized_parameter_stat(layer_mean_abs_value(layer), max_mean_abs_value);
    let alpha = 0.08 + 0.22 * intensity;
    let border_alpha = 0.18 + 0.16 * intensity;
    format!(
        "border-color: rgba(79, 216, 255, {border_alpha:.3}); background: linear-gradient(90deg, rgba(79, 216, 255, {alpha:.3}), rgba(7, 10, 14, 0.94) 46%);"
    )
}

fn layer_stage_tint_style(
    class_suffix: &str,
    stats: ParameterStats,
    max_mean_abs_value: f64,
) -> String {
    let (red, green, blue) = layer_stage_color(class_suffix);
    let intensity = normalized_parameter_stat(stats.mean_abs_value(), max_mean_abs_value);
    let alpha = 0.06 + 0.18 * intensity;
    let border_alpha = 0.26 + 0.22 * intensity;
    format!(
        "border-color: rgba({red}, {green}, {blue}, {border_alpha:.3}); background: linear-gradient(135deg, rgba({red}, {green}, {blue}, {alpha:.3}), rgba(9, 13, 18, 0.98) 66%);"
    )
}

fn non_parameter_stage_tint_style(class_suffix: &str) -> String {
    let (red, green, blue) = layer_stage_color(class_suffix);
    format!(
        "border-color: rgba({red}, {green}, {blue}, 0.38); background: linear-gradient(135deg, rgba({red}, {green}, {blue}, 0.14), rgba(9, 13, 18, 0.98) 66%);"
    )
}

fn layer_stage_color(class_suffix: &str) -> (u8, u8, u8) {
    match class_suffix {
        "norm" => (170, 164, 145),
        "attention" => (79, 216, 255),
        "mlp-expand" => (116, 214, 168),
        "mlp-gate" | "mlp-gate-off" => (246, 211, 101),
        "mlp-mix" => (224, 64, 251),
        "mlp-project" => (255, 77, 109),
        _ => (240, 232, 208),
    }
}

fn format_enabled(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

fn metric(label: &str, tooltip: &str, value: String) -> Element {
    rsx! {
        div { class: "metric",
            div { class: "metric-label", title: "{tooltip}", "{label}" }
            div { class: "metric-value", "{value}" }
        }
    }
}

fn config_number_input(
    label: &'static str,
    value: usize,
    is_editable: bool,
    field: TrainingRunConfigField,
    mut state: Signal<AppState>,
) -> Element {
    let tooltip = config_label_tooltip(label);
    rsx! {
        div { class: "field",
            label { title: "{tooltip}", "{label}" }
            input {
                class: "text-input",
                r#type: "number",
                min: "1",
                value: "{value}",
                disabled: !is_editable,
                oninput: move |event| {
                    state.write().set_training_run_config_value(field, event.value());
                }
            }
        }
    }
}

fn config_checkbox_input(
    label: &'static str,
    value: bool,
    is_editable: bool,
    field: TrainingRunConfigToggleField,
    mut state: Signal<AppState>,
) -> Element {
    let tooltip = config_label_tooltip(label);
    rsx! {
        label { class: "checkbox-field", title: "{tooltip}",
            input {
                r#type: "checkbox",
                checked: value,
                disabled: !is_editable,
                onchange: move |event| {
                    state.write().set_training_run_config_toggle(field, event.checked());
                }
            }
            span { "{label}" }
        }
    }
}

fn config_label_tooltip(label: &str) -> &'static str {
    match label {
        "Validation interval steps" => {
            "How many training steps run between validation loss evaluations."
        }
        "Docs per batch" => "Number of training documents sampled for each optimizer step.",
        "Max total docs" => {
            "Maximum number of filtered source documents available for training and validation."
        }
        "Validation docs divisor" => {
            "Validation set size before capping: total filtered documents divided by this value."
        }
        "Validation docs cap" => {
            "Maximum number of documents reserved for the fixed validation set."
        }
        "Context size" => "Maximum number of characters the model can condition on at once.",
        "Layers" => "Number of repeated Transformer blocks in the model.",
        "Attention heads" => "Number of attention heads used in each Transformer layer.",
        "Embedding size" => "Width of token, hidden, and attention projection vectors.",
        "MLP expansion factor" => {
            "Multiplier from embedding width to the feed-forward hidden width inside each MLP."
        }
        "Learned biases" => {
            "Enable trainable bias vectors in linear projections and output logits."
        }
        "RoPE positions" => "Use rotary position embeddings in attention queries and keys.",
        "Absolute positions" => "Add learned absolute position embeddings to token embeddings.",
        "Residual dropout" => {
            "Apply dropout to attention and MLP residual updates during training."
        }
        "Learned RMSNorm gain" => "Use trainable per-channel gain values in RMSNorm layers.",
        "Final RMSNorm" => "Normalize the final hidden state before producing output logits.",
        "SwiGLU MLP" => "Use the gated SwiGLU feed-forward block instead of a single activation.",
        "GELU MLP activation" => {
            "When SwiGLU is disabled, use GELU instead of ReLU in the feed-forward block."
        }
        "Tied output embeddings" => {
            "Share token embedding weights with the output language-model head."
        }
        "Gradient clipping" => "Limit gradient norm before Adam updates to reduce unstable steps.",
        "Weight decay" => "Apply AdamW-style parameter decay during optimization.",
        "Warmup/cosine LR" => {
            "Use warmup followed by cosine learning-rate decay instead of linear decay."
        }
        _ => "Training configuration value.",
    }
}

fn document_match_score(document: &str, query: &str, terms: &[&str]) -> usize {
    let mut score = if document.contains(query) {
        1_000 + query.len() * 4
    } else {
        ordered_character_score(document, query)
    };

    for term in terms {
        if document.contains(term) {
            score += 100 + term.len() * 8;
            score += document.matches(term).count() * 25;
        } else {
            score += ordered_character_score(document, term);
        }
    }

    score
}

fn ordered_character_score(document: &str, query: &str) -> usize {
    let mut score = 0;
    let mut document_chars = document.chars();
    for query_character in query.chars() {
        if document_chars.any(|document_character| document_character == query_character) {
            score += 1;
        }
    }
    score
}

fn loss_history_chart(progress_history: &[MicrogptTrainingProgress]) -> Element {
    if progress_history.is_empty() {
        return rsx! { div { class: "model-summary", "Waiting for first training step" } };
    }

    let plotted_losses: Vec<_> = progress_history
        .iter()
        .flat_map(|progress| [Some(progress.loss), progress.validation_loss])
        .flatten()
        .collect();
    let min_loss = plotted_losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max_loss = plotted_losses
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let loss_range = (max_loss - min_loss).max(0.000_001);
    let training_points = polyline_points(
        progress_history,
        |progress| Some(progress.loss),
        min_loss,
        loss_range,
    );
    let validation_points = polyline_points(
        progress_history,
        |progress| progress.validation_loss,
        min_loss,
        loss_range,
    );
    let running_mean_points = running_mean_loss_points(progress_history, min_loss, loss_range);
    let latest_running_mean = running_mean_loss(progress_history).unwrap_or(0.0);
    let latest = progress_history.last().expect("non-empty progress history");

    rsx! {
        div {
            div { class: "model-summary", "max {format_loss(max_loss)} | min {format_loss(min_loss)} | running mean {format_loss(latest_running_mean)}" }
            svg { class: "chart", view_box: "0 0 1000 220", preserve_aspect_ratio: "none",
                line { x1: "30", y1: "190", x2: "980", y2: "190", stroke: "#8b846f", stroke_width: "1" }
                line { x1: "30", y1: "20", x2: "30", y2: "190", stroke: "#8b846f", stroke_width: "1" }
                for y in [20, 62, 105, 147, 190] {
                    line { x1: "30", y1: "{y}", x2: "980", y2: "{y}", stroke: "#26333a", stroke_width: "1" }
                }
                polyline {
                    points: "{training_points}",
                    fill: "none",
                    stroke: "#4fd8ff",
                    stroke_width: "2",
                    stroke_linecap: "round",
                    stroke_linejoin: "round"
                }
                if !validation_points.is_empty() {
                    polyline {
                        points: "{validation_points}",
                        fill: "none",
                        stroke: "#ff4d6d",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round"
                    }
                }
                polyline {
                    points: "{running_mean_points}",
                    fill: "none",
                    stroke: "#f6d365",
                    stroke_width: "1.5",
                    stroke_dasharray: "10 8",
                    stroke_linecap: "round",
                    stroke_linejoin: "round"
                }
            }
            div { class: "model-summary",
                "Train {format_loss(latest.loss)} | Mean {format_loss(latest_running_mean)} | Step {latest.completed_step_count} / {latest.training_step_count}"
            }
        }
    }
}

fn generation_inspection_panel(
    trace: &SampleGenerationTrace,
    tokenizer: &CharacterTokenizer,
    selected_token_index: usize,
    mut state: Signal<AppState>,
) -> Element {
    if trace.tokens.is_empty() {
        return rsx! {
            div { class: "inspection-panel",
                div { class: "model-summary", "No generated tokens to inspect" }
            }
        };
    }

    let selected_token_index = selected_token_index.min(trace.tokens.len().saturating_sub(1));
    let selected_token = &trace.tokens[selected_token_index];
    let selected_label = token_label(tokenizer, selected_token.token_id);
    let selected_source = if selected_token.is_prefix {
        "prefix"
    } else {
        "sampled"
    };
    let mut distribution = selected_token
        .probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    distribution.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let max_probability = distribution
        .first()
        .map(|(_, probability)| *probability)
        .unwrap_or(1.0)
        .max(1e-12);

    rsx! {
        div { class: "inspection-panel",
            div { class: "inspection-token-row",
                for (token_index, token) in trace.tokens.iter().enumerate() {
                    button {
                        class: "{inspection_token_class(token.is_prefix, token_index == selected_token_index)}",
                        style: "{confidence_token_style(token.probability)}",
                        title: "position {token.position_id} | probability {format_probability(token.probability)} | entropy {format_rate(token.entropy)} bits",
                        onclick: move |_| state.write().select_inspection_token(token_index),
                        span { class: "inspection-token-text", "{token_label(tokenizer, token.token_id)}" }
                        if token.is_prefix {
                            span { class: "inspection-prefix-marker", "_" }
                        }
                    }
                }
            }
            div { class: "inspection-summary",
                div { "selected {selected_label}" }
                div { "source {selected_source}" }
                div { "position {selected_token.position_id}" }
                div { "confidence {format_probability(selected_token.probability)}" }
                div { "entropy {format_rate(selected_token.entropy)} bits" }
            }
            div { class: "distribution-list",
                for (token_id, probability) in distribution {
                    div {
                        class: "{distribution_row_class(token_id == selected_token.token_id)}",
                        title: "{token_title(tokenizer, token_id)}",
                        div { class: "distribution-token", "{token_label(tokenizer, token_id)}" }
                        div { class: "distribution-track",
                            div {
                                class: "distribution-fill",
                                style: "width: {format_percent_style(probability / max_probability)};"
                            }
                        }
                        div { class: "distribution-value", "{format_probability(probability)}" }
                    }
                }
            }
        }
    }
}

fn token_embedding_snapshot_for_session(
    session: &TrainingSession,
) -> Result<TokenEmbeddingSnapshot, String> {
    let tokenizer = session.tokenizer();
    let rows = match session {
        TrainingSession::Cpu(session) => session
            .trained_microgpt
            .model
            .token_embedding
            .iter()
            .map(|row| row.iter().map(|value| value.data()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        TrainingSession::Mlx(session) => {
            let embedding = &session.trained_microgpt.model.token_embedding;
            embedding.eval().map_err(|error| error.to_string())?;
            let shape = embedding.shape();
            let row_count = shape.first().copied().unwrap_or(0).max(0) as usize;
            let column_count = shape.get(1).copied().unwrap_or(0).max(0) as usize;
            embedding
                .as_slice::<f32>()
                .chunks(column_count.max(1))
                .take(row_count)
                .map(|row| row.iter().map(|value| *value as f64).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        }
    };

    let embedding_size = rows.first().map(Vec::len).unwrap_or(0);
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut max_abs_value = 0.0_f64;
    let rows = rows
        .into_iter()
        .enumerate()
        .map(|(token_id, values)| {
            for value in &values {
                min_value = min_value.min(*value);
                max_value = max_value.max(*value);
                max_abs_value = max_abs_value.max(value.abs());
            }
            let label = token_label(tokenizer, token_id);
            let title = token_title(tokenizer, token_id);
            let l2_norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            TokenEmbeddingRow {
                label,
                title,
                values,
                l2_norm,
            }
        })
        .collect::<Vec<_>>();

    if rows.is_empty() || embedding_size == 0 {
        min_value = 0.0;
        max_value = 0.0;
    }

    Ok(TokenEmbeddingSnapshot {
        rows,
        embedding_size,
        min_value,
        max_value,
        max_abs_value: max_abs_value.max(1e-12),
    })
}

fn token_embedding_table(snapshot: Option<&Result<TokenEmbeddingSnapshot, String>>) -> Element {
    match snapshot {
        None => rsx! { div { class: "model-summary", "Waiting for a training session" } },
        Some(Err(error)) => rsx! {
            div { class: "model-summary", "Could not read token embeddings: {error}" }
        },
        Some(Ok(snapshot)) if snapshot.rows.is_empty() || snapshot.embedding_size == 0 => {
            rsx! { div { class: "model-summary", "No token embeddings available" } }
        }
        Some(Ok(snapshot)) => {
            let column_ticks = embedding_column_ticks(snapshot.embedding_size);
            rsx! {
                div { class: "embedding-panel",
                    div { class: "embedding-summary",
                        div { "rows {snapshot.rows.len()} tokens" }
                        div { "columns {snapshot.embedding_size} dimensions" }
                        div { "range {format_embedding_value(snapshot.min_value)} to {format_embedding_value(snapshot.max_value)}" }
                    }
                    div { class: "embedding-legend",
                        span { "negative" }
                        div { class: "embedding-legend-ramp" }
                        span { "positive" }
                    }
                    div { class: "embedding-scroll",
                        div {
                            class: "embedding-table",
                            style: "grid-template-columns: 64px repeat({snapshot.embedding_size}, 12px) 56px;",
                            div { class: "embedding-corner", "token" }
                            for column_index in 0..snapshot.embedding_size {
                                div {
                                    class: "embedding-column-label",
                                    if column_ticks.contains(&column_index) {
                                        "{column_index}"
                                    }
                                }
                            }
                            div { class: "embedding-norm-label", "norm" }
                            for row in &snapshot.rows {
                                div {
                                    class: "embedding-token-label",
                                    title: "{row.title}",
                                    "{row.label}"
                                }
                                for value in &row.values {
                                    div {
                                        class: "embedding-cell",
                                        title: "{row.title} | {format_embedding_value(*value)}",
                                        style: "{embedding_cell_style(*value, snapshot.max_abs_value)}"
                                    }
                                }
                                div {
                                    class: "embedding-norm",
                                    title: "L2 norm of this token embedding vector.",
                                    "{format_embedding_value(row.l2_norm)}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn token_label(tokenizer: &CharacterTokenizer, token_id: usize) -> String {
    if token_id == tokenizer.sequence_boundary_token_id {
        return "<B>".into();
    }
    match tokenizer.unique_characters.get(token_id).copied() {
        Some('\n') => "\\n".into(),
        Some('\t') => "\\t".into(),
        Some(character) => character.to_string(),
        None => "?".into(),
    }
}

fn token_title(tokenizer: &CharacterTokenizer, token_id: usize) -> String {
    if token_id == tokenizer.sequence_boundary_token_id {
        return format!("token {token_id}: sequence boundary");
    }
    match tokenizer.unique_characters.get(token_id).copied() {
        Some(' ') => format!("token {token_id}: space"),
        Some(character) => format!("token {token_id}: '{character}'"),
        None => format!("token {token_id}: unknown"),
    }
}

fn embedding_column_ticks(embedding_size: usize) -> Vec<usize> {
    if embedding_size <= 16 {
        return (0..embedding_size).collect();
    }
    let step = (embedding_size / 8).max(1);
    let mut ticks = (0..embedding_size).step_by(step).collect::<Vec<_>>();
    if !ticks.contains(&(embedding_size - 1)) {
        ticks.push(embedding_size - 1);
    }
    ticks
}

fn embedding_cell_style(value: f64, max_abs_value: f64) -> String {
    let magnitude = (value.abs() / max_abs_value.max(1e-12)).clamp(0.0, 1.0);
    let alpha = 0.08 + 0.84 * magnitude;
    if value >= 0.0 {
        format!("background-color: rgba(79, 216, 255, {alpha:.3});")
    } else {
        format!("background-color: rgba(255, 77, 109, {alpha:.3});")
    }
}

fn confidence_token_style(probability: f64) -> String {
    let confidence = probability.clamp(0.0, 1.0).sqrt();
    let hue = 345.0 + 62.0 * confidence;
    let lightness = 82.0 - 18.0 * confidence;
    format!("background-color: hsl({hue:.1} 86% {lightness:.1}%);")
}

fn inspection_token_class(is_prefix: bool, is_selected: bool) -> &'static str {
    match (is_prefix, is_selected) {
        (true, true) => "inspection-token prefix-token selected",
        (true, false) => "inspection-token prefix-token",
        (false, true) => "inspection-token selected",
        (false, false) => "inspection-token",
    }
}

fn distribution_row_class(is_chosen: bool) -> &'static str {
    if is_chosen {
        "distribution-row chosen"
    } else {
        "distribution-row"
    }
}

fn format_probability(probability: f64) -> String {
    format!("{:.1}%", probability.clamp(0.0, 1.0) * 100.0)
}

fn format_embedding_value(value: f64) -> String {
    format!("{value:.3}")
}

fn running_mean_loss_points(
    progress_history: &[MicrogptTrainingProgress],
    min_loss: f64,
    loss_range: f64,
) -> String {
    let smoothed_losses = running_mean_loss_values(progress_history);
    let last_index = smoothed_losses.len().saturating_sub(1).max(1);
    smoothed_losses
        .iter()
        .enumerate()
        .map(|(index, mean_loss)| {
            let x = 30.0 + 950.0 * index as f64 / last_index as f64;
            let normalized_loss = (mean_loss - min_loss) / loss_range;
            let y = 190.0 - 170.0 * normalized_loss;
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn polyline_points(
    progress_history: &[MicrogptTrainingProgress],
    value: impl Fn(&MicrogptTrainingProgress) -> Option<f64>,
    min_loss: f64,
    loss_range: f64,
) -> String {
    let last_index = progress_history.len().saturating_sub(1).max(1);
    progress_history
        .iter()
        .enumerate()
        .filter_map(|(index, progress)| {
            value(progress).map(|loss| {
                let x = 30.0 + 950.0 * index as f64 / last_index as f64;
                let normalized_loss = (loss - min_loss) / loss_range;
                let y = 190.0 - 170.0 * normalized_loss;
                format!("{x:.2},{y:.2}")
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_rate(value: f64) -> String {
    format!("{value:.1}")
}

fn format_elapsed_training_time(milliseconds: u128) -> String {
    let total_seconds = milliseconds / 1_000;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else {
        format!("{minutes}m {seconds:02}s")
    }
}
