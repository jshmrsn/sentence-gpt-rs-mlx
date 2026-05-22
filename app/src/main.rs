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
    generate_samples as generate_cpu_samples, MicrogptTrainingProgress, TransformerConfig,
};
use sentence_gpt_rs_mlx_lib::mlx_microgpt::generate_samples as generate_mlx_samples;
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
    is_generating_samples: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    throughput_start_step: usize,
    prefix: String,
    document_browser_dataset: DocumentBrowserDataset,
    training_document_search: String,
    cached_browser_search_matches: Vec<(usize, String)>,
    training_document_page: usize,
    training_config_expanded: bool,
    temperature: f64,
    samples: Vec<String>,
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
}

struct GenerationResult {
    samples: Vec<String>,
    sample_rng: ChaCha8Rng,
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
                        {metric("Status", status)}
                        {metric("Backend", backend_label.into())}
                        {metric("Model params", format_count(snapshot.session.as_ref().map(TrainingSession::parameter_count).unwrap_or(0)))}
                        {metric("Learning rate", snapshot.session.as_ref().map(|session| format_learning_rate(session.current_learning_rate())).unwrap_or_else(|| "pending".into()))}
                        {metric("Training step", format!("{} / {}", snapshot.completed_step_count(), snapshot.training_step_count()))}
                        {metric("Train loss", latest_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
                        {metric("Validation", latest_validation_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
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
        let transformer_config = TransformerConfig::new_with_features(
            training_run_config.layer_count,
            training_run_config.embedding_size,
            training_run_config.context_window_size,
            training_run_config.attention_heads,
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
                                is_generating_samples: false,
                                next_validation_step: training_run_config.validation_step_interval,
                                accumulated_training_millis: 0,
                                throughput_start_step: 0,
                                prefix: String::new(),
                                document_browser_dataset: DocumentBrowserDataset::Training,
                                training_document_search: String::new(),
                                cached_browser_search_matches: Vec::new(),
                                training_document_page: 0,
                                training_config_expanded: true,
                                temperature: 0.5,
                                samples: Vec::new(),
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
            is_generating_samples: false,
            next_validation_step: training_run_config.validation_step_interval,
            accumulated_training_millis: 0,
            throughput_start_step: 0,
            prefix: String::new(),
            document_browser_dataset: DocumentBrowserDataset::Training,
            training_document_search: String::new(),
            cached_browser_search_matches: Vec::new(),
            training_document_page: 0,
            training_config_expanded: true,
            temperature: 0.5,
            samples: Vec::new(),
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
        self.is_generating_samples = true;
        self.initialization_error = None;
    }

    fn take_generation_work(&mut self) -> Option<GenerationWork> {
        if self.is_training_busy || !self.generation_requested {
            return None;
        }
        let session = self.session.as_ref()?;
        self.generation_requested = false;
        Some(GenerationWork {
            trained_microgpt: session.trained_snapshot(),
            prefix: self.prefix.clone(),
            temperature: self.temperature,
            sample_rng: self.sample_rng.clone(),
        })
    }

    fn apply_generation_result(&mut self, generation_result: GenerationResult) {
        self.samples = generation_result.samples;
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

fn metric(label: &str, value: String) -> Element {
    let tooltip = metric_tooltip(label);
    rsx! {
        div { class: "metric",
            div { class: "metric-label", title: "{tooltip}", "{label}" }
            div { class: "metric-value", "{value}" }
        }
    }
}

fn metric_tooltip(label: &str) -> &'static str {
    match label {
        "Status" => "Current worker state for training or sample generation.",
        "Backend" => "Execution backend used for training and inference.",
        "Model params" => "Number of trainable model parameters.",
        "Learning rate" => "Current optimizer learning rate after scheduling.",
        "Training step" => "Completed training steps out of the configured training budget.",
        "Train loss" => "Latest cross-entropy loss measured on training data.",
        "Validation" => "Latest cross-entropy loss measured on the full fixed validation set.",
        _ => "Metric reported by the current training session.",
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
                line { x1: "30", y1: "190", x2: "980", y2: "190", stroke: "#80958a", stroke_width: "1" }
                line { x1: "30", y1: "20", x2: "30", y2: "190", stroke: "#80958a", stroke_width: "1" }
                for y in [20, 62, 105, 147, 190] {
                    line { x1: "30", y1: "{y}", x2: "980", y2: "{y}", stroke: "#d3ded7", stroke_width: "1" }
                }
                polyline {
                    points: "{training_points}",
                    fill: "none",
                    stroke: "#1f6feb",
                    stroke_width: "4",
                    stroke_linecap: "round",
                    stroke_linejoin: "round"
                }
                if !validation_points.is_empty() {
                    polyline {
                        points: "{validation_points}",
                        fill: "none",
                        stroke: "#c62828",
                        stroke_width: "4",
                        stroke_linecap: "round",
                        stroke_linejoin: "round"
                    }
                }
                polyline {
                    points: "{running_mean_points}",
                    fill: "none",
                    stroke: "#b7791f",
                    stroke_width: "3",
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
