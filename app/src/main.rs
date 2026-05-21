// Dioxus desktop app for watching the tiny Transformer train.
//
// The ML concepts live in `microgpt-lib`; this file is mainly about turning
// training into an interactive learning tool. The important production lesson
// here is scheduling: training and sample generation are CPU/GPU-heavy work, so
// they run in blocking worker tasks while the UI stays responsive.

use dioxus::prelude::*;
use microgpt_config::{
    get_optimizer_config, AdamOptimizerConfig, ATTENTION_HEADS, CONTEXT_WINDOW_SIZE,
    EMBEDDING_SIZE, LAYER_COUNT, MAX_DOCUMENT_COUNT, MAX_TRAINING_STEP_COUNT,
    TRAINING_DOCUMENT_BATCH_SIZE, TRAINING_FRAME_BUDGET, VALIDATION_EVALUATION_DOCUMENT_COUNT,
    VALIDATION_SET_DIVISOR, VALIDATION_STEP_INTERVAL,
};
use microgpt_lib::microgpt::{
    attach_validation_loss as attach_cpu_validation_loss,
    calculate_training_loss_baseline as calculate_cpu_training_loss_baseline,
    calculate_validation_loss as calculate_cpu_validation_loss, create_microgpt_training_session,
    generate_samples as generate_cpu_samples, train_microgpt_step, CharacterTokenizer, Matrix,
    MicrogptTrainingProgress, MicrogptTrainingSession, TrainedMicrogpt, TransformerConfig,
};
use microgpt_lib::mlx_microgpt::{
    attach_validation_loss as attach_mlx_validation_loss,
    calculate_validation_loss as calculate_mlx_validation_loss,
    create_mlx_microgpt_training_session, generate_samples as generate_mlx_samples,
    matrix_heatmaps as build_mlx_matrix_heatmaps, train_mlx_microgpt_step, MlxMatrixHeatmap,
    MlxMicrogptTrainingSession, MlxTrainedMicrogpt,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const CSS: &str = r#"
:root {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #17202a;
    background: #edf4ef;
}

body {
    margin: 0;
}

button, input {
    font: inherit;
}

.app {
    min-height: 100vh;
    background: #edf4ef;
}

.shell {
    width: min(1180px, calc(100vw - 32px));
    margin: 0 auto;
    padding: 20px 0 40px;
}

.topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}

.title {
    margin: 0;
    font-size: 24px;
    line-height: 1.2;
    font-weight: 760;
}

.subtitle {
    margin: 6px 0 0;
    color: #53645c;
    font-size: 14px;
}

.actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
}

.button {
    border: 1px solid #28533f;
    background: #28533f;
    color: #fff;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
}

.button.secondary {
    background: #fff;
    color: #28533f;
}

.button:disabled {
    opacity: 0.55;
    cursor: default;
}

.panel {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #cfdcd3;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
}

.metric {
    border-left: 3px solid #28533f;
    padding-left: 10px;
}

.metric-label {
    color: #53645c;
    font-size: 12px;
}

.metric-value {
    font-size: 18px;
    font-weight: 720;
}

.progress-track {
    height: 10px;
    border-radius: 999px;
    background: #d5e2d9;
    overflow: hidden;
    margin: 12px 0 8px;
}

.progress-fill {
    height: 100%;
    background: #28533f;
}

.chart {
    width: 100%;
    height: 180px;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    background: #fbfdfb;
}

.controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px auto;
    gap: 12px;
    align-items: end;
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.field label {
    display: block;
    margin-bottom: 5px;
    font-size: 12px;
    color: #53645c;
}

.text-input {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid #b9c8bf;
    border-radius: 6px;
    padding: 8px 10px;
    background: #fff;
}

.range {
    width: 100%;
}

.samples {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.sample {
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 8px 10px;
    min-height: 22px;
}

.document-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.document-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 12px;
}

.page-button {
    border: 1px solid #b9c8bf;
    background: #fff;
    color: #28533f;
    border-radius: 6px;
    padding: 5px 8px;
    cursor: pointer;
    min-width: 34px;
}

.document-item {
    display: grid;
    grid-template-columns: 46px minmax(0, 1fr);
    gap: 8px;
    background: #fbfdfb;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    padding: 8px 10px;
}

.document-index {
    color: #53645c;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}

.document-text {
    color: #17202a;
    font-size: 13px;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.section-title {
    margin: 0 0 8px;
    font-size: 16px;
    font-weight: 720;
}

.model-summary {
    color: #53645c;
    font-size: 13px;
    margin-bottom: 12px;
}

.heatmap-groups {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.heatmap-card {
    min-width: 0;
}

.heatmap-label {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: #314238;
    font-size: 12px;
    margin-bottom: 4px;
}

.heatmap {
    display: grid;
    height: 120px;
    border: 1px solid #d3ded7;
    border-radius: 6px;
    overflow: hidden;
    background: #f7faf8;
}

.cell {
    min-width: 1px;
    min-height: 1px;
}

.layer {
    margin-top: 16px;
}

@media (max-width: 820px) {
    .topbar, .controls {
        display: block;
    }

    .actions {
        justify-content: flex-start;
        margin-top: 12px;
    }

    .status-grid, .heatmap-groups, .samples, .document-list {
        grid-template-columns: 1fr;
    }

    .field {
        margin-bottom: 10px;
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Mlx,
    Cpu,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Backend::Mlx => "MLX",
            Backend::Cpu => "CPU",
        }
    }

    fn toggled(self) -> Self {
        match self {
            Backend::Mlx => Backend::Cpu,
            Backend::Cpu => Backend::Mlx,
        }
    }
}

#[derive(Clone)]
enum TrainingSession {
    Mlx(MlxMicrogptTrainingSession),
    Cpu(MicrogptTrainingSession),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DocumentBrowserDataset {
    Training,
    Validation,
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

impl TrainingSession {
    fn backend(&self) -> Backend {
        match self {
            TrainingSession::Mlx(_) => Backend::Mlx,
            TrainingSession::Cpu(_) => Backend::Cpu,
        }
    }

    fn is_complete(&self) -> bool {
        match self {
            TrainingSession::Mlx(session) => session.is_complete(),
            TrainingSession::Cpu(session) => session.is_complete(),
        }
    }

    fn completed_step_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.completed_step_count,
            TrainingSession::Cpu(session) => session.completed_step_count,
        }
    }

    fn training_step_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.training_step_count,
            TrainingSession::Cpu(session) => session.training_step_count,
        }
    }

    fn latest_loss(&self) -> Option<f64> {
        match self {
            TrainingSession::Mlx(session) => session.latest_loss,
            TrainingSession::Cpu(session) => session.latest_loss,
        }
    }

    fn latest_validation_loss(&self) -> Option<f64> {
        match self {
            TrainingSession::Mlx(session) => session.latest_validation_loss,
            TrainingSession::Cpu(session) => session.latest_validation_loss,
        }
    }

    fn training_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.documents.len(),
            TrainingSession::Cpu(session) => session.documents.len(),
        }
    }

    fn training_documents(&self) -> &[String] {
        match self {
            TrainingSession::Mlx(session) => session.documents.as_slice(),
            TrainingSession::Cpu(session) => session.documents.as_slice(),
        }
    }

    fn validation_documents(&self) -> &[String] {
        match self {
            TrainingSession::Mlx(session) => session.validation_documents.as_slice(),
            TrainingSession::Cpu(session) => session.validation_documents.as_slice(),
        }
    }

    fn documents_for_browser(&self, dataset: DocumentBrowserDataset) -> &[String] {
        match dataset {
            DocumentBrowserDataset::Training => self.training_documents(),
            DocumentBrowserDataset::Validation => self.validation_documents(),
        }
    }

    fn validation_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session.validation_documents.len(),
            TrainingSession::Cpu(session) => session.validation_documents.len(),
        }
    }

    fn validation_evaluation_document_count(&self) -> usize {
        match self {
            TrainingSession::Mlx(session) => session
                .validation_evaluation_document_count
                .min(session.validation_documents.len()),
            TrainingSession::Cpu(session) => session
                .validation_evaluation_document_count
                .min(session.validation_documents.len()),
        }
    }

    fn tokenizer(&self) -> &CharacterTokenizer {
        match self {
            TrainingSession::Mlx(session) => &session.trained_microgpt.tokenizer,
            TrainingSession::Cpu(session) => &session.trained_microgpt.tokenizer,
        }
    }

    fn config(&self) -> &TransformerConfig {
        match self {
            TrainingSession::Mlx(session) => &session.trained_microgpt.config,
            TrainingSession::Cpu(session) => &session.trained_microgpt.config,
        }
    }

    fn progress_history(&self) -> &[MicrogptTrainingProgress] {
        match self {
            TrainingSession::Mlx(session) => session.progress_history.as_slice(),
            TrainingSession::Cpu(session) => session.progress_history.as_slice(),
        }
    }

    fn trained_snapshot(&self) -> TrainedSnapshot {
        match self {
            TrainingSession::Mlx(session) => TrainedSnapshot::Mlx(session.trained_microgpt.clone()),
            TrainingSession::Cpu(session) => TrainedSnapshot::Cpu(session.trained_microgpt.clone()),
        }
    }
}

#[derive(Clone)]
enum TrainedSnapshot {
    Mlx(MlxTrainedMicrogpt),
    Cpu(TrainedMicrogpt),
}

#[derive(Clone)]
struct ModelHeatmap {
    label: String,
    rows: usize,
    columns: usize,
    values: Vec<f32>,
    min: f32,
    max: f32,
    mean_abs: f32,
}

#[derive(Clone)]
struct AppState {
    backend: Backend,
    session: Option<TrainingSession>,
    // Full parameter heatmaps require copying model values to the UI and drawing
    // many small cells. That is excellent for inspection but expensive during
    // training, so it is opt-in.
    model_heatmaps: Vec<ModelHeatmap>,
    visualize_network_values: bool,
    is_training_active: bool,
    is_training_busy: bool,
    manual_training_chunk_requested: bool,
    // Generation is queued behind training so MLX is not asked to train and
    // sample from the same model concurrently.
    generation_requested: bool,
    is_generating_samples: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    prefix: String,
    document_browser_dataset: DocumentBrowserDataset,
    training_document_search: String,
    training_document_page: usize,
    temperature: f64,
    samples: Vec<String>,
    initialization_error: Option<String>,
    sample_rng: ChaCha8Rng,
    training_document_page_rng: ChaCha8Rng,
}

#[derive(Deserialize)]
struct Story {
    story: String,
    source: String,
}

struct TrainingChunkResult {
    session: TrainingSession,
    // Empty unless `visualize_network_values` was enabled when the worker began.
    model_heatmaps: Vec<ModelHeatmap>,
    visualized_network_values: bool,
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
    dioxus::launch(App);
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

            if let Some((session, next_validation_step, visualize_network_values)) = training_work {
                // `spawn_blocking` keeps the Dioxus/Tokio UI runtime from being
                // monopolized by MLX or CPU matrix math.
                match tokio::task::spawn_blocking(move || {
                    train_session_until_budget(
                        session,
                        next_validation_step,
                        visualize_network_values,
                    )
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
    let validation_batch_count = snapshot
        .session
        .as_ref()
        .map(TrainingSession::validation_evaluation_document_count)
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
    let is_complete = snapshot
        .session
        .as_ref()
        .is_some_and(TrainingSession::is_complete);
    let visible_documents = selected_browser_documents(&snapshot);
    let is_document_search_empty = snapshot.training_document_search.trim().is_empty();
    let browser_document_count = snapshot.browser_document_count();
    let browser_dataset_label = snapshot.document_browser_dataset.label();
    let browser_dataset_label_lowercase = browser_dataset_label.to_lowercase();
    let document_page_count = snapshot.document_page_count();
    let current_document_page = snapshot
        .training_document_page
        .min(document_page_count.saturating_sub(1));

    rsx! {
        style { "{CSS}" }
        main { class: "app",
            div { class: "shell",
                div { class: "topbar",
                    div {
                        h1 { class: "title", "microgpt Rust Visualized" }
                        p { class: "subtitle", "Transformer training ported from Kotlin Multiplatform Compose to Rust and Dioxus, with MLX and dry Rust CPU backends." }
                    }
                    div { class: "actions",
                        button {
                            class: "button",
                            disabled: snapshot.session.is_none() || is_complete,
                            onclick: move |_| state.write().toggle_training(),
                            if snapshot.is_training_active { "Pause" } else { "Start" }
                        }
                        button {
                            class: "button secondary",
                            disabled: snapshot.session.is_none() || is_complete || snapshot.is_training_busy,
                            onclick: move |_| state.write().request_training_chunk(),
                            "Step chunk"
                        }
                        button {
                            class: "button secondary",
                            disabled: snapshot.is_training_busy || snapshot.is_generating_samples,
                            onclick: move |_| state.write().toggle_backend(),
                            "Backend: {backend_label}"
                        }
                        button {
                            class: "button secondary",
                            disabled: snapshot.session.is_none(),
                            onclick: move |_| state.write().toggle_network_value_visualization(),
                            if snapshot.visualize_network_values { "Values: On" } else { "Values: Off" }
                        }
                        button {
                            class: "button secondary",
                            onclick: move |_| {
                                let backend = state.read().backend;
                                state.set(AppState::initialize_with_backend(backend));
                            },
                            "Reset"
                        }
                    }
                }

                if let Some(error) = &snapshot.initialization_error {
                    div { class: "panel", "Error: {error}" }
                }

                section { class: "panel",
                    div { class: "status-grid",
                        {metric("Status", status)}
                        {metric("Backend", backend_label.into())}
                        {metric("Step", format!("{} / {}", snapshot.completed_step_count(), snapshot.training_step_count()))}
                        {metric("Train loss", latest_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
                        {metric("Validation", latest_validation_loss.map(format_loss).unwrap_or_else(|| "pending".into()))}
                    }
                    div { class: "progress-track",
                        div { class: "progress-fill", style: "width: {format_percent_style(progress)};" }
                    }
                    div { class: "model-summary",
                        "Document trains {completed_document_train_count} / {total_document_train_count} | running avg {format_rate(document_trains_per_minute)}/min | elapsed {format_elapsed_training_time(snapshot.accumulated_training_millis)}"
                    }
                    div { class: "model-summary",
                        "Train examples {training_example_count} | validation examples {validation_example_count} | validation batch {validation_batch_count} | train batch {TRAINING_DOCUMENT_BATCH_SIZE}"
                    }
                    if let Some(loss) = latest_loss {
                        {loss_metric_text("Train loss", loss, vocabulary_size)}
                    }
                    if let Some(validation_loss) = latest_validation_loss {
                        {loss_metric_text("Validation loss", validation_loss, vocabulary_size)}
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
                        label { "Search {browser_dataset_label_lowercase} examples" }
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
                            label { "Prefix" }
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
                            label { "Temperature {format_rate(snapshot.temperature)}" }
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
                        for sample in snapshot.samples.iter() {
                            div { class: "sample", "{sample}" }
                        }
                    }
                }

                section { class: "panel",
                    div { class: "model-header",
                        h2 { class: "section-title", "Model values" }
                        button {
                            class: "button secondary",
                            disabled: snapshot.session.is_none(),
                            onclick: move |_| state.write().toggle_network_value_visualization(),
                            if snapshot.visualize_network_values { "Hide values" } else { "Show values" }
                        }
                    }
                    {model_visualization(
                        snapshot.session.as_ref(),
                        snapshot.visualize_network_values,
                        &snapshot.model_heatmaps,
                        snapshot.completed_step_count(),
                    )}
                }
            }
        }
    }
}

impl AppState {
    fn initialize() -> Self {
        Self::initialize_with_backend(Backend::Mlx)
    }

    fn initialize_with_backend(backend: Backend) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let transformer_config = TransformerConfig::new(
            LAYER_COUNT,
            EMBEDDING_SIZE,
            CONTEXT_WINDOW_SIZE,
            ATTENTION_HEADS,
        )
        .expect("valid built-in transformer config");
        let optimizer_config = get_optimizer_config();

        let documents = load_input_documents();
        match documents {
            Ok(input_documents) => {
                let session_result = create_training_session(
                    input_documents,
                    &mut rng,
                    backend,
                    transformer_config,
                    optimizer_config,
                );

                let session = match session_result {
                    Ok(session) => session,
                    Err(error) => {
                        return Self {
                            backend,
                            session: None,
                            model_heatmaps: Vec::new(),
                            visualize_network_values: false,
                            is_training_active: false,
                            is_training_busy: false,
                            manual_training_chunk_requested: false,
                            generation_requested: false,
                            is_generating_samples: false,
                            next_validation_step: VALIDATION_STEP_INTERVAL,
                            accumulated_training_millis: 0,
                            prefix: String::new(),
                            document_browser_dataset: DocumentBrowserDataset::Training,
                            training_document_search: String::new(),
                            training_document_page: 0,
                            temperature: 0.5,
                            samples: Vec::new(),
                            initialization_error: Some(error),
                            sample_rng: ChaCha8Rng::seed_from_u64(1),
                            training_document_page_rng: ChaCha8Rng::seed_from_u64(2),
                        };
                    }
                };
                Self {
                    backend,
                    session: Some(session),
                    model_heatmaps: Vec::new(),
                    visualize_network_values: false,
                    is_training_active: false,
                    is_training_busy: false,
                    manual_training_chunk_requested: false,
                    generation_requested: false,
                    is_generating_samples: false,
                    next_validation_step: VALIDATION_STEP_INTERVAL,
                    accumulated_training_millis: 0,
                    prefix: String::new(),
                    document_browser_dataset: DocumentBrowserDataset::Training,
                    training_document_search: String::new(),
                    training_document_page: 0,
                    temperature: 0.5,
                    samples: Vec::new(),
                    initialization_error: None,
                    sample_rng: ChaCha8Rng::seed_from_u64(1),
                    training_document_page_rng: ChaCha8Rng::seed_from_u64(2),
                }
            }
            Err(error) => Self {
                backend,
                session: None,
                model_heatmaps: Vec::new(),
                visualize_network_values: false,
                is_training_active: false,
                is_training_busy: false,
                manual_training_chunk_requested: false,
                generation_requested: false,
                is_generating_samples: false,
                next_validation_step: VALIDATION_STEP_INTERVAL,
                accumulated_training_millis: 0,
                prefix: String::new(),
                document_browser_dataset: DocumentBrowserDataset::Training,
                training_document_search: String::new(),
                training_document_page: 0,
                temperature: 0.5,
                samples: Vec::new(),
                initialization_error: Some(error),
                sample_rng: ChaCha8Rng::seed_from_u64(1),
                training_document_page_rng: ChaCha8Rng::seed_from_u64(2),
            },
        }
    }

    fn set_training_document_search(&mut self, search: String) {
        self.training_document_search = search;
        self.training_document_page = 0;
    }

    fn toggle_document_browser_dataset(&mut self) {
        self.document_browser_dataset = self.document_browser_dataset.toggled();
        self.training_document_page = 0;
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
            .map(|session| {
                session
                    .documents_for_browser(self.document_browser_dataset)
                    .len()
            })
            .unwrap_or(0)
    }

    fn document_page_count(&self) -> usize {
        self.browser_document_count().div_ceil(10)
    }

    fn browser_documents(&self) -> &[String] {
        self.session
            .as_ref()
            .map(|session| session.documents_for_browser(self.document_browser_dataset))
            .unwrap_or(&[])
    }

    fn toggle_backend(&mut self) {
        if self.is_training_busy || self.is_generating_samples {
            return;
        }
        *self = Self::initialize_with_backend(self.backend.toggled());
    }

    fn toggle_network_value_visualization(&mut self) {
        // Turning visualization on builds one snapshot immediately. Future
        // training chunks will refresh it only while the toggle stays on.
        self.visualize_network_values = !self.visualize_network_values;
        if self.visualize_network_values {
            self.model_heatmaps = self
                .session
                .as_ref()
                .map(build_model_heatmaps)
                .unwrap_or_default();
        } else {
            self.model_heatmaps.clear();
        }
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
        self.is_training_active = !self.is_training_active;
    }

    fn request_training_chunk(&mut self) {
        self.manual_training_chunk_requested = true;
    }

    fn take_training_work(&mut self) -> Option<(TrainingSession, usize, bool)> {
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
        Some((
            session.clone(),
            self.next_validation_step,
            self.visualize_network_values,
        ))
    }

    fn apply_training_chunk(&mut self, chunk_result: TrainingChunkResult) {
        // If the user toggled visualization while a worker was running, honor the
        // latest UI state. That avoids drawing stale heatmaps after values were
        // hidden, and can build a fresh snapshot if values were just enabled.
        let is_complete = chunk_result.session.is_complete();
        self.accumulated_training_millis += chunk_result.elapsed_millis;
        self.next_validation_step = chunk_result.next_validation_step;
        self.is_training_active = !is_complete && self.is_training_active;
        self.is_training_busy = false;
        if self.visualize_network_values {
            self.model_heatmaps = if chunk_result.visualized_network_values {
                chunk_result.model_heatmaps
            } else {
                build_model_heatmaps(&chunk_result.session)
            };
        } else {
            self.model_heatmaps.clear();
        }
        self.session = Some(chunk_result.session);
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
        self.completed_step_count() * TRAINING_DOCUMENT_BATCH_SIZE
    }

    fn total_document_train_count(&self) -> usize {
        self.training_step_count() * TRAINING_DOCUMENT_BATCH_SIZE
    }

    fn document_trains_per_minute(&self) -> f64 {
        if self.accumulated_training_millis == 0 {
            return 0.0;
        }
        self.completed_document_train_count() as f64 * 60_000.0
            / self.accumulated_training_millis as f64
    }

    fn progress_history(&self) -> &[MicrogptTrainingProgress] {
        self.session
            .as_ref()
            .map(TrainingSession::progress_history)
            .unwrap_or(&[])
    }
}

fn create_training_session(
    input_documents: Vec<String>,
    rng: &mut ChaCha8Rng,
    backend: Backend,
    transformer_config: TransformerConfig,
    optimizer_config: AdamOptimizerConfig,
) -> Result<TrainingSession, String> {
    match backend {
        Backend::Mlx => create_mlx_microgpt_training_session(
            input_documents,
            rng,
            MAX_TRAINING_STEP_COUNT,
            VALIDATION_SET_DIVISOR,
            VALIDATION_EVALUATION_DOCUMENT_COUNT,
            transformer_config,
            optimizer_config,
        )
        .with_initial_progress()
        .map(TrainingSession::Mlx)
        .map_err(|error| error.to_string()),
        Backend::Cpu => {
            let mut session = create_microgpt_training_session(
                input_documents,
                rng,
                MAX_TRAINING_STEP_COUNT,
                VALIDATION_SET_DIVISOR,
                VALIDATION_EVALUATION_DOCUMENT_COUNT,
                transformer_config,
                optimizer_config,
            );
            let train_loss = calculate_cpu_training_loss_baseline(&session);
            let validation_loss =
                calculate_cpu_validation_loss(&session, 0, VALIDATION_STEP_INTERVAL);
            session = session.with_initial_progress(train_loss, validation_loss);
            Ok(TrainingSession::Cpu(session))
        }
    }
}

fn train_session_until_budget(
    session: TrainingSession,
    mut next_validation_step: usize,
    visualize_network_values: bool,
) -> Result<TrainingChunkResult, String> {
    // One background chunk trains until the frame budget expires. The app then
    // yields back to the UI, updates metrics, and queues another chunk if
    // continuous training is active.
    let chunk_start = Instant::now();
    let frame_start = Instant::now();
    let session = match session {
        TrainingSession::Mlx(session) => TrainingSession::Mlx(train_mlx_until_budget(
            session,
            &mut next_validation_step,
            frame_start,
        )?),
        TrainingSession::Cpu(session) => TrainingSession::Cpu(train_cpu_until_budget(
            session,
            &mut next_validation_step,
            frame_start,
        )),
    };

    let model_heatmaps = if visualize_network_values {
        build_model_heatmaps(&session)
    } else {
        Vec::new()
    };
    Ok(TrainingChunkResult {
        session,
        model_heatmaps,
        visualized_network_values: visualize_network_values,
        next_validation_step,
        elapsed_millis: chunk_start.elapsed().as_millis(),
    })
}

fn train_mlx_until_budget(
    mut session: MlxMicrogptTrainingSession,
    next_validation_step: &mut usize,
    frame_start: Instant,
) -> Result<MlxMicrogptTrainingSession, String> {
    loop {
        if session.is_complete() {
            break;
        }

        let mut result = train_mlx_microgpt_step(session, TRAINING_DOCUMENT_BATCH_SIZE)
            .map_err(|error| error.to_string())?
            .expect("incomplete MLX session should produce a training step");

        if result.session.completed_step_count >= *next_validation_step {
            let validation_loss = calculate_mlx_validation_loss(
                &result.session,
                result.session.completed_step_count,
                VALIDATION_STEP_INTERVAL,
            )
            .map_err(|error| error.to_string())?;
            result = attach_mlx_validation_loss(result, validation_loss);
            *next_validation_step += VALIDATION_STEP_INTERVAL;
        }

        let should_stop = result.session.is_complete()
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
) -> MicrogptTrainingSession {
    loop {
        if session.is_complete() {
            break;
        }

        let mut result = train_microgpt_step(session, TRAINING_DOCUMENT_BATCH_SIZE)
            .expect("incomplete CPU session should produce a training step");

        if result.session.completed_step_count >= *next_validation_step {
            let validation_loss = calculate_cpu_validation_loss(
                &result.session,
                result.session.completed_step_count,
                VALIDATION_STEP_INTERVAL,
            );
            result = attach_cpu_validation_loss(result, validation_loss);
            *next_validation_step += VALIDATION_STEP_INTERVAL;
        }

        let should_stop = result.session.is_complete()
            || result.session.completed_step_count >= *next_validation_step
            || frame_start.elapsed() >= TRAINING_FRAME_BUDGET;
        session = result.session;

        if should_stop {
            break;
        }
    }

    session
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

fn load_input_documents() -> Result<Vec<String>, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let stories_path =
        root.join("shared/src/commonMain/composeResources/files/input-stories-00.json");
    let names_path = root.join("shared/src/commonMain/composeResources/files/input-names.txt");

    if stories_path.exists() {
        let stories_json = std::fs::read_to_string(&stories_path)
            .map_err(|error| format!("could not read {}: {error}", stories_path.display()))?;
        let stories: Vec<Story> = serde_json::from_str(&stories_json)
            .map_err(|error| format!("could not parse {}: {error}", stories_path.display()))?;
        let documents = stories_to_sentences(stories);
        if !documents.is_empty() {
            return Ok(documents);
        }
    }

    let names = std::fs::read_to_string(&names_path)
        .map_err(|error| format!("could not read fallback {}: {error}", names_path.display()))?;
    let documents = names
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_lowercase)
        .take(MAX_DOCUMENT_COUNT)
        .collect::<Vec<_>>();
    if documents.is_empty() {
        Err("no input documents found".into())
    } else {
        Ok(documents)
    }
}

fn stories_to_sentences(stories: Vec<Story>) -> Vec<String> {
    const EXCLUDE_CHARACTERS: &[char] = &[
        '$', '&', '"', '“', '”', '(', ')', '*', '\'', '_', '-', '–', '…', '%', '~', '`', '[', ']',
        '{', '}', '\\', ';', '|', '—', 'é', '/', '’', '‘', ':', '0', '1', '2', '3', '4', '5', '6',
        '7', '8', '9',
    ];

    stories
        .into_iter()
        .filter(|story| story.source == "GPT-4")
        .flat_map(|story| {
            story
                .story
                .replace(['!', '?'], ".")
                .split('.')
                .map(|sentence| sentence.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|sentence| {
            !EXCLUDE_CHARACTERS
                .iter()
                .any(|excluded| sentence.contains(*excluded))
        })
        .map(|sentence| sentence.replace(['\n', ','], "").trim().to_lowercase())
        .filter(|sentence| {
            sentence.len() > 10
                && sentence.contains(' ')
                && sentence.chars().count() < CONTEXT_WINDOW_SIZE
        })
        .take(MAX_DOCUMENT_COUNT)
        .collect()
}

fn metric(label: &str, value: String) -> Element {
    rsx! {
        div { class: "metric",
            div { class: "metric-label", "{label}" }
            div { class: "metric-value", "{value}" }
        }
    }
}

fn selected_browser_documents(state: &AppState) -> Vec<(usize, String)> {
    let documents = state.browser_documents();
    let query = state.training_document_search.trim().to_lowercase();
    if query.is_empty() {
        let page_start = state
            .training_document_page
            .min(state.document_page_count().saturating_sub(1))
            * 10;
        return documents
            .iter()
            .enumerate()
            .skip(page_start)
            .take(10)
            .map(|(index, document)| (index, document.clone()))
            .collect();
    }

    let terms = query
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    let mut scored_documents = documents
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
    scored_documents.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    scored_documents
        .into_iter()
        .take(10)
        .map(|(_, index, document)| (index, document))
        .collect()
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

fn loss_metric_text(label: &str, loss: f64, vocabulary_size: usize) -> Element {
    let estimated_accuracy = estimated_accuracy_from_loss(loss, vocabulary_size);
    let random_accuracy = if vocabulary_size > 0 {
        1.0 / vocabulary_size as f64
    } else {
        0.0
    };

    rsx! {
        div { class: "model-summary",
            "{label} estimated accuracy {format_percent(estimated_accuracy)} | random {format_percent(random_accuracy)} | vocab {vocabulary_size}"
        }
    }
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
    let latest = progress_history.last().expect("non-empty progress history");

    rsx! {
        div {
            div { class: "model-summary", "max {format_loss(max_loss)} | min {format_loss(min_loss)}" }
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
            }
            div { class: "model-summary",
                "Train {format_loss(latest.loss)} | Step {latest.completed_step_count} / {latest.training_step_count}"
            }
        }
    }
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

fn model_visualization(
    session: Option<&TrainingSession>,
    visualize_network_values: bool,
    heatmaps: &[ModelHeatmap],
    completed_step_count: usize,
) -> Element {
    // Even when heatmaps are hidden, keep the compact architecture summary
    // visible so learners can relate loss curves to model size.
    let Some(session) = session else {
        return rsx! { div { class: "model-summary", "Initializing model" } };
    };

    let config = session.config();
    let tokenizer = session.tokenizer();
    let backend = session.backend().label();

    rsx! {
        div {
            div { class: "model-summary",
                "{backend} | Step {completed_step_count} | layers {config.layer_count} | embedding {config.embedding_size} | heads {config.attention_head_count} x {config.attention_head_size} | context {config.context_window_size} | vocab {tokenizer.vocabulary_size()}"
            }
            if visualize_network_values {
                h3 { class: "section-title", "Parameter arrays" }
                div { class: "heatmap-groups",
                    for heatmap in heatmaps.iter() {
                        {matrix_heatmap(heatmap, None)}
                    }
                }
            } else {
                div { class: "model-summary", "Parameter heatmaps are hidden to reduce drawing and training overhead." }
            }
        }
    }
}

fn build_model_heatmaps(session: &TrainingSession) -> Vec<ModelHeatmap> {
    match session {
        TrainingSession::Mlx(session) => build_mlx_matrix_heatmaps(&session.trained_microgpt.model)
            .into_iter()
            .map(ModelHeatmap::from)
            .collect(),
        TrainingSession::Cpu(session) => build_cpu_model_heatmaps(&session.trained_microgpt),
    }
}

fn build_cpu_model_heatmaps(trained_microgpt: &TrainedMicrogpt) -> Vec<ModelHeatmap> {
    let model = &trained_microgpt.model;
    let mut heatmaps = vec![
        matrix_heatmap_data("Token embedding", &model.token_embedding),
        matrix_heatmap_data("Position embedding", &model.position_embedding),
        matrix_heatmap_data("Language head", &model.language_model_head),
    ];
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let prefix = format!("Layer {}", layer_index + 1);
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} Q"),
            &layer.attention.query_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} K"),
            &layer.attention.key_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} V"),
            &layer.attention.value_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} Attn out"),
            &layer.attention.output_projection_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} FF expand"),
            &layer.feed_forward.expansion_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} FF gate"),
            &layer.feed_forward.gate_weights,
        ));
        heatmaps.push(matrix_heatmap_data(
            &format!("{prefix} FF project"),
            &layer.feed_forward.projection_weights,
        ));
    }
    heatmaps
}

fn matrix_heatmap_data(label: &str, matrix: &Matrix) -> ModelHeatmap {
    let rows = matrix.len();
    let columns = matrix.first().map(Vec::len).unwrap_or(0);
    let values = matrix
        .iter()
        .flat_map(|row| row.iter().map(|value| value.data() as f32))
        .collect::<Vec<_>>();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f32>() / values.len().max(1) as f32;
    ModelHeatmap {
        label: label.into(),
        rows,
        columns,
        values,
        min,
        max,
        mean_abs,
    }
}

impl From<MlxMatrixHeatmap> for ModelHeatmap {
    fn from(heatmap: MlxMatrixHeatmap) -> Self {
        Self {
            label: heatmap.label,
            rows: heatmap.rows,
            columns: heatmap.columns,
            values: heatmap.values,
            min: heatmap.min,
            max: heatmap.max,
            mean_abs: heatmap.mean_abs,
        }
    }
}

fn matrix_heatmap(matrix: &ModelHeatmap, scale: Option<f64>) -> Element {
    let label = &matrix.label;
    let rows = matrix.rows;
    let columns = matrix.columns;
    if rows == 0 || columns == 0 {
        return rsx! { div { class: "heatmap-card", "{label}: empty" } };
    }

    let scale = scale.unwrap_or_else(|| matrix_max_abs(matrix).max(0.001));
    let stats = matrix_stats(matrix);
    let grid_style = format!("grid-template-columns: repeat({columns}, minmax(1px, 1fr));");

    rsx! {
        div { class: "heatmap-card",
            div { class: "heatmap-label",
                span { "{label} {rows}x{columns}" }
                span { "{stats}" }
            }
            div { class: "heatmap", style: "{grid_style}",
                for value in matrix.values.iter() {
                    div { class: "cell", style: "{weight_style(*value, scale)}" }
                }
            }
        }
    }
}

fn matrix_max_abs(matrix: &ModelHeatmap) -> f64 {
    matrix
        .values
        .iter()
        .map(|value| value.abs() as f64)
        .fold(0.0, f64::max)
}

fn matrix_stats(matrix: &ModelHeatmap) -> String {
    if matrix.values.is_empty() {
        return "empty".into();
    }
    format!(
        "min {} max {} mean |w| {}",
        format_compact(matrix.min as f64),
        format_compact(matrix.max as f64),
        format_compact(matrix.mean_abs as f64)
    )
}

fn weight_style(value: f32, scale: f64) -> String {
    let value = value as f64;
    let strength = (value.abs() / scale).min(1.0);
    let alpha = 0.12 + strength * 0.88;
    let color = if value >= 0.0 {
        format!("rgba(25, 118, 210, {alpha:.3})")
    } else {
        format!("rgba(198, 40, 40, {alpha:.3})")
    };
    format!("background: {color};")
}

fn format_loss(loss: f64) -> String {
    format!("{loss:.4}")
}

fn format_percent(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

fn format_percent_style(value: f64) -> String {
    format!("{:.3}%", (value * 100.0).clamp(0.0, 100.0))
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

fn estimated_accuracy_from_loss(loss: f64, vocabulary_size: usize) -> f64 {
    let random_loss = (vocabulary_size as f64).ln();
    if random_loss <= 0.0 {
        return 1.0;
    }
    (1.0 - loss / random_loss).clamp(0.0, 1.0)
}

fn format_compact(value: f64) -> String {
    format!("{value:.3}")
}
