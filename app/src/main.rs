use dioxus::prelude::*;
use microgpt_lib::{
    attach_validation_loss, calculate_training_loss_baseline, calculate_validation_loss,
    create_microgpt_training_session, generate_samples, train_microgpt_step, AdamOptimizerConfig,
    Matrix, MicrogptTrainingProgress, MicrogptTrainingSession, TrainedMicrogpt, TransformerConfig,
    Value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const TRAINING_FRAME_BUDGET: Duration = Duration::from_millis(500);
const VALIDATION_STEP_INTERVAL: usize = 50;
const TRAINING_DOCUMENT_BATCH_SIZE: usize = 16;
const MAX_DOCUMENT_COUNT: usize = 3000;
const MAX_TRAINING_STEP_COUNT: usize = 8_000;
const VALIDATION_SET_DIVISOR: usize = 20;
const VALIDATION_EVALUATION_DOCUMENT_COUNT: usize = 8;
const CONTEXT_WINDOW_SIZE: usize = 64;
const LAYER_COUNT: usize = 3;
const ATTENTION_HEADS: usize = 8;
const EMBEDDING_SIZE: usize = 64;

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
    grid-template-columns: repeat(4, minmax(0, 1fr));
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

    .status-grid, .heatmap-groups, .samples {
        grid-template-columns: 1fr;
    }

    .field {
        margin-bottom: 10px;
    }
}
"#;

#[derive(Clone)]
struct AppState {
    session: Option<MicrogptTrainingSession>,
    is_training_active: bool,
    is_training_busy: bool,
    manual_training_chunk_requested: bool,
    is_generating_samples: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    prefix: String,
    temperature: f64,
    samples: Vec<String>,
    initialization_error: Option<String>,
    sample_rng: ChaCha8Rng,
}

#[derive(Deserialize)]
struct Story {
    story: String,
    source: String,
}

struct TrainingChunkResult {
    session: MicrogptTrainingSession,
    next_validation_step: usize,
    elapsed_millis: u128,
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let mut state = use_signal(AppState::initialize);

    use_future(move || async move {
        loop {
            let training_work = {
                let mut current = state.write();
                current.take_training_work()
            };

            if let Some((session, next_validation_step)) = training_work {
                match tokio::task::spawn_blocking(move || {
                    train_session_until_budget(session, next_validation_step)
                })
                .await
                {
                    Ok(chunk_result) => state.write().apply_training_chunk(chunk_result),
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
    let training_example_count = snapshot
        .session
        .as_ref()
        .map(|session| session.documents.len())
        .unwrap_or(0);
    let validation_example_count = snapshot
        .session
        .as_ref()
        .map(|session| session.validation_documents.len())
        .unwrap_or(0);
    let validation_batch_count = snapshot
        .session
        .as_ref()
        .map(|session| {
            session
                .validation_evaluation_document_count
                .min(session.validation_documents.len())
        })
        .unwrap_or(0);
    let latest_loss = snapshot
        .session
        .as_ref()
        .and_then(|session| session.latest_loss);
    let latest_validation_loss = snapshot
        .session
        .as_ref()
        .and_then(|session| session.latest_validation_loss);
    let vocabulary_size = snapshot
        .session
        .as_ref()
        .map(|session| session.trained_microgpt.tokenizer.vocabulary_size())
        .unwrap_or(0);
    let is_complete = snapshot
        .session
        .as_ref()
        .is_some_and(MicrogptTrainingSession::is_complete);
    let trained = snapshot
        .session
        .as_ref()
        .map(|session| &session.trained_microgpt);

    rsx! {
        style { "{CSS}" }
        main { class: "app",
            div { class: "shell",
                div { class: "topbar",
                    div {
                        h1 { class: "title", "microgpt Rust Visualized" }
                        p { class: "subtitle", "CPU scalar autograd Transformer training, ported from Kotlin Multiplatform Compose to Rust and Dioxus." }
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
                            onclick: move |_| state.set(AppState::initialize()),
                            "Reset"
                        }
                    }
                }

                if let Some(error) = &snapshot.initialization_error {
                    div { class: "panel", "Initialization failed: {error}" }
                }

                section { class: "panel",
                    div { class: "status-grid",
                        {metric("Status", status)}
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
                    h2 { class: "section-title", "Generate samples" }
                    div { class: "controls",
                        div { class: "field",
                            label { "Prefix" }
                            input {
                                class: "text-input",
                                value: "{snapshot.prefix}",
                                oninput: move |event| state.write().prefix = event.value()
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
                        for (index, sample) in snapshot.samples.iter().enumerate() {
                            div { class: "sample", "Sample {index + 1}: {sample}" }
                        }
                    }
                }

                section { class: "panel",
                    {model_visualization(trained, snapshot.completed_step_count())}
                }
            }
        }
    }
}

impl AppState {
    fn initialize() -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let transformer_config = TransformerConfig::new(LAYER_COUNT, EMBEDDING_SIZE, CONTEXT_WINDOW_SIZE, ATTENTION_HEADS)
            .expect("valid built-in transformer config");
        let optimizer_config = AdamOptimizerConfig {
            learning_rate: 0.006,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1e-8,
        };

        let documents = load_input_documents();
        match documents {
            Ok(input_documents) => {
                let mut session = create_microgpt_training_session(
                    input_documents,
                    &mut rng,
                    MAX_TRAINING_STEP_COUNT,
                    VALIDATION_SET_DIVISOR,
                    VALIDATION_EVALUATION_DOCUMENT_COUNT,
                    transformer_config,
                    optimizer_config,
                );

                let train_loss = calculate_training_loss_baseline(&session);
                let validation_loss =
                    calculate_validation_loss(&session, 0, VALIDATION_STEP_INTERVAL);
                session = session.with_initial_progress(train_loss, validation_loss);

                Self {
                    session: Some(session),
                    is_training_active: false,
                    is_training_busy: false,
                    manual_training_chunk_requested: false,
                    is_generating_samples: false,
                    next_validation_step: VALIDATION_STEP_INTERVAL,
                    accumulated_training_millis: 0,
                    prefix: String::new(),
                    temperature: 0.5,
                    samples: Vec::new(),
                    initialization_error: None,
                    sample_rng: ChaCha8Rng::seed_from_u64(1),
                }
            }
            Err(error) => Self {
                session: None,
                is_training_active: false,
                is_training_busy: false,
                manual_training_chunk_requested: false,
                is_generating_samples: false,
                next_validation_step: VALIDATION_STEP_INTERVAL,
                accumulated_training_millis: 0,
                prefix: String::new(),
                temperature: 0.5,
                samples: Vec::new(),
                initialization_error: Some(error),
                sample_rng: ChaCha8Rng::seed_from_u64(1),
            },
        }
    }

    fn toggle_training(&mut self) {
        if self
            .session
            .as_ref()
            .is_some_and(MicrogptTrainingSession::is_complete)
        {
            self.is_training_active = false;
            return;
        }
        self.is_training_active = !self.is_training_active;
    }

    fn request_training_chunk(&mut self) {
        self.manual_training_chunk_requested = true;
    }

    fn take_training_work(&mut self) -> Option<(MicrogptTrainingSession, usize)> {
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
        self.accumulated_training_millis += chunk_result.elapsed_millis;
        self.next_validation_step = chunk_result.next_validation_step;
        self.is_training_active = !is_complete && self.is_training_active;
        self.is_training_busy = false;
        self.session = Some(chunk_result.session);
    }

    fn generate(&mut self) {
        let Some(session) = &self.session else {
            return;
        };
        self.is_generating_samples = true;
        self.samples = generate_samples(
            &session.trained_microgpt.model,
            &session.trained_microgpt.config,
            &session.trained_microgpt.tokenizer,
            &self.prefix,
            10,
            self.temperature,
            &mut self.sample_rng,
        );
        self.is_generating_samples = false;
    }

    fn status_label(&self) -> String {
        match &self.session {
            None => "Initializing".into(),
            Some(session) if session.is_complete() => "Ready".into(),
            Some(_) if self.is_training_busy => "Training".into(),
            Some(_) if self.is_training_active => "Training queued".into(),
            Some(_) => "Paused".into(),
        }
    }

    fn completed_step_count(&self) -> usize {
        self.session
            .as_ref()
            .map(|session| session.completed_step_count)
            .unwrap_or(0)
    }

    fn training_step_count(&self) -> usize {
        self.session
            .as_ref()
            .map(|session| session.training_step_count)
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
            .map(|session| session.progress_history.as_slice())
            .unwrap_or(&[])
    }
}

fn train_session_until_budget(
    mut session: MicrogptTrainingSession,
    mut next_validation_step: usize,
) -> TrainingChunkResult {
    let chunk_start = Instant::now();
    let frame_start = Instant::now();
    let mut latest_result = None;

    loop {
        let Some(result) = train_microgpt_step(session.clone(), TRAINING_DOCUMENT_BATCH_SIZE)
        else {
            break;
        };
        session = result.session.clone();
        latest_result = Some(result);

        if session.is_complete()
            || session.completed_step_count >= next_validation_step
            || frame_start.elapsed() >= TRAINING_FRAME_BUDGET
        {
            break;
        }
    }

    if let Some(mut result) = latest_result {
        if session.completed_step_count >= next_validation_step {
            let validation_loss = calculate_validation_loss(
                &session,
                session.completed_step_count,
                VALIDATION_STEP_INTERVAL,
            );
            result = attach_validation_loss(result, validation_loss);
            session = result.session;
            next_validation_step += VALIDATION_STEP_INTERVAL;
        }
    }

    TrainingChunkResult {
        session,
        next_validation_step,
        elapsed_millis: chunk_start.elapsed().as_millis(),
    }
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
    trained_microgpt: Option<&TrainedMicrogpt>,
    completed_step_count: usize,
) -> Element {
    let Some(trained_microgpt) = trained_microgpt else {
        return rsx! { div { class: "model-summary", "Initializing model" } };
    };

    let config = &trained_microgpt.config;
    let tokenizer = &trained_microgpt.tokenizer;

    rsx! {
        div {
            h2 { class: "section-title", "Model values" }
            div { class: "model-summary",
                "Step {completed_step_count} | layers {config.layer_count} | embedding {config.embedding_size} | heads {config.attention_head_count} x {config.attention_head_size} | context {config.context_window_size} | vocab {tokenizer.vocabulary_size()}"
            }
            h3 { class: "section-title", "Embeddings and output" }
            div { class: "heatmap-groups",
                {matrix_heatmap("Token embedding", &trained_microgpt.model.token_embedding, None)}
                {matrix_heatmap("Position embedding", &trained_microgpt.model.position_embedding, None)}
                {matrix_heatmap("Language head", &trained_microgpt.model.language_model_head, None)}
            }
            h3 { class: "section-title layer", "Transformer layers" }
            for (layer_index, layer) in trained_microgpt.model.layers.iter().enumerate() {
                div { class: "layer",
                    div { class: "model-summary", "Layer {layer_index + 1}" }
                    div { class: "heatmap-groups",
                        {matrix_heatmap("Q", &layer.attention.query_weights, None)}
                        {matrix_heatmap("K", &layer.attention.key_weights, None)}
                        {matrix_heatmap("V", &layer.attention.value_weights, None)}
                        {matrix_heatmap("Attn out", &layer.attention.output_projection_weights, None)}
                        {matrix_heatmap("FF expand", &layer.feed_forward.expansion_weights, None)}
                        {matrix_heatmap("FF project", &layer.feed_forward.projection_weights, None)}
                    }
                }
            }
        }
    }
}

fn matrix_heatmap(label: &str, matrix: &Matrix, scale: Option<f64>) -> Element {
    let rows = matrix.len();
    let columns = matrix.first().map(Vec::len).unwrap_or(0);
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
                for row in matrix.iter() {
                    for value in row.iter() {
                        div { class: "cell", style: "{weight_style(value, scale)}" }
                    }
                }
            }
        }
    }
}

fn matrix_max_abs(matrix: &Matrix) -> f64 {
    matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.data().abs())
        .fold(0.0, f64::max)
}

fn matrix_stats(matrix: &Matrix) -> String {
    let values: Vec<_> = matrix
        .iter()
        .flat_map(|row| row.iter().map(Value::data))
        .collect();
    if values.is_empty() {
        return "empty".into();
    }
    let min_value = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64;
    format!(
        "min {} max {} mean |w| {}",
        format_compact(min_value),
        format_compact(max_value),
        format_compact(mean_abs)
    )
}

fn weight_style(value: &Value, scale: f64) -> String {
    let strength = (value.data().abs() / scale).min(1.0);
    let alpha = 0.12 + strength * 0.88;
    let color = if value.data() >= 0.0 {
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
