// Ratatui terminal app for the same training loop as the Dioxus UI.
//
// This is intentionally useful over SSH or in a plain terminal: it shows loss,
// samples, backend choice, and optional parameter summaries without needing a
// desktop renderer. The same production rule applies here: training runs on a
// worker thread so input handling and rendering remain responsive.

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Wrap},
    Frame, Terminal,
};
use sentence_gpt_rs_mlx_config::{
    create_training_session, format_compact, format_count, format_learning_rate, format_loss,
    get_optimizer_config, load_input_documents, next_validation_step_after, running_mean_loss,
    train_session_until_budget as train_shared_session_until_budget, Backend, TrainingSession,
    ATTENTION_HEADS, CONTEXT_WINDOW_SIZE, EMBEDDING_SIZE, LAYER_COUNT,
    TRAINING_DOCUMENT_BATCH_SIZE, VALIDATION_STEP_INTERVAL,
};
use sentence_gpt_rs_mlx_lib::checkpoint::{load_checkpoint_from_path, save_checkpoint_to_path};
use sentence_gpt_rs_mlx_lib::microgpt::{
    generate_samples as generate_cpu_samples, Matrix, MicrogptTrainingProgress, TrainedMicrogpt,
    TransformerConfig, Vector,
};
use sentence_gpt_rs_mlx_lib::mlx_microgpt::{
    generate_samples as generate_mlx_samples, matrix_summaries as build_mlx_matrix_summaries,
    MlxMatrixSummary,
};
use std::{
    io::{self, Stdout},
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

#[derive(Clone)]
struct MatrixSummary {
    label: String,
    rows: usize,
    columns: usize,
    min: f32,
    max: f32,
    mean_abs: f32,
}

struct TrainingChunkResult {
    session: TrainingSession,
    matrix_summaries: Vec<MatrixSummary>,
    visualized_network_values: bool,
    next_validation_step: usize,
    elapsed_millis: u128,
}

struct App {
    session: TrainingSession,
    // Matrix summaries are hidden by default because collecting them requires
    // reading parameter tensors back for display. Press `v` when you want to
    // inspect value ranges.
    matrix_summaries: Vec<MatrixSummary>,
    visualize_network_values: bool,
    is_training_active: bool,
    is_training_busy: bool,
    generation_requested: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    throughput_start_step: usize,
    prefix: String,
    temperature: f64,
    samples: Vec<String>,
    sample_rng: ChaCha8Rng,
    training_receiver: Option<Receiver<Result<TrainingChunkResult, String>>>,
    status_message: String,
}

fn main() -> io::Result<()> {
    let mut app = App::initialize().map_err(io::Error::other)?;
    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal, &mut app);
    restore_terminal(&mut terminal)?;
    result
}

fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()
}

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> io::Result<()> {
    loop {
        app.poll_training_worker();
        app.start_training_worker_if_needed();
        terminal.draw(|frame| render(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if app.handle_key(key) => break,
                Event::Key(_) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

impl App {
    fn initialize() -> Result<Self, String> {
        Self::initialize_with_backend(Backend::Mlx)
    }

    fn initialize_with_backend(backend: Backend) -> Result<Self, String> {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let transformer_config = TransformerConfig::new(
            LAYER_COUNT,
            EMBEDDING_SIZE,
            CONTEXT_WINDOW_SIZE,
            ATTENTION_HEADS,
        )
        .expect("valid built-in transformer config");
        let optimizer_config = get_optimizer_config();
        let input_documents = load_input_documents()?;
        let session = create_training_session(
            input_documents,
            &mut rng,
            backend,
            transformer_config,
            optimizer_config,
        )?;
        Ok(Self {
            session,
            matrix_summaries: Vec::new(),
            visualize_network_values: false,
            is_training_active: false,
            is_training_busy: false,
            generation_requested: false,
            next_validation_step: VALIDATION_STEP_INTERVAL,
            accumulated_training_millis: 0,
            throughput_start_step: 0,
            prefix: String::new(),
            temperature: 0.5,
            samples: Vec::new(),
            sample_rng: ChaCha8Rng::seed_from_u64(1),
            training_receiver: None,
            status_message: "Ready".into(),
        })
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return true;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => true,
            KeyCode::Char('s') => {
                self.toggle_training();
                false
            }
            KeyCode::Char('c') => {
                self.request_training_chunk();
                false
            }
            KeyCode::Char('b') => {
                self.toggle_backend();
                false
            }
            KeyCode::Char('v') => {
                self.toggle_network_value_visualization();
                false
            }
            KeyCode::F(5) => {
                self.export_checkpoint();
                false
            }
            KeyCode::F(6) => {
                self.import_checkpoint();
                false
            }
            KeyCode::Char('r') => {
                if !self.is_training_busy {
                    match App::initialize_with_backend(self.session.backend()) {
                        Ok(next) => *self = next,
                        Err(error) => self.status_message = error,
                    }
                }
                false
            }
            KeyCode::Char('g') | KeyCode::Enter => {
                self.request_generation();
                false
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.temperature = (self.temperature + 0.1).min(2.0);
                false
            }
            KeyCode::Char('-') => {
                self.temperature = (self.temperature - 0.1).max(0.1);
                false
            }
            KeyCode::Backspace => {
                self.prefix.pop();
                false
            }
            KeyCode::Char(character) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    self.prefix.push(character);
                }
                false
            }
            _ => false,
        }
    }

    fn toggle_backend(&mut self) {
        if self.is_training_busy {
            return;
        }
        match App::initialize_with_backend(self.session.backend().toggled()) {
            Ok(next) => *self = next,
            Err(error) => self.status_message = error,
        }
    }

    fn export_checkpoint(&mut self) {
        if self.is_training_busy {
            return;
        }
        let path = checkpoint_file_path();
        match self
            .session
            .export_checkpoint()
            .and_then(|checkpoint| save_checkpoint_to_path(&checkpoint, &path))
        {
            Ok(()) => self.status_message = format!("Exported {}", path.display()),
            Err(error) => self.status_message = format!("Export failed: {error}"),
        }
    }

    fn import_checkpoint(&mut self) {
        if self.is_training_busy {
            return;
        }
        let path = checkpoint_file_path();
        match load_checkpoint_from_path(&path)
            .and_then(|checkpoint| TrainingSession::import_checkpoint(&checkpoint))
        {
            Ok(session) => {
                self.session = session;
                self.next_validation_step =
                    next_validation_step_after(self.session.completed_step_count());
                self.accumulated_training_millis = 0;
                self.throughput_start_step = self.session.completed_step_count();
                self.is_training_active = false;
                self.generation_requested = false;
                self.training_receiver = None;
                self.matrix_summaries = if self.visualize_network_values {
                    build_matrix_summaries(&self.session)
                } else {
                    Vec::new()
                };
                self.status_message = format!("Imported {}", path.display());
            }
            Err(error) => self.status_message = format!("Import failed: {error}"),
        }
    }

    fn toggle_network_value_visualization(&mut self) {
        // Build summaries immediately when toggled on; future training chunks
        // will refresh them only while visualization remains enabled.
        self.visualize_network_values = !self.visualize_network_values;
        if self.visualize_network_values {
            self.matrix_summaries = build_matrix_summaries(&self.session);
            self.status_message = "Model values visible".into();
        } else {
            self.matrix_summaries.clear();
            self.status_message = "Model values hidden".into();
        }
    }

    fn toggle_training(&mut self) {
        if self.session.is_complete() {
            self.is_training_active = false;
            return;
        }
        self.is_training_active = !self.is_training_active;
        self.status_message = if self.is_training_active {
            "Training queued".into()
        } else {
            "Paused".into()
        };
    }

    fn request_training_chunk(&mut self) {
        if self.session.is_complete() || self.is_training_busy {
            return;
        }
        self.spawn_training_worker();
    }

    fn start_training_worker_if_needed(&mut self) {
        // Generation waits until the current training worker finishes. This
        // avoids concurrent MLX access to the same logical model state.
        if self.generation_requested && !self.is_training_busy {
            self.generation_requested = false;
            self.generate_now();
            return;
        }

        if self.is_training_active && !self.is_training_busy && !self.session.is_complete() {
            self.spawn_training_worker();
        }
    }

    fn spawn_training_worker(&mut self) {
        // The worker owns a clone of the current session and returns a complete
        // replacement session. The UI thread never mutates model parameters while
        // training math is running.
        let session = self.session.clone();
        let next_validation_step = self.next_validation_step;
        let visualize_network_values = self.visualize_network_values;
        let (sender, receiver) = mpsc::channel();
        self.is_training_busy = true;
        self.training_receiver = Some(receiver);
        self.status_message = "Training".into();

        thread::spawn(move || {
            let result =
                train_session_until_budget(session, next_validation_step, visualize_network_values);
            let _ = sender.send(result);
        });
    }

    fn poll_training_worker(&mut self) {
        let Some(receiver) = self.training_receiver.take() else {
            return;
        };

        match receiver.try_recv() {
            Ok(Ok(result)) => {
                self.accumulated_training_millis += result.elapsed_millis;
                self.next_validation_step = result.next_validation_step;
                self.session = result.session;
                if self.visualize_network_values {
                    self.matrix_summaries = if result.visualized_network_values {
                        result.matrix_summaries
                    } else {
                        build_matrix_summaries(&self.session)
                    };
                } else {
                    self.matrix_summaries.clear();
                }
                self.is_training_busy = false;

                if self.generation_requested {
                    self.generation_requested = false;
                    self.generate_now();
                } else if self.session.is_complete() {
                    self.is_training_active = false;
                    self.status_message = "Ready".into();
                } else if self.is_training_active {
                    self.status_message = "Training queued".into();
                } else {
                    self.status_message = "Paused".into();
                }
            }
            Ok(Err(error)) => {
                self.is_training_active = false;
                self.is_training_busy = false;
                self.status_message = format!("Training failed: {error}");
            }
            Err(mpsc::TryRecvError::Empty) => {
                self.training_receiver = Some(receiver);
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.is_training_active = false;
                self.is_training_busy = false;
                self.status_message = "Training worker disconnected".into();
            }
        }
    }

    fn request_generation(&mut self) {
        if self.is_training_busy {
            self.generation_requested = true;
            self.status_message = "Generate queued".into();
            return;
        }
        self.generate_now();
    }

    fn generate_now(&mut self) {
        let result = match &self.session {
            TrainingSession::Mlx(session) => generate_mlx_samples(
                &session.trained_microgpt.model,
                &session.trained_microgpt.config,
                &session.trained_microgpt.tokenizer,
                &self.prefix,
                10,
                self.temperature,
                &mut self.sample_rng,
            )
            .map_err(|error| error.to_string()),
            TrainingSession::Cpu(session) => Ok(generate_cpu_samples(
                &session.trained_microgpt.model,
                &session.trained_microgpt.config,
                &session.trained_microgpt.tokenizer,
                &self.prefix,
                10,
                self.temperature,
                &mut self.sample_rng,
            )),
        };
        match result {
            Ok(samples) => {
                self.samples = samples;
                self.status_message = "Generated samples".into();
            }
            Err(error) => {
                self.status_message = format!("Generate failed: {error}");
            }
        }
    }

    fn progress_fraction(&self) -> f64 {
        self.session.completed_step_count() as f64
            / self.session.training_step_count().max(1) as f64
    }

    fn document_trains_per_minute(&self) -> f64 {
        if self.accumulated_training_millis == 0 {
            return 0.0;
        }
        let completed_steps_since_rate_start = self
            .session
            .completed_step_count()
            .saturating_sub(self.throughput_start_step);
        completed_steps_since_rate_start as f64 * TRAINING_DOCUMENT_BATCH_SIZE as f64 * 60_000.0
            / self.accumulated_training_millis as f64
    }
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Min(10),
            Constraint::Length(8),
        ])
        .split(frame.area());

    render_header(frame, app, root[0]);
    render_progress(frame, app, root[1]);
    render_loss(frame, app, root[2]);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(root[3]);
    render_model(frame, app, middle[0]);
    render_samples(frame, app, middle[1]);
    render_help(frame, root[4]);
}

fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let config = app.session.config();
    let backend = app.session.backend().label();
    let text = vec![
        Line::from(vec![
            Span::styled(
                "sentence-gpt-rs-mlx TUI",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  {}", app.status_message)),
        ]),
        Line::from(format!(
            "backend {} | params {} | lr {} | values {} | layers {} | embedding {} | heads {} x {} | context {} | vocab {} | prefix {:?} | temp {:.1}",
            backend,
            format_count(app.session.parameter_count()),
            format_learning_rate(app.session.current_learning_rate()),
            if app.visualize_network_values { "on" } else { "off" },
            config.layer_count,
            config.embedding_size,
            config.attention_head_count,
            config.attention_head_size,
            config.context_window_size,
            app.session.tokenizer_vocabulary_size(),
            app.prefix,
            app.temperature
        )),
    ];
    frame.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn render_progress(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let latest_loss = app
        .session
        .latest_loss()
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let latest_validation_loss = app
        .session
        .latest_validation_loss()
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let label = format!(
        "step {} / {} | train loss {} | validation {} | {:.1} doc-trains/min",
        app.session.completed_step_count(),
        app.session.training_step_count(),
        latest_loss,
        latest_validation_loss,
        app.document_trains_per_minute()
    );
    frame.render_widget(
        Gauge::default()
            .block(Block::default().title("Training").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(app.progress_fraction().clamp(0.0, 1.0))
            .label(label),
        area,
    );
}

fn render_loss(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let progress_history = app.session.progress_history();
    let losses = sparkline_losses(progress_history);
    let running_mean = running_mean_loss(progress_history)
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let train_examples = app.session.training_document_count();
    let validation_examples = app.session.validation_document_count();
    frame.render_widget(
        Sparkline::default()
            .block(
                Block::default()
                    .title(format!(
                        "Loss history | mean {} | train docs {} | validation docs {}",
                        running_mean, train_examples, validation_examples
                    ))
                    .borders(Borders::ALL),
            )
            .style(Style::default().fg(Color::Blue))
            .data(&losses),
        area,
    );
}

fn render_model(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Keeping this panel mounted but empty-by-default makes the cost model
    // obvious to the user: values are available, but not free.
    if !app.visualize_network_values {
        frame.render_widget(
            Paragraph::new("Hidden. Press v to show model value summaries.")
                .block(Block::default().title("Model values").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }

    let lines = app
        .matrix_summaries
        .iter()
        .map(|summary| {
            format!(
                "{:<18} {:>3}x{:<3} min {} max {} mean |w| {}",
                summary.label,
                summary.rows,
                summary.columns,
                format_compact(summary.min as f64),
                format_compact(summary.max as f64),
                format_compact(summary.mean_abs as f64)
            )
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines.join("\n"))
            .block(Block::default().title("Model values").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_samples(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let items = if app.samples.is_empty() {
        vec![ListItem::new("Press g or Enter to generate.")]
    } else {
        app.samples
            .iter()
            .enumerate()
            .map(|(index, sample)| ListItem::new(format!("{}: {}", index + 1, sample)))
            .collect()
    };
    frame.render_widget(
        List::new(items).block(Block::default().title("Samples").borders(Borders::ALL)),
        area,
    );
}

fn render_help(frame: &mut Frame<'_>, area: Rect) {
    let help = "s start/pause | c step chunk | g/Enter generate | b backend | v values | F5 export | F6 import | type prefix | Backspace edit | +/- temp | r reset | q/Esc quit";
    frame.render_widget(
        Paragraph::new(help).block(Block::default().title("Keys").borders(Borders::ALL)),
        area,
    );
}

fn train_session_until_budget(
    session: TrainingSession,
    next_validation_step: usize,
    visualize_network_values: bool,
) -> Result<TrainingChunkResult, String> {
    // A chunk is deliberately bounded by wall-clock time. Continuous training is
    // implemented as many small chunks, which lets the terminal process input
    // and redraw metrics between updates.
    let training_result = train_shared_session_until_budget(session, next_validation_step)?;

    let matrix_summaries = if visualize_network_values {
        build_matrix_summaries(&training_result.session)
    } else {
        Vec::new()
    };
    Ok(TrainingChunkResult {
        session: training_result.session,
        matrix_summaries,
        visualized_network_values: visualize_network_values,
        next_validation_step: training_result.next_validation_step,
        elapsed_millis: training_result.elapsed_millis,
    })
}

fn build_matrix_summaries(session: &TrainingSession) -> Vec<MatrixSummary> {
    match session {
        TrainingSession::Mlx(session) => {
            build_mlx_matrix_summaries(&session.trained_microgpt.model)
                .into_iter()
                .map(MatrixSummary::from)
                .collect()
        }
        TrainingSession::Cpu(session) => build_cpu_matrix_summaries(&session.trained_microgpt),
    }
}

fn build_cpu_matrix_summaries(trained_microgpt: &TrainedMicrogpt) -> Vec<MatrixSummary> {
    let model = &trained_microgpt.model;
    let mut summaries = vec![
        matrix_summary("Token embedding", &model.token_embedding),
        matrix_summary("Position embedding", &model.position_embedding),
        matrix_summary("Language head", &model.language_model_head),
        vector_summary("Language head bias", &model.language_model_head_biases),
        vector_summary("Final norm gain", &model.final_norm_gain),
    ];
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let prefix = format!("Layer {}", layer_index + 1);
        summaries.push(vector_summary(
            &format!("{prefix} attention norm gain"),
            &layer.attention_norm_gain,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} Q"),
            &layer.attention.query_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} Q bias"),
            &layer.attention.query_biases,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} K"),
            &layer.attention.key_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} K bias"),
            &layer.attention.key_biases,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} V"),
            &layer.attention.value_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} V bias"),
            &layer.attention.value_biases,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} Attn out"),
            &layer.attention.output_projection_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} Attn out bias"),
            &layer.attention.output_projection_biases,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} feed-forward norm gain"),
            &layer.feed_forward_norm_gain,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF expand"),
            &layer.feed_forward.expansion_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} FF expand bias"),
            &layer.feed_forward.expansion_biases,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF gate"),
            &layer.feed_forward.gate_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} FF gate bias"),
            &layer.feed_forward.gate_biases,
        ));
        summaries.push(matrix_summary(
            &format!("{prefix} FF project"),
            &layer.feed_forward.projection_weights,
        ));
        summaries.push(vector_summary(
            &format!("{prefix} FF project bias"),
            &layer.feed_forward.projection_biases,
        ));
    }
    summaries
}

fn matrix_summary(label: &str, matrix: &Matrix) -> MatrixSummary {
    let rows = matrix.len();
    let columns = matrix.first().map(Vec::len).unwrap_or(0);
    let values = matrix
        .iter()
        .flat_map(|row| row.iter().map(|value| value.data() as f32))
        .collect::<Vec<_>>();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f32>() / values.len().max(1) as f32;
    MatrixSummary {
        label: label.into(),
        rows,
        columns,
        min,
        max,
        mean_abs,
    }
}

fn vector_summary(label: &str, vector: &Vector) -> MatrixSummary {
    let values = vector
        .iter()
        .map(|value| value.data() as f32)
        .collect::<Vec<_>>();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f32>() / values.len().max(1) as f32;
    MatrixSummary {
        label: label.into(),
        rows: 1,
        columns: vector.len(),
        min,
        max,
        mean_abs,
    }
}

impl From<MlxMatrixSummary> for MatrixSummary {
    fn from(summary: MlxMatrixSummary) -> Self {
        Self {
            label: summary.label,
            rows: summary.rows,
            columns: summary.columns,
            min: summary.min,
            max: summary.max,
            mean_abs: summary.mean_abs,
        }
    }
}

fn checkpoint_file_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("sentence-gpt-rs-mlx-checkpoint.bin")
}

fn sparkline_losses(progress_history: &[MicrogptTrainingProgress]) -> Vec<u64> {
    if progress_history.is_empty() {
        return vec![0];
    }
    let losses: Vec<f64> = progress_history
        .iter()
        .map(|progress| progress.loss)
        .collect();
    let min = losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.000_001);
    losses
        .iter()
        .map(|loss| (((loss - min) / range) * 100.0) as u64)
        .collect()
}
