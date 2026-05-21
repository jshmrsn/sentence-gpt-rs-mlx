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
use serde::Deserialize;
use std::{
    io::{self, Stdout},
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use microgpt_lib::microgpt::{attach_validation_loss, calculate_training_loss_baseline, calculate_validation_loss, create_microgpt_training_session, generate_samples, train_microgpt_step, AdamOptimizerConfig, Matrix, MicrogptTrainingProgress, MicrogptTrainingSession, TransformerConfig};
use microgpt_lib::value::Value;

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

struct App {
    session: MicrogptTrainingSession,
    is_training_active: bool,
    is_training_busy: bool,
    next_validation_step: usize,
    accumulated_training_millis: u128,
    prefix: String,
    temperature: f64,
    samples: Vec<String>,
    sample_rng: ChaCha8Rng,
    training_receiver: Option<Receiver<TrainingChunkResult>>,
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
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let transformer_config = TransformerConfig::new(LAYER_COUNT, EMBEDDING_SIZE, CONTEXT_WINDOW_SIZE, ATTENTION_HEADS)
            .expect("valid built-in transformer config");
        let optimizer_config = AdamOptimizerConfig {
            learning_rate: 0.006,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1e-8,
        };
        let input_documents = load_input_documents()?;
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
        let validation_loss = calculate_validation_loss(&session, 0, VALIDATION_STEP_INTERVAL);
        session = session.with_initial_progress(train_loss, validation_loss);

        Ok(Self {
            session,
            is_training_active: false,
            is_training_busy: false,
            next_validation_step: VALIDATION_STEP_INTERVAL,
            accumulated_training_millis: 0,
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
            KeyCode::Char('r') => {
                if !self.is_training_busy {
                    match App::initialize() {
                        Ok(next) => *self = next,
                        Err(error) => self.status_message = error,
                    }
                }
                false
            }
            KeyCode::Char('g') | KeyCode::Enter => {
                self.generate();
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
        if self.is_training_active && !self.is_training_busy && !self.session.is_complete() {
            self.spawn_training_worker();
        }
    }

    fn spawn_training_worker(&mut self) {
        let session = self.session.clone();
        let next_validation_step = self.next_validation_step;
        let (sender, receiver) = mpsc::channel();
        self.is_training_busy = true;
        self.training_receiver = Some(receiver);
        self.status_message = "Training".into();

        thread::spawn(move || {
            let result = train_session_until_budget(session, next_validation_step);
            let _ = sender.send(result);
        });
    }

    fn poll_training_worker(&mut self) {
        let Some(receiver) = self.training_receiver.take() else {
            return;
        };

        match receiver.try_recv() {
            Ok(result) => {
                self.accumulated_training_millis += result.elapsed_millis;
                self.next_validation_step = result.next_validation_step;
                self.session = result.session;
                self.is_training_busy = false;
                if self.session.is_complete() {
                    self.is_training_active = false;
                    self.status_message = "Ready".into();
                } else if self.is_training_active {
                    self.status_message = "Training queued".into();
                } else {
                    self.status_message = "Paused".into();
                }
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

    fn generate(&mut self) {
        self.samples = generate_samples(
            &self.session.trained_microgpt.model,
            &self.session.trained_microgpt.config,
            &self.session.trained_microgpt.tokenizer,
            &self.prefix,
            10,
            self.temperature,
            &mut self.sample_rng,
        );
        self.status_message = "Generated samples".into();
    }

    fn progress_fraction(&self) -> f64 {
        self.session.completed_step_count as f64 / self.session.training_step_count.max(1) as f64
    }

    fn document_trains_per_minute(&self) -> f64 {
        if self.accumulated_training_millis == 0 {
            return 0.0;
        }
        self.session.completed_step_count as f64 * TRAINING_DOCUMENT_BATCH_SIZE as f64 * 60_000.0
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
    let config = &app.session.trained_microgpt.config;
    let tokenizer = &app.session.trained_microgpt.tokenizer;
    let text = vec![
        Line::from(vec![
            Span::styled(
                "microgpt Rust TUI",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  {}", app.status_message)),
        ]),
        Line::from(format!(
            "layers {} | embedding {} | heads {} x {} | context {} | vocab {} | prefix {:?} | temp {:.1}",
            config.layer_count,
            config.embedding_size,
            config.attention_head_count,
            config.attention_head_size,
            config.context_window_size,
            tokenizer.vocabulary_size(),
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
        .latest_loss
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let latest_validation_loss = app
        .session
        .latest_validation_loss
        .map(format_loss)
        .unwrap_or_else(|| "pending".into());
    let label = format!(
        "step {} / {} | train loss {} | validation {} | {:.1} doc-trains/min",
        app.session.completed_step_count,
        app.session.training_step_count,
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
    let losses = sparkline_losses(&app.session.progress_history);
    let train_examples = app.session.documents.len();
    let validation_examples = app.session.validation_documents.len();
    frame.render_widget(
        Sparkline::default()
            .block(
                Block::default()
                    .title(format!(
                        "Loss history | train docs {} | validation docs {}",
                        train_examples, validation_examples
                    ))
                    .borders(Borders::ALL),
            )
            .style(Style::default().fg(Color::Blue))
            .data(&losses),
        area,
    );
}

fn render_model(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let model = &app.session.trained_microgpt.model;
    let mut lines = vec![
        matrix_line("token embedding", &model.token_embedding),
        matrix_line("position embedding", &model.position_embedding),
        matrix_line("language head", &model.language_model_head),
    ];
    for (index, layer) in model.layers.iter().enumerate() {
        lines.push(format!("layer {}", index + 1));
        lines.push(matrix_line("  q", &layer.attention.query_weights));
        lines.push(matrix_line("  k", &layer.attention.key_weights));
        lines.push(matrix_line("  v", &layer.attention.value_weights));
        lines.push(matrix_line(
            "  attn out",
            &layer.attention.output_projection_weights,
        ));
        lines.push(matrix_line(
            "  ff expand",
            &layer.feed_forward.expansion_weights,
        ));
        lines.push(matrix_line(
            "  ff project",
            &layer.feed_forward.projection_weights,
        ));
    }
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
    let help = "s start/pause | c step chunk | g/Enter generate | type prefix | Backspace edit | +/- temperature | r reset | q/Esc quit";
    frame.render_widget(
        Paragraph::new(help).block(Block::default().title("Keys").borders(Borders::ALL)),
        area,
    );
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

fn matrix_line(label: &str, matrix: &Matrix) -> String {
    let rows = matrix.len();
    let columns = matrix.first().map(Vec::len).unwrap_or(0);
    let stats = matrix_stats(matrix);
    format!("{label:<16} {rows:>3}x{columns:<3} {stats}")
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

fn format_loss(loss: f64) -> String {
    format!("{loss:.4}")
}

fn format_compact(value: f64) -> String {
    format!("{value:.3}")
}
