//! egui desktop frontend. Reuses the whole backend (config, llm streaming +
//! parser, tools, training, persistence); this module is just the window, the
//! agentic orchestration, and the widgets.

use std::path::PathBuf;
use std::time::Duration;

use egui::{Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use tokio::runtime::Handle;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::task::AbortHandle;

use gene_core::config::Config;
use gene_core::llm::{StreamEvent, WireMessage};
use gene_core::model::{Conversation, Message, Role, TrainingExample};
use gene_core::persist;
use gene_core::runs::Metric;
use gene_core::tools::{self, run_command, ExecResult};
use gene_core::train::{self, TrainMsg};

const USER: Color32 = Color32::from_rgb(120, 190, 255);
const ASSISTANT: Color32 = Color32::from_rgb(140, 220, 150);
const TOOL: Color32 = Color32::from_rgb(220, 200, 120);
const CODE_BG: Color32 = Color32::from_rgb(28, 30, 34);

/// Repaint cadence while a background task (stream/exec/train) is in flight, so
/// streamed tokens and logs show up without waiting for an input event.
const BUSY_REPAINT_INTERVAL: Duration = Duration::from_millis(50);

/// Cap on retained fine-tune log lines.
const TRAIN_LOG_MAX: usize = 200;

/// A saved-conversation list entry: (id, title, updated_at).
type ConvEntry = (String, String, String);

/// Which persona/behavior the assistant uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Assistant,
    Tech,
    Convo,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Assistant => "assistant (runs commands)",
            Mode::Tech => "tech-guy (talks, no commands)",
            Mode::Convo => "convo (casual chat)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Busy {
    Idle,
    Streaming,
    Executing,
    Training,
}

/// Events from background tokio tasks, drained each frame.
enum AppEvent {
    Stream(StreamEvent),
    Exec(ExecResult),
    Train(TrainMsg),
    Models(Vec<String>),
}

pub struct GuiApp {
    config: Config,
    config_path: PathBuf,
    conv: Conversation,
    mode: Mode,

    http: reqwest::Client,
    rt: Handle,
    ctx: egui::Context,
    tx: UnboundedSender<AppEvent>,
    rx: UnboundedReceiver<AppEvent>,

    busy: Busy,
    streaming_index: Option<usize>,
    stream_abort: Option<AbortHandle>,
    tool_rounds: usize,
    auto_run: bool,

    input: String,
    refocus_input: bool,

    /// When `Some`, a command is awaiting confirmation; the string is editable.
    pending_command: Option<String>,

    /// When `Some(idx)`, editing assistant message `idx`; `edit_buf` is the field.
    editing: Option<usize>,
    edit_buf: String,
    edit_original: String,

    show_model_picker: bool,
    models: Option<Vec<String>>,
    show_settings: bool,
    show_help: bool,

    conversations: Vec<ConvEntry>,
    train_log: Vec<String>,
    train_metrics: Vec<Metric>,
    status: String,

    conversations_dir: PathBuf,
    dataset_path: PathBuf,
    work_dir: PathBuf,
    runs_dir: PathBuf,
}

impl GuiApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: Config,
        config_path: PathBuf,
        rt: Handle,
    ) -> Self {
        let (tx, rx) = unbounded_channel();
        let conv = Conversation::new(config.model.clone(), config.system_prompt.clone());
        let conversations_dir = config.conversations_dir().unwrap_or_default();
        let dataset_path = config.dataset_path().unwrap_or_default();
        let work_dir = config.work_dir().unwrap_or_default();
        let runs_dir = config.runs_dir().unwrap_or_default();
        let auto_run = config.agent.auto_run;
        let conversations = persist::list_conversations(&conversations_dir);

        GuiApp {
            config,
            config_path,
            conv,
            mode: Mode::Assistant,
            http: reqwest::Client::new(),
            rt,
            ctx: cc.egui_ctx.clone(),
            tx,
            rx,
            busy: Busy::Idle,
            streaming_index: None,
            stream_abort: None,
            tool_rounds: 0,
            auto_run,
            input: String::new(),
            refocus_input: true,
            pending_command: None,
            editing: None,
            edit_buf: String::new(),
            edit_original: String::new(),
            show_model_picker: false,
            models: None,
            show_settings: false,
            show_help: false,
            conversations,
            train_log: Vec::new(),
            train_metrics: Vec::new(),
            status: "ready".into(),
            conversations_dir,
            dataset_path,
            work_dir,
            runs_dir,
        }
    }

    fn effective_system_prompt(&self) -> String {
        match self.mode {
            Mode::Assistant => self.conv.system_prompt.clone(),
            Mode::Tech => self.config.tech_system_prompt.clone(),
            Mode::Convo => self.config.convo_system_prompt.clone(),
        }
    }

    fn build_wire(&self) -> Vec<WireMessage> {
        let mut wire = vec![WireMessage {
            role: "system".into(),
            content: self.effective_system_prompt(),
        }];
        for m in &self.conv.messages {
            if m.content.trim().is_empty() {
                continue;
            }
            let (role, content) = match m.role {
                Role::System => continue,
                Role::User => ("user", m.content.clone()),
                Role::Assistant => ("assistant", m.content.clone()),
                Role::Tool => (
                    "user",
                    format!(
                        "[output of `{}`]\n{}",
                        m.command.as_deref().unwrap_or(""),
                        m.content
                    ),
                ),
            };
            wire.push(WireMessage {
                role: role.into(),
                content,
            });
        }
        wire
    }

    fn submit_input(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.input.clear();
        self.conv.push(Message::new(Role::User, text));
        self.tool_rounds = 0;
        self.spawn_turn();
    }

    fn spawn_turn(&mut self) {
        let wire = self.build_wire();
        let idx = self.conv.push(Message::new(Role::Assistant, ""));
        self.streaming_index = Some(idx);
        self.busy = Busy::Streaming;

        let provider = self.config.chat_provider(self.http.clone());
        let request = self.config.chat_request(wire);
        let detect = self.mode == Mode::Assistant;
        let app_tx = self.tx.clone();
        let ctx = self.ctx.clone();
        let handle = self.rt.spawn(async move {
            let (s_tx, mut s_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
            let producer = provider.chat_stream(request, detect, s_tx);
            let forward = async {
                while let Some(ev) = s_rx.recv().await {
                    if app_tx.send(AppEvent::Stream(ev)).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            };
            tokio::join!(producer, forward);
        });
        self.stream_abort = Some(handle.abort_handle());
    }

    fn abort_stream(&mut self) {
        if let Some(h) = self.stream_abort.take() {
            h.abort();
        }
    }

    fn finalize_streaming(&mut self) {
        if let Some(i) = self.streaming_index {
            if let Some(m) = self.conv.messages.get(i) {
                if m.content.trim().is_empty() && m.thinking.is_none() {
                    self.conv.messages.remove(i);
                }
            }
        }
        self.streaming_index = None;
    }

    fn on_tool_call(&mut self, command: String) {
        self.abort_stream();
        if let Some(i) = self.streaming_index {
            let m = &mut self.conv.messages[i];
            if !m.content.is_empty() && !m.content.ends_with('\n') {
                m.content.push('\n');
            }
            m.content.push_str(&format!("```run\n{command}\n```\n"));
        }
        self.streaming_index = None;
        let denied = tools::is_denied(&command, &self.config.agent.denylist);
        if self.auto_run && !denied {
            self.exec_command(command);
        } else {
            self.pending_command = Some(command);
            self.busy = Busy::Idle;
            self.status = if denied {
                "command matches denylist — review carefully".into()
            } else {
                "command proposed — review and run".into()
            };
        }
    }

    fn exec_command(&mut self, command: String) {
        self.busy = Busy::Executing;
        self.pending_command = None;
        self.status = format!("running: {command}");
        let timeout = self.config.agent.exec_timeout_secs;
        let app_tx = self.tx.clone();
        let ctx = self.ctx.clone();
        self.rt.spawn(async move {
            let res = run_command(command, timeout).await;
            let _ = app_tx.send(AppEvent::Exec(res));
            ctx.request_repaint();
        });
    }

    fn deny_command(&mut self) {
        if let Some(cmd) = self.pending_command.take() {
            self.conv
                .push(Message::tool(cmd, "[command denied by user]".into(), None));
        }
        self.tool_rounds += 1;
        self.continue_or_idle();
    }

    fn continue_or_idle(&mut self) {
        if self.tool_rounds <= self.config.agent.max_tool_rounds {
            self.spawn_turn();
        } else {
            self.busy = Busy::Idle;
            self.status = "reached max tool rounds".into();
            self.save_conv();
        }
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                AppEvent::Stream(s) => self.on_stream(s),
                AppEvent::Exec(res) => {
                    let feedback = res.as_feedback();
                    self.conv
                        .push(Message::tool(res.command, feedback, res.exit_code));
                    self.tool_rounds += 1;
                    self.continue_or_idle();
                }
                AppEvent::Train(msg) => self.on_train(msg),
                AppEvent::Models(m) => {
                    if m.is_empty() {
                        self.status = "no models found (is Ollama running?)".into();
                    }
                    self.models = Some(m);
                }
            }
        }
    }

    fn on_stream(&mut self, ev: StreamEvent) {
        match ev {
            StreamEvent::AnswerDelta(s) => {
                if let Some(i) = self.streaming_index {
                    self.conv.messages[i].content.push_str(&s);
                }
            }
            StreamEvent::ThinkStart | StreamEvent::ThinkDelta(_) => {
                if let (Some(i), text) = (self.streaming_index, think_text(&ev)) {
                    let m = &mut self.conv.messages[i];
                    m.thinking.get_or_insert_with(String::new).push_str(&text);
                }
            }
            StreamEvent::ThinkEnd => {}
            StreamEvent::ToolCall(tc) => self.on_tool_call(tc.command),
            StreamEvent::Done => {
                self.abort_stream();
                self.finalize_streaming();
                if self.busy == Busy::Streaming {
                    self.busy = Busy::Idle;
                    self.status = "ready".into();
                }
                self.save_conv();
            }
            StreamEvent::Error(e) => {
                self.abort_stream();
                self.finalize_streaming();
                self.busy = Busy::Idle;
                self.status = format!("error: {e}");
            }
        }
    }

    fn on_train(&mut self, msg: TrainMsg) {
        match msg {
            TrainMsg::Log(line) => {
                self.train_log.push(line);
                if self.train_log.len() > TRAIN_LOG_MAX {
                    self.train_log.remove(0);
                }
            }
            TrainMsg::Metric(m) => {
                self.train_metrics.push(m);
            }
            TrainMsg::Done {
                ok,
                message,
                new_base_url,
                new_model,
            } => {
                self.busy = Busy::Idle;
                self.status = message;
                if ok {
                    if let Some(url) = new_base_url {
                        self.config.base_url = url;
                    }
                    if let Some(model) = new_model {
                        self.config.model = model.clone();
                        self.conv.model = model;
                    }
                }
            }
        }
    }

    fn start_training(&mut self) {
        if self.busy == Busy::Training {
            return;
        }
        self.busy = Busy::Training;
        self.train_log.clear();
        self.train_metrics.clear();
        self.status = "starting fine-tune…".into();
        let (t_tx, mut t_rx) = unbounded_channel::<TrainMsg>();
        train::start_training(
            self.config.clone(),
            self.work_dir.clone(),
            self.dataset_path.clone(),
            self.runs_dir.clone(),
            t_tx,
        );
        let app_tx = self.tx.clone();
        let ctx = self.ctx.clone();
        self.rt.spawn(async move {
            while let Some(m) = t_rx.recv().await {
                let _ = app_tx.send(AppEvent::Train(m));
                ctx.request_repaint();
            }
        });
    }

    fn fetch_models(&mut self) {
        self.models = None;
        let provider = self.config.chat_provider(self.http.clone());
        let app_tx = self.tx.clone();
        let ctx = self.ctx.clone();
        self.rt.spawn(async move {
            let m = provider.list_models().await;
            let _ = app_tx.send(AppEvent::Models(m));
            ctx.request_repaint();
        });
    }

    fn start_edit(&mut self, idx: usize) {
        self.edit_original = self.conv.messages[idx].content.clone();
        self.edit_buf.clear();
        self.editing = Some(idx);
    }

    fn commit_edit(&mut self) {
        let Some(idx) = self.editing else { return };
        let text = self.edit_buf.trim().to_string();
        if text.is_empty() {
            self.editing = None;
            self.status = "empty — edit discarded".into();
            return;
        }
        if let Some(m) = self.conv.messages.get_mut(idx) {
            if m.original_content.is_none() {
                m.original_content = Some(m.content.clone());
            }
            m.content = text;
            m.edited = true;
        }
        self.editing = None;
        match self.append_training_example(idx) {
            Ok(true) => self.status = "reply edited and added to training dataset".into(),
            _ => self.status = "reply edited".into(),
        }
        self.save_conv();
    }

    fn append_training_example(&self, idx: usize) -> anyhow::Result<bool> {
        match TrainingExample::from_conversation(&self.conv, idx) {
            Some(ex) => {
                persist::append_dataset(&self.dataset_path, &ex)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn new_conversation(&mut self) {
        self.save_conv();
        self.abort_stream();
        self.conv = Conversation::new(self.config.model.clone(), self.config.system_prompt.clone());
        self.streaming_index = None;
        self.busy = Busy::Idle;
        self.pending_command = None;
        self.refresh_conversations();
        self.status = "new conversation".into();
    }

    fn load_conversation(&mut self, id: &str) {
        match persist::load_conversation(&self.conversations_dir, id) {
            Ok(conv) => {
                self.abort_stream();
                self.streaming_index = None;
                self.busy = Busy::Idle;
                self.conv = conv;
                self.status = "conversation loaded".into();
            }
            Err(e) => self.status = format!("load failed: {e}"),
        }
    }

    fn save_conv(&mut self) {
        if let Err(e) = persist::save_conversation(&self.conversations_dir, &self.conv) {
            self.status = format!("save failed: {e}");
        }
        self.refresh_conversations();
    }

    fn save_config(&mut self) {
        match toml::to_string_pretty(&self.config) {
            Ok(s) => match std::fs::write(&self.config_path, s) {
                Ok(()) => self.status = "config saved".into(),
                Err(e) => self.status = format!("config save failed: {e}"),
            },
            Err(e) => self.status = format!("config serialize failed: {e}"),
        }
    }

    fn refresh_conversations(&mut self) {
        self.conversations = persist::list_conversations(&self.conversations_dir);
    }
}

fn think_text(ev: &StreamEvent) -> String {
    match ev {
        StreamEvent::ThinkDelta(s) => s.clone(),
        _ => String::new(),
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        if self.busy != Busy::Idle {
            ctx.request_repaint_after(BUSY_REPAINT_INTERVAL);
        }

        self.top_bar(ctx);
        self.sidebar(ctx);
        self.input_bar(ctx);
        self.chat_panel(ctx);

        self.confirm_window(ctx);
        self.edit_window(ctx);
        self.model_window(ctx);
        self.settings_window(ctx);
        self.help_window(ctx);
        self.training_window(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = persist::save_conversation(&self.conversations_dir, &self.conv);
    }
}

impl GuiApp {
    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("gene");
                ui.separator();

                egui::ComboBox::from_id_salt("mode")
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.mode,
                            Mode::Assistant,
                            Mode::Assistant.label(),
                        );
                        ui.selectable_value(&mut self.mode, Mode::Tech, Mode::Tech.label());
                        ui.selectable_value(&mut self.mode, Mode::Convo, Mode::Convo.label());
                    });

                if ui.button(format!("model: {}", self.config.model)).clicked() {
                    self.show_model_picker = true;
                    if self.models.is_none() {
                        self.fetch_models();
                    }
                }

                ui.checkbox(&mut self.auto_run, "auto-run").on_hover_text(
                    "run proposed commands without confirming (denylist still asks)",
                );

                ui.separator();
                if ui.button("new").clicked() {
                    self.new_conversation();
                }
                if ui.button("fine-tune").clicked() {
                    self.start_training();
                }
                if ui.button("settings").clicked() {
                    self.show_settings = true;
                }
                if ui.button("help").clicked() {
                    self.show_help = true;
                }
                if self.busy == Busy::Streaming && ui.button("stop").clicked() {
                    self.abort_stream();
                    self.finalize_streaming();
                    self.busy = Busy::Idle;
                    self.status = "stopped".into();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let busy = match self.busy {
                        Busy::Idle => "",
                        Busy::Streaming => "streaming… ",
                        Busy::Executing => "running… ",
                        Busy::Training => "training… ",
                    };
                    ui.label(RichText::new(format!("{busy}{}", self.status)).weak());
                });
            });
        });
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("conversations")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("conversations").strong());
                    if ui.small_button("⟳").on_hover_text("refresh").clicked() {
                        self.refresh_conversations();
                    }
                });
                ui.separator();
                let entries = self.conversations.clone();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (id, title, updated) in &entries {
                            let date = updated.get(..10).unwrap_or(updated);
                            let title = if title.is_empty() {
                                "(untitled)"
                            } else {
                                title.as_str()
                            };
                            let selected = id == &self.conv.id.to_string();
                            let label = format!("{date}\n{title}");
                            if ui.selectable_label(selected, label).clicked() {
                                self.load_conversation(id);
                            }
                        }
                    });
            });
    }

    fn input_bar(&mut self, ctx: &egui::Context) {
        let modal_open = self.pending_command.is_some()
            || self.editing.is_some()
            || self.show_settings
            || self.show_model_picker
            || self.show_help;

        egui::TopBottomPanel::bottom("input").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let send_clicked = ui.button("send").clicked();
                let hint = "Enter to send · Shift+Enter for newline";
                let resp = ui.add_sized(
                    [ui.available_width(), 56.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text(hint)
                        .desired_rows(2)
                        .id_salt("input"),
                );
                if self.refocus_input {
                    resp.request_focus();
                    self.refocus_input = false;
                }
                let enter = resp.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                if enter {
                    // TextEdit inserted a newline this frame; drop it before sending.
                    if self.input.ends_with('\n') {
                        self.input.pop();
                    }
                }
                if (send_clicked || enter) && !modal_open {
                    self.submit_input();
                    self.refocus_input = true;
                }
            });
            ui.add_space(4.0);
        });
    }

    fn chat_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self
                        .conv
                        .messages
                        .iter()
                        .all(|m| matches!(m.role, Role::System))
                    {
                        ui.label(
                            RichText::new("Start chatting — ask a question or request a command.")
                                .weak(),
                        );
                    }
                    let mut edit_request = None;
                    for (i, m) in self.conv.messages.iter().enumerate() {
                        if let Some(idx) = render_message(ui, m, i) {
                            edit_request = Some(idx);
                        }
                    }
                    if let Some(i) = edit_request {
                        self.start_edit(i);
                    }
                });
        });
    }

    fn confirm_window(&mut self, ctx: &egui::Context) {
        if self.pending_command.is_none() {
            return;
        }
        let denied = self
            .pending_command
            .as_ref()
            .map(|c| tools::is_denied(c, &self.config.agent.denylist))
            .unwrap_or(false);
        let mut run = false;
        let mut deny = false;
        egui::Window::new("Run command?")
            .collapsible(false)
            .resizable(true)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if denied {
                    ui.label(
                        RichText::new("⚠ matches denylist — review carefully").color(Color32::RED),
                    );
                }
                if let Some(cmd) = self.pending_command.as_mut() {
                    ui.add(egui::TextEdit::multiline(cmd).desired_rows(2).code_editor());
                }
                ui.horizontal(|ui| {
                    if ui.button("Run").clicked() {
                        run = true;
                    }
                    if ui.button("Deny").clicked() {
                        deny = true;
                    }
                });
            });
        if run {
            if let Some(cmd) = self.pending_command.take() {
                self.exec_command(cmd);
            }
        } else if deny {
            self.deny_command();
        }
    }

    fn edit_window(&mut self, ctx: &egui::Context) {
        if self.editing.is_none() {
            return;
        }
        let mut save = false;
        let mut cancel = false;
        let mut load_original = false;
        egui::Window::new("Edit reply")
            .collapsible(false)
            .resizable(true)
            .default_width(560.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("Type the corrected reply, or load the original to edit it.")
                        .weak(),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.edit_buf)
                        .desired_rows(10)
                        .desired_width(f32::INFINITY)
                        .hint_text("corrected reply…"),
                );
                ui.horizontal(|ui| {
                    if ui.button("Save → train").clicked() {
                        save = true;
                    }
                    if ui.button("Load original").clicked() {
                        load_original = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if load_original {
            self.edit_buf = self.edit_original.clone();
        }
        if save {
            self.commit_edit();
        } else if cancel {
            self.editing = None;
        }
    }

    fn model_window(&mut self, ctx: &egui::Context) {
        if !self.show_model_picker {
            return;
        }
        let mut open = true;
        let mut chosen = None;
        egui::Window::new("Select model")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                if ui.button("⟳ refresh").clicked() {
                    self.fetch_models();
                }
                ui.separator();
                match &self.models {
                    None => {
                        ui.label("loading… (is Ollama running?)");
                    }
                    Some(list) if list.is_empty() => {
                        ui.label("no models found");
                    }
                    Some(list) => {
                        egui::ScrollArea::vertical()
                            .max_height(300.0)
                            .show(ui, |ui| {
                                for m in list {
                                    if ui.selectable_label(m == &self.config.model, m).clicked() {
                                        chosen = Some(m.clone());
                                    }
                                }
                            });
                    }
                }
            });
        if let Some(m) = chosen {
            self.config.model = m.clone();
            self.conv.model = m.clone();
            self.status = format!("model → {m}");
            self.show_model_picker = false;
        }
        if !open {
            self.show_model_picker = false;
        }
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }
        let mut open = true;
        let mut save_cfg = false;
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(true)
            .default_width(620.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(440.0)
                    .show(ui, |ui| {
                        ui.label("Chat endpoint (base URL)");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.config.base_url)
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(6.0);
                        ui.add(
                            egui::Slider::new(&mut self.config.generation.temperature, 0.0..=2.0)
                                .text("temperature"),
                        );
                        ui.add_space(10.0);

                        ui.label(RichText::new("Assistant system prompt (runs commands)").strong());
                        ui.add(
                            egui::TextEdit::multiline(&mut self.config.system_prompt)
                                .desired_rows(4)
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(8.0);
                        ui.label(RichText::new("Tech-guy system prompt").strong());
                        ui.add(
                            egui::TextEdit::multiline(&mut self.config.tech_system_prompt)
                                .desired_rows(3)
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(8.0);
                        ui.label(RichText::new("Convo system prompt").strong());
                        ui.add(
                            egui::TextEdit::multiline(&mut self.config.convo_system_prompt)
                                .desired_rows(3)
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(8.0);
                        ui.label(RichText::new("Fine-tune MLX base").strong());
                        ui.add(
                            egui::TextEdit::singleline(&mut self.config.finetune.mlx_base)
                                .desired_width(f32::INFINITY),
                        );
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save to config.toml").clicked() {
                        save_cfg = true;
                    }
                    ui.label(RichText::new("(changes apply this session immediately)").weak());
                });
            });
        if save_cfg {
            // Assistant prompt also drives the current conversation.
            self.conv.system_prompt = self.config.system_prompt.clone();
            self.save_config();
        }
        if !open {
            self.show_settings = false;
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = true;
        egui::Window::new("Help").open(&mut open).resizable(true).show(ctx, |ui| {
            ui.label("• Modes: assistant runs shell commands (with confirm); tech-guy and convo just talk.");
            ui.label("• Edit a reply with its 'edit' button to correct it — saved replies train the model.");
            ui.label("• 'fine-tune' runs a real LoRA pass on your edited dataset (needs mlx-lm).");
            ui.label("• Pick past chats from the left sidebar; 'new' starts a fresh one.");
            ui.label("• Enter sends · Shift+Enter inserts a newline · 'stop' halts a response.");
        });
        if !open {
            self.show_help = false;
        }
    }

    fn training_window(&mut self, ctx: &egui::Context) {
        if self.busy != Busy::Training && self.train_log.is_empty() {
            return;
        }
        let mut open = true;
        egui::Window::new("Fine-tune")
            .open(&mut open)
            .resizable(true)
            .default_width(560.0)
            .show(ctx, |ui| {
                ui.label(if self.busy == Busy::Training {
                    "training in progress…"
                } else {
                    "last run"
                });
                ui.separator();
                if !self.train_metrics.is_empty() {
                    let series = |field: &str| -> Vec<[f64; 2]> {
                        self.train_metrics
                            .iter()
                            .filter_map(|m| m.fields.get(field).map(|v| [m.step as f64, *v]))
                            .collect()
                    };
                    let train = series("train_loss");
                    let val = series("val_loss");
                    Plot::new("loss")
                        .height(180.0)
                        .legend(Legend::default())
                        .show(ui, |plot_ui| {
                            if !train.is_empty() {
                                plot_ui.line(Line::new(PlotPoints::from(train)).name("train loss"));
                            }
                            if !val.is_empty() {
                                plot_ui.line(Line::new(PlotPoints::from(val)).name("val loss"));
                            }
                        });
                    ui.separator();
                }
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.train_log {
                            ui.label(RichText::new(line).monospace().small());
                        }
                    });
            });
        if !open && self.busy != Busy::Training {
            self.train_log.clear();
        }
    }
}

/// Render one message. Returns `Some(idx)` when the user clicks its edit button.
fn render_message(ui: &mut egui::Ui, m: &Message, idx: usize) -> Option<usize> {
    let mut edit_request = None;
    match m.role {
        Role::System => {}
        Role::User => {
            ui.add_space(8.0);
            ui.label(RichText::new("you").color(USER).strong());
            render_content(ui, &m.content);
        }
        Role::Assistant => {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("gene").color(ASSISTANT).strong());
                if m.edited {
                    ui.label(RichText::new("(edited)").weak().small());
                }
                if ui.small_button("edit").clicked() {
                    edit_request = Some(idx);
                }
            });
            if let Some(think) = &m.thinking {
                egui::CollapsingHeader::new(RichText::new("thinking").weak().small())
                    .id_salt(("think", idx))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(RichText::new(think).weak().italics());
                    });
            }
            render_content(ui, &m.content);
        }
        Role::Tool => {
            ui.add_space(4.0);
            let cmd = m.command.as_deref().unwrap_or("");
            ui.label(RichText::new(format!("$ {cmd}")).color(TOOL).monospace());
            code_block(ui, &m.content);
        }
    }
    edit_request
}

/// Render message text, splitting fenced code blocks into monospace frames.
fn render_content(ui: &mut egui::Ui, content: &str) {
    let mut buf = String::new();
    let mut in_code = false;
    for line in content.split('\n') {
        if line.trim_start().starts_with("```") {
            flush_segment(ui, &buf, in_code);
            buf.clear();
            in_code = !in_code;
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush_segment(ui, &buf, in_code);
}

fn flush_segment(ui: &mut egui::Ui, buf: &str, code: bool) {
    let text = buf.trim_end_matches('\n');
    if text.is_empty() {
        return;
    }
    if code {
        code_block(ui, text);
    } else {
        ui.add(egui::Label::new(text).wrap());
    }
}

fn code_block(ui: &mut egui::Ui, text: &str) {
    egui::Frame::group(ui.style()).fill(CODE_BG).show(ui, |ui| {
        ui.add(egui::Label::new(RichText::new(text).monospace()).wrap());
    });
}
