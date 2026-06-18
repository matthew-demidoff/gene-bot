//! gene: a local-first toolkit for chatting with, evaluating, and fine-tuning
//! open LLMs. This binary is the CLI plus — with the default `gui` feature — the
//! egui desktop frontend; all engine logic lives in the `gene-core` library.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use gene_core::config::Config;

#[cfg(feature = "gui")]
mod app;
mod cli;

#[derive(Parser)]
#[command(
    name = "gene",
    version,
    about = "Local-first toolkit for chatting with, evaluating, and fine-tuning open LLMs"
)]
struct Cli {
    /// Path to a config file (defaults to the platform config dir).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the model tag (legacy single-endpoint setups).
    #[arg(long)]
    model: Option<String>,
    /// Override the chat endpoint base URL (legacy single-endpoint setups).
    #[arg(long)]
    base_url: Option<String>,
    /// Use a named provider profile for this run (overrides [roles].chat).
    #[arg(long, global = true)]
    provider: Option<String>,
    /// Emit machine-readable JSON instead of human-readable text.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Chat with the model (one-shot via --message, or piped on stdin).
    Chat(ChatArgs),
    /// List models available from the active provider.
    Models,
    /// Inspect tracked training/eval runs.
    Run {
        #[command(subcommand)]
        cmd: RunCmd,
    },
    /// Show or locate the configuration.
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },
    /// Report whether the chat + fine-tuning prerequisites are installed.
    Doctor,
}

#[derive(Args)]
struct ChatArgs {
    /// The prompt. If omitted, the prompt is read from stdin.
    #[arg(short, long)]
    message: Option<String>,
    /// Persona: assistant | tech | convo.
    #[arg(long, default_value = "tech")]
    mode: String,
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    max_tokens: Option<u32>,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Subcommand)]
enum RunCmd {
    /// List all runs, newest first.
    List,
    /// Show one run by id.
    Show { id: String },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Print the config file path.
    Path,
    /// Print the current configuration.
    Show,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let (mut cfg, cfg_path) = Config::load(cli.config.as_deref())?;
    if let Some(m) = cli.model {
        cfg.model = m;
    }
    if let Some(u) = cli.base_url {
        cfg.base_url = u;
    }
    if let Some(p) = cli.provider {
        cfg.roles.chat = Some(p);
    }
    let json = cli.json;

    match cli.command {
        Some(Cmd::Chat(a)) => {
            cli::chat(
                &cfg,
                &a.mode,
                a.message,
                a.temperature,
                a.max_tokens,
                a.seed,
                json,
            )
            .await
        }
        Some(Cmd::Models) => cli::models(&cfg, json).await,
        Some(Cmd::Run { cmd }) => match cmd {
            RunCmd::List => cli::run_list(&cfg, json),
            RunCmd::Show { id } => cli::run_show(&cfg, &id, json),
        },
        Some(Cmd::Config { cmd }) => match cmd {
            ConfigCmd::Path => cli::config_path(&cfg_path),
            ConfigCmd::Show => cli::config_show(&cfg, &cfg_path, json),
        },
        Some(Cmd::Doctor) => cli::doctor(&cfg, json).await,
        // No subcommand launches the desktop GUI (when this build includes it).
        None => launch_gui(cfg, cfg_path),
    }
}

#[cfg(feature = "gui")]
fn launch_gui(cfg: Config, cfg_path: PathBuf) -> Result<()> {
    init_tracing(&cfg);
    tracing::info!("config loaded from {}", cfg_path.display());

    // The GUI event loop runs on this (main) thread; background async work runs
    // on the tokio runtime's worker threads via this handle.
    let rt = tokio::runtime::Handle::current();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 740.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("gene"),
        ..Default::default()
    };

    eframe::run_native(
        "gene",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::GuiApp::new(cc, cfg, cfg_path, rt)))),
    )
    .map_err(|e| anyhow::anyhow!("gui error: {e}"))?;
    Ok(())
}

#[cfg(not(feature = "gui"))]
fn launch_gui(_cfg: Config, _cfg_path: PathBuf) -> Result<()> {
    anyhow::bail!(
        "this build has no GUI (compiled with `--no-default-features`); \
         run a subcommand such as `gene doctor` or `gene chat`"
    )
}

#[cfg(feature = "gui")]
fn init_tracing(cfg: &Config) {
    use tracing_subscriber::EnvFilter;
    let Ok(path) = cfg.log_path() else { return };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let make_writer = move || file.try_clone().expect("clone log file handle");
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(make_writer)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();
}
