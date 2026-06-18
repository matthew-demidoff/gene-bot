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
    /// Manage the training dataset.
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCmd,
    },
    /// Run a fine-tune (LoRA/DoRA/full) and record it as a tracked run.
    Train(TrainArgs),
    /// Evaluate the model against a prompt set.
    Eval {
        #[command(subcommand)]
        cmd: EvalCmd,
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

#[derive(Args)]
struct DatasetFile {
    /// Dataset JSONL file (defaults to the configured dataset path).
    #[arg(long)]
    file: Option<PathBuf>,
}

#[derive(Subcommand)]
enum DatasetCmd {
    /// Summary counts for the dataset.
    Stats {
        #[command(flatten)]
        file: DatasetFile,
    },
    /// Remove duplicate examples (by normalized content).
    Dedup {
        #[command(flatten)]
        file: DatasetFile,
        /// Report what would change without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Import examples from another format (appends unless --replace).
    Import {
        #[command(flatten)]
        file: DatasetFile,
        /// Source file to read.
        #[arg(long)]
        from: PathBuf,
        /// Source format: gene | mlx | openai | sharegpt.
        #[arg(long, default_value = "openai")]
        format: String,
        /// Replace the dataset instead of appending.
        #[arg(long)]
        replace: bool,
    },
    /// Export the dataset to another format.
    Export {
        #[command(flatten)]
        file: DatasetFile,
        /// Destination file.
        #[arg(long)]
        to: PathBuf,
        /// Target format: gene | mlx | openai | sharegpt.
        #[arg(long, default_value = "mlx")]
        format: String,
    },
    /// Write an immutable, content-addressed snapshot of the dataset.
    Snapshot {
        #[command(flatten)]
        file: DatasetFile,
    },
}

#[derive(Args)]
struct TrainArgs {
    /// Fine-tune method: lora | dora | full.
    #[arg(long)]
    method: Option<String>,
    /// Override the iteration count.
    #[arg(long)]
    iters: Option<u32>,
    /// Override the learning rate.
    #[arg(long)]
    learning_rate: Option<String>,
    /// Print the resolved subprocess commands without running anything.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum EvalCmd {
    /// Run an eval set against the active provider and record an Eval run.
    Run(EvalRunArgs),
    /// Run an eval set across several named providers and compare side by side.
    Compare(EvalCompareArgs),
}

#[derive(Args)]
struct EvalRunArgs {
    /// Eval-set JSON file.
    #[arg(long)]
    set: PathBuf,
    /// Grader: none | exact | contains | judge.
    #[arg(long, default_value = "none")]
    grader: String,
    /// Judge provider profile (for --grader judge); defaults to [roles].judge.
    #[arg(long)]
    judge: Option<String>,
    /// Maximum concurrent requests.
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
}

#[derive(Args)]
struct EvalCompareArgs {
    /// Eval-set JSON file.
    #[arg(long)]
    set: PathBuf,
    /// Comma-separated named provider profiles to compare (e.g. base,finetuned).
    #[arg(long)]
    providers: String,
    /// Grader: none | exact | contains | judge.
    #[arg(long, default_value = "none")]
    grader: String,
    /// Judge provider profile (for --grader judge); defaults to [roles].judge.
    #[arg(long)]
    judge: Option<String>,
    /// Maximum concurrent requests.
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    reset_sigpipe();
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

    let result = match cli.command {
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
            ConfigCmd::Show => cli::config_show(&cfg, json),
        },
        Some(Cmd::Dataset { cmd }) => match cmd {
            DatasetCmd::Stats { file } => cli::dataset_stats(&cfg, file.file, json),
            DatasetCmd::Dedup { file, dry_run } => {
                cli::dataset_dedup(&cfg, file.file, dry_run, json)
            }
            DatasetCmd::Import {
                file,
                from,
                format,
                replace,
            } => cli::dataset_import(&cfg, file.file, &from, &format, replace, json),
            DatasetCmd::Export { file, to, format } => {
                cli::dataset_export(&cfg, file.file, &to, &format, json)
            }
            DatasetCmd::Snapshot { file } => cli::dataset_snapshot(&cfg, file.file, json),
        },
        Some(Cmd::Train(a)) => {
            cli::train(&cfg, a.method, a.iters, a.learning_rate, a.dry_run, json).await
        }
        Some(Cmd::Eval { cmd }) => match cmd {
            EvalCmd::Run(a) => {
                cli::eval_run(&cfg, &a.set, &a.grader, a.judge, a.concurrency, json).await
            }
            EvalCmd::Compare(a) => {
                cli::eval_compare(
                    &cfg,
                    &a.set,
                    &a.providers,
                    &a.grader,
                    a.judge,
                    a.concurrency,
                    json,
                )
                .await
            }
        },
        Some(Cmd::Doctor) => cli::doctor(&cfg, json).await,
        // No subcommand launches the desktop GUI (when this build includes it).
        None => launch_gui(cfg, cfg_path),
    };

    // With --json, surface errors as JSON on stderr (consistent machine-readable
    // contract) and exit non-zero; otherwise let anyhow render them as text.
    match result {
        Err(e) if json => {
            let obj = serde_json::json!({ "error": format!("{e:#}") });
            eprintln!(
                "{}",
                serde_json::to_string(&obj).unwrap_or_else(|_| format!("{{\"error\":\"{e}\"}}"))
            );
            std::process::exit(1);
        }
        other => other,
    }
}

/// Restore the default SIGPIPE disposition so `gene … | head` (and other closed
/// pipes) terminate cleanly instead of panicking on a failed stdout write.
#[cfg(unix)]
fn reset_sigpipe() {
    // SAFETY: resetting a signal to its default handler before any threads rely
    // on a custom disposition is sound.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

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
