//! gene: a local-first toolkit for chatting with, evaluating, and fine-tuning
//! open LLMs. This binary is the CLI plus — with the default `gui` feature — the
//! egui desktop frontend; all engine logic lives in the `gene-core` library.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};

use gene_core::config::Config;

#[cfg(feature = "gui")]
mod app;

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
    /// Override the model tag.
    #[arg(long)]
    model: Option<String>,
    /// Override the chat endpoint base URL.
    #[arg(long)]
    base_url: Option<String>,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Report whether the chat + fine-tuning prerequisites are installed.
    Doctor,
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

    match cli.command {
        Some(Cmd::Doctor) => doctor(&cfg).await,
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
         run a subcommand such as `gene doctor`"
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

async fn doctor(cfg: &Config) -> Result<()> {
    println!("gene doctor — prerequisite check\n");

    line(arch_is_arm64(), "Apple Silicon (arm64)", &arch_string());

    let ollama = cmd_version("ollama", &["--version"]);
    line(
        ollama.is_some(),
        "ollama (chat host)",
        ollama
            .as_deref()
            .unwrap_or("not found — https://ollama.com"),
    );

    let reachable = ollama_reachable(&cfg.base_url).await;
    line(
        reachable,
        "ollama server reachable",
        if reachable {
            &cfg.base_url
        } else {
            "not reachable (run `ollama serve`)"
        },
    );

    let py = cmd_version("python3", &["--version"]);
    line(
        py.is_some(),
        "python3 (for MLX)",
        py.as_deref().unwrap_or("not found"),
    );

    let mlx = cmd_version(
        "python3",
        &[
            "-c",
            "import mlx_lm; print(getattr(mlx_lm,'__version__','?'))",
        ],
    );
    line(
        mlx.is_some(),
        "mlx-lm (LoRA trainer)",
        mlx.as_deref().unwrap_or("not found — `pip install mlx-lm`"),
    );

    println!("\nchat model: {}", cfg.model);
    println!("mlx base:   {}", cfg.finetune.mlx_base);
    println!("dataset:    {}", cfg.dataset_path()?.display());
    Ok(())
}

fn line(ok: bool, name: &str, detail: &str) {
    let mark = if ok { "✓" } else { "✗" };
    println!("[{mark}] {name:<26} {detail}");
}

fn cmd_version(bin: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let mut s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        s = String::from_utf8_lossy(&out.stderr).trim().to_string();
    }
    s.lines().next().map(|l| l.to_string())
}

fn arch_string() -> String {
    cmd_version("uname", &["-m"]).unwrap_or_else(|| "unknown".into())
}

fn arch_is_arm64() -> bool {
    arch_string().contains("arm64")
}

async fn ollama_reachable(base_url: &str) -> bool {
    let tags_url = base_url
        .split("/v1/")
        .next()
        .map(|root| format!("{root}/api/tags"))
        .unwrap_or_default();
    if tags_url.is_empty() {
        return false;
    }
    let client = reqwest::Client::new();
    client
        .get(&tags_url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}
