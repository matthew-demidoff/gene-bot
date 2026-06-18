//! Headless CLI commands. Compiled into every build (no GUI dependencies), so a
//! researcher can drive gene from scripts, CI, or ssh. Each command supports a
//! `--json` mode for machine-readable output.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

use gene_core::chat::{build_wire, Mode};
use gene_core::config::Config;
use gene_core::llm::StreamEvent;
use gene_core::model::{Message, Role};
use gene_core::provider::http_client;
use gene_core::runs::RunStore;

fn parse_mode(s: &str) -> Result<Mode> {
    match s {
        "assistant" => Ok(Mode::Assistant),
        "tech" => Ok(Mode::Tech),
        "convo" => Ok(Mode::Convo),
        other => bail!("unknown mode '{other}' (expected assistant | tech | convo)"),
    }
}

fn read_stdin() -> Result<String> {
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .context("reading prompt from stdin")?;
    Ok(s)
}

fn ensure_provider_resolvable(cfg: &Config) -> Result<()> {
    if cfg.chat_role_is_dangling() {
        bail!(
            "the chat provider profile was not found in [providers] — \
             check --provider or [roles].chat in the config"
        );
    }
    Ok(())
}

pub async fn chat(
    cfg: &Config,
    mode: &str,
    message: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
    json: bool,
) -> Result<()> {
    ensure_provider_resolvable(cfg)?;
    let mode = parse_mode(mode)?;
    let prompt = match message {
        Some(m) => m,
        None => read_stdin()?,
    };
    if prompt.trim().is_empty() {
        bail!("empty prompt — pass --message or pipe text on stdin");
    }

    let system = mode.system_prompt(cfg, &cfg.system_prompt);
    let messages = vec![Message::new(Role::User, prompt)];
    let mut request = cfg.chat_request(build_wire(&system, &messages));
    if let Some(t) = temperature {
        request.sampling.temperature = Some(t);
    }
    if let Some(m) = max_tokens {
        request.sampling.max_tokens = Some(m);
    }
    if let Some(s) = seed {
        request.sampling.seed = Some(s);
    }

    let provider = cfg.chat_provider(http_client());
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let producer = provider.chat_stream(request, mode.detect_commands(), tx);

    let consume = async {
        let mut answer = String::new();
        let mut thinking = String::new();
        let mut commands: Vec<String> = Vec::new();
        let mut error: Option<String> = None;
        let mut stdout = std::io::stdout();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::AnswerDelta(s) => {
                    if json {
                        answer.push_str(&s);
                    } else {
                        print!("{s}");
                        let _ = stdout.flush();
                    }
                }
                StreamEvent::ThinkDelta(s) => {
                    if json {
                        thinking.push_str(&s);
                    }
                }
                StreamEvent::ToolCall(tc) => {
                    if json {
                        commands.push(tc.command);
                    } else {
                        print!("\n[suggested command]\n{}\n", tc.command);
                        let _ = stdout.flush();
                    }
                }
                StreamEvent::Error(e) => error = Some(e),
                StreamEvent::ThinkStart | StreamEvent::ThinkEnd | StreamEvent::Done => {}
            }
        }
        (answer, thinking, commands, error)
    };

    let (_, (answer, thinking, commands, error)) = tokio::join!(producer, consume);

    if json {
        let mut obj = serde_json::json!({ "model": cfg.chat_model(), "answer": answer });
        if !thinking.is_empty() {
            obj["thinking"] = serde_json::Value::String(thinking);
        }
        if !commands.is_empty() {
            obj["commands"] = serde_json::json!(commands);
        }
        if let Some(e) = &error {
            obj["error"] = serde_json::Value::String(e.clone());
        }
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!();
        if let Some(e) = error {
            bail!("{e}");
        }
    }
    Ok(())
}

pub async fn models(cfg: &Config, json: bool) -> Result<()> {
    ensure_provider_resolvable(cfg)?;
    let provider = cfg.chat_provider(http_client());
    let models = provider.list_models().await;
    if json {
        println!("{}", serde_json::to_string(&models)?);
    } else if models.is_empty() {
        println!("no models found at {}", provider.endpoint());
    } else {
        for m in models {
            println!("{m}");
        }
    }
    Ok(())
}

fn summary_line(summary: &std::collections::BTreeMap<String, f64>) -> String {
    summary
        .iter()
        .map(|(k, v)| format!("{k}={v:.4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn run_list(cfg: &Config, json: bool) -> Result<()> {
    let store = RunStore::new(cfg.runs_dir()?);
    let runs = store.list();
    if json {
        println!("{}", serde_json::to_string(&runs)?);
        return Ok(());
    }
    if runs.is_empty() {
        println!("no runs yet");
        return Ok(());
    }
    for r in &runs {
        println!(
            "{}  {:?}  {:?}  {}  {}",
            r.id,
            r.kind,
            r.status,
            r.base_model,
            summary_line(&r.summary)
        );
    }
    Ok(())
}

pub fn run_show(cfg: &Config, id: &str, json: bool) -> Result<()> {
    let store = RunStore::new(cfg.runs_dir()?);
    let run = store.load(id)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&run)?);
        return Ok(());
    }
    println!("id:        {}", run.id);
    println!("kind:      {:?}", run.kind);
    println!("status:    {:?}", run.status);
    println!("base:      {}", run.base_model);
    println!("created:   {}", run.created_at);
    if let Some(finished) = run.finished_at {
        println!("finished:  {finished}");
    }
    if let Some(d) = &run.dataset {
        println!(
            "dataset:   {} examples ({} edited), hash {}",
            d.n_examples, d.n_edited, d.content_hash
        );
    }
    if !run.summary.is_empty() {
        println!("summary:   {}", summary_line(&run.summary));
    }
    println!("metrics:   {} points", store.metrics(id).len());
    if let Some(e) = &run.error {
        println!("error:     {e}");
    }
    Ok(())
}

pub fn config_path(cfg_path: &Path) -> Result<()> {
    println!("{}", cfg_path.display());
    Ok(())
}

pub fn config_show(cfg: &Config, cfg_path: &Path, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(cfg)?);
        return Ok(());
    }
    // Prefer the actual on-disk file; fall back to the resolved config as JSON.
    match std::fs::read_to_string(cfg_path) {
        Ok(text) => print!("{text}"),
        Err(_) => println!("{}", serde_json::to_string_pretty(cfg)?),
    }
    Ok(())
}

pub async fn doctor(cfg: &Config, json: bool) -> Result<()> {
    let report = gene_core::doctor::report(cfg).await;
    if json {
        let checks: Vec<_> = report
            .checks
            .iter()
            .map(|c| serde_json::json!({ "name": c.name, "ok": c.ok, "detail": c.detail }))
            .collect();
        let obj = serde_json::json!({
            "ok": report.all_ok(),
            "checks": checks,
            "chat_model": report.chat_model,
            "chat_endpoint": report.chat_endpoint,
            "mlx_base": report.mlx_base,
            "dataset_path": report.dataset_path,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("gene doctor — prerequisite check\n");
        for c in &report.checks {
            let mark = if c.ok { "✓" } else { "✗" };
            println!("[{mark}] {:<26} {}", c.name, c.detail);
        }
        println!("\nchat model: {}", report.chat_model);
        println!("endpoint:   {}", report.chat_endpoint);
        println!("mlx base:   {}", report.mlx_base);
        println!("dataset:    {}", report.dataset_path);
    }
    Ok(())
}
