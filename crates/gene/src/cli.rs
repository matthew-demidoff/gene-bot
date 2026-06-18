//! Headless CLI commands. Compiled into every build (no GUI dependencies), so a
//! researcher can drive gene from scripts, CI, or ssh. Each command supports a
//! `--json` mode for machine-readable output.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use gene_core::chat::{build_wire, Mode};
use gene_core::config::Config;
use gene_core::dataset::{self, format::Format};
use gene_core::llm::StreamEvent;
use gene_core::model::{Message, Role};
use gene_core::provider::http_client;
use gene_core::runs::{DatasetRef, RunStore};

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
    // NB: `consume` accumulates for --json and streams live otherwise; a
    // stream/provider error is surfaced as a non-zero exit after output (below).

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
    }
    // Non-zero exit on a stream/provider error in both modes (the JSON body
    // above still carries the error for machine consumers).
    if let Some(e) = error {
        bail!("{e}");
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
    store.reconcile();
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
    store.reconcile();
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

/// A clone of the config with API keys masked — `config show` writes to stdout
/// (CI logs, scrollback, pasted issues), which must never carry a real token.
fn redacted(cfg: &Config) -> Config {
    let mut c = cfg.clone();
    if !c.api_key.is_empty() {
        c.api_key = "<redacted>".into();
    }
    for p in c.providers.values_mut() {
        if !p.api_key.is_empty() {
            p.api_key = "<redacted>".into();
        }
    }
    c
}

pub fn config_show(cfg: &Config, json: bool) -> Result<()> {
    let cfg = redacted(cfg);
    if json {
        println!("{}", serde_json::to_string(&cfg)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&cfg)?);
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
    if !report.all_ok() {
        bail!("some prerequisite checks failed");
    }
    Ok(())
}

/// Resolve the dataset file: an explicit `--file`, else the configured path.
fn dataset_file(cfg: &Config, file: Option<PathBuf>) -> Result<PathBuf> {
    match file {
        Some(f) => Ok(f),
        None => cfg.dataset_path(),
    }
}

pub fn dataset_stats(cfg: &Config, file: Option<PathBuf>, json: bool) -> Result<()> {
    let path = dataset_file(cfg, file)?;
    let examples = dataset::load(&path)?;
    let s = dataset::stats(&examples);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "total": s.total,
                "edited": s.edited,
                "conversations": s.conversations,
                "by_source": s.by_source,
            }))?
        );
    } else {
        println!("path:          {}", path.display());
        println!("total:         {}", s.total);
        println!("edited:        {}", s.edited);
        println!("conversations: {}", s.conversations);
        let sources = s
            .by_source
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("  ");
        println!("sources:       {sources}");
    }
    Ok(())
}

pub fn dataset_dedup(cfg: &Config, file: Option<PathBuf>, dry_run: bool, json: bool) -> Result<()> {
    let path = dataset_file(cfg, file)?;
    let mut examples = dataset::load(&path)?;
    let removed = dataset::dedup(&mut examples);
    if !dry_run && removed > 0 {
        dataset::save(&path, &examples)?;
    }
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "removed": removed,
                "remaining": examples.len(),
                "written": !dry_run && removed > 0,
            }))?
        );
    } else if dry_run {
        println!(
            "{removed} duplicate(s) would be removed, {} remaining (dry run)",
            examples.len()
        );
    } else {
        println!(
            "removed {removed} duplicate(s), {} remaining",
            examples.len()
        );
    }
    Ok(())
}

pub fn dataset_import(
    cfg: &Config,
    file: Option<PathBuf>,
    from: &Path,
    format: &str,
    replace: bool,
    json: bool,
) -> Result<()> {
    let path = dataset_file(cfg, file)?;
    let fmt = Format::parse(format)?;
    let text =
        std::fs::read_to_string(from).with_context(|| format!("reading {}", from.display()))?;
    let mut incoming = dataset::format::import(&text, fmt)?;
    let added = incoming.len();
    let mut examples = if replace {
        Vec::new()
    } else {
        dataset::load(&path).unwrap_or_default()
    };
    examples.append(&mut incoming);
    dataset::save(&path, &examples)?;
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "imported": added,
                "total": examples.len(),
                "replaced": replace,
                "path": path.display().to_string(),
            }))?
        );
    } else {
        let verb = if replace { "replaced with" } else { "appended" };
        println!(
            "{verb} {added} example(s); dataset now has {}",
            examples.len()
        );
    }
    Ok(())
}

pub fn dataset_export(
    cfg: &Config,
    file: Option<PathBuf>,
    to: &Path,
    format: &str,
    json: bool,
) -> Result<()> {
    let path = dataset_file(cfg, file)?;
    let fmt = Format::parse(format)?;
    let examples = dataset::load(&path)?;
    let text = dataset::format::export(&examples, fmt)?;
    std::fs::write(to, text).with_context(|| format!("writing {}", to.display()))?;
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "exported": examples.len(),
                "to": to.display().to_string(),
                "format": format,
            }))?
        );
    } else {
        println!("exported {} example(s) to {}", examples.len(), to.display());
    }
    Ok(())
}

pub fn dataset_snapshot(cfg: &Config, file: Option<PathBuf>, json: bool) -> Result<()> {
    let path = dataset_file(cfg, file)?;
    // Content-addressed: identical datasets snapshot to the same file.
    let dref = DatasetRef::from_dataset(&path)?;
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("snapshots");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(format!("{}.jsonl", dref.content_hash));
    std::fs::copy(&path, &dest).with_context(|| format!("writing {}", dest.display()))?;
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "snapshot": dest.display().to_string(),
                "content_hash": dref.content_hash,
                "examples": dref.n_examples,
            }))?
        );
    } else {
        println!(
            "snapshot {} ({} examples) -> {}",
            dref.content_hash,
            dref.n_examples,
            dest.display()
        );
    }
    Ok(())
}
