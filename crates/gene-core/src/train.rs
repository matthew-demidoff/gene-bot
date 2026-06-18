//! `/train` pipeline orchestration: export the dataset to MLX chat-format JSONL,
//! run `mlx_lm.lora` then `mlx_lm.fuse` as subprocesses, then deploy the result
//! (serve via `mlx_lm.server`, or convert to GGUF and register with Ollama).
//!
//! Every command is a configurable template, so the trainer/backend is pluggable.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{anyhow, bail, Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

use crate::config::Config;
use crate::model::dataset::TrainingExample;

/// Progress and completion messages from the fine-tune pipeline.
pub enum TrainMsg {
    Log(String),
    Done {
        ok: bool,
        message: String,
        new_base_url: Option<String>,
        new_model: Option<String>,
    },
}

/// Spawn the whole pipeline. Progress and completion arrive as `TrainMsg`.
pub fn start_training(
    cfg: Config,
    work_dir: PathBuf,
    dataset_path: PathBuf,
    tx: UnboundedSender<TrainMsg>,
) {
    tokio::spawn(async move {
        let result = run_pipeline(&cfg, &work_dir, &dataset_path, &tx).await;
        let done = match result {
            Ok((new_base_url, new_model)) => TrainMsg::Done {
                ok: true,
                message: "fine-tune complete".into(),
                new_base_url,
                new_model,
            },
            Err(e) => TrainMsg::Done {
                ok: false,
                message: format!("fine-tune failed: {e}"),
                new_base_url: None,
                new_model: None,
            },
        };
        let _ = tx.send(done);
    });
}

async fn run_pipeline(
    cfg: &Config,
    work_dir: &Path,
    dataset_path: &Path,
    tx: &UnboundedSender<TrainMsg>,
) -> Result<(Option<String>, Option<String>)> {
    let examples = read_examples(dataset_path)?;
    if examples.len() < cfg.finetune.min_examples {
        bail!(
            "need at least {} examples, have {} — edit more replies first",
            cfg.finetune.min_examples,
            examples.len()
        );
    }
    log(tx, format!("preparing {} examples", examples.len()));

    let data_dir = work_dir.join("data");
    let adapters_dir = work_dir.join("adapters");
    let fused_dir = work_dir.join("fused");
    std::fs::create_dir_all(&data_dir).context("creating data dir")?;

    write_split(&examples, &data_dir, cfg.finetune.valid_fraction)?;

    let vars = template_vars(cfg, &data_dir, &adapters_dir, &fused_dir, work_dir);

    run_cmd("train", &fill(&cfg.finetune.train_command, &vars), tx).await?;
    run_cmd("fuse", &fill(&cfg.finetune.fuse_command, &vars), tx).await?;

    deploy(cfg, &fused_dir, work_dir, &vars, tx).await
}

fn read_examples(path: &Path) -> Result<Vec<TrainingExample>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading dataset {}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(ex) = serde_json::from_str::<TrainingExample>(line) {
            out.push(ex);
        }
    }
    Ok(out)
}

/// Write `train.jsonl` / `valid.jsonl` in MLX chat format (messages only).
fn write_split(examples: &[TrainingExample], data_dir: &Path, valid_fraction: f64) -> Result<()> {
    let n = examples.len();
    let mut valid_n = ((n as f64) * valid_fraction).round() as usize;
    valid_n = valid_n.clamp(1, n.saturating_sub(1).max(1));
    let split = n - valid_n;

    write_jsonl(&data_dir.join("train.jsonl"), &examples[..split])?;
    write_jsonl(&data_dir.join("valid.jsonl"), &examples[split..])?;
    Ok(())
}

fn write_jsonl(path: &Path, examples: &[TrainingExample]) -> Result<()> {
    let mut buf = String::new();
    for ex in examples {
        let line = serde_json::to_string(&serde_json::json!({ "messages": ex.messages }))?;
        buf.push_str(&line);
        buf.push('\n');
    }
    std::fs::write(path, buf).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn template_vars(
    cfg: &Config,
    data_dir: &Path,
    adapters_dir: &Path,
    fused_dir: &Path,
    _work_dir: &Path,
) -> Vec<(&'static str, String)> {
    vec![
        ("base", cfg.finetune.mlx_base.clone()),
        ("data", data_dir.display().to_string()),
        ("adapters", adapters_dir.display().to_string()),
        ("fused", fused_dir.display().to_string()),
        ("iters", cfg.finetune.iters.to_string()),
        ("batch", cfg.finetune.batch.to_string()),
        ("layers", cfg.finetune.layers.to_string()),
        ("lr", cfg.finetune.learning_rate.clone()),
        ("port", cfg.finetune.mlx_server_port.to_string()),
        ("llama_cpp", cfg.finetune.llama_cpp_dir.clone()),
        ("tag", cfg.finetune.ollama_tag.clone()),
    ]
}

fn fill(template: &str, vars: &[(&str, String)]) -> String {
    let mut s = template.to_string();
    for (k, v) in vars {
        s = s.replace(&format!("{{{k}}}"), v);
    }
    s
}

async fn deploy(
    cfg: &Config,
    fused_dir: &Path,
    work_dir: &Path,
    vars: &[(&str, String)],
    tx: &UnboundedSender<TrainMsg>,
) -> Result<(Option<String>, Option<String>)> {
    match cfg.finetune.deploy_mode.as_str() {
        "mlx_server" => {
            log(tx, "starting mlx_lm.server with the fused model".into());
            let cmd = fill(&cfg.finetune.mlx_server_command, vars);
            let parts = shell_words::split(&cmd).map_err(|e| anyhow!("bad server command: {e}"))?;
            if parts.is_empty() {
                bail!("empty mlx server command");
            }
            // Detached: default kill_on_drop is false, so the server outlives this task.
            Command::new(&parts[0])
                .args(&parts[1..])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| anyhow!("failed to start mlx server: {e}"))?;

            let base_url = format!(
                "http://localhost:{}/v1/chat/completions",
                cfg.finetune.mlx_server_port
            );
            Ok((Some(base_url), Some(fused_dir.display().to_string())))
        }
        "ollama_gguf" => {
            if cfg.finetune.llama_cpp_dir.is_empty() {
                bail!("deploy_mode=ollama_gguf needs finetune.llama_cpp_dir set in config");
            }
            run_cmd(
                "convert-gguf",
                &fill(&cfg.finetune.gguf_convert_command, vars),
                tx,
            )
            .await?;

            let gguf = fused_dir.join("model-f16.gguf");
            let modelfile = work_dir.join("Modelfile");
            let content = format!(
                "FROM {}\nSYSTEM \"\"\"{}\"\"\"\n",
                gguf.display(),
                cfg.system_prompt
            );
            std::fs::write(&modelfile, content).context("writing Modelfile")?;

            let mut create_vars = vars.to_vec();
            create_vars.push(("modelfile", modelfile.display().to_string()));
            run_cmd(
                "ollama-create",
                &fill(&cfg.finetune.ollama_create_command, &create_vars),
                tx,
            )
            .await?;

            Ok((None, Some(cfg.finetune.ollama_tag.clone())))
        }
        other => bail!("unknown deploy_mode: {other}"),
    }
}

/// Lines of stderr retained for the error message when a subprocess fails.
const ERR_TAIL_LINES: usize = 40;

/// Run a subprocess, streaming stdout as progress logs; on failure include the
/// tail of stderr in the error.
async fn run_cmd(label: &str, command: &str, tx: &UnboundedSender<TrainMsg>) -> Result<()> {
    log(tx, format!("{label}: starting"));
    let parts = shell_words::split(command).map_err(|e| anyhow!("bad {label} command: {e}"))?;
    if parts.is_empty() {
        bail!("empty {label} command");
    }

    let mut child = Command::new(&parts[0])
        .args(&parts[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| anyhow!("failed to start {} ({label}): {e}", parts[0]))?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let tx_out = tx.clone();
    let label_out = label.to_string();
    let out_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = tx_out.send(TrainMsg::Log(format!("{label_out}: {line}")));
        }
    });

    let mut err_lines: Vec<String> = Vec::new();
    let mut err_reader = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = err_reader.next_line().await {
        err_lines.push(line);
        if err_lines.len() > ERR_TAIL_LINES {
            err_lines.remove(0);
        }
    }

    let status = child.wait().await.context("waiting on subprocess")?;
    let _ = out_task.await;

    if !status.success() {
        bail!("{label} failed ({status}):\n{}", err_lines.join("\n"));
    }
    Ok(())
}

fn log(tx: &UnboundedSender<TrainMsg>, msg: String) {
    let _ = tx.send(TrainMsg::Log(msg));
}
