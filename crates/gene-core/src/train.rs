//! `/train` pipeline orchestration: export the dataset to MLX chat-format JSONL,
//! run `mlx_lm.lora` then `mlx_lm.fuse` as subprocesses, then deploy the result
//! (serve via `mlx_lm.server`, or convert to GGUF and register with Ollama).
//!
//! Every command is a configurable template, so the trainer/backend is pluggable.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

use crate::config::Config;
use crate::dataset::{self, SplitSpec, SplitStrategy, TrainingExample};
use crate::runs::{DatasetRef, Metric, RunKind, RunStatus, RunStore};

/// Progress and completion messages from the fine-tune pipeline.
pub enum TrainMsg {
    Log(String),
    Metric(Metric),
    Done {
        ok: bool,
        message: String,
        new_base_url: Option<String>,
        new_model: Option<String>,
    },
}

/// Spawn the whole pipeline. Progress and completion arrive as `TrainMsg`; the
/// run is recorded in `runs_dir` (best-effort — tracking failures never abort
/// the training itself).
pub fn start_training(
    cfg: Config,
    work_dir: PathBuf,
    dataset_path: PathBuf,
    runs_dir: PathBuf,
    tx: UnboundedSender<TrainMsg>,
) {
    tokio::spawn(async move {
        let store = RunStore::new(runs_dir);
        let hyperparams = serde_json::json!({
            "method": "lora",
            "iters": cfg.finetune.iters,
            "batch": cfg.finetune.batch,
            "layers": cfg.finetune.layers,
            "learning_rate": cfg.finetune.learning_rate,
            "valid_fraction": cfg.finetune.valid_fraction,
            "deploy_mode": cfg.finetune.deploy_mode,
        });
        let dataset = DatasetRef::from_dataset(&dataset_path).ok();
        let run = store.create(
            RunKind::Train,
            cfg.finetune.mlx_base.clone(),
            hyperparams,
            dataset,
        );
        let run_id = match &run {
            Ok(r) => {
                log(&tx, format!("run {}", r.id));
                Some(r.id.clone())
            }
            Err(e) => {
                log(&tx, format!("run tracking unavailable: {e}"));
                None
            }
        };

        let result = run_pipeline(
            &cfg,
            &work_dir,
            &dataset_path,
            &store,
            run_id.as_deref(),
            &tx,
        )
        .await;

        if let Ok(mut r) = run {
            r.summary = summarize(&store.metrics(&r.id));
            r.finished_at = Some(Utc::now());
            r.status = match &result {
                Ok(_) => RunStatus::Succeeded,
                Err(e) => {
                    r.error = Some(e.to_string());
                    RunStatus::Failed
                }
            };
            let _ = store.save(&r);
        }

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
    store: &RunStore,
    run_id: Option<&str>,
    tx: &UnboundedSender<TrainMsg>,
) -> Result<(Option<String>, Option<String>)> {
    let examples = dataset::load(dataset_path)?;
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

    run_cmd(
        "train",
        &fill(&cfg.finetune.train_command, &vars),
        store,
        run_id,
        tx,
    )
    .await?;
    run_cmd(
        "fuse",
        &fill(&cfg.finetune.fuse_command, &vars),
        store,
        run_id,
        tx,
    )
    .await?;

    deploy(cfg, &fused_dir, work_dir, &vars, store, run_id, tx).await
}

/// Write `train.jsonl` / `valid.jsonl` in MLX chat format (messages only).
///
/// Splits by conversation so no conversation's examples straddle the train/valid
/// boundary (the old tail-slice could leak). Falls back to a plain tail hold-out
/// for degenerate datasets (e.g. a single conversation) so neither side is empty.
fn write_split(examples: &[TrainingExample], data_dir: &Path, valid_fraction: f64) -> Result<()> {
    let spec = SplitSpec {
        strategy: SplitStrategy::ByConversation { seed: 0 },
        valid: valid_fraction,
        test: 0.0,
    };
    let split = dataset::make_split(examples, &spec);
    let (mut train_idx, mut valid_idx) = (split.train, split.valid);

    let n = examples.len();
    if n > 1 && (train_idx.is_empty() || valid_idx.is_empty()) {
        let valid_n = (((n as f64) * valid_fraction).round() as usize).clamp(1, n - 1);
        train_idx = (0..n - valid_n).collect();
        valid_idx = (n - valid_n..n).collect();
    }

    let pick = |idxs: &[usize]| -> Vec<TrainingExample> {
        idxs.iter().map(|&i| examples[i].clone()).collect()
    };
    write_jsonl(&data_dir.join("train.jsonl"), &pick(&train_idx))?;
    write_jsonl(&data_dir.join("valid.jsonl"), &pick(&valid_idx))?;
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
    store: &RunStore,
    run_id: Option<&str>,
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
                store,
                run_id,
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
                store,
                run_id,
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
async fn run_cmd(
    label: &str,
    command: &str,
    store: &RunStore,
    run_id: Option<&str>,
    tx: &UnboundedSender<TrainMsg>,
) -> Result<()> {
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
    let store_out = store.clone();
    let run_out = run_id.map(str::to_string);
    let out_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some((step, fields)) = parse_mlx_metric(&line) {
                let metric = Metric {
                    step,
                    at: Utc::now(),
                    fields,
                };
                if let Some(id) = &run_out {
                    let _ = store_out.append_metric(id, &metric);
                }
                let _ = tx_out.send(TrainMsg::Metric(metric));
            }
            let text = format!("{label_out}: {line}");
            if let Some(id) = &run_out {
                let _ = store_out.append_log(id, &text);
            }
            let _ = tx_out.send(TrainMsg::Log(text));
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

/// Parse an `mlx_lm.lora` progress line such as
/// `Iter 10: Train loss 2.413, Learning Rate 1.000e-05, Tokens/sec 213.5` or
/// `Iter 200: Val loss 1.890` into `(iteration, fields)`. Lenient: returns
/// `None` for any line without a recognizable iteration and metric.
fn parse_mlx_metric(line: &str) -> Option<(u64, BTreeMap<String, f64>)> {
    let step = number_after(line, "Iter ")? as u64;
    let mut fields = BTreeMap::new();
    for (key, name) in [
        ("Train loss", "train_loss"),
        ("Val loss", "val_loss"),
        ("Learning Rate", "learning_rate"),
        ("Tokens/sec", "tokens_per_sec"),
    ] {
        if let Some(value) = number_after(line, key) {
            fields.insert(name.to_string(), value);
        }
    }
    if fields.is_empty() {
        None
    } else {
        Some((step, fields))
    }
}

/// The first number after `key` in `line`, tolerating a trailing comma/unit and
/// scientific notation (e.g. `1.000e-05,`).
fn number_after(line: &str, key: &str) -> Option<f64> {
    let token = line.split(key).nth(1)?.split_whitespace().next()?;
    let token = token.trim_end_matches(|c: char| {
        !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
    });
    // Reject NaN/±inf (e.g. an overflowing "1e999") — a non-finite metric would
    // serialize to JSON `null` and fail to deserialize back, silently dropping it.
    token.parse().ok().filter(|v: &f64| v.is_finite())
}

/// Reduce a metric series to the headline numbers stored on the run record.
fn summarize(metrics: &[Metric]) -> BTreeMap<String, f64> {
    let mut summary = BTreeMap::new();
    if let Some(train) = metrics
        .iter()
        .rev()
        .find_map(|m| m.fields.get("train_loss"))
    {
        summary.insert("final_train_loss".into(), *train);
    }
    let val: Vec<f64> = metrics
        .iter()
        .filter_map(|m| m.fields.get("val_loss").copied())
        .collect();
    if let Some(&last) = val.last() {
        summary.insert("final_val_loss".into(), last);
    }
    if let Some(min) = val.iter().copied().reduce(f64::min) {
        summary.insert("min_val_loss".into(), min);
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: Option<&f64>, expected: f64) -> bool {
        actual.is_some_and(|v| (v - expected).abs() < 1e-9)
    }

    #[test]
    fn parses_train_loss_line() {
        let (step, fields) = parse_mlx_metric(
            "Iter 10: Train loss 2.413, Learning Rate 1.000e-05, Tokens/sec 213.5",
        )
        .unwrap();
        assert_eq!(step, 10);
        assert!(approx(fields.get("train_loss"), 2.413));
        assert!(approx(fields.get("learning_rate"), 1.0e-5));
        assert!(approx(fields.get("tokens_per_sec"), 213.5));
        assert!(!fields.contains_key("val_loss"));
    }

    #[test]
    fn parses_val_loss_line() {
        let (step, fields) = parse_mlx_metric("Iter 200: Val loss 1.890, Val took 3.1s").unwrap();
        assert_eq!(step, 200);
        assert!(approx(fields.get("val_loss"), 1.890));
        assert!(!fields.contains_key("train_loss"));
    }

    #[test]
    fn ignores_lines_without_metrics() {
        assert!(parse_mlx_metric("Loading pretrained model").is_none());
        assert!(parse_mlx_metric("train: starting").is_none());
        assert!(parse_mlx_metric("Iter 5: nothing useful here").is_none());
    }

    #[test]
    fn rejects_overflow_and_tolerates_whitespace() {
        // An overflowing exponent parses to inf, which is filtered out, leaving
        // no usable field.
        assert!(parse_mlx_metric("Iter 1: Train loss 1e999").is_none());
        let (step, fields) = parse_mlx_metric("Iter 7:   Train loss   3.0").unwrap();
        assert_eq!(step, 7);
        assert!(approx(fields.get("train_loss"), 3.0));
    }

    #[test]
    fn summarizes_series() {
        let point = |step, key: &str, value: f64| Metric {
            step,
            at: Utc::now(),
            fields: BTreeMap::from([(key.to_string(), value)]),
        };
        let series = vec![
            point(10, "train_loss", 2.5),
            point(20, "val_loss", 2.0),
            point(30, "train_loss", 1.8),
            point(40, "val_loss", 2.3),
        ];
        let summary = summarize(&series);
        assert!(approx(summary.get("final_train_loss"), 1.8));
        assert!(approx(summary.get("final_val_loss"), 2.3));
        assert!(approx(summary.get("min_val_loss"), 2.0));
    }

    fn sft(conversation: &str, content: &str) -> TrainingExample {
        use crate::dataset::{ChatMsg, Meta};
        TrainingExample {
            messages: vec![ChatMsg {
                role: "user".into(),
                content: content.into(),
            }],
            meta: Meta {
                conversation_id: conversation.into(),
                model: "m".into(),
                created_at: Utc::now(),
                edited: false,
                source: "test".into(),
                original_assistant: None,
            },
        }
    }

    #[test]
    fn write_split_keeps_conversations_whole() {
        let dir =
            std::env::temp_dir().join(format!("gene-split-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        // 4 conversations, 2 examples each, interleaved; content encodes the conv.
        let mut examples = Vec::new();
        for round in 0..2 {
            for conv in ["a", "b", "c", "d"] {
                examples.push(sft(conv, &format!("{conv}-{round}")));
            }
        }
        write_split(&examples, &dir, 0.25).unwrap();

        let convs = |file: &str| -> std::collections::BTreeSet<char> {
            std::fs::read_to_string(dir.join(file))
                .unwrap()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let v: serde_json::Value = serde_json::from_str(l).unwrap();
                    v["messages"][0]["content"]
                        .as_str()
                        .unwrap()
                        .chars()
                        .next()
                        .unwrap()
                })
                .collect()
        };
        let train = convs("train.jsonl");
        let valid = convs("valid.jsonl");
        assert!(!train.is_empty() && !valid.is_empty());
        // No conversation lands in both train and valid.
        assert!(train.is_disjoint(&valid));

        std::fs::remove_dir_all(&dir).ok();
    }
}
