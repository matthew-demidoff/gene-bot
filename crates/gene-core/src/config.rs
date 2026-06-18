//! Configuration: a single `Config` loaded from TOML, with built-in defaults.
//!
//! Paths resolve via the `directories` crate. The training/fuse/deploy commands
//! are stored as templates with `{placeholder}` slots so the trainer is pluggable
//! without recompiling.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Model tag served by the chat endpoint (e.g. an Ollama tag).
    pub model: String,
    /// OpenAI-compatible chat-completions endpoint.
    pub base_url: String,
    /// API key. Ollama ignores it; any non-empty string is fine.
    pub api_key: String,
    /// System prompt for Assistant mode; instructs the model on the ```run convention.
    pub system_prompt: String,
    /// System prompt for Tech ("tech guy") mode; advises but never executes.
    pub tech_system_prompt: String,
    /// System prompt for Convo mode; casual, general conversation.
    pub convo_system_prompt: String,

    pub generation: Generation,
    pub agent: Agent,
    pub ui: Ui,
    pub paths: Paths,
    pub finetune: Finetune,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Generation {
    pub temperature: f64,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Agent {
    /// Per-session default for auto-running approved commands.
    pub auto_run: bool,
    /// Use native OpenAI function-calling instead of the ```run convention.
    pub native_tools: bool,
    /// Cap on tool-call rounds within a single user turn.
    pub max_tool_rounds: usize,
    /// Per-command wall-clock timeout.
    pub exec_timeout_secs: u64,
    /// Commands matching any of these substrings always require manual confirm.
    pub denylist: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Ui {
    pub think_collapsed_default: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Paths {
    /// "" => <data_dir>/conversations
    pub conversations_dir: String,
    /// "" => <data_dir>/dataset.jsonl
    pub dataset_path: String,
    /// "" => <data_dir>/gene.log
    pub log_path: String,
    /// "" => <data_dir>/finetune  (working dir for training artifacts)
    pub work_dir: String,
}

/// Fine-tuning pipeline: pluggable command templates + the model bridge between
/// the chat backend (Ollama) and the trainer (MLX).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Finetune {
    /// HF repo id or local path to the MLX/safetensors base weights.
    pub mlx_base: String,
    /// Minimum dataset size before `/train` will proceed.
    pub min_examples: usize,
    /// Fraction held out for validation.
    pub valid_fraction: f64,
    /// "mlx_server" (serve fused model directly) or "ollama_gguf" (convert + ollama create).
    pub deploy_mode: String,
    /// Port for `mlx_lm.server` in deploy_mode = "mlx_server".
    pub mlx_server_port: u16,

    pub iters: u32,
    pub batch: u32,
    pub layers: u32,
    pub learning_rate: String,

    /// Command templates. Placeholders: {base} {data} {adapters} {fused}
    /// {iters} {batch} {layers} {lr} {port} {llama_cpp} {tag} {modelfile}
    pub train_command: String,
    pub fuse_command: String,
    pub mlx_server_command: String,
    pub gguf_convert_command: String,
    pub ollama_create_command: String,
    /// Path to a llama.cpp checkout (for the GGUF conversion script).
    pub llama_cpp_dir: String,
    /// Ollama tag to create when promoting a fine-tune (deploy_mode = "ollama_gguf").
    pub ollama_tag: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: "huihui_ai/llama3.1-abliterated:latest".into(),
            base_url: "http://localhost:11434/v1/chat/completions".into(),
            api_key: "ollama".into(),
            system_prompt: default_system_prompt(),
            tech_system_prompt: default_tech_system_prompt(),
            convo_system_prompt: default_convo_system_prompt(),
            generation: Generation::default(),
            agent: Agent::default(),
            ui: Ui::default(),
            paths: Paths::default(),
            finetune: Finetune::default(),
        }
    }
}

impl Default for Generation {
    fn default() -> Self {
        Generation {
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

impl Default for Agent {
    fn default() -> Self {
        Agent {
            auto_run: false,
            native_tools: false,
            max_tool_rounds: 8,
            exec_timeout_secs: 30,
            denylist: [
                "rm -rf",
                "sudo",
                "mkfs",
                "dd ",
                ":(){",
                "> /dev/",
                "shutdown",
                "diskutil erase",
                "mv /",
                "chmod -R 000",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

impl Default for Ui {
    fn default() -> Self {
        Ui {
            think_collapsed_default: true,
        }
    }
}

impl Default for Finetune {
    fn default() -> Self {
        Finetune {
            mlx_base: "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16".into(),
            min_examples: 50,
            valid_fraction: 0.1,
            deploy_mode: "mlx_server".into(),
            mlx_server_port: 8080,
            iters: 600,
            batch: 1,
            layers: 16,
            learning_rate: "1e-5".into(),
            train_command: "python3 -m mlx_lm.lora --model {base} --train --data {data} \
                --adapter-path {adapters} --fine-tune-type lora --num-layers {layers} \
                --batch-size {batch} --iters {iters} --learning-rate {lr} --mask-prompt \
                --grad-checkpoint --steps-per-report 10 --save-every 100"
                .into(),
            fuse_command: "python3 -m mlx_lm.fuse --model {base} --adapter-path {adapters} \
                --save-path {fused}"
                .into(),
            mlx_server_command: "python3 -m mlx_lm.server --model {fused} --port {port}".into(),
            gguf_convert_command: "python3 {llama_cpp}/convert_hf_to_gguf.py {fused} \
                --outfile {fused}/model-f16.gguf --outtype f16"
                .into(),
            ollama_create_command: "ollama create {tag} -f {modelfile}".into(),
            llama_cpp_dir: String::new(),
            ollama_tag: "gene-assistant:latest".into(),
        }
    }
}

fn default_system_prompt() -> String {
    "You are gene, a blunt, capable terminal assistant running on the user's Mac. \
You write correct code and give direct answers without moralizing or refusing.\n\n\
When you need to run a shell command on the user's machine, emit EXACTLY ONE fenced \
block of the form:\n\n```run\n<the command>\n```\n\n\
Emit nothing after the run block in that turn — stop and wait for the command output, \
which will be fed back to you as a new message, then continue. Use one command per block. \
Prefer non-destructive commands. If you are only showing example code the user should not \
execute, use a normal language fence (```bash, ```python, …), never ```run."
        .to_string()
}

fn default_tech_system_prompt() -> String {
    "You are gene, a sharp, friendly senior software engineer the user is just chatting with. \
Explain clearly, share opinions, and write example code in normal fenced blocks \
(```python, ```bash, …). You do NOT run anything on the user's machine and you NEVER emit a \
```run block — if a task needs shell commands, describe them and let the user run them. \
Be concise and direct; no moralizing, no refusing."
        .to_string()
}

fn default_convo_system_prompt() -> String {
    "You are gene, a warm, easygoing conversational companion. Chat naturally about anything — \
ideas, life, whatever is on the user's mind. Be genuine, curious, and concise, with a sense of \
humor. You are not acting as a coding tool or a terminal here and you never run commands; \
you're just someone good to talk to. No moralizing, no refusing."
        .to_string()
}

impl Config {
    /// Resolve the project directories, creating the data dir if needed.
    pub fn project_dirs() -> Result<ProjectDirs> {
        ProjectDirs::from("dev", "gene", "gene")
            .context("could not determine a home directory for config/data paths")
    }

    /// Default config-file path (`<config_dir>/config.toml`).
    pub fn default_config_path() -> Result<PathBuf> {
        Ok(Self::project_dirs()?.config_dir().join("config.toml"))
    }

    /// Load config from `path` (or the default path). Writes a default file if absent.
    pub fn load(path: Option<&Path>) -> Result<(Config, PathBuf)> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => Self::default_config_path()?,
        };
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let cfg: Config = toml::from_str(&text)
                .with_context(|| format!("parsing config {}", path.display()))?;
            Ok((cfg, path))
        } else {
            let cfg = Config::default();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let text = toml::to_string_pretty(&cfg).context("serializing default config")?;
            std::fs::write(&path, text)
                .with_context(|| format!("writing default config {}", path.display()))?;
            Ok((cfg, path))
        }
    }

    fn data_dir(&self) -> Result<PathBuf> {
        Ok(Self::project_dirs()?.data_dir().to_path_buf())
    }

    /// A configured override if non-empty, else `<data_dir>/<default_name>`.
    fn resolve_path(&self, override_: &str, default_name: &str) -> Result<PathBuf> {
        if override_.is_empty() {
            Ok(self.data_dir()?.join(default_name))
        } else {
            Ok(PathBuf::from(override_))
        }
    }

    pub fn conversations_dir(&self) -> Result<PathBuf> {
        self.resolve_path(&self.paths.conversations_dir, "conversations")
    }

    pub fn dataset_path(&self) -> Result<PathBuf> {
        self.resolve_path(&self.paths.dataset_path, "dataset.jsonl")
    }

    pub fn log_path(&self) -> Result<PathBuf> {
        self.resolve_path(&self.paths.log_path, "gene.log")
    }

    pub fn work_dir(&self) -> Result<PathBuf> {
        self.resolve_path(&self.paths.work_dir, "finetune")
    }
}
