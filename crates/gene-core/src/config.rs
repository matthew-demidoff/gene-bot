//! Configuration: a single `Config` loaded from TOML, with built-in defaults.
//!
//! Paths resolve via the `directories` crate. The training/fuse/deploy commands
//! are stored as templates with `{placeholder}` slots so the trainer is pluggable
//! without recompiling.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::llm::types::{ChatRequest, Sampling};
use crate::llm::WireMessage;
use crate::provider::{Provider, ProviderKind};

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

    /// Named inference endpoints. Empty => use the top-level model/base_url/api_key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, ProviderProfile>,
    /// Which provider profile each activity (chat/eval/judge) uses.
    #[serde(default, skip_serializing_if = "Roles::is_empty")]
    pub roles: Roles,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Generation {
    pub temperature: f64,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
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
    /// "" => <data_dir>/runs  (experiment-tracking run store)
    pub runs_dir: String,
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
    /// Fine-tune method: "lora" | "dora" | "full". Sets {fine_tune_type}; "full"
    /// trains all weights and skips the fuse step.
    pub method: String,
    /// Extra arguments appended to the train command (the {extra_args} slot).
    pub extra_args: String,

    /// Command templates. Placeholders: {base} {data} {adapters} {fused}
    /// {iters} {batch} {layers} {lr} {fine_tune_type} {extra_args} {port}
    /// {llama_cpp} {tag} {modelfile}
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

/// A named inference endpoint: which backend, where, and the default model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderProfile {
    pub kind: ProviderKind,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl Default for ProviderProfile {
    fn default() -> Self {
        ProviderProfile {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434/v1/chat/completions".into(),
            api_key: "ollama".into(),
            model: String::new(),
        }
    }
}

/// Maps each activity to a named provider profile. Unset => fall back to chat,
/// then to the first configured provider, then to the top-level fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Roles {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<String>,
}

impl Roles {
    fn is_empty(&self) -> bool {
        self.chat.is_none() && self.eval.is_none() && self.judge.is_none()
    }
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
            providers: BTreeMap::new(),
            roles: Roles::default(),
        }
    }
}

impl Default for Generation {
    fn default() -> Self {
        Generation {
            temperature: 0.7,
            max_tokens: 4096,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl Generation {
    /// The wire sampling parameters for this configuration.
    pub fn to_sampling(&self) -> Sampling {
        Sampling {
            temperature: Some(self.temperature),
            max_tokens: Some(self.max_tokens),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            stop: self.stop.clone(),
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
            method: "lora".into(),
            extra_args: String::new(),
            train_command: "python3 -m mlx_lm.lora --model {base} --train --data {data} \
                --adapter-path {adapters} --fine-tune-type {fine_tune_type} --num-layers {layers} \
                --batch-size {batch} --iters {iters} --learning-rate {lr} --mask-prompt \
                --grad-checkpoint --steps-per-report 10 --save-every 100 {extra_args}"
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

impl Finetune {
    /// Whether a fuse step is needed: LoRA/DoRA produce adapters to merge into
    /// the base; a full fine-tune already produces complete weights.
    pub fn needs_fuse(&self) -> bool {
        self.method != "full"
    }

    /// Error if the method can't take effect: a pre-0.2 config whose
    /// `train_command` hardcodes `--fine-tune-type` would skip fuse (for "full")
    /// yet still train LoRA. Shared by every training entry point so both the
    /// CLI and the GUI are protected.
    pub fn check_method(&self) -> Result<()> {
        if self.method != "lora" && !self.train_command.contains("{fine_tune_type}") {
            bail!(
                "the configured train_command hardcodes --fine-tune-type and won't honor \
                 method '{}' — add the {{fine_tune_type}} placeholder (see the default config)",
                self.method
            );
        }
        Ok(())
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

    pub fn runs_dir(&self) -> Result<PathBuf> {
        self.resolve_path(&self.paths.runs_dir, "runs")
    }

    /// The provider profile used for chat, falling back to the top-level
    /// model/base_url/api_key when no named providers are configured.
    fn chat_profile(&self) -> ProviderProfile {
        let legacy = || ProviderProfile {
            kind: ProviderKind::Ollama,
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
        };
        if self.providers.is_empty() {
            return legacy();
        }
        self.roles
            .chat
            .as_deref()
            .and_then(|name| self.providers.get(name))
            .or_else(|| self.providers.values().next())
            .cloned()
            .unwrap_or_else(legacy)
    }

    /// Build the chat [`Provider`] from the active profile.
    pub fn chat_provider(&self, http: reqwest::Client) -> Provider {
        let p = self.chat_profile();
        Provider::new(p.kind, http, p.base_url, p.api_key)
    }

    /// Build a streaming chat request for `messages` using the active profile's
    /// model and the configured sampling parameters.
    pub fn chat_request(&self, messages: Vec<WireMessage>) -> ChatRequest {
        ChatRequest {
            model: self.chat_profile().model,
            messages,
            stream: true,
            sampling: self.generation.to_sampling(),
        }
    }

    /// The model id of the active chat profile (for display/diagnostics).
    pub fn chat_model(&self) -> String {
        self.chat_profile().model
    }

    /// Whether `roles.chat` names a provider that doesn't exist. The CLI can
    /// surface this as an error instead of silently falling back.
    pub fn chat_role_is_dangling(&self) -> bool {
        match self.roles.chat.as_deref() {
            Some(name) => !self.providers.contains_key(name),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_uses_legacy_fields_when_no_providers() {
        let cfg = Config {
            model: "legacy-model".into(),
            ..Config::default()
        };
        let req = cfg.chat_request(vec![]);
        assert_eq!(req.model, "legacy-model");
        assert!(req.stream);
    }

    #[test]
    fn chat_request_uses_named_profile_via_role() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "remote".into(),
            ProviderProfile {
                kind: ProviderKind::OpenAiCompat,
                base_url: "https://api.example/v1/chat/completions".into(),
                api_key: "k".into(),
                model: "gpt-x".into(),
            },
        );
        let cfg = Config {
            providers,
            roles: Roles {
                chat: Some("remote".into()),
                ..Default::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg.chat_request(vec![]).model, "gpt-x");
    }

    #[test]
    fn unknown_role_falls_back_to_first_provider() {
        let mut providers = BTreeMap::new();
        providers.insert(
            "only".into(),
            ProviderProfile {
                kind: ProviderKind::Ollama,
                base_url: "x".into(),
                api_key: String::new(),
                model: "m-only".into(),
            },
        );
        let cfg = Config {
            providers,
            roles: Roles {
                chat: Some("typo".into()),
                ..Default::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg.chat_request(vec![]).model, "m-only");
    }

    #[test]
    fn dangling_chat_role_detected() {
        let dangling = Config {
            roles: Roles {
                chat: Some("nope".into()),
                ..Default::default()
            },
            ..Config::default()
        };
        assert!(dangling.chat_role_is_dangling());
        assert!(!Config::default().chat_role_is_dangling());
    }

    #[test]
    fn check_method_guards_hardcoded_template() {
        // full + a hardcoded --fine-tune-type template → refused
        let hardcoded = Finetune {
            method: "full".into(),
            train_command: "mlx_lm.lora --fine-tune-type lora --iters {iters}".into(),
            ..Default::default()
        };
        assert!(hardcoded.check_method().is_err());

        // full + a templated train_command → allowed
        let templated = Finetune {
            method: "full".into(),
            train_command: "mlx_lm.lora --fine-tune-type {fine_tune_type}".into(),
            ..Default::default()
        };
        assert!(templated.check_method().is_ok());

        // lora is always fine (matches any legacy template)
        assert!(Finetune::default().check_method().is_ok());
    }
}
