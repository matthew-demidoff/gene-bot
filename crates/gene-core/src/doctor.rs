//! Prerequisite checks for chat + fine-tuning, returned as structured data so a
//! frontend can render them as text, `--json`, or a settings-panel checklist.

use crate::config::Config;

/// One prerequisite check.
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// The result of a `doctor` run.
pub struct DoctorReport {
    pub checks: Vec<Check>,
    pub chat_model: String,
    pub chat_endpoint: String,
    pub mlx_base: String,
    pub dataset_path: String,
}

impl DoctorReport {
    pub fn all_ok(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }
}

/// Probe the environment and the *active* chat provider.
pub async fn report(config: &Config) -> DoctorReport {
    let mut checks = Vec::new();

    let arch = arch_string();
    checks.push(Check {
        name: "Apple Silicon (arm64)".into(),
        ok: arch.contains("arm64"),
        detail: arch,
    });

    let ollama = cmd_version("ollama", &["--version"]);
    checks.push(Check {
        name: "ollama (chat host)".into(),
        ok: ollama.is_some(),
        detail: ollama.unwrap_or_else(|| "not found — https://ollama.com".into()),
    });

    // Probe the provider the chat path actually uses, via its own discovery
    // endpoint (Ollama /api/tags or OpenAI-compatible /v1/models) — not a
    // hardcoded legacy URL.
    let provider = config.chat_provider(crate::provider::http_client());
    let reachable = provider.reachable().await;
    let endpoint = provider.endpoint().to_string();
    checks.push(Check {
        name: "chat provider reachable".into(),
        ok: reachable,
        detail: if reachable {
            endpoint.clone()
        } else {
            format!("not reachable: {endpoint}")
        },
    });

    let py = cmd_version("python3", &["--version"]);
    checks.push(Check {
        name: "python3 (for MLX)".into(),
        ok: py.is_some(),
        detail: py.unwrap_or_else(|| "not found".into()),
    });

    let mlx = cmd_version(
        "python3",
        &[
            "-c",
            "import mlx_lm; print(getattr(mlx_lm,'__version__','?'))",
        ],
    );
    checks.push(Check {
        name: "mlx-lm (LoRA trainer)".into(),
        ok: mlx.is_some(),
        detail: mlx.unwrap_or_else(|| "not found — `pip install mlx-lm`".into()),
    });

    DoctorReport {
        checks,
        chat_model: config.chat_model(),
        chat_endpoint: endpoint,
        mlx_base: config.finetune.mlx_base.clone(),
        dataset_path: config
            .dataset_path()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
    }
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
