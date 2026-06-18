//! Tool-call model and the confirm-gated shell executor.

pub mod exec;

pub use exec::{run_command, ExecResult};

use serde::{Deserialize, Serialize};

/// A shell command the model wants to run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub command: String,
}

/// True if the command matches any denylist substring. Denylisted commands
/// always require a manual confirm, even when auto-run is on.
pub fn is_denied(command: &str, denylist: &[String]) -> bool {
    denylist.iter().any(|d| !d.is_empty() && command.contains(d.as_str()))
}
