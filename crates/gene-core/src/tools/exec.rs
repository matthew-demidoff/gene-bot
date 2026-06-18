//! Run a shell command with a timeout, capturing (and truncating) output.

use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

/// Per-stream cap on captured output before it is fed back to the model.
const MAX_STREAM_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone)]
pub struct ExecResult {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

impl ExecResult {
    /// Render the result as the text fed back into the conversation.
    pub fn as_feedback(&self) -> String {
        let mut out = String::new();
        if self.timed_out {
            out.push_str("[command timed out]\n");
        }
        if !self.stdout.trim().is_empty() {
            out.push_str(&self.stdout);
            if !self.stdout.ends_with('\n') {
                out.push('\n');
            }
        }
        if !self.stderr.trim().is_empty() {
            out.push_str("[stderr]\n");
            out.push_str(&self.stderr);
            if !self.stderr.ends_with('\n') {
                out.push('\n');
            }
        }
        match self.exit_code {
            Some(0) | None if !out.trim().is_empty() => {}
            Some(code) => out.push_str(&format!("[exit code {code}]\n")),
            None => {}
        }
        if out.trim().is_empty() {
            out.push_str("[no output]\n");
        }
        out
    }
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_STREAM_BYTES {
        // Cut on a char boundary at or below the cap.
        let mut end = MAX_STREAM_BYTES;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
        s.push_str("\n…[truncated]");
    }
    s
}

/// Execute `command` via `sh -lc`, capturing stdout/stderr with a timeout.
/// `kill_on_drop` ensures a timed-out child is killed when the future is dropped.
pub async fn run_command(command: String, timeout_secs: u64) -> ExecResult {
    let mut cmd = Command::new("sh");
    cmd.arg("-lc")
        .arg(&command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ExecResult {
                command,
                stdout: String::new(),
                stderr: format!("failed to spawn command: {e}"),
                exit_code: None,
                timed_out: false,
            };
        }
    };

    match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output()).await {
        Ok(Ok(output)) => ExecResult {
            command,
            stdout: truncate(String::from_utf8_lossy(&output.stdout).into_owned()),
            stderr: truncate(String::from_utf8_lossy(&output.stderr).into_owned()),
            exit_code: output.status.code(),
            timed_out: false,
        },
        Ok(Err(e)) => ExecResult {
            command,
            stdout: String::new(),
            stderr: format!("error while running command: {e}"),
            exit_code: None,
            timed_out: false,
        },
        Err(_elapsed) => ExecResult {
            command,
            stdout: String::new(),
            stderr: format!("command exceeded {timeout_secs}s timeout"),
            exit_code: None,
            timed_out: true,
        },
    }
}
