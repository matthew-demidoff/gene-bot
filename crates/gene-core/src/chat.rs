//! Shared chat plumbing used by every frontend: the conversation persona
//! ([`Mode`]) and the conversation → OpenAI wire-message conversion. Keeping
//! this in the core means the CLI and GUI build identical requests.

use crate::config::Config;
use crate::llm::WireMessage;
use crate::model::{Message, Role};

/// Conversation persona — selects the system prompt and whether ```run blocks
/// are parsed as executable commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Can run shell commands (parses ```run blocks).
    Assistant,
    /// Advises but never executes.
    Tech,
    /// Casual conversation.
    Convo,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Assistant => "assistant (runs commands)",
            Mode::Tech => "tech-guy (talks, no commands)",
            Mode::Convo => "convo (casual chat)",
        }
    }

    /// Whether this mode parses ```run command blocks out of the stream.
    pub fn detect_commands(self) -> bool {
        matches!(self, Mode::Assistant)
    }

    /// The system prompt for this mode. `conv_system_prompt` is the
    /// conversation's own (assistant) prompt; Tech/Convo use the config's.
    pub fn system_prompt(self, config: &Config, conv_system_prompt: &str) -> String {
        match self {
            Mode::Assistant => conv_system_prompt.to_string(),
            Mode::Tech => config.tech_system_prompt.clone(),
            Mode::Convo => config.convo_system_prompt.clone(),
        }
    }
}

/// Build the OpenAI wire messages: the system prompt first, then the
/// conversation — empty messages skipped, tool outputs folded into user turns.
pub fn build_wire(system_prompt: &str, messages: &[Message]) -> Vec<WireMessage> {
    let mut wire = vec![WireMessage {
        role: "system".into(),
        content: system_prompt.to_string(),
    }];
    for m in messages {
        if m.content.trim().is_empty() {
            continue;
        }
        let (role, content) = match m.role {
            Role::System => continue,
            Role::User => ("user", m.content.clone()),
            Role::Assistant => ("assistant", m.content.clone()),
            Role::Tool => (
                "user",
                format!(
                    "[output of `{}`]\n{}",
                    m.command.as_deref().unwrap_or(""),
                    m.content
                ),
            ),
        };
        wire.push(WireMessage {
            role: role.into(),
            content,
        });
    }
    wire
}
