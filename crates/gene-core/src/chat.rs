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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Message;

    #[test]
    fn build_wire_shapes_the_conversation() {
        let mut tool = Message::new(Role::Tool, "the output");
        tool.command = Some("ls".into());
        let messages = vec![
            Message::new(Role::User, "hi"),
            Message::new(Role::Assistant, "hello"),
            tool,
            Message::new(Role::User, "   "), // blank -> skipped
        ];

        let wire = build_wire("SYS", &messages);

        assert_eq!(wire.len(), 4); // system + user + assistant + tool-as-user
        assert_eq!(
            (wire[0].role.as_str(), wire[0].content.as_str()),
            ("system", "SYS")
        );
        assert_eq!(
            (wire[1].role.as_str(), wire[1].content.as_str()),
            ("user", "hi")
        );
        assert_eq!(wire[2].role, "assistant");
        assert_eq!(wire[3].role, "user");
        assert!(wire[3].content.starts_with("[output of `ls`]"));
    }

    #[test]
    fn modes_select_prompt_and_command_detection() {
        assert!(Mode::Assistant.detect_commands());
        assert!(!Mode::Tech.detect_commands());
        assert!(!Mode::Convo.detect_commands());
    }
}
