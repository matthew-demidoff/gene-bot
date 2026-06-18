//! Shared chat plumbing: the conversation → OpenAI wire-message conversion,
//! used by both frontends so they build identical requests.

use crate::llm::WireMessage;
use crate::model::{Message, Role};

/// Build the OpenAI wire messages: the system prompt first (skipped when empty),
/// then the conversation — empty messages skipped, tool outputs folded into user
/// turns.
pub fn build_wire(system_prompt: &str, messages: &[Message]) -> Vec<WireMessage> {
    let mut wire = Vec::new();
    if !system_prompt.trim().is_empty() {
        wire.push(WireMessage {
            role: "system".into(),
            content: system_prompt.to_string(),
        });
    }
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
    fn empty_system_prompt_is_skipped() {
        let wire = build_wire("   ", &[Message::new(Role::User, "hi")]);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
    }
}
