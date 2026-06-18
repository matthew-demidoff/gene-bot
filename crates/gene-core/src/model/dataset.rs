//! The training-dataset record. One JSON object per line; the last entry in
//! `messages` is the *ideal* assistant reply (the kept or edited text). This is
//! the contract consumed by the MLX LoRA pipeline (`mlx_lm.lora` chat format).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::conversation::{Conversation, Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub conversation_id: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub edited: bool,
    /// "edit" | "accept"
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_assistant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub messages: Vec<ChatMsg>,
    pub meta: Meta,
}

impl TrainingExample {
    /// Build an example from the conversation up to and including the assistant
    /// message at `assistant_idx`. Tool outputs are folded in as user turns so
    /// each example is self-contained.
    pub fn from_conversation(conv: &Conversation, assistant_idx: usize) -> Option<TrainingExample> {
        let target = conv.messages.get(assistant_idx)?;
        if !matches!(target.role, Role::Assistant) || target.content.trim().is_empty() {
            return None;
        }

        let mut messages = vec![ChatMsg {
            role: "system".into(),
            content: conv.system_prompt.clone(),
        }];

        for m in &conv.messages[..=assistant_idx] {
            let role = match m.role {
                Role::System => continue, // system prompt already emitted once
                Role::User => "user",
                Role::Assistant => "assistant",
                // Tool output becomes a user turn so any base model template accepts it.
                Role::Tool => "user",
            };
            let content = match m.role {
                Role::Tool => format!(
                    "[output of `{}`]\n{}",
                    m.command.as_deref().unwrap_or(""),
                    m.content
                ),
                _ => m.content.clone(),
            };
            if content.trim().is_empty() {
                continue;
            }
            messages.push(ChatMsg { role: role.into(), content });
        }

        Some(TrainingExample {
            messages,
            meta: Meta {
                conversation_id: conv.id.to_string(),
                model: conv.model.clone(),
                created_at: Utc::now(),
                edited: target.edited,
                source: if target.edited { "edit".into() } else { "accept".into() },
                original_assistant: target.original_content.clone(),
            },
        })
    }
}
