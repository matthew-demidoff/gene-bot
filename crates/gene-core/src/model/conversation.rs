//! The durable conversation model. An assistant message keeps both its edited
//! `content` and the model's `original_content`, which is what makes editing a
//! reply usable as a training signal.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub role: Role,
    /// Canonical, visible content (the edited text for an edited assistant turn).
    pub content: String,
    /// Set once, when an assistant reply is first edited: the model's raw output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_content: Option<String>,
    #[serde(default)]
    pub edited: bool,
    /// Captured `<think>` / reasoning text (assistant turns only). Not sent back upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default)]
    pub think_collapsed: bool,
    /// For tool messages: the command whose output this message carries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub created_at: DateTime<Utc>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            id: Uuid::new_v4(),
            role,
            content: content.into(),
            original_content: None,
            edited: false,
            thinking: None,
            think_collapsed: false,
            command: None,
            exit_code: None,
            created_at: Utc::now(),
        }
    }

    pub fn tool(command: String, content: String, exit_code: Option<i32>) -> Self {
        let mut m = Message::new(Role::Tool, content);
        m.command = Some(command);
        m.exit_code = exit_code;
        m
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Uuid,
    pub title: String,
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Conversation {
    pub fn new(model: String, system_prompt: String) -> Self {
        let now = Utc::now();
        Conversation {
            id: Uuid::new_v4(),
            title: "new conversation".into(),
            model,
            system_prompt,
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn push(&mut self, msg: Message) -> usize {
        if matches!(msg.role, Role::User) && self.title == "new conversation" {
            self.title = msg.content.lines().next().unwrap_or("").chars().take(60).collect();
        }
        self.messages.push(msg);
        self.updated_at = Utc::now();
        self.messages.len() - 1
    }
}
