//! OpenAI-compatible chat-completions wire types. Every response field is
//! optional/defaulted so non-conformant local models don't break deserialization.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub delta: Delta,
}

#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    /// Reasoning models expose chain-of-thought in a separate field; accept both spellings.
    #[serde(default, alias = "reasoning_content")]
    pub reasoning: Option<String>,
}
