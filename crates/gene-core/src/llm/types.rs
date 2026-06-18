//! OpenAI-compatible chat-completions wire types. Every response field is
//! optional/defaulted so non-conformant local models don't break deserialization.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
}

/// Sampling parameters for a chat request. Every field is optional: a `None`
/// value (or an empty `stop` list) is omitted from the JSON body, so a backend
/// never sees — and never rejects — a knob the caller didn't set.
///
/// Note: `top_k`, `min_p`, and `repetition_penalty` are local-backend extensions
/// (Ollama, vLLM, llama.cpp). A strict hosted OpenAI-compatible endpoint may
/// reject them with HTTP 400 if set, so leave them unset for such providers.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    pub stream: bool,
    #[serde(flatten)]
    pub sampling: Sampling,
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
