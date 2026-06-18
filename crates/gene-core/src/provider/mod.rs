//! Inference providers: a thin layer over OpenAI-compatible chat endpoints
//! (Ollama, vLLM, llama.cpp `server`, LM Studio, hosted OpenAI-compatible APIs).
//! The chat path is identical across them; only model discovery differs by
//! [`ProviderKind`].

use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::llm::types::{ChatChunk, ChatRequest};
use crate::llm::{StreamEvent, StreamParser};

/// A shared HTTP client with a connection timeout, so an unreachable or stalled
/// server fails fast at connect time instead of hanging the app. There is no
/// total-request timeout — streaming chat responses are long-lived by design.
pub fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

/// Which backend a provider talks to. Determines model discovery; the chat path
/// is OpenAI-compatible for every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProviderKind {
    /// A local Ollama server (discovers models via `/api/tags`).
    #[default]
    #[serde(rename = "ollama")]
    Ollama,
    /// A generic OpenAI-compatible server (discovers models via `/v1/models`):
    /// vLLM, llama.cpp `server`, LM Studio, or a hosted API.
    #[serde(rename = "openai_compat", alias = "openai", alias = "open_ai_compat")]
    OpenAiCompat,
}

/// A configured inference endpoint.
pub struct Provider {
    kind: ProviderKind,
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl Provider {
    pub fn new(
        kind: ProviderKind,
        http: reqwest::Client,
        base_url: String,
        api_key: String,
    ) -> Self {
        Provider {
            kind,
            http,
            base_url,
            api_key,
        }
    }

    /// Drive one streaming chat request to completion, forwarding parsed events
    /// over `tx`. Intended to be `tokio::spawn`ed; aborting the task cancels the
    /// request.
    pub async fn chat_stream(
        &self,
        request: ChatRequest,
        detect_commands: bool,
        tx: mpsc::Sender<StreamEvent>,
    ) {
        let resp = match self
            .http
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx
                    .send(StreamEvent::Error(format!("request failed: {e}")))
                    .await;
                return;
            }
        };

        let resp = match resp.error_for_status() {
            Ok(r) => r,
            Err(e) => {
                let _ = tx
                    .send(StreamEvent::Error(format!("server returned an error: {e}")))
                    .await;
                return;
            }
        };

        let mut parser = StreamParser::new(detect_commands);
        let mut sse = resp.bytes_stream().eventsource();

        while let Some(event) = sse.next().await {
            match event {
                Ok(ev) => {
                    if ev.data == "[DONE]" {
                        break;
                    }
                    let chunk: ChatChunk = match serde_json::from_str(&ev.data) {
                        Ok(c) => c,
                        Err(_) => continue, // ignore keep-alives / non-JSON frames
                    };
                    for choice in chunk.choices {
                        if let Some(reason) = choice.delta.reasoning {
                            for out in parser.push_reasoning(&reason) {
                                if tx.send(out).await.is_err() {
                                    return;
                                }
                            }
                        }
                        if let Some(content) = choice.delta.content {
                            for out in parser.push(&content) {
                                if tx.send(out).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(StreamEvent::Error(format!("stream error: {e}")))
                        .await;
                    break;
                }
            }
        }

        for out in parser.flush() {
            if tx.send(out).await.is_err() {
                return;
            }
        }
        let _ = tx.send(StreamEvent::Done).await;
    }

    /// The chat endpoint URL (for diagnostics).
    pub fn endpoint(&self) -> &str {
        &self.base_url
    }

    /// The model-discovery URL for this backend (Ollama `/api/tags`,
    /// OpenAI-compatible `/v1/models`). Empty if the base URL is unusable.
    fn discovery_url(&self) -> String {
        // Split at the first "/v1" so both ".../v1/chat/completions" and a bare
        // ".../v1" yield the same root (no doubled "/v1/v1/models").
        let root = self
            .base_url
            .split_once("/v1")
            .map(|(root, _)| root)
            .unwrap_or(&self.base_url);
        if root.is_empty() {
            return String::new();
        }
        match self.kind {
            ProviderKind::Ollama => format!("{root}/api/tags"),
            ProviderKind::OpenAiCompat => format!("{root}/v1/models"),
        }
    }

    /// Whether the discovery endpoint responds successfully within 2s — lets
    /// `doctor` probe the *active* provider rather than a hardcoded URL.
    pub async fn reachable(&self) -> bool {
        let url = self.discovery_url();
        if url.is_empty() {
            return false;
        }
        self.http
            .get(&url)
            .bearer_auth(&self.api_key)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// List available model ids (best-effort; returns empty on any failure).
    pub async fn list_models(&self) -> Vec<String> {
        let url = self.discovery_url();
        if url.is_empty() {
            return vec![];
        }
        let Ok(resp) = self.http.get(&url).bearer_auth(&self.api_key).send().await else {
            return vec![];
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };
        // Ollama returns {"models":[{"name":...}]}; OpenAI {"data":[{"id":...}]}.
        let (array_key, name_field) = match self.kind {
            ProviderKind::Ollama => ("models", "name"),
            ProviderKind::OpenAiCompat => ("data", "id"),
        };
        json.get(array_key)
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get(name_field).and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }
}
