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

    /// List available model ids (best-effort; returns empty on any failure).
    pub async fn list_models(&self) -> Vec<String> {
        match self.kind {
            ProviderKind::Ollama => self.list_models_ollama().await,
            ProviderKind::OpenAiCompat => self.list_models_openai().await,
        }
    }

    async fn list_models_ollama(&self) -> Vec<String> {
        let tags_url = self
            .base_url
            .split("/v1/")
            .next()
            .map(|root| format!("{root}/api/tags"))
            .unwrap_or_default();
        if tags_url.is_empty() {
            return vec![];
        }
        let Ok(resp) = self.http.get(&tags_url).send().await else {
            return vec![];
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };
        json.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn list_models_openai(&self) -> Vec<String> {
        let models_url = self
            .base_url
            .split("/v1/")
            .next()
            .map(|root| format!("{root}/v1/models"))
            .unwrap_or_default();
        if models_url.is_empty() {
            return vec![];
        }
        let Ok(resp) = self
            .http
            .get(&models_url)
            .bearer_auth(&self.api_key)
            .send()
            .await
        else {
            return vec![];
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };
        json.get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }
}
