//! The LLM client: POST a streaming chat-completions request and decode the SSE
//! response into `StreamEvent`s via the incremental parser.

pub mod stream;
pub mod types;

pub use stream::{StreamEvent, StreamParser};
pub use types::WireMessage;

use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::config::Config;
use types::{ChatChunk, ChatRequest};

/// Drive one streaming request to completion, forwarding parsed events over `tx`.
/// Intended to be `tokio::spawn`ed; aborting the task cancels the request.
pub async fn run_stream(
    http: reqwest::Client,
    cfg: Config,
    messages: Vec<WireMessage>,
    detect_commands: bool,
    tx: mpsc::Sender<StreamEvent>,
) {
    let request = ChatRequest {
        model: cfg.model.clone(),
        messages,
        stream: true,
        temperature: Some(cfg.generation.temperature),
        max_tokens: Some(cfg.generation.max_tokens),
    };

    let resp = match http
        .post(&cfg.base_url)
        .bearer_auth(&cfg.api_key)
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

/// List local Ollama model tags (best-effort; returns empty on any failure).
pub async fn list_models(http: &reqwest::Client, base_url: &str) -> Vec<String> {
    let tags_url = base_url
        .split("/v1/")
        .next()
        .map(|root| format!("{root}/api/tags"))
        .unwrap_or_default();
    if tags_url.is_empty() {
        return vec![];
    }
    let Ok(resp) = http.get(&tags_url).send().await else {
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
