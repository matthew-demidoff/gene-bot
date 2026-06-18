//! Convert between gene's native dataset and common chat-dataset file formats.
//!
//! - **gene** — the native [`TrainingExample`] JSONL (messages + provenance meta)
//! - **mlx / openai** — `{"messages": [{"role","content"}, …]}` per line
//! - **sharegpt** — `{"conversations": [{"from","value"}, …]}` per line

use anyhow::{anyhow, bail, Result};
use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use crate::model::dataset::{ChatMsg, Meta, TrainingExample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Gene,
    Mlx,
    OpenAi,
    ShareGpt,
}

impl Format {
    pub fn parse(s: &str) -> Result<Format> {
        match s {
            "gene" => Ok(Format::Gene),
            "mlx" => Ok(Format::Mlx),
            "openai" => Ok(Format::OpenAi),
            "sharegpt" => Ok(Format::ShareGpt),
            other => bail!("unknown format '{other}' (gene | mlx | openai | sharegpt)"),
        }
    }
}

/// Import JSONL `text` in `format` into examples. Errors on the first record
/// that isn't valid JSON or lacks the expected array; a message missing
/// `role`/`content` (ShareGPT: `from`/`value`) defaults that field to empty
/// rather than failing. `mlx` is treated as the OpenAI chat-messages shape —
/// prompt/completion or plain-text MLX files are not parsed.
pub fn import(text: &str, format: Format) -> Result<Vec<TrainingExample>> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        let messages = match format {
            Format::Gene => {
                out.push(serde_json::from_value(value)?);
                continue;
            }
            Format::Mlx | Format::OpenAi => messages_from_openai(&value)?,
            Format::ShareGpt => messages_from_sharegpt(&value)?,
        };
        out.push(imported(messages));
    }
    Ok(out)
}

/// Export examples to JSONL text in `format`.
pub fn export(examples: &[TrainingExample], format: Format) -> Result<String> {
    let mut buf = String::new();
    for ex in examples {
        let line = match format {
            Format::Gene => serde_json::to_string(ex)?,
            Format::Mlx | Format::OpenAi => {
                serde_json::to_string(&serde_json::json!({ "messages": ex.messages }))?
            }
            Format::ShareGpt => serde_json::to_string(
                &serde_json::json!({ "conversations": to_sharegpt(&ex.messages) }),
            )?,
        };
        buf.push_str(&line);
        buf.push('\n');
    }
    Ok(buf)
}

fn imported(messages: Vec<ChatMsg>) -> TrainingExample {
    TrainingExample {
        messages,
        meta: Meta {
            // A unique id per imported record: imports carry no conversation
            // grouping, so each is its own conversation and a later
            // ByConversation split won't collapse the whole import to one side.
            conversation_id: format!("import-{}", Uuid::new_v4().simple()),
            model: String::new(),
            created_at: Utc::now(),
            edited: false,
            source: "import".into(),
            original_assistant: None,
        },
    }
}

fn messages_from_openai(value: &Value) -> Result<Vec<ChatMsg>> {
    let arr = value
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow!("record has no 'messages' array"))?;
    Ok(arr
        .iter()
        .map(|m| ChatMsg {
            role: str_field(m, "role"),
            content: str_field(m, "content"),
        })
        .collect())
}

fn messages_from_sharegpt(value: &Value) -> Result<Vec<ChatMsg>> {
    let arr = value
        .get("conversations")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow!("record has no 'conversations' array"))?;
    Ok(arr
        .iter()
        .map(|m| ChatMsg {
            role: match str_field(m, "from").as_str() {
                "human" => "user".into(),
                "gpt" => "assistant".into(),
                other => other.to_string(),
            },
            content: str_field(m, "value"),
        })
        .collect())
}

fn to_sharegpt(messages: &[ChatMsg]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let from = match m.role.as_str() {
                "user" => "human",
                "assistant" => "gpt",
                other => other,
            };
            serde_json::json!({ "from": from, "value": m.content })
        })
        .collect()
}

fn str_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharegpt_round_trips_through_gene() {
        let sharegpt =
            r#"{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"yo"}]}"#;
        let examples = import(sharegpt, Format::ShareGpt).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].messages[0].role, "user");
        assert_eq!(examples[0].messages[1].role, "assistant");
        assert_eq!(examples[0].meta.source, "import");

        // Export back to ShareGPT maps roles in reverse.
        let out = export(&examples, Format::ShareGpt).unwrap();
        assert!(out.contains("\"from\":\"human\""));
        assert!(out.contains("\"from\":\"gpt\""));
    }

    #[test]
    fn openai_import_reads_messages() {
        let line =
            r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}]}"#;
        let examples = import(line, Format::OpenAi).unwrap();
        assert_eq!(examples[0].messages.len(), 2);
        assert_eq!(examples[0].messages[0].role, "system");
    }

    #[test]
    fn mlx_export_emits_messages_only() {
        let examples = import(
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            Format::Mlx,
        )
        .unwrap();
        let out = export(&examples, Format::Mlx).unwrap();
        assert!(out.contains("\"messages\""));
        assert!(!out.contains("\"meta\"")); // MLX form drops provenance
    }
}
