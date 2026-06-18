//! Disk persistence: conversations as JSON (atomic writes) and the append-only
//! training dataset as JSONL.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use crate::model::{Conversation, TrainingExample};

/// Write a conversation atomically to `<dir>/<id>.json`.
pub fn save_conversation(dir: &Path, conv: &Conversation) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let final_path = dir.join(format!("{}.json", conv.id));
    let tmp_path = dir.join(format!(".{}.json.tmp", conv.id));
    let json = serde_json::to_string_pretty(conv).context("serializing conversation")?;
    fs::write(&tmp_path, json).with_context(|| format!("writing {}", tmp_path.display()))?;
    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("renaming to {}", final_path.display()))?;
    Ok(())
}

/// Load a conversation by id.
pub fn load_conversation(dir: &Path, id: &str) -> Result<Conversation> {
    let path = dir.join(format!("{id}.json"));
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let conv =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(conv)
}

/// List saved conversations as (id, title, updated_at) newest-first.
pub fn list_conversations(dir: &Path) -> Vec<(String, String, String)> {
    let Ok(entries) = fs::read_dir(dir) else {
        return vec![];
    };
    let mut out: Vec<(String, String, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&path) {
            if let Ok(conv) = serde_json::from_str::<Conversation>(&text) {
                out.push((
                    conv.id.to_string(),
                    conv.title,
                    conv.updated_at.to_rfc3339(),
                ));
            }
        }
    }
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

/// Append one training example to the dataset JSONL.
pub fn append_dataset(path: &Path, example: &TrainingExample) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let line = serde_json::to_string(example).context("serializing training example")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening dataset {}", path.display()))?;
    writeln!(file, "{line}").context("appending to dataset")?;
    Ok(())
}
