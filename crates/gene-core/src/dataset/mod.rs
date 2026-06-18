//! Dataset management: load, inspect, and transform the training dataset, and
//! (in [`format`]) convert to/from common chat-dataset file formats.
//!
//! The example type lives in [`crate::model::dataset`] and is re-exported here so
//! `gene_core::dataset::TrainingExample` is the one canonical path going forward.

pub mod format;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result};

pub use crate::model::dataset::{ChatMsg, Meta, TrainingExample};

/// Read a JSONL dataset, skipping blank and unparseable lines.
pub fn load(path: &Path) -> Result<Vec<TrainingExample>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading dataset {}", path.display()))?;
    Ok(parse(&text))
}

/// Parse JSONL text into examples (blank/unparseable lines skipped).
pub fn parse(text: &str) -> Vec<TrainingExample> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Write examples as JSONL (one object per line), creating parent dirs.
///
/// Atomic (tmp-then-rename): a crash or ENOSPC mid-write must not truncate the
/// existing dataset — the in-place `dedup`/`import` overwrites depend on this.
pub fn save(path: &Path, examples: &[TrainingExample]) -> Result<()> {
    let mut buf = String::new();
    for ex in examples {
        buf.push_str(&serde_json::to_string(ex)?);
        buf.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset.jsonl");
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&tmp, buf).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("finalizing {}", path.display()))?;
    Ok(())
}

/// Summary counts for a dataset.
#[derive(Debug, Clone)]
pub struct Stats {
    pub total: usize,
    pub edited: usize,
    pub conversations: usize,
    pub by_source: BTreeMap<String, usize>,
}

pub fn stats(examples: &[TrainingExample]) -> Stats {
    let mut by_source = BTreeMap::new();
    let mut conversations = BTreeSet::new();
    let mut edited = 0;
    for ex in examples {
        if ex.meta.edited {
            edited += 1;
        }
        *by_source.entry(ex.meta.source.clone()).or_insert(0) += 1;
        conversations.insert(ex.meta.conversation_id.clone());
    }
    Stats {
        total: examples.len(),
        edited,
        conversations: conversations.len(),
        by_source,
    }
}

/// A normalized content key: role-tagged, whitespace-trimmed message text. Case
/// is preserved (code is case-sensitive). Returns the full string (not a hash)
/// so dedup is collision-free.
fn content_key(messages: &[ChatMsg]) -> String {
    let mut buf = String::new();
    for m in messages {
        buf.push_str(&m.role);
        buf.push('\u{1}');
        buf.push_str(m.content.trim());
        buf.push('\u{2}');
    }
    buf
}

/// Remove examples whose normalized message content duplicates an earlier one.
/// Keeps the first occurrence; returns the number removed.
pub fn dedup(examples: &mut Vec<TrainingExample>) -> usize {
    let mut seen = BTreeSet::new();
    let before = examples.len();
    examples.retain(|ex| seen.insert(content_key(&ex.messages)));
    before - examples.len()
}

/// How to partition a dataset into train / validation / test.
#[derive(Debug, Clone, Copy)]
pub enum SplitStrategy {
    /// Shuffle examples (seeded), then slice by fraction.
    Random { seed: u64 },
    /// Assign whole conversations to one side (no example from a conversation
    /// leaks across the split), shuffling conversations (seeded).
    ByConversation { seed: u64 },
}

#[derive(Debug, Clone)]
pub struct SplitSpec {
    pub strategy: SplitStrategy,
    pub valid: f64,
    pub test: f64,
}

/// Index partition produced by [`make_split`].
#[derive(Debug, Clone, Default)]
pub struct Split {
    pub train: Vec<usize>,
    pub valid: Vec<usize>,
    pub test: Vec<usize>,
}

/// Partition example indices into train/valid/test per `spec`. Deterministic
/// given the seed.
pub fn make_split(examples: &[TrainingExample], spec: &SplitSpec) -> Split {
    let n = examples.len();
    if n == 0 {
        return Split::default();
    }
    let mut valid_target = ((n as f64) * spec.valid).round() as usize;
    let mut test_target = ((n as f64) * spec.test).round() as usize;
    // A positive fraction shouldn't round away to an empty split when there is
    // data to spare.
    if spec.valid > 0.0 && valid_target == 0 && n >= 2 {
        valid_target = 1;
    }
    if spec.test > 0.0 && test_target == 0 && n.saturating_sub(valid_target) >= 1 {
        test_target = 1;
    }

    match spec.strategy {
        SplitStrategy::Random { seed } => {
            let mut order: Vec<usize> = (0..n).collect();
            shuffle(&mut order, seed);
            let valid_n = valid_target.min(n);
            let test_n = test_target.min(n - valid_n);
            let mut split = Split::default();
            for (rank, &i) in order.iter().enumerate() {
                if rank < valid_n {
                    split.valid.push(i);
                } else if rank < valid_n + test_n {
                    split.test.push(i);
                } else {
                    split.train.push(i);
                }
            }
            split
        }
        SplitStrategy::ByConversation { seed } => {
            let mut groups = group_by_conversation(examples);
            let mut order: Vec<usize> = (0..groups.len()).collect();
            shuffle(&mut order, seed);
            let mut split = Split::default();
            let (mut in_valid, mut in_test) = (0usize, 0usize);
            for g in order {
                let members = std::mem::take(&mut groups[g]);
                if in_valid < valid_target {
                    in_valid += members.len();
                    split.valid.extend(members);
                } else if in_test < test_target {
                    in_test += members.len();
                    split.test.extend(members);
                } else {
                    split.train.extend(members);
                }
            }
            split
        }
    }
}

/// Group example indices by `conversation_id`, preserving first-seen order.
fn group_by_conversation(examples: &[TrainingExample]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pos: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, ex) in examples.iter().enumerate() {
        let cid = ex.meta.conversation_id.as_str();
        match pos.get(cid) {
            Some(&g) => groups[g].push(i),
            None => {
                pos.insert(cid, groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups
}

/// Deterministic in-place shuffle (splitmix64 + Fisher–Yates) — reproducible
/// from `seed`, no `rand` dependency.
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn ex(conversation: &str, user: &str) -> TrainingExample {
        TrainingExample {
            messages: vec![ChatMsg {
                role: "user".into(),
                content: user.into(),
            }],
            meta: Meta {
                conversation_id: conversation.into(),
                model: "m".into(),
                created_at: Utc::now(),
                edited: false,
                source: "test".into(),
                original_assistant: None,
            },
        }
    }

    #[test]
    fn dedup_removes_content_duplicates() {
        let mut data = vec![ex("a", "hello"), ex("b", " hello "), ex("c", "world")];
        // "hello" and " hello " normalize equal; "world" is distinct.
        assert_eq!(dedup(&mut data), 1);
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn by_conversation_split_has_no_leakage() {
        // 3 conversations, 2 examples each.
        let data: Vec<_> = ["a", "b", "c"]
            .iter()
            .flat_map(|c| [ex(c, &format!("{c}-1")), ex(c, &format!("{c}-2"))])
            .collect();
        let split = make_split(
            &data,
            &SplitSpec {
                strategy: SplitStrategy::ByConversation { seed: 7 },
                valid: 0.34,
                test: 0.0,
            },
        );
        let conv = |idxs: &[usize]| -> BTreeSet<String> {
            idxs.iter()
                .map(|&i| data[i].meta.conversation_id.clone())
                .collect()
        };
        let train_convs = conv(&split.train);
        let valid_convs = conv(&split.valid);
        // No conversation appears on both sides, and neither side is empty.
        assert!(train_convs.is_disjoint(&valid_convs));
        assert!(!split.train.is_empty());
        assert!(!split.valid.is_empty());
        assert_eq!(split.train.len() + split.valid.len(), data.len());
    }

    #[test]
    fn random_split_is_deterministic() {
        let data: Vec<_> = (0..10).map(|i| ex(&format!("c{i}"), "x")).collect();
        let spec = SplitSpec {
            strategy: SplitStrategy::Random { seed: 42 },
            valid: 0.2,
            test: 0.0,
        };
        let a = make_split(&data, &spec);
        let b = make_split(&data, &spec);
        assert_eq!(a.valid, b.valid);
        assert_eq!(a.valid.len(), 2);
    }

    #[test]
    fn tiny_valid_fraction_still_holds_out_one() {
        // n = 3, valid 0.1 rounds to 0 — but a positive fraction must hold out
        // at least one example rather than yield an empty validation set.
        let data: Vec<_> = (0..3).map(|i| ex(&format!("c{i}"), "x")).collect();
        let split = make_split(
            &data,
            &SplitSpec {
                strategy: SplitStrategy::Random { seed: 1 },
                valid: 0.1,
                test: 0.0,
            },
        );
        assert_eq!(split.valid.len(), 1);
        assert_eq!(split.train.len(), 2);
    }

    #[test]
    fn dedup_keeps_first_occurrence() {
        let mut data = vec![ex("a", "dup"), ex("b", "dup"), ex("c", "unique")];
        assert_eq!(dedup(&mut data), 1);
        // The first "dup" (conversation a) is kept; the second (b) is dropped.
        assert_eq!(data[0].meta.conversation_id, "a");
        assert_eq!(data[1].meta.conversation_id, "c");
    }
}
