//! Experiment tracking. Every training (and, later, eval) run is recorded on
//! disk as a directory under `<data_dir>/runs/<id>/`:
//!
//! - `run.json`     — the [`Run`] record (config snapshot, dataset provenance, status)
//! - `metrics.jsonl` — an append-only time-series of [`Metric`]s (loss curves, …)
//! - `run.log`      — the raw subprocess log
//!
//! Storage is deliberately dependency-light (serde_json + JSONL, atomic
//! tmp-then-rename writes), mirroring [`crate::persist`]. A `RunStore` is just a
//! root directory; listing scans it, so there is no index to keep in sync.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::dataset::TrainingExample;

/// A sortable run identifier: `20260618T142233-a1b2c3`.
pub type RunId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    Train,
    Eval,
    Sweep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Aborted,
}

/// What a run trained or evaluated on — pins the exact dataset bytes (so a model
/// is traceable to its data) and the conversations that produced them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRef {
    pub path: String,
    /// Content hash of the dataset file — its "version" without a VCS.
    pub content_hash: String,
    pub n_examples: usize,
    pub n_edited: usize,
    pub source_conversations: Vec<String>,
}

impl DatasetRef {
    /// Summarize a dataset JSONL file: hash its bytes and read provenance from
    /// each example's `meta`.
    pub fn from_dataset(path: &Path) -> Result<DatasetRef> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading dataset {}", path.display()))?;
        let mut n_examples = 0;
        let mut n_edited = 0;
        let mut conversations = std::collections::BTreeSet::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(ex) = serde_json::from_str::<TrainingExample>(line) {
                n_examples += 1;
                if ex.meta.edited {
                    n_edited += 1;
                }
                conversations.insert(ex.meta.conversation_id);
            }
        }
        Ok(DatasetRef {
            path: path.display().to_string(),
            content_hash: format!("{:016x}", crate::hash::fnv1a(text.as_bytes())),
            n_examples,
            n_edited,
            source_conversations: conversations.into_iter().collect(),
        })
    }
}

/// One point in a run's metric time-series, e.g. `{train_loss, val_loss}` at an iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub step: u64,
    pub at: DateTime<Utc>,
    pub fields: BTreeMap<String, f64>,
}

/// A tracked run: its configuration snapshot, dataset provenance, and outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: RunId,
    pub kind: RunKind,
    pub status: RunStatus,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub base_model: String,
    /// Hyperparameters as a flat JSON object, for cheap listing/diffing.
    pub hyperparams: serde_json::Value,
    pub dataset: Option<DatasetRef>,
    /// Final/summary metrics (e.g. final train loss, min val loss).
    pub summary: BTreeMap<String, f64>,
    pub error: Option<String>,
    pub gene_version: String,
}

/// An on-disk store of runs, rooted at a directory (`<data_dir>/runs`).
#[derive(Debug, Clone)]
pub struct RunStore {
    root: PathBuf,
}

impl RunStore {
    pub fn new(root: PathBuf) -> Self {
        RunStore { root }
    }

    fn run_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    /// Mint a new run, create its directory, and write the initial `run.json`.
    pub fn create(
        &self,
        kind: RunKind,
        base_model: String,
        hyperparams: serde_json::Value,
        dataset: Option<DatasetRef>,
    ) -> Result<Run> {
        let created_at = Utc::now();
        let id = format!(
            "{}-{}",
            created_at.format("%Y%m%dT%H%M%S"),
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let run = Run {
            id,
            kind,
            status: RunStatus::Running,
            created_at,
            finished_at: None,
            base_model,
            hyperparams,
            dataset,
            summary: BTreeMap::new(),
            error: None,
            gene_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        fs::create_dir_all(self.run_dir(&run.id))
            .with_context(|| format!("creating run dir for {}", run.id))?;
        self.save(&run)?;
        Ok(run)
    }

    /// Atomically (tmp-then-rename) write a run's `run.json`.
    pub fn save(&self, run: &Run) -> Result<()> {
        let dir = self.run_dir(&run.id);
        fs::create_dir_all(&dir)?;
        let path = dir.join("run.json");
        let tmp = dir.join(".run.json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(run)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("finalizing {}", path.display()))?;
        Ok(())
    }

    /// Append one metric point to the run's `metrics.jsonl`.
    pub fn append_metric(&self, id: &str, metric: &Metric) -> Result<()> {
        let path = self.run_dir(id).join("metrics.jsonl");
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{}", serde_json::to_string(metric)?)?;
        Ok(())
    }

    /// Append one line to the run's `run.log`.
    pub fn append_log(&self, id: &str, line: &str) -> Result<()> {
        let path = self.run_dir(id).join("run.log");
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    /// Write an arbitrary artifact file into the run's directory.
    pub fn write_artifact(&self, id: &str, name: &str, content: &str) -> Result<()> {
        let dir = self.run_dir(id);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join(name), content).with_context(|| format!("writing artifact {name}"))?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Run> {
        let path = self.run_dir(id).join("run.json");
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// All metric points for a run, oldest first (empty if none recorded).
    pub fn metrics(&self, id: &str) -> Vec<Metric> {
        let path = self.run_dir(id).join("metrics.jsonl");
        let Ok(text) = fs::read_to_string(path) else {
            return vec![];
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    /// Every run, newest first. Tolerant: unreadable runs are skipped.
    pub fn list(&self) -> Vec<Run> {
        let Ok(entries) = fs::read_dir(&self.root) else {
            return vec![];
        };
        let mut runs: Vec<Run> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .filter_map(|id| self.load(&id).ok())
            .collect();
        runs.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        runs
    }

    /// Mark any run still `Running` as `Aborted` — call at startup, where a
    /// `Running` run means a previous process died mid-run. Returns the count.
    ///
    /// Assumes a single running instance: a concurrent live run would be
    /// false-aborted, though it self-heals when that run next saves its status.
    pub fn reconcile(&self) -> usize {
        let mut aborted = 0;
        for mut run in self.list() {
            if run.status == RunStatus::Running {
                run.status = RunStatus::Aborted;
                run.finished_at = Some(Utc::now());
                if run.error.is_none() {
                    run.error = Some("process exited before the run finished".into());
                }
                if self.save(&run).is_ok() {
                    aborted += 1;
                }
            }
        }
        aborted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("gene-runs-test-{}", Uuid::new_v4().simple()))
    }

    #[test]
    fn create_append_load_roundtrip() {
        let root = temp_root();
        let store = RunStore::new(root.clone());

        let run = store
            .create(
                RunKind::Train,
                "mlx-community/Llama".into(),
                serde_json::json!({ "iters": 600 }),
                None,
            )
            .unwrap();
        assert_eq!(run.status, RunStatus::Running);

        store
            .append_metric(
                &run.id,
                &Metric {
                    step: 10,
                    at: Utc::now(),
                    fields: BTreeMap::from([("train_loss".into(), 2.5)]),
                },
            )
            .unwrap();

        let loaded = store.load(&run.id).unwrap();
        assert_eq!(loaded.id, run.id);
        assert_eq!(loaded.base_model, "mlx-community/Llama");

        let metrics = store.metrics(&run.id);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].step, 10);

        let listed = store.list();
        assert_eq!(listed.len(), 1);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dataset_ref_reads_provenance() {
        let path =
            std::env::temp_dir().join(format!("gene-ds-test-{}.jsonl", Uuid::new_v4().simple()));
        let lines = concat!(
            r#"{"messages":[{"role":"user","content":"hi"}],"meta":{"conversation_id":"c1","model":"m","created_at":"2026-01-01T00:00:00Z","edited":true,"source":"edit"}}"#,
            "\n",
            r#"{"messages":[{"role":"user","content":"hey"}],"meta":{"conversation_id":"c2","model":"m","created_at":"2026-01-01T00:00:00Z","edited":false,"source":"accept"}}"#,
            "\n",
        );
        fs::write(&path, lines).unwrap();

        let d = DatasetRef::from_dataset(&path).unwrap();
        assert_eq!(d.n_examples, 2);
        assert_eq!(d.n_edited, 1);
        assert_eq!(
            d.source_conversations,
            vec!["c1".to_string(), "c2".to_string()]
        );
        assert!(!d.content_hash.is_empty());

        fs::remove_file(&path).ok();
    }

    #[test]
    fn reconcile_aborts_only_stale_running() {
        let root = temp_root();
        let store = RunStore::new(root.clone());

        let stale = store
            .create(RunKind::Train, "m".into(), serde_json::json!({}), None)
            .unwrap();
        let mut finished = store
            .create(RunKind::Eval, "m".into(), serde_json::json!({}), None)
            .unwrap();
        finished.status = RunStatus::Succeeded;
        store.save(&finished).unwrap();

        assert_eq!(store.reconcile(), 1);
        let stale = store.load(&stale.id).unwrap();
        assert_eq!(stale.status, RunStatus::Aborted);
        assert!(stale.error.is_some());
        assert_eq!(
            store.load(&finished.id).unwrap().status,
            RunStatus::Succeeded
        );

        fs::remove_dir_all(&root).ok();
    }
}
