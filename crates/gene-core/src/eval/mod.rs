//! Evaluation harness: run a fixed prompt set through a provider/model, grade
//! the outputs, and summarize. Results persist as an `Eval` run in the run store.
//!
//! Eval-set file format (JSON):
//! ```json
//! { "name": "smoke", "system_prompt": "...", "temperature": 0.0,
//!   "items": [ { "id": "q1", "prompt": "2+2?", "reference": "4" } ] }
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::llm::types::{ChatRequest, Sampling};
use crate::llm::WireMessage;
use crate::provider::Provider;
use crate::runs::{RunId, RunKind, RunStatus, RunStore};

/// How to score an output against an item's `reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Grader {
    /// Capture outputs only (no score).
    #[default]
    None,
    /// Output must equal the reference (both trimmed).
    Exact,
    /// Output must contain the reference (trimmed).
    Contains,
    /// An LLM judge scores the answer PASS/FAIL (uses a judge provider). Works
    /// for open-ended items with no exact reference.
    Judge,
}

impl Grader {
    pub fn parse(s: &str) -> anyhow::Result<Grader> {
        match s {
            "none" => Ok(Grader::None),
            "exact" => Ok(Grader::Exact),
            "contains" => Ok(Grader::Contains),
            "judge" => Ok(Grader::Judge),
            other => anyhow::bail!("unknown grader '{other}' (none | exact | contains | judge)"),
        }
    }
}

/// A configured LLM judge: the provider and model used to score answers.
pub struct Judge<'a> {
    pub provider: &'a Provider,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalItem {
    pub id: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Per-item grader override; falls back to the run-level grader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grader: Option<Grader>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSet {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Pinned for comparability across runs (default 0.0 = greedy).
    #[serde(default)]
    pub temperature: f64,
    pub items: Vec<EvalItem>,
}

impl EvalSet {
    pub fn load(path: &Path) -> anyhow::Result<EvalSet> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading eval set {}: {e}", path.display()))?;
        serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing eval set {}: {e}", path.display()))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    pub item_id: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    /// An inference or judge error occurred — distinct from a graded FAIL, so a
    /// broken backend doesn't masquerade as a wrong answer.
    pub error: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub set: String,
    pub model: String,
    pub n: usize,
    /// How many items produced a pass/fail verdict.
    pub scored: usize,
    pub passed: usize,
    /// Items where inference or judging errored — NOT counted as failures, so a
    /// down provider reports `errored`, not a misleading 0.0%.
    pub errored: usize,
    /// Pass rate over scored items (None if nothing was graded).
    pub mean_score: Option<f64>,
    pub items: Vec<EvalResult>,
}

/// Aggregate per-item results into a report (extracted so it's unit-testable).
fn report_from(set: &str, model: &str, items: Vec<EvalResult>) -> EvalReport {
    let scored = items.iter().filter(|r| r.passed.is_some()).count();
    let passed = items.iter().filter(|r| r.passed == Some(true)).count();
    let errored = items.iter().filter(|r| r.error).count();
    let mean_score = (scored > 0).then(|| passed as f64 / scored as f64);
    EvalReport {
        set: set.to_string(),
        model: model.to_string(),
        n: items.len(),
        scored,
        passed,
        errored,
        mean_score,
        items,
    }
}

fn grade(grader: Grader, item: &EvalItem, output: &str) -> Option<bool> {
    let reference = item.reference.as_deref()?;
    match grader {
        Grader::None => None,
        Grader::Exact => Some(output.trim() == reference.trim()),
        Grader::Contains => Some(output.contains(reference.trim())),
        // Judge is async and handled in run_eval; never reaches the sync path.
        Grader::Judge => None,
    }
}

fn messages(system: &Option<String>, prompt: &str) -> Vec<WireMessage> {
    let mut out = Vec::new();
    if let Some(s) = system {
        if !s.trim().is_empty() {
            out.push(WireMessage {
                role: "system".into(),
                content: s.clone(),
            });
        }
    }
    out.push(WireMessage {
        role: "user".into(),
        content: prompt.to_string(),
    });
    out
}

/// Whether a judge's reply is a PASS. The judge is asked for one word; FAIL wins
/// if both somehow appear.
fn parse_verdict(text: &str) -> bool {
    let upper = text.to_uppercase();
    if upper.contains("FAIL") {
        false
    } else {
        upper.contains("PASS")
    }
}

/// Score an answer with the LLM judge — uses the item's `reference` as criteria
/// when present, else asks for a general correctness judgement (so open-ended
/// items with no exact reference are still gradable).
async fn judge_grade(judge: &Judge<'_>, item: &EvalItem, output: &str) -> Option<bool> {
    let criteria = item
        .reference
        .as_deref()
        .unwrap_or("answer the question correctly and helpfully");
    let prompt = format!(
        "You are grading a model's answer. Reply with exactly one word: PASS or FAIL.\n\n\
         Question:\n{}\n\nExpected answer or criteria:\n{}\n\nCandidate answer:\n{}\n\n\
         Does the candidate answer satisfy the criteria? Reply PASS or FAIL.",
        item.prompt, criteria, output
    );
    let request = ChatRequest {
        model: judge.model.clone(),
        messages: vec![WireMessage {
            role: "user".into(),
            content: prompt,
        }],
        stream: true,
        sampling: Sampling {
            temperature: Some(0.0),
            ..Default::default()
        },
    };
    let verdict = judge.provider.complete(request).await.ok()?;
    Some(parse_verdict(&verdict))
}

/// Run the eval set through `provider`/`model`, grading with `grader` (per-item
/// graders win). Inference fans out up to `concurrency` requests at a time.
pub async fn run_eval(
    set: &EvalSet,
    provider: &Provider,
    model: &str,
    grader: Grader,
    judge: Option<&Judge<'_>>,
    concurrency: usize,
) -> EvalReport {
    use futures_util::stream::{self, StreamExt};

    let mut items: Vec<EvalResult> = stream::iter(set.items.iter())
        .map(|item| {
            let model = model.to_string();
            let system = set.system_prompt.clone();
            let temperature = set.temperature;
            async move {
                let request = ChatRequest {
                    model,
                    messages: messages(&system, &item.prompt),
                    stream: true,
                    sampling: Sampling {
                        temperature: Some(temperature),
                        ..Default::default()
                    },
                };
                let (output, inference_error) = match provider.complete(request).await {
                    Ok(o) => (o, false),
                    Err(e) => (format!("[error: {e}]"), true),
                };
                let effective = item.grader.unwrap_or(grader);
                let passed = if inference_error {
                    None
                } else {
                    match effective {
                        Grader::Judge => match judge {
                            Some(j) => judge_grade(j, item, &output).await,
                            None => None,
                        },
                        other => grade(other, item, &output),
                    }
                };
                // A judge that was needed but failed (or wasn't configured) is an
                // error, not "ungraded" — otherwise it silently shrinks the
                // denominator and inflates the pass rate.
                let judge_error =
                    !inference_error && matches!(effective, Grader::Judge) && passed.is_none();
                EvalResult {
                    item_id: item.id.clone(),
                    output,
                    passed,
                    error: inference_error || judge_error,
                }
            }
        })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;

    // buffer_unordered yields out of order — restore the eval-set order.
    items.sort_by_key(|r| {
        set.items
            .iter()
            .position(|i| i.id == r.item_id)
            .unwrap_or(usize::MAX)
    });

    report_from(&set.name, model, items)
}

/// Record an eval report as an `Eval` run: summary metrics on `run.json` plus a
/// `results.jsonl` artifact of every per-item output. Returns the run id.
pub fn persist(store: &RunStore, report: &EvalReport, grader: Grader) -> anyhow::Result<RunId> {
    let mut run = store.create(
        RunKind::Eval,
        report.model.clone(),
        serde_json::json!({ "set": report.set, "grader": grader }),
        None,
    )?;
    let mut summary = BTreeMap::new();
    if let Some(ms) = report.mean_score {
        summary.insert("mean_score".to_string(), ms);
    }
    summary.insert("n".to_string(), report.n as f64);
    summary.insert("passed".to_string(), report.passed as f64);
    summary.insert("errored".to_string(), report.errored as f64);
    run.summary = summary;
    run.status = RunStatus::Succeeded;
    run.finished_at = Some(chrono::Utc::now());
    store.save(&run)?;

    let mut jsonl = String::new();
    for item in &report.items {
        jsonl.push_str(&serde_json::to_string(item)?);
        jsonl.push('\n');
    }
    store.write_artifact(&run.id, "results.jsonl", &jsonl)?;
    Ok(run.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(reference: Option<&str>) -> EvalItem {
        EvalItem {
            id: "q".into(),
            prompt: "p".into(),
            reference: reference.map(String::from),
            grader: None,
        }
    }

    #[test]
    fn grading_exact_and_contains() {
        let it = item(Some("hello"));
        assert_eq!(grade(Grader::Exact, &it, " hello "), Some(true));
        assert_eq!(grade(Grader::Exact, &it, "hello world"), Some(false));
        assert_eq!(grade(Grader::Contains, &it, "well hello there"), Some(true));
        assert_eq!(grade(Grader::None, &it, "hello"), None);
        // no reference -> not gradable
        assert_eq!(grade(Grader::Exact, &item(None), "x"), None);
    }

    #[test]
    fn eval_set_parses_minimal_json() {
        let json = r#"{"name":"t","items":[{"id":"a","prompt":"hi","reference":"yo"}]}"#;
        let set: EvalSet = serde_json::from_str(json).unwrap();
        assert_eq!(set.name, "t");
        assert_eq!(set.items.len(), 1);
        assert_eq!(set.items[0].reference.as_deref(), Some("yo"));
        assert_eq!(set.temperature, 0.0);
    }

    #[test]
    fn judge_verdict_parsing() {
        assert!(parse_verdict("PASS"));
        assert!(parse_verdict("pass — the answer is correct"));
        assert!(!parse_verdict("FAIL"));
        assert!(!parse_verdict("This should FAIL because ..."));
        assert!(!parse_verdict("unclear")); // neither word -> not a pass
        assert!(!parse_verdict("PASS? no, FAIL")); // both present -> FAIL wins
    }

    #[test]
    fn report_separates_errors_from_failures() {
        let r = |id: &str, passed: Option<bool>, error: bool| EvalResult {
            item_id: id.into(),
            output: String::new(),
            passed,
            error,
        };
        let report = report_from(
            "s",
            "m",
            vec![
                r("a", Some(true), false),  // pass
                r("b", Some(false), false), // fail
                r("c", None, true),         // errored (e.g. down provider)
                r("d", None, false),        // ungraded
            ],
        );
        assert_eq!(report.n, 4);
        assert_eq!(report.scored, 2); // pass + fail only
        assert_eq!(report.passed, 1);
        assert_eq!(report.errored, 1); // the down item is NOT a failure
        assert_eq!(report.mean_score, Some(0.5)); // 1/2, not 1/3 or 1/4
    }
}
