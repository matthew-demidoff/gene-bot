//! Incremental parser that splits a model's token stream into answer text,
//! `<think>` reasoning, and ```run shell-command tool calls.
//!
//! Markers arrive split across arbitrary chunk boundaries (`<th` | `ink>`), so
//! the parser keeps a `carry` tail equal to the longest suffix that could be the
//! start of a marker, and only emits text that cannot be a partial marker.

use crate::tools::ToolCall;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    AnswerDelta(String),
    ThinkDelta(String),
    ThinkStart,
    ThinkEnd,
    ToolCall(ToolCall),
    Done,
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Answer,
    Think,
    Run,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const RUN_OPEN: &str = "```run";
const FENCE_CLOSE: &str = "```";

pub struct StreamParser {
    mode: Mode,
    /// Unemitted text that may contain a partial marker.
    carry: String,
    /// Accumulated body of an open ```run block.
    fence_buf: String,
    /// Whether to treat ```run blocks as executable tool calls (Assistant mode)
    /// or as ordinary code (Chat mode).
    detect_commands: bool,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new(true)
    }
}

impl StreamParser {
    pub fn new(detect_commands: bool) -> Self {
        StreamParser {
            mode: Mode::Answer,
            carry: String::new(),
            fence_buf: String::new(),
            detect_commands,
        }
    }

    /// Feed text from a `delta.reasoning` field: always reasoning, no inline markers.
    pub fn push_reasoning(&self, text: &str) -> Vec<StreamEvent> {
        if text.is_empty() {
            vec![]
        } else {
            vec![StreamEvent::ThinkDelta(text.to_string())]
        }
    }

    /// Feed a chunk of `delta.content`. Returns 0..N parsed events.
    pub fn push(&mut self, text: &str) -> Vec<StreamEvent> {
        self.carry.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.mode {
                Mode::Answer => {
                    let markers: &[&str] = if self.detect_commands {
                        &[THINK_OPEN, RUN_OPEN]
                    } else {
                        &[THINK_OPEN]
                    };
                    if let Some((pos, marker)) = earliest_marker(&self.carry, markers) {
                        if pos > 0 {
                            out.push(StreamEvent::AnswerDelta(self.carry[..pos].to_string()));
                        }
                        self.carry.drain(..pos + marker.len());
                        if marker == THINK_OPEN {
                            self.mode = Mode::Think;
                            out.push(StreamEvent::ThinkStart);
                        } else {
                            // Entering a run block: drop one leading newline if present.
                            if self.carry.starts_with('\n') {
                                self.carry.drain(..1);
                            } else if self.carry.starts_with("\r\n") {
                                self.carry.drain(..2);
                            }
                            self.mode = Mode::Run;
                        }
                        continue;
                    }
                    let keep = partial_suffix_len(&self.carry, markers);
                    let emit_to = self.carry.len() - keep;
                    if emit_to > 0 {
                        out.push(StreamEvent::AnswerDelta(self.carry[..emit_to].to_string()));
                        self.carry.drain(..emit_to);
                    }
                    break;
                }
                Mode::Think => {
                    if let Some(pos) = self.carry.find(THINK_CLOSE) {
                        if pos > 0 {
                            out.push(StreamEvent::ThinkDelta(self.carry[..pos].to_string()));
                        }
                        self.carry.drain(..pos + THINK_CLOSE.len());
                        self.mode = Mode::Answer;
                        out.push(StreamEvent::ThinkEnd);
                        continue;
                    }
                    let keep = partial_suffix_len(&self.carry, &[THINK_CLOSE]);
                    let emit_to = self.carry.len() - keep;
                    if emit_to > 0 {
                        out.push(StreamEvent::ThinkDelta(self.carry[..emit_to].to_string()));
                        self.carry.drain(..emit_to);
                    }
                    break;
                }
                Mode::Run => {
                    if let Some(pos) = self.carry.find(FENCE_CLOSE) {
                        self.fence_buf.push_str(&self.carry[..pos]);
                        self.carry.drain(..pos + FENCE_CLOSE.len());
                        let command = self.fence_buf.trim().to_string();
                        self.fence_buf.clear();
                        self.mode = Mode::Answer;
                        if !command.is_empty() {
                            out.push(StreamEvent::ToolCall(ToolCall { command }));
                        }
                        continue;
                    }
                    let keep = partial_suffix_len(&self.carry, &[FENCE_CLOSE]);
                    let emit_to = self.carry.len() - keep;
                    if emit_to > 0 {
                        self.fence_buf.push_str(&self.carry[..emit_to]);
                        self.carry.drain(..emit_to);
                    }
                    break;
                }
            }
        }
        out
    }

    /// Drain any buffered text at end of stream (handles unterminated tags).
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        match self.mode {
            Mode::Answer => {
                if !self.carry.is_empty() {
                    out.push(StreamEvent::AnswerDelta(std::mem::take(&mut self.carry)));
                }
            }
            Mode::Think => {
                if !self.carry.is_empty() {
                    out.push(StreamEvent::ThinkDelta(std::mem::take(&mut self.carry)));
                }
                out.push(StreamEvent::ThinkEnd);
            }
            Mode::Run => {
                // Unterminated run fence: surface what we have as plain answer text.
                let mut leftover = std::mem::take(&mut self.fence_buf);
                leftover.push_str(&std::mem::take(&mut self.carry));
                if !leftover.is_empty() {
                    out.push(StreamEvent::AnswerDelta(leftover));
                }
            }
        }
        self.mode = Mode::Answer;
        out
    }
}

/// Find the earliest complete occurrence of any marker; returns (byte_pos, marker).
fn earliest_marker<'a>(buf: &str, markers: &[&'a str]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &str)> = None;
    for m in markers {
        if let Some(p) = buf.find(m) {
            match best {
                Some((bp, _)) if bp <= p => {}
                _ => best = Some((p, m)),
            }
        }
    }
    best
}

/// Longest suffix of `buf` that is a *proper* prefix of some marker (so it might
/// complete into a marker on the next chunk). Markers are ASCII, so a valid
/// partial only contains ASCII bytes; we guard char boundaries for safety.
fn partial_suffix_len(buf: &str, markers: &[&str]) -> usize {
    let max = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    let start = max.saturating_sub(1).min(buf.len());
    for len in (1..=start).rev() {
        let idx = buf.len() - len;
        if !buf.is_char_boundary(idx) {
            continue;
        }
        let suffix = &buf[idx..];
        if markers
            .iter()
            .any(|m| m.len() > len && m.starts_with(suffix))
        {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(events: &[StreamEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::AnswerDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    fn drive(chunks: &[&str]) -> Vec<StreamEvent> {
        drive_mode(chunks, true)
    }

    fn drive_mode(chunks: &[&str], detect: bool) -> Vec<StreamEvent> {
        let mut p = StreamParser::new(detect);
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c));
        }
        out.extend(p.flush());
        out
    }

    #[test]
    fn plain_answer() {
        let ev = drive(&["hello ", "world"]);
        assert_eq!(answer(&ev), "hello world");
    }

    #[test]
    fn think_block_split_across_chunks() {
        let ev = drive(&["pre <th", "ink>reasoning</thi", "nk> post"]);
        assert_eq!(answer(&ev), "pre  post");
        let thinking: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "reasoning");
        assert!(ev.contains(&StreamEvent::ThinkStart));
        assert!(ev.contains(&StreamEvent::ThinkEnd));
    }

    #[test]
    fn run_fence_becomes_tool_call() {
        let ev = drive(&["sure:\n```", "run\nls -la\n``", "`\n"]);
        let calls: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall(t) => Some(t.command.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec!["ls -la".to_string()]);
        assert!(answer(&ev).contains("sure:"));
    }

    #[test]
    fn chat_mode_keeps_run_block_as_text() {
        // In Chat mode, a ```run fence is ordinary text, not a tool call.
        let ev = drive_mode(&["try:\n```run\nls\n```\n"], false);
        let calls = ev
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCall(_)))
            .count();
        assert_eq!(calls, 0);
        assert!(answer(&ev).contains("```run"));
        assert!(answer(&ev).contains("ls"));
    }

    #[test]
    fn marker_at_exact_boundary() {
        let ev = drive(&["<think>", "x", "</think>", "y"]);
        assert_eq!(answer(&ev), "y");
    }

    #[test]
    fn unterminated_think_flushes() {
        let ev = drive(&["<think>still thinking"]);
        let thinking: String = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "still thinking");
        assert!(ev.contains(&StreamEvent::ThinkEnd));
    }
}
