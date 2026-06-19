# gene

A local-first toolkit for chatting with, evaluating, and fine-tuning open LLMs.

`gene` is a native GUI (built with egui) plus a headless CLI over a shared Rust
engine. You chat with a local, uncensored, code-capable model; **edit its replies**
when they're wrong to build a correction dataset; **fine-tune** it (LoRA, DoRA, or
full) via MLX; **evaluate** models against a fixed prompt set; and keep every
training/eval **run** tracked on disk. It talks to any OpenAI-compatible backend,
so you can point chat, eval, and judging at different providers.

Everything runs locally by default. The model has no content filters because it's
an open-weight model you choose and run yourself.

## What it is

- **Chat** with a local model. You write the system prompt (no built-in
  personas); opt into a confirm-gated agentic shell with `agent.run_commands`.
  Streams token-by-token; reasoning models that emit `<think>…</think>` (or a
  `reasoning` field) get a collapsible thinking section.
- **Edit-to-correct dataset building.** When a reply is wrong, edit it; the
  correction is appended to the training dataset as the *ideal* answer, with the
  model's original output kept as provenance.
- **Fine-tune** the model on your dataset: LoRA, DoRA, or a full fine-tune, run
  through `mlx_lm` on Apple Silicon. Every step is a configurable command
  template, so the trainer/backend is pluggable without recompiling.
- **Evaluate & compare** models against an eval set with a pluggable grader
  (`exact`, `contains`, an LLM `judge`, or capture-only), fanned out
  concurrently; run the same set across several providers to rank them.
- **Experiment tracking.** Each training and eval run is recorded under
  `runs/<id>/` with a config snapshot, dataset provenance (content hash), the
  loss-curve metric series, and the raw log.
- **Multi-provider.** Named `[providers]` profiles (Ollama or any
  OpenAI-compatible server). `[roles].chat` picks the chat provider, `eval
  compare` runs across the providers you name, and an LLM judge uses
  `[roles].judge` — so chat, the model under eval, and the judge can differ.
- **GUI + headless CLI.** The default build ships the desktop window; a
  `--no-default-features` build is a lean CLI with no windowing dependencies, for
  scripts, CI, or ssh. Every CLI command supports `--json`.

## Prerequisites

`gene` **detects** most of these but never installs them — run `gene doctor` to
check (it probes the chat host, Python, and `mlx-lm`; not llama.cpp).

| For | Install |
| --- | --- |
| **Chatting** | [Ollama](https://ollama.com) (`ollama pull <tag>`), or any OpenAI-compatible server (vLLM, llama.cpp `server`, LM Studio, a hosted API) |
| **Fine-tuning** | `python3` + `pip install mlx-lm`, on **Apple Silicon** (arm64) |
| **Promote a fine-tune to Ollama** (optional) | a [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout for GGUF conversion |

A good default pairing keeps the chat host and the trainer aligned (same base in
both worlds) — e.g. a Llama-3.1-8B *abliterated* tag in Ollama with the matching
`mlx-community` base in `config.toml`. Llama bases have the smoothest fine-tune
round-trip; Qwen2.5-Coder *abliterated* is stronger at code but needs the
llama.cpp GGUF path to get back into Ollama.

> Fine-tuning an 8B model wants ~40–60 GB free disk and substantial unified
> memory. On ≤16 GB Macs, point `finetune.mlx_base` at a 3B (or quantized) base.

## Install

Prebuilt binaries for each tagged release (desktop builds for macOS and Windows,
a portable headless build for Linux) are attached to the
[GitHub Releases](https://github.com/matthew-demidoff/gene-bot/releases), each
with a `.sha256` to verify the download.

Or build from source:

```sh
cargo build --release                           # release binary (target/release/gene)
cargo run -p gene-ai                            # launch the GUI (default `gui` feature)
cargo build -p gene-ai --no-default-features    # headless CLI, no windowing deps
```

The package is `gene-ai`; the binary is `gene`.

## Quickstart

```sh
gene                                  # launch the desktop GUI (no subcommand)
gene setup                            # interactively write config.toml (model, prompt, …)
gene doctor                           # check chat + fine-tune prerequisites
gene chat -m "explain mmap in two sentences"
gene dataset stats                    # counts for the accumulated dataset
gene train --dry-run                  # print the exact subprocess commands, run nothing
gene eval run --set evals/smoke.json --grader contains
```

`--provider <name>` and `--json` are global (they work after any subcommand);
`--config <path>`, `--model <tag>`, and `--base-url <url>` go before the
subcommand. The full command reference is in [docs/cli.md](docs/cli.md).

## Configuration

A `config.toml` is created on first run under your platform config dir:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/dev.gene.gene/config.toml` |
| Linux | `~/.config/gene/config.toml` |
| Windows | `%APPDATA%\gene\gene\config\config.toml` |

`gene config path` prints the resolved path; `gene config show` prints the current
config (API keys redacted). It controls the model/endpoint, sampling parameters
(`temperature`, `max_tokens`, `top_p`, `top_k`, `min_p`, `repetition_penalty`,
`seed`, `stop`), the agent denylist + timeouts, the data/runs paths, the
`[providers]`/`[roles]` multi-provider mapping, and the fine-tune method and
command templates. See [docs/config.md](docs/config.md) for an annotated example.

## Safety

The agentic shell is **off by default** — set `agent.run_commands` (or tick "run
commands" in the GUI) to let the model propose `` ```run `` blocks. Approved
commands run with your full user privileges — there is no sandbox.
Confirm-before-run is on by default; **auto-run** is opt-in and per-session, and
the denylist (`rm -rf`, `sudo`, `mkfs`, `dd `, fork bombs, …) still forces a
manual confirm for dangerous patterns. Review commands before approving.

## Workspace layout

A Cargo workspace with two crates:

- `crates/gene-core/` — the UI-agnostic **engine**: `config`, `llm/` (streaming
  client + the incremental `<think>`/```run parser), `provider/`
  (OpenAI-compatible backends), `chat`, `tools/` (confirm-gated shell exec),
  `model/` + `persist` (conversations + dataset), `dataset/` (load/stats/dedup/
  split + format conversion), `eval/` (the eval harness + graders), `train`
  (MLX fine-tune orchestration), `runs/` (experiment tracking), and `doctor`.
- `crates/gene/` — the **binary** (`gene`): the headless CLI (`cli.rs`) plus the
  egui desktop frontend (`app.rs`) behind the default `gui` feature, so the
  engine builds with zero GUI dependencies.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build -p gene-ai --no-default-features     # the headless build must stay green too
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the green-gate, the workspace layout,
and how to add a provider kind, a dataset format, or an eval grader.
