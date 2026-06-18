# gene

A personal desktop AI assistant (native GUI, built with egui). Chat with a
local, uncensored, code-capable model; **edit its replies** when they're wrong
so it learns your style; let it **run shell commands** as an agent (with a
confirm step); and **fine-tune** it (real LoRA) on your corrected conversations.

Everything runs locally. The model has no content filters because it's an
open-weight model you choose and run yourself.

## How it works

```
you ─chat─►  Ollama (uncensored model)  ─stream─►  GUI window
                                                     │  edit a reply ──► dataset.jsonl
                                                     │  model emits ```run ──► confirm ──► shell ──► fed back
                                                     ▼
                                    fine-tune ──► MLX LoRA ──► fused model ──► serve / promote to Ollama
```

- **Modes** (top-bar dropdown): *assistant* (can run shell commands), *tech-guy*
  (talks/advises, never executes), *convo* (casual chat). Only assistant mode
  parses ```` ```run ```` blocks as commands.
- **Chat** streams token-by-token; the view sticks to the bottom as it grows.
  Reasoning models that emit `<think>…</think>` (or a `reasoning` field) get a
  collapsible *thinking* section.
- **Edit to teach:** click **edit** on any reply, type a correction (or *Load
  original* to edit from it), **Save → train**. The corrected text replaces it
  and is appended to the training dataset as the *ideal* answer (the model's
  original output is kept as provenance).
- **Agent:** when the model wants to run a command, a confirm window shows it —
  **Run / Deny**, and you can edit it first. Output is fed back so it can
  continue. The **auto-run** checkbox runs approved commands without asking;
  denylisted patterns always prompt.
- **Fine-tune:** the **fine-tune** button exports the dataset to MLX chat-format
  JSONL, runs `mlx_lm.lora` + `mlx_lm.fuse` (progress in a log window), then
  serves the result (or promotes it to Ollama as GGUF). Every command is a
  configurable template.
- **Past chats:** the left sidebar lists saved conversations — click to load;
  **new** starts a fresh one.

## Prerequisites

The app **detects** these but never installs them — run `gene doctor` to check.

| For | Install |
| --- | --- |
| **Chatting** | [Ollama](https://ollama.com), then `ollama pull <tag>` for an uncensored model |
| **Fine-tuning** (Apple Silicon) | `python3` + `pip install mlx-lm` |
| **Promote to Ollama** (optional) | a [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout for GGUF conversion |

A good default model pairing keeps Ollama and the trainer aligned (same base in
both worlds) — e.g. a Llama-3.1-8B *abliterated* tag in Ollama with the matching
`mlx-community` base in `config.toml`. Llama bases have the smoothest fine-tune
round-trip; Qwen2.5-Coder *abliterated* is stronger at code but needs the
llama.cpp GGUF path to get back into Ollama.

> Fine-tuning an 8B model wants ~40–60 GB free disk and substantial unified
> memory. On ≤16 GB Macs, point `finetune.mlx_base` at a 3B (or quantized) base.

## Usage

```sh
gene                              # launch the GUI window
gene doctor                       # check prerequisites (CLI, no window)
gene --model qwen2.5-coder:7b     # override the model for one run
```

In the window: type and press **Enter** to send (**Shift+Enter** for a newline),
or click **send**. Use the top-bar dropdown to switch mode, the **model** button
to pick a local model, **stop** to halt a response, and **settings** to edit the
system prompts / endpoint (and save them to `config.toml`).

Fine-tuning trains on the accumulated dataset and refuses below `min_examples`
(default 50) — edit/accept more replies first to build it up.

## Configuration

A `config.toml` is created on first run under your platform config dir
(`~/Library/Application Support/dev.gene.gene/` on macOS). It controls the
model/endpoint, generation params, the agent denylist + timeouts, and the
fine-tune command templates and deploy mode (`mlx_server` vs `ollama_gguf`).
`gene doctor` prints the resolved paths.

## Safety

Approved commands run with your full user privileges — there is no sandbox.
Confirm-before-run is on by default; **auto-run** is opt-in and per-session, and
the denylist still forces a manual confirm for dangerous patterns. Review
commands before approving.

## Development

```sh
cargo test --workspace                          # engine + parser tests
cargo clippy --workspace
cargo run -p gene-ai                            # launch the GUI (default features)
cargo build -p gene-ai --no-default-features    # headless CLI, no windowing deps
```

Layout: a Cargo workspace. `crates/gene-core/` is the UI-agnostic engine —
`config`, `llm/` (streaming client + the incremental `<think>`/```run parser),
`provider/` (OpenAI-compatible backends), `tools/` (confirm-gated shell exec),
`model/` + `persist.rs` (conversations + dataset), `train.rs` (MLX LoRA
orchestration), and `runs/` (experiment tracking). `crates/gene/` is the binary:
the CLI plus the egui frontend (`app.rs`) behind a default `gui` feature, so the
engine builds with zero GUI dependencies.
