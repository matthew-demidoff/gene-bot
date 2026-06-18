# Contributing

## Toolchain

Stable Rust (edition 2021). No nightly features. Install via
[rustup](https://rustup.rs); `cargo` is all you need to build and test.

The GUI build pulls in `eframe`/`egui`; the headless build (below) needs none of
that, which is why CI and most engine work can run without windowing libraries.

## Green gate

A change is done when all of these pass:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build -p gene-ai --no-default-features    # the headless CLI must stay buildable
```

The last line matters: the engine and CLI must compile with **zero** GUI
dependencies. It's easy to break by referencing something behind the `gui`
feature from shared code — the `--no-default-features` build catches it.

## Workspace layout

A two-crate Cargo workspace:

- **`crates/gene-core/`** — the UI-agnostic engine. Frontends share all logic
  through it.
  - `config.rs` — the `Config` struct, defaults, path resolution, provider/role
    resolution.
  - `llm/` — the streaming chat client and the incremental `<think>` / ```run
    parser.
  - `provider/` — OpenAI-compatible backends and model discovery (`ProviderKind`).
  - `chat.rs` — the persona `Mode` and conversation → wire-message conversion.
  - `tools/` — confirm-gated shell execution.
  - `model/` + `persist.rs` — conversation and dataset types and storage.
  - `dataset/` — load/stats/dedup/split (`mod.rs`) and format conversion
    (`format.rs`).
  - `eval/` — the eval harness, `EvalSet`/`EvalReport`, and the `Grader` enum.
  - `train.rs` — MLX fine-tune orchestration (template fill, subprocess runner,
    metric parsing).
  - `runs/` — experiment tracking (`Run`, `RunStore`, `DatasetRef`).
  - `doctor.rs` — prerequisite checks.
- **`crates/gene/`** — the binary (package `gene-ai`, binary `gene`).
  - `main.rs` — the clap CLI surface (the source of truth for commands/flags).
  - `cli.rs` — the headless command implementations (always compiled, no GUI
    deps).
  - `app.rs` — the egui desktop frontend, behind the default `gui` feature.

Keep new engine logic in `gene-core` and wire it into both `cli.rs` and `app.rs`,
rather than putting behavior in the binary. The CLI is the place to add a
`--json` contract; the GUI consumes the same core functions.

## Extending the pluggable points

Three small enums are the extension seams. Adding a variant means updating the
enum, its `parse`, and the per-variant logic — the compiler's exhaustiveness
checks will point you at the rest.

- **A provider kind** — `ProviderKind` in `crates/gene-core/src/provider/mod.rs`.
  Add the variant (with its `#[serde]` rename), then handle it in `discovery_url`
  and `list_models` (the chat path is OpenAI-compatible for every kind, so it
  needs nothing there). Surface it as a `kind` value in `[providers]`.

- **A dataset format** — `Format` in
  `crates/gene-core/src/dataset/format.rs`. Add the variant and its `parse`
  string, then handle it in `import` and `export`. It becomes a valid
  `--format` value for `gene dataset import`/`export`.

- **An eval grader** — `Grader` in `crates/gene-core/src/eval/mod.rs`. Add the
  variant and its `parse` string, then the scoring branch in `grade`. It becomes
  a valid `--grader` value and a valid per-item `grader` in an eval set.

When you change the CLI surface (`main.rs`) or any `--json` shape, update
`docs/cli.md`; when you change `Config`, update `docs/config.md`.

## MLX training is Apple-Silicon-only

The fine-tune pipeline shells out to `mlx_lm` (LoRA/DoRA/full) and, optionally,
llama.cpp for GGUF conversion. MLX runs on Apple Silicon (arm64) only, so the
end-to-end training path is **validated locally, not in CI** — CI builds and runs
the Rust tests but does not invoke `mlx_lm`. The Rust side is testable without
it: `gene train --dry-run` and the `planned_commands`/metric-parsing tests cover
the orchestration logic without running a trainer. If you change the training
pipeline, run a real fine-tune on an Apple Silicon machine before merging.
