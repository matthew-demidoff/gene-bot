# CLI reference

`gene` is the binary (package `gene-ai`). With no subcommand it launches the
desktop GUI (in a build that includes the default `gui` feature); a
`--no-default-features` build is CLI-only and prints an error if you run it with
no subcommand.

Every subcommand below is available in both builds — the CLI carries no GUI
dependencies.

## Global flags

These precede the subcommand (and `--provider`/`--json` also work after it):

| Flag | Effect |
| --- | --- |
| `--config <path>` | Use this config file instead of the platform default. |
| `--model <tag>` | Override the chat model tag (legacy single-endpoint setups). |
| `--base_url <url>` | Override the chat endpoint base URL (legacy single-endpoint setups). |
| `--provider <name>` | Use a named `[providers]` profile for this run (overrides `[roles].chat`). |
| `--json` | Emit machine-readable JSON instead of human-readable text. Errors are emitted as `{"error": "..."}` on stderr with a non-zero exit. |

```sh
gene --config ./gene.toml --provider remote chat -m "hi"
gene --json doctor
```

---

## chat

Chat with the model: one-shot via `--message`, or read the prompt from stdin.

| Flag | Default | Effect |
| --- | --- | --- |
| `-m`, `--message <text>` | (stdin) | The prompt. If omitted, read from stdin. |
| `--mode <persona>` | `tech` | `assistant` (parses ```run blocks as commands), `tech` (advises, never executes), `convo` (casual). |
| `--temperature <f>` | config | Override sampling temperature for this run. |
| `--max-tokens <n>` | config | Override the max-tokens cap. |
| `--seed <n>` | config | Override the sampling seed. |

```sh
gene chat -m "write a bash one-liner to find large files"
echo "summarize this" | gene chat --mode convo
```

> Note: the CLI `chat` does not execute ```run blocks — in `assistant` mode it
> prints suggested commands. Command execution (confirm-gated) is a GUI feature.

`--json` shape (fields present only when non-empty):

```json
{
  "model": "huihui_ai/llama3.1-abliterated:latest",
  "answer": "…",
  "thinking": "…",
  "commands": ["ls -la"],
  "error": "…"
}
```

---

## models

List the model ids the active provider advertises (Ollama `/api/tags`,
OpenAI-compatible `/v1/models`).

```sh
gene models
gene --provider remote --json models   # -> ["gpt-x", "…"]
```

---

## run

Inspect tracked training/eval runs in the run store. Listing first reconciles any
run left `running` by a dead process to `aborted`.

```sh
gene run list                 # all runs, newest first
gene run show <id>            # one run by id
```

`run list` text columns: `id  kind  status  base_model  <summary k=v …>`.
`--json` on `list` emits the array of run records; on `show`, the single record
(id, kind, status, timestamps, base model, hyperparams, dataset ref, summary
metrics, error).

---

## config

```sh
gene config path     # print the config file path
gene config show     # print the current config (API keys redacted)
```

`config show --json` emits the config as compact JSON (pretty without `--json`).
API keys — top-level and every `[providers]` key — are replaced with
`<redacted>` so the output is safe to paste.

---

## dataset

Manage the training dataset (JSONL of `{messages, meta}` examples). Every
subcommand accepts `--file <path>` to target a dataset other than the configured
one.

### dataset stats

```sh
gene dataset stats
gene dataset stats --file other.jsonl --json
```

`--json` shape:

```json
{
  "path": "/…/dataset.jsonl",
  "total": 312,
  "edited": 47,
  "conversations": 28,
  "by_source": { "edit": 47, "accept": 265 }
}
```

### dataset dedup

Remove examples whose normalized message content duplicates an earlier one
(keeps the first). `--dry-run` reports without writing.

```sh
gene dataset dedup --dry-run
gene dataset dedup            # -> {"removed":3,"remaining":309,"written":true}
```

### dataset import

Read examples from another format and append (or `--replace`).

| Flag | Default | Effect |
| --- | --- | --- |
| `--from <path>` | (required) | Source file. |
| `--format <fmt>` | `openai` | `gene` \| `mlx` \| `openai` \| `sharegpt`. |
| `--replace` | off | Replace the dataset instead of appending (refused if the import is empty). |

```sh
gene dataset import --from sharegpt_dump.jsonl --format sharegpt
gene dataset import --from openai.jsonl --replace --json
```

### dataset export

Write the dataset to another format.

| Flag | Default | Effect |
| --- | --- | --- |
| `--to <path>` | (required) | Destination file. |
| `--format <fmt>` | `mlx` | `gene` \| `mlx` \| `openai` \| `sharegpt`. |

`mlx`/`openai` emit `{"messages":[…]}` per line (provenance dropped); `sharegpt`
emits `{"conversations":[{"from","value"}, …]}`; `gene` keeps the native form.

```sh
gene dataset export --to train.jsonl --format mlx
```

### dataset snapshot

Write an immutable, content-addressed copy under `snapshots/<hash>.jsonl` next to
the dataset — identical datasets snapshot to the same file.

```sh
gene dataset snapshot     # -> snapshot 9f1c… (312 examples) -> /…/snapshots/9f1c….jsonl
```

---

## train

Run a fine-tune (LoRA / DoRA / full) through `mlx_lm` and record it as a tracked
run. Overrides apply on top of the `[finetune]` config for this run only.

| Flag | Default | Effect |
| --- | --- | --- |
| `--method <m>` | config | `lora` \| `dora` \| `full`. `full` trains all weights and skips the fuse step. |
| `--iters <n>` | config | Override the iteration count. |
| `--learning-rate <lr>` | config | Override the learning rate (string, e.g. `1e-5`). |
| `--dry-run` | off | Print the resolved subprocess commands in order and run nothing. |

```sh
gene train --dry-run                       # preview the train/fuse/deploy commands
gene train --method dora --iters 800       # actually train
```

A real run streams progress logs and `[iter N] …` metric lines on **stderr**, so
stdout carries only the final result line (and the served URL/model if deployed).
Training refuses below `finetune.min_examples`.

`--dry-run --json` shape:

```json
{
  "method": "lora",
  "commands": [
    { "label": "train", "command": "python3 -m mlx_lm.lora --model … --fine-tune-type lora …" },
    { "label": "fuse",  "command": "python3 -m mlx_lm.fuse …" },
    { "label": "serve", "command": "python3 -m mlx_lm.server …" }
  ]
}
```

Completed-run `--json` shape:

```json
{ "ok": true, "message": "fine-tune complete", "served_url": "http://localhost:8080/v1/chat/completions", "served_model": "/…/fused" }
```

---

## eval run

Run an eval set against the active provider, grade the outputs, and record an
`Eval` run.

| Flag | Default | Effect |
| --- | --- | --- |
| `--set <path>` | (required) | Eval-set JSON file. |
| `--grader <g>` | `none` | `none` (capture only) \| `exact` \| `contains`. Per-item graders in the set override this. |
| `--concurrency <n>` | `4` | Maximum concurrent inference requests. |

```sh
gene eval run --set evals/smoke.json --grader contains --concurrency 8
```

Eval-set JSON (`temperature` defaults to `0.0` for comparability; `reference` and
per-item `grader` are optional):

```json
{
  "name": "smoke",
  "system_prompt": "You are a terse calculator.",
  "temperature": 0.0,
  "items": [
    { "id": "add",  "prompt": "2+2?",        "reference": "4", "grader": "contains" },
    { "id": "caps", "prompt": "capital of France?", "reference": "Paris" },
    { "id": "open", "prompt": "tell me a joke" }
  ]
}
```

`--json` emits the full report (per-item outputs, `passed`, `mean_score`,
`scored`/`passed` counts) plus the `run_id` it was saved under. The eval also
writes a `results.jsonl` artifact into the run directory.

---

## doctor

Report whether the chat + fine-tuning prerequisites are installed and the active
chat provider is reachable. Exits non-zero if any check fails.

```sh
gene doctor
gene --json doctor
```

Checks: Apple Silicon (arm64), `ollama`, the active chat provider's reachability,
`python3`, and `mlx-lm`. The report also prints the resolved chat model/endpoint,
the MLX base, and the dataset path.
