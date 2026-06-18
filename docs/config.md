# Configuration

`gene` loads a single `config.toml`. It's created with defaults on first run and
lives in the platform config dir:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/dev.gene.gene/config.toml` |
| Linux | `~/.config/gene/config.toml` |
| Windows | `%APPDATA%\gene\gene\config\config.toml` |

`gene config path` prints the resolved path; `gene config show` prints the loaded
config with API keys redacted. Pass `--config <path>` to use a different file.

Anything you omit falls back to the built-in default, so a minimal config only
needs the fields you want to change.

## Top-level fields

```toml
# Legacy single-endpoint fields. Used when [providers] is empty (or as the
# fallback when a role names a missing provider).
model    = "huihui_ai/llama3.1-abliterated:latest"   # model tag served by the chat endpoint
base_url = "http://localhost:11434/v1/chat/completions"
api_key  = "ollama"   # Ollama ignores it; any non-empty string is fine

# System prompts per persona (see `--mode`).
system_prompt       = "…"   # assistant mode: instructs the model on the ```run convention
tech_system_prompt  = "…"   # tech mode: advises, never emits ```run
convo_system_prompt = "…"   # convo mode: casual conversation
```

## [generation] — sampling

```toml
[generation]
temperature = 0.7
max_tokens  = 4096
# All of the following are optional — omit to leave unset:
# top_p              = 0.95
# top_k              = 40
# min_p              = 0.05
# repetition_penalty = 1.1
# seed               = 42
# stop               = ["</s>", "###"]
```

`temperature`/`max_tokens` can be overridden per `gene chat` call with
`--temperature` / `--max-tokens` / `--seed`.

## [agent] — confirm-gated shell execution

Governs the assistant persona's command execution (a GUI feature; the CLI prints
suggested commands rather than running them).

```toml
[agent]
auto_run          = false   # per-session default for auto-running approved commands
native_tools      = false   # use OpenAI function-calling instead of the ```run convention
max_tool_rounds   = 8       # cap on tool-call rounds in one user turn
exec_timeout_secs = 30      # per-command wall-clock timeout
# Commands matching any of these substrings always require a manual confirm,
# even with auto_run on:
denylist = ["rm -rf", "sudo", "mkfs", "dd ", ":(){", "> /dev/",
            "shutdown", "diskutil erase", "mv /", "chmod -R 000"]
```

## [ui]

```toml
[ui]
think_collapsed_default = true   # start the <think> section collapsed
```

## [paths] — on-disk locations

Each field is an override; an empty string resolves to the default under the
platform **data** dir (e.g. `~/Library/Application Support/dev.gene.gene/` on
macOS).

```toml
[paths]
conversations_dir = ""   # "" => <data_dir>/conversations
dataset_path      = ""   # "" => <data_dir>/dataset.jsonl
log_path          = ""   # "" => <data_dir>/gene.log
work_dir          = ""   # "" => <data_dir>/finetune  (training scratch)
runs_dir          = ""   # "" => <data_dir>/runs      (experiment tracking)
```

## [finetune] — the fine-tune pipeline

The trainer is driven by command templates with `{placeholder}` slots, so you can
swap the trainer/backend without recompiling.

```toml
[finetune]
mlx_base        = "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"  # HF repo id or local path
min_examples    = 50        # train refuses below this
valid_fraction  = 0.1       # fraction held out for validation (split by conversation)
deploy_mode     = "mlx_server"   # "mlx_server" (serve fused model) or "ollama_gguf" (convert + ollama create)
mlx_server_port = 8080      # port for mlx_lm.server in mlx_server mode

iters         = 600
batch         = 1
layers        = 16
learning_rate = "1e-5"      # string (passed through verbatim)
method        = "lora"      # "lora" | "dora" | "full"; "full" trains all weights and skips fuse
extra_args    = ""          # appended to the train command via {extra_args}

# Command templates (defaults shown abbreviated):
train_command         = "python3 -m mlx_lm.lora --model {base} --train --data {data} --adapter-path {adapters} --fine-tune-type {fine_tune_type} --num-layers {layers} --batch-size {batch} --iters {iters} --learning-rate {lr} --mask-prompt --grad-checkpoint --steps-per-report 10 --save-every 100 {extra_args}"
fuse_command          = "python3 -m mlx_lm.fuse --model {base} --adapter-path {adapters} --save-path {fused}"
mlx_server_command    = "python3 -m mlx_lm.server --model {fused} --port {port}"
gguf_convert_command  = "python3 {llama_cpp}/convert_hf_to_gguf.py {fused} --outfile {fused}/model-f16.gguf --outtype f16"
ollama_create_command = "ollama create {tag} -f {modelfile}"
llama_cpp_dir         = ""                       # llama.cpp checkout (required for ollama_gguf)
ollama_tag            = "gene-assistant:latest"  # tag created when promoting (ollama_gguf)
```

### Template placeholders

| Placeholder | Filled with |
| --- | --- |
| `{base}` | `finetune.mlx_base` |
| `{data}` | the prepared data dir (`train.jsonl` / `valid.jsonl`) |
| `{adapters}` | the adapter output dir |
| `{fused}` | the fused-model dir (the deployed model; for `full`, the adapter dir) |
| `{iters}` | `finetune.iters` |
| `{batch}` | `finetune.batch` |
| `{layers}` | `finetune.layers` |
| `{lr}` | `finetune.learning_rate` |
| `{fine_tune_type}` | `finetune.method` (`lora` / `dora` / `full`) |
| `{extra_args}` | `finetune.extra_args` |
| `{port}` | `finetune.mlx_server_port` |
| `{llama_cpp}` | `finetune.llama_cpp_dir` |
| `{tag}` | `finetune.ollama_tag` |
| `{modelfile}` | the generated `Modelfile` path (deploy time, `ollama_gguf`) |

`method` only takes effect if the `train_command` carries the `{fine_tune_type}`
placeholder — a pre-0.2 template that hardcodes `--fine-tune-type` is refused for
`dora`/`full` rather than silently training LoRA. `full` skips the fuse step (it
produces complete weights, not adapters). `gene train --dry-run` prints the exact
resolved commands without running anything.

## [providers] and [roles] — multi-provider

`[providers]` defines named inference endpoints; `[roles]` maps each activity
(chat / eval / judge) to one. When `[providers]` is empty, `gene` uses the
top-level `model` / `base_url` / `api_key` instead.

```toml
[providers.local]
kind     = "ollama"                                       # "ollama" | "openai_compat"
base_url = "http://localhost:11434/v1/chat/completions"
api_key  = "ollama"
model    = "huihui_ai/llama3.1-abliterated:latest"

[providers.remote]
kind     = "openai_compat"
base_url = "https://api.example.com/v1/chat/completions"
api_key  = "env:OPENAI_API_KEY"                           # read from the environment, not stored in plaintext
model    = "gpt-x"

[roles]
chat  = "local"    # interactive chat / `gene chat`
eval  = "local"    # the model under eval
judge = "remote"   # a stronger model for grading (when used)
```

- `kind` selects model discovery: `ollama` uses `/api/tags`, `openai_compat`
  (aliases: `openai`, `open_ai_compat`) uses `/v1/models`. The chat path itself
  is OpenAI-compatible for both, so this covers Ollama, vLLM, llama.cpp `server`,
  LM Studio, and hosted APIs.
- `api_key` accepts an `env:VAR` form so secrets stay out of the file. `gene
  config show` redacts keys regardless.
- An unset role falls back to `chat`, then to the first configured provider, then
  to the top-level fields. A `[roles].chat` naming a provider that doesn't exist
  is reported as an error (not silently fallen back) by `gene chat`/`doctor`.
- `--provider <name>` on the command line overrides `[roles].chat` for one run.

## On-disk layout

Under the data dir (or your `[paths]` overrides):

```
<data_dir>/
  config.toml? (config lives in the config dir, not here)
  conversations/             saved chats
  dataset.jsonl              the training dataset ({messages, meta} per line)
  snapshots/<hash>.jsonl     content-addressed dataset snapshots
  finetune/                  training scratch
    data/{train,valid}.jsonl prepared MLX chat-format split
    adapters/                LoRA/DoRA adapters
    fused/                   the fused model (and model-f16.gguf for ollama_gguf)
    Modelfile                generated for the ollama_gguf deploy
  runs/<id>/                 one tracked run per dir (id like 20260618T142233-a1b2c3)
    run.json                 record: config snapshot, dataset ref, status, summary
    metrics.jsonl            append-only metric series (train_loss, val_loss, …)
    run.log                  raw subprocess log
    results.jsonl            (eval runs) per-item outputs
  gene.log                   app log
```
