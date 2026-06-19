# Examples

Runnable snippets for the things people most often want to copy.

## `evals/capitals.json` — an eval set

A small [eval set](../docs/cli.md#eval-run) mixing graders: `contains` and
`exact` for closed answers, and an LLM `judge` for the open-ended item.

```sh
# Score the active provider against the set (the judge item needs a judge — see below).
gene eval run --set examples/evals/capitals.json --grader contains

# Compare two named providers side by side (e.g. a base vs. a fine-tune).
gene eval compare --set examples/evals/capitals.json --providers local,finetuned --grader contains
```

Each run is recorded under `runs/<id>/` with a `results.jsonl`; inspect it with
`gene run show <id>`.

## `multi-provider.toml` — chat, eval, and judge on different backends

Copy the `[providers]` / `[roles]` blocks into your own config (`gene config
path` prints where it lives). With named providers configured:

```sh
gene --provider finetuned chat -m "ping"          # one-off override of the chat role
gene eval run --set examples/evals/capitals.json --grader judge   # judge item uses [roles].judge
```

> Keep real API keys in your private config.toml, never in a file you commit.
