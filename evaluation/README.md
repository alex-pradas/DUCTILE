# DUCTILE Evaluation

Evaluation pipeline for the DUCTILE agentic loads-processing + HPC-submission application.

## Setup

```bash
cd evaluation
uv sync
cp .env.example .env  # then edit with your keys
```

Required environment variables (see `.env.example`):

- `ANTHROPIC_API_KEY` — Claude models (`haiku`, `sonnet`, `opus`)
- `FIREWORKS_API_KEY` — Kimi K2.5 via Fireworks
- `LOGFIRE_TOKEN` — optional, exports traces to Pydantic Logfire

## Defaults

- **Solver:** `sonnet` (`anthropic:claude-sonnet-4-5`)
- **Judge:** `kimi` (`accounts/fireworks/models/kimi-k2p5`)

Override either with `--model` / `--judge`.

## Scenarios

Four stages of the same OEM v2 task (process loads + submit to HPC):

| ID | Stage | Seed | Evaluators |
|---|---|---|---|
| `loads_processing` | fresh | — | `deviations_handled`, `numerical_match` |
| `create_new_script` | loads done | `seeds/seed_after_loads.json` | `runscript_loadcases` |
| `call_hpc` | loads + script done | `seeds/seed_after_runscript.json` | `hpc_submit` |
| `end_to_end` | fresh | — | all four |

Seeded scenarios hand the agent a pre-recorded conversation up to a checkpoint and ask it to *continue*. The agent must infer the remaining work from the original combined instruction — we never say "edit runscript" or "submit to HPC" explicitly. That inference is the test.

## Recording seeds (one-time)

The seed JSONs are reproducible fixtures captured from an oracle (high-capability solver) running `end_to_end`. Re-record them only when design practices or `ductile-loads` change.

```bash
uv run python evaluator.py --record-seeds --model opus
```

This produces:

- `seeds/seed_after_loads.json`
- `seeds/seed_after_runscript.json`
- `seeds/runscript_modified.ans` (pre-staged for the `call_hpc` scenario)

If the oracle does not write `runscript.ans` via `write_file`, the run dumps the full message history to `seeds/_oracle_full.json` for manual splitting.

## Running evaluations

```bash
# single scenario, defaults (sonnet solver, kimi judge)
uv run python evaluator.py --scenario loads_processing -n 3

# specific solver and judge
uv run python evaluator.py --model kimi --judge opus --scenario end_to_end -n 1

# full matrix
uv run python evaluator.py --all -n 3

# solve-only (no evaluators)
uv run python evaluator.py --scenario create_new_script --solve-only

# JSON output for archival
uv run python evaluator.py --all -n 10 --output json > results/run.json
```

## Logfire — seeing the whole conversation

The evaluator instruments Pydantic AI and ships traces to the **`jmd-genai`** project (org `alex-pradas`, EU region) under `service.name = ductile-evaluator`.

Project URL: <https://logfire-eu.pydantic.dev/alex-pradas/jmd-genai>

One-time setup (already done for this checkout — credentials live in `evaluation/.logfire/logfire_credentials.json`, which is gitignored):

```bash
uv run logfire auth                                       # browser-based auth
uv run logfire projects use jmd-genai --org alex-pradas   # bind this dir to the project
```

Any evaluator run from this directory now pushes traces automatically (no `LOGFIRE_TOKEN` env var needed — the credentials file is picked up by `send_to_logfire="if-token-present"`).

In the UI:

1. Open <https://logfire-eu.pydantic.dev/alex-pradas/jmd-genai>.
2. Filter the Live view on `service.name = ductile-evaluator`.
3. Open a `ductile_evaluation_run` span. The attributes panel includes:
   - `scenario`, `stage`, `model`, `judge`, `work_dir`
   - `seeded_history` — the **full prior conversation** handed to the agent at the start of this run (user/assistant/tool turns from the oracle recording)
4. Child spans show the live activity: `agent run`, individual `chat` requests, every tool call (incl. MCP `folios` and `mock-gkn-hpc` calls), and each evaluator's judgment span.

Together, `seeded_history` + live child spans give you the entire conversation end-to-end.

## Token cost & prompt caching

The wire-level chat-completions API requires the full message history on every
turn, so the per-chat input-token count grows as the agent accumulates tool
calls — this is intrinsic to the protocol. What you *can* control is what
fraction of those tokens get re-prefilled each turn:

- **Fireworks (`kimi`, `kimi26`)** — prompt caching is automatic and on by
  default ([Fireworks docs](https://docs.fireworks.ai/guides/prompt-caching)).
  Kimi K2.6 charges $0.95 / 1M input vs $0.16 / 1M cached input — a ~83 %
  discount on the cached prefix.
- **Anthropic (`haiku`, `sonnet`, `opus`)** — caching is opt-in. The evaluator
  passes `AnthropicModelSettings(anthropic_cache_instructions=True,
  anthropic_cache_tool_definitions=True, anthropic_cache=True)` for the solver,
  and `anthropic_cache_instructions=True` for the LLMJudge. These cache the
  system prompt, the tool surface, and the message-history breakpoint — three
  of the four breakpoints Anthropic allows per request.

Per-run cache statistics are visible in two places:

- **Logfire** — under each `ductile_evaluation_run` span, look for a sibling
  `run_usage` event with `input_tokens`, `output_tokens`, `cached_tokens`, and
  `cache_hit_ratio`.
- **Terminal** — `--solve-only` prints a one-line summary:
  `Tokens: input=160,000  cached=148,000  output=3,400  cache_hit_ratio=93%`.
