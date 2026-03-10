# Evaluation Results

Results from the DUCTILE evaluation pipeline (see `../evaluator.py`).

## Latest Run: Opus 4.6 — Scenario v2 (n=10)

**Date**: 2026-03-10
**Agent model**: Claude Opus 4.6 (`claude-opus-4-6`)
**Judge model**: Claude Opus 4.6 (`claude-opus-4-6`)
**Scenario**: v2 — OEM loads delivery v2 processing

| Check | Pass Rate |
|-------|-----------|
| `deviations_handled` (LLM-as-a-judge) | 10/10 |
| `numerical_match` (deterministic) | 10/10 |
| **Overall (both must pass)** | **10/10 (100%)** |

Raw results: [`results_opus_v2_n10.json`](results_opus_v2_n10.json)

### Evaluation methodology

Each run:
1. Sets up an isolated working directory with input files
2. Starts a folios MCP server (design practice documents)
3. Runs a Pydantic AI agent with file I/O, code execution, and MCP tools
4. Judges the output with two complementary evaluators:
   - **LLM-as-a-judge**: Checks all 4 deviations handled (YAML→JSON, unit conversion, node renaming, 1.04 Fx correction) and that the certified `ductile-loads` tool was used
   - **Deterministic comparison**: Envelope extreme values compared against expert reference (Engineer 1's validated output)
5. A run passes only if **both** checks succeed

## How to reproduce

```bash
cd evaluation
uv sync
cp .env.example .env  # Add ANTHROPIC_API_KEY
uv run python evaluator.py --model opus --scenario v2 -n 10 --output json > results/results_opus_v2_n10.json
```

See the [main README](../README.md) for full instructions.
