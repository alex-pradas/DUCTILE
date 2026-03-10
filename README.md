# DUCTILE

**Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled engineering analysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18836517.svg)](https://doi.org/10.5281/zenodo.18836517)

## About

This repository contains the implementation, evaluation pipeline, and session transcripts for the DUCTILE approach to agentic LLM orchestration of engineering analysis, accompanying the paper (currently in review):

> A. Pradas-Gomez, A. Brahma, and O. Isaksson, "DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice," *Journal of Mechanical Design*, ASME, 2026.
> <!-- DOI: [10.1115/1.4XXXXXX](https://doi.org/10.1115/1.4XXXXXX) -->

DUCTILE separates *interpretation* (handled by an LLM) from *computation* (handled by verified tools), enabling agentic engineering analysis that is traceable, auditable, and adaptable to evolving product development contexts.

## Repository structure

```
.
├── agent/                     # The agentic application
│   ├── CLAUDE.md              # System prompt (agent configuration)
│   ├── .mcp.json              # MCP server config (folios + internal-tool-docs)
│   ├── documents/             # Design practice served via folios MCP
│   ├── docs/                  # Tool documentation served via internal-tool-docs MCP
│   ├── ref_documents/         # Task description (+ design practice PDF for engineer)
│   ├── previous_run/          # Reference outputs from previous analysis
│   └── inputs/                # OEM load deliveries (v2, v3 YAML files)
│
├── evaluation/                # Automated evaluation pipeline
│   ├── evaluator.py           # Pydantic AI + Logfire evaluator
│   ├── pyproject.toml         # Python dependencies
│   ├── .env.example           # API key template
│   └── results/               # Evaluation outputs (n=10 runs)
│
├── transcripts/               # Engineer session transcripts
│   ├── engineer_1.md          # Heavy delegation style
│   └── engineer_2.md          # Interactive checking style
│
├── CITATION.cff
├── .zenodo.json
└── LICENSE
```

## Reproducing the evaluation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management (`uvx` runs MCP servers and inline-script dependencies)
- API keys for Anthropic (Claude)
- PyPI packages (fetched automatically via `uvx`/script frontmatter):
  - [`ductile-loads`](https://pypi.org/project/ductile-loads/) — certified loads processing tool
  - [`folios`](https://pypi.org/project/folios/) — MCP server for design practice documents
  - [`internal-tool-docs-mcp`](https://pypi.org/project/internal-tool-docs-mcp/) — MCP server for tool API documentation

### Setup

```bash
cd evaluation
uv sync
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Quick test: single model, single scenario, no evaluation
uv run python evaluator.py --model sonnet --scenario v2 --solve-only

# Development evaluation (k=3, per paper Sec. 4.3)
uv run python evaluator.py --model sonnet --scenario v2 -n 3

# Deployment evaluation (k=10, per paper Sec. 4.3)
uv run python evaluator.py --all -n 10

# Export results as JSON
uv run python evaluator.py --all -n 10 --output json > results/results_n10.json
```

### Viewing traces in Logfire

All evaluation runs are instrumented with [Pydantic Logfire](https://logfire.pydantic.dev/). If you set `LOGFIRE_TOKEN` in `.env`, traces are exported to your Logfire project where you can inspect:

- Each model invocation (prompt, response, tokens, latency)
- Tool calls and their results
- Evaluation scores and judge reasoning

## The agent application

The `agent/` directory contains everything needed to run the agentic loads processing task:

1. **`CLAUDE.md`** — System prompt defining the agent's role, scope, and behavioral guidelines
2. **`.mcp.json`** — MCP server configuration for two servers: [folios](https://pypi.org/project/folios/) (design practice) and [internal-tool-docs-mcp](https://pypi.org/project/internal-tool-docs-mcp/) (tool API docs)
3. **`documents/`** — Design practice in folios Markdown format (versioned, chapter-level access via MCP)
4. **`docs/`** — Tool API documentation (`ductile-loads.txt`) served via internal-tool-docs MCP
5. **`ref_documents/`** — Task description PDF (+ design practice PDF for engineer reference)
6. **`previous_run/`** — Reference outputs from a previous analysis and the processing script
7. **`inputs/`** — Two OEM load delivery files in YAML format to be processed

To use with Claude Code, copy the `agent/` contents to a working directory and start a session. The agent accesses the design practice via folios MCP, queries tool documentation via internal-tool-docs MCP, and processes load deliveries using the certified [`ductile-loads`](https://pypi.org/project/ductile-loads/) package. All MCP servers and the loads tool are fetched automatically via `uvx` and script frontmatter; no manual installation is required.

## Citation

If you use this code, please cite:

```bibtex
@article{pradasgomez2026ductile,
  author  = {Pradas-G{\'o}mez, Alejandro and Brahma, Arindam and Isaksson, Ola},
  title   = {{DUCTILE}: Agentic {LLM} Orchestration of Engineering Analysis in Product Development Practice},
  journal = {Journal of Mechanical Design},
  year    = {2026},
  publisher = {ASME},
  note    = {DOI forthcoming}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
