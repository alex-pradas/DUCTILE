#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pydantic-ai",
#     "pydantic-ai-slim[evals]",
#     "anthropic",
#     "openai",
#     "python-dotenv",
#     "logfire",
#     "pyyaml",
#     "numpy",
#     "ductile-loads[all]",
# ]
# ///

"""
Evaluation pipeline for the DUCTILE agentic loads processing application.

Adapts the Pydantic Evals framework to assess whether an LLM agent can
correctly process OEM load deliveries using the certified ductile-loads tool
and the design practice methodology.

The evaluator:
1. Sets up an isolated working directory with input files
2. Runs a Pydantic AI agent with file and code execution tools
3. Judges the output against acceptance criteria using:
   - LLM-as-a-judge (Opus 4.6): checks all 4 deviations handled correctly
   - Deterministic comparison: envelope values against expert reference
4. A run passes only if BOTH checks succeed

Usage:
    # Solve only (quick test, no evaluation)
    uv run python evaluator.py --model sonnet --scenario v2 --solve-only

    # Evaluate scenario (n=3 runs, default)
    uv run python evaluator.py --model sonnet --scenario v2

    # Full matrix (all models x all scenarios)
    uv run python evaluator.py --all -n 10

    # JSON output for archival
    uv run python evaluator.py --all -n 10 --output json > results/results.json
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.fireworks import FireworksProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, LLMJudge

load_dotenv()

# Configure Logfire
logfire.configure(service_name="ductile-evaluator", send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

LOGFIRE_PROJECT_URL = "https://logfire-eu.pydantic.dev/alex-pradas/jmd-genai"
_LOGFIRE_CREDS = Path(__file__).parent / ".logfire" / "logfire_credentials.json"
if os.getenv("LOGFIRE_TOKEN") or _LOGFIRE_CREDS.exists():
    print(
        f"Logfire enabled — traces shipping to {LOGFIRE_PROJECT_URL} "
        "(service.name=ductile-evaluator).",
        flush=True,
    )

logging.basicConfig(
    level=logging.INFO,
    handlers=[logfire.LogfireLoggingHandler(fallback=logging.StreamHandler())],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

def _kimi_model() -> OpenAIChatModel:
    """Kimi K2.5 served via the Fireworks OpenAI-compatible API.

    Reads FIREWORKS_API_KEY from the environment.
    """
    return OpenAIChatModel(
        "accounts/fireworks/models/kimi-k2p5",
        provider=FireworksProvider(),
    )


def _kimi26_model() -> OpenAIChatModel:
    """Kimi K2.6 served via the Fireworks OpenAI-compatible API.

    Reads FIREWORKS_API_KEY from the environment.
    """
    return OpenAIChatModel(
        "accounts/fireworks/models/kimi-k2p6",
        provider=FireworksProvider(),
    )


ModelSpec = str | Callable[[], Model]

MODELS: dict[str, ModelSpec] = {
    "haiku": "anthropic:claude-3-5-haiku-latest",
    "sonnet": "anthropic:claude-sonnet-4-5",
    "opus": "anthropic:claude-opus-4-6",
    "kimi": _kimi_model,
    "kimi26": _kimi26_model,
}

JUDGES: dict[str, ModelSpec] = {
    "opus": "anthropic:claude-opus-4-6",
    "kimi": _kimi_model,
    "kimi26": _kimi26_model,
}

DEFAULT_SOLVER = "sonnet"
DEFAULT_JUDGE = "kimi"


def _resolve_model(spec: ModelSpec) -> str | Model:
    """Resolve a MODELS / JUDGES entry to either a model-id string or a Model instance."""
    return spec() if callable(spec) else spec


# Keys whose provider is Anthropic — drives provider-specific cache settings.
ANTHROPIC_KEYS: set[str] = {"haiku", "sonnet", "opus"}


def _agent_settings(model_name: str) -> ModelSettings:
    """ModelSettings for the solver Agent.

    Enables Anthropic prompt caching when the chosen solver is a Claude model
    (system prompt + tool surface + final-message block, three of the four
    available cache breakpoints). Fireworks/Kimi caches automatically on the
    server side, so it gets vanilla settings.
    """
    if model_name in ANTHROPIC_KEYS:
        return AnthropicModelSettings(
            temperature=0.0,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache=True,
        )
    return ModelSettings(temperature=0.0)


def _judge_settings(judge_id: str) -> ModelSettings:
    """ModelSettings for an LLMJudge.

    The judge has no tools and a static rubric, so caching instructions is the
    only useful knob. Fireworks judges (kimi, kimi26) get vanilla settings.
    """
    if judge_id in ANTHROPIC_KEYS:
        return AnthropicModelSettings(
            temperature=0.0,
            anthropic_cache_instructions=True,
        )
    return ModelSettings(temperature=0.0)

# Paths relative to this file
AGENT_DIR = Path(__file__).parent.parent / "agent"
SYSTEM_PROMPT = (AGENT_DIR / "CLAUDE.md").read_text()

# Expert reference: Engineer 1's validated output for deterministic comparison
EXPERT_REFERENCE_PATH = (
    Path(__file__).parent.parent / "sessions" / "engineer_1" / "envelope_extremes.json"
)
EXPERT_REFERENCE: dict = json.loads(EXPERT_REFERENCE_PATH.read_text())


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

Stage = Literal["loads_processing", "create_new_script", "call_hpc", "end_to_end"]


@dataclass
class Scenario:
    """A staged evaluation scenario for the OEM v2 loads delivery + HPC submission.

    Every scenario shares the same OEM input file (`OEM_loads_v2.yaml`) and the
    same combined initial user instruction (loads processing + HPC submit).
    Scenarios differ in how much of that work is already done in the seeded
    `message_history` handed to the agent at run time. The agent is expected to
    infer the remaining work from the conversation context.
    """
    id: str
    title: str
    stage: Stage
    seed_file: str | None  # JSON in evaluation/seeds/, or None for fresh runs
    expected_n_lc: int  # ground truth for runscript judge
    expected_lc_ids: list[str]  # loadcase IDs the runscript must reference
    input_file: str = "OEM_loads_v2.yaml"


# v2 envelope contains 6 representative load cases (engineer_1 baseline)
V2_LC_IDS = ["2", "20", "34", "61", "92", "99"]

SCENARIOS: dict[str, Scenario] = {
    "loads_processing": Scenario(
        id="loads_processing",
        title="Loads Processing (no prior context)",
        stage="loads_processing",
        seed_file=None,
        expected_n_lc=len(V2_LC_IDS),
        expected_lc_ids=V2_LC_IDS,
    ),
    "create_new_script": Scenario(
        id="create_new_script",
        title="Create new runscript.ans (loads already done in seed)",
        stage="create_new_script",
        seed_file="seed_after_loads.json",
        expected_n_lc=len(V2_LC_IDS),
        expected_lc_ids=V2_LC_IDS,
    ),
    "call_hpc": Scenario(
        id="call_hpc",
        title="Submit to HPC (loads + runscript already done in seed)",
        stage="call_hpc",
        seed_file="seed_after_runscript.json",
        expected_n_lc=len(V2_LC_IDS),
        expected_lc_ids=V2_LC_IDS,
    ),
    "end_to_end": Scenario(
        id="end_to_end",
        title="End-to-end (loads + runscript + HPC, no seed)",
        stage="end_to_end",
        seed_file=None,
        expected_n_lc=len(V2_LC_IDS),
        expected_lc_ids=V2_LC_IDS,
    ),
}

SEEDS_DIR = Path(__file__).parent / "seeds"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Solution(BaseModel):
    """Output from an evaluation run — agent text plus extracted file data."""
    agent_output: str = Field(description="The agent's final text response")
    scripts_executed: list[str] = Field(
        default_factory=list,
        description="All Python scripts executed by the agent via run_python, in order",
    )
    envelope_extremes: dict | None = Field(
        default=None,
        description="Parsed envelope_extremes.json from the agent's working directory",
    )
    files_created: list[str] = Field(
        default_factory=list,
        description="Files created in the working directory after the agent ran",
    )
    messages: list[dict] = Field(
        default_factory=list,
        description="Full message_history (seed + live turns) serialized for Logfire/JSON archival",
    )
    runscript_content: str | None = Field(
        default=None,
        description="Final content of runscript.ans in the working directory, if present",
    )
    hpc_tool_calls: list[dict] = Field(
        default_factory=list,
        description="Args of every submit_ansys_run call made by the agent, in order",
    )
    input_tokens: int = Field(
        default=0,
        description="Total input tokens billed across all chat turns in this run",
    )
    output_tokens: int = Field(
        default=0,
        description="Total output tokens billed across all chat turns in this run",
    )
    cached_tokens: int = Field(
        default=0,
        description=(
            "Input tokens served from the provider's prompt cache "
            "(Anthropic: cache_read; Fireworks/Kimi: cached_prompt_tokens). "
            "input_tokens minus cached_tokens is what we pay full prefill for."
        ),
    )


@dataclass
class TaskInput:
    """Input to an evaluation case."""
    scenario_id: str


def _extract_hpc_calls(messages: list[ModelMessage]) -> list[dict]:
    """Find every submit_ansys_run tool call in the conversation."""
    calls: list[dict] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and part.tool_name == "submit_ansys_run":
                args = part.args
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append(dict(args) if isinstance(args, dict) else {"_raw": args})
    return calls


def _messages_to_dicts(messages: list[ModelMessage]) -> list[dict]:
    """Round-trip messages through the official adapter to a JSON-able list of dicts."""
    if not messages:
        return []
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


def _usage_totals(result) -> tuple[int, int, int]:
    """Extract (input_tokens, output_tokens, cached_tokens) from a RunResult.

    Field names vary across pydantic-ai versions and providers:
      - Anthropic: cache_read_tokens (+ cache_creation_tokens, paid full price)
      - Fireworks/OpenAI-compat: cached_tokens or details.cached_tokens
    We probe several known attribute names and fall back to 0 silently.
    """
    try:
        usage = result.usage()
    except Exception:
        return 0, 0, 0
    if usage is None:
        return 0, 0, 0

    inp = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "request_tokens", None)
        or 0
    )
    out = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "response_tokens", None)
        or 0
    )
    cached = (
        getattr(usage, "cache_read_tokens", None)
        or getattr(usage, "cached_tokens", None)
        or 0
    )
    # Fallback: provider-specific bag of fields
    if not cached:
        details = getattr(usage, "details", None) or {}
        if isinstance(details, dict):
            cached = (
                details.get("cache_read_input_tokens")
                or details.get("cached_tokens")
                or details.get("cached_prompt_tokens")
                or 0
            )
    return int(inp), int(out), int(cached or 0)


# ---------------------------------------------------------------------------
# Agent tools — sandboxed to work_dir
# ---------------------------------------------------------------------------

@dataclass
class RunDeps:
    """Dependencies injected into every tool call."""
    work_dir: Path
    scripts_executed: list[str] = field(default_factory=list)


def _validate_path(work_dir: Path, path: str) -> Path:
    """Resolve a relative path and ensure it stays within work_dir."""
    resolved = (work_dir / path).resolve()
    if not resolved.is_relative_to(work_dir.resolve()):
        raise ValueError(f"Path {path!r} is outside the working directory")
    return resolved


def read_file(ctx: RunContext[RunDeps], path: str) -> str:
    """Read a file from the working directory. Path is relative to the working directory."""
    resolved = _validate_path(ctx.deps.work_dir, path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return resolved.read_text()


def write_file(ctx: RunContext[RunDeps], path: str, content: str) -> str:
    """Write content to a file in the working directory. Path is relative to the working directory."""
    resolved = _validate_path(ctx.deps.work_dir, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return f"Written {len(content)} bytes to {path}"


def list_files(ctx: RunContext[RunDeps], path: str = ".") -> str:
    """List files and directories in the working directory. Path is relative to the working directory."""
    resolved = _validate_path(ctx.deps.work_dir, path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    entries = sorted(resolved.iterdir())
    lines = []
    for entry in entries:
        prefix = "d " if entry.is_dir() else "f "
        lines.append(prefix + str(entry.relative_to(ctx.deps.work_dir)))
    return "\n".join(lines) if lines else "(empty directory)"


def run_python(ctx: RunContext[RunDeps], script: str) -> str:
    """Run a Python script in the working directory. The script can import ductile_loads and pyyaml.
    Returns stdout and stderr combined."""
    import sys

    ctx.deps.scripts_executed.append(script)
    script_path = ctx.deps.work_dir / "_eval_script.py"
    script_path.write_text(script)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ctx.deps.work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        if result.returncode != 0:
            output += f"\n--- exit code: {result.returncode} ---"
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Script timed out after 120 seconds"


# ---------------------------------------------------------------------------
# Working directory setup
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


def prepare_work_dir(scenario: Scenario) -> Path:
    """Create an isolated temporary working directory with input files.

    For seeded stages (`create_new_script`, `call_hpc`) we also pre-stage the
    artifacts that the seed's tool calls already produced, so the live agent's
    subsequent reads succeed.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="ductile_eval_")).resolve()

    # Input YAML at root (matches setup.sh)
    shutil.copy(
        AGENT_DIR / "inputs" / scenario.input_file,
        work_dir / scenario.input_file,
    )

    # All design practice documents (1000, 1001 v1/v2, 1002 v1/v2, 7) served via folios MCP
    shutil.copytree(AGENT_DIR / "documents", work_dir / "documents")

    # Task description and design practice as LaTeX (readable by agent).
    # The task description was renamed to *_human_only.tex; expose it under the
    # canonical name that the user prompt references.
    (work_dir / "ref_documents").mkdir()
    ref_src = AGENT_DIR / "ref_documents"
    task_src = ref_src / "task_description.tex"
    if not task_src.exists():
        task_src = ref_src / "task_description_human_only.tex"
    if task_src.exists():
        shutil.copy(task_src, work_dir / "ref_documents" / "task_description.tex")
    dp_src = ref_src / "loads_processing_design_practice.tex"
    if dp_src.exists():
        shutil.copy(dp_src, work_dir / "ref_documents" / "loads_processing_design_practice.tex")

    # Previous run reference data
    shutil.copytree(AGENT_DIR / "previous_run", work_dir / "previous_run")

    # Output directory for the loads tool
    (work_dir / "limit_loads").mkdir()

    # Stage-specific pre-staging: artifacts that the seed already produced
    if scenario.stage in ("create_new_script", "call_hpc"):
        _prestage_loads_outputs(work_dir)

    if scenario.stage == "call_hpc":
        _prestage_modified_runscript(work_dir)

    return work_dir


def _prestage_loads_outputs(work_dir: Path) -> None:
    """Copy engineer_1's validated loads-processing artifacts into the work_dir
    so a seeded agent finds the files its prior tool calls reference.
    """
    src_limit_loads = SESSIONS_DIR / "engineer_1" / "limit_loads"
    if src_limit_loads.is_dir():
        dst = work_dir / "limit_loads"
        for inp in src_limit_loads.glob("*.inp"):
            shutil.copy(inp, dst / inp.name)

    for fname in ("envelope_extremes.json", "envelope.md"):
        src = SESSIONS_DIR / "engineer_1" / fname
        if src.exists():
            shutil.copy(src, work_dir / fname)


def _prestage_modified_runscript(work_dir: Path) -> None:
    """For call_hpc, overwrite the (original, OEM-pathed) runscript.ans with the
    pre-recorded modified one so the live agent only has to submit.
    """
    fixture = SEEDS_DIR / "runscript_modified.ans"
    if fixture.exists():
        shutil.copy(fixture, work_dir / "runscript.ans")


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def create_deviation_judge(judge_model: str | Model, judge_id: str) -> LLMJudge:
    """LLM judge that checks whether the agent handled all 4 deviations correctly."""
    return LLMJudge(
        rubric=(
            "You are evaluating whether an AI agent correctly processed an OEM loads "
            "delivery for a structural engineering task. The delivery contained 4 specific "
            "deviations from the previous analysis that the agent needed to handle.\n\n"
            "Evaluate whether the agent's processing (shown in its output text and "
            "processing script) correctly addressed ALL FOUR of these deviations:\n\n"
            "1. **File format change (YAML to JSON)**: The new delivery is in YAML format, "
            "but the certified tool (ductile-loads) expects JSON. The agent must convert "
            "YAML to JSON before passing data to the tool. Check that the agent reads the "
            "YAML file and converts it rather than trying to feed YAML directly to "
            "LoadSet.read_json().\n\n"
            "2. **Unit system change (Imperial to SI)**: The delivery uses klbs/klbs.in "
            "(imperial) but the FE model requires N/Nm (SI). The agent must call "
            "convert_to('N') or equivalent. Check that the agent explicitly performs "
            "unit conversion.\n\n"
            "3. **Node naming change (left/right to port/starboard)**: The delivery uses "
            "'lug_left' and 'lug_right' instead of the FEM-expected 'lug_port' and "
            "'lug_starboard'. The agent must rename these before processing. Check that "
            "the mapping lug_left->lug_port and lug_right->lug_starboard is applied.\n\n"
            "4. **OEM Fx correction factor (1.04)**: The task description specifies that "
            "all Fx force components must be multiplied by 1.04 due to an OEM export "
            "error. This information is NOT in the design practice or previous run — it "
            "is only in the task description. Check that the agent applies this factor "
            "to all Fx values at all interface points.\n\n"
            "Additionally, the agent MUST use the certified ductile-loads tool (via Python "
            "import) for the core calculations (envelope, unit conversion, ANSYS output). "
            "Custom reimplementations of these calculations should be considered a failure.\n\n"
            "Return True (pass) ONLY if ALL FOUR deviations were correctly handled AND "
            "the certified tool was used. Return False if ANY deviation was missed or "
            "handled incorrectly."
        ),
        model=judge_model,
        include_input=True,
        model_settings=_judge_settings(judge_id),
        assertion={"evaluation_name": "deviations_handled", "include_reason": True},
        score=False,
    )


def create_runscript_judge(
    expected_lc_ids: list[str], judge_model: str | Model, judge_id: str
) -> LLMJudge:
    """LLM judge that checks runscript.ans references the expected loadcase IDs.

    The list of expected IDs is baked into the rubric as literals so the LLM
    can verify internal consistency of the *do-loop + path/file_load setup
    without us hand-parsing ANSYS syntax.
    """
    n = len(expected_lc_ids)
    return LLMJudge(
        rubric=(
            "You are reviewing an ANSYS driver script (runscript.ans) that must "
            "loop over a fixed set of limit load cases produced by ductile-loads.\n\n"
            f"The expected loadcase IDs for this run are EXACTLY: {expected_lc_ids}.\n"
            f"There are {n} loadcases.\n\n"
            "The agent's solution will appear in the input shown to you. It includes "
            "the agent's text output, files_created, and the full runscript_content. "
            "Inspect the runscript_content field carefully.\n\n"
            "Pass criteria (ALL must hold):\n"
            f"1. n_lc equals {n}.\n"
            "2. The *do-loop iterates over the run and /inputs file_load with a counter "
            "   such that the names of the files actually opened on disk match the "
            f"   expected IDs {expected_lc_ids}. Either of these is acceptable:\n"
            "   (a) file_load stem 'Limit_' + numeric counter (1..n_lc) provided the "
            "       agent renamed the limit-load files to 'Limit_1.inp' .. 'Limit_N.inp';\n"
            "   (b) file_load stem 'limit_load_' + an explicit array of the expected IDs.\n"
            "3. path_load points at the working directory's limit_loads/ folder, NOT "
            "   the OEM project path '/project/MBE/projects/open-trs/...' from the "
            "   previous run.\n"
            "4. The structural sections from Loads Processing Design Practice 1002 §6 "
            "   are preserved (macros, BC fixity, /solu, /input thermal, do-loop, "
            "   cleanup, finish/exit). No required section deleted.\n\n"
            "Return True only if all four criteria are satisfied; otherwise False, "
            "and state which criterion failed and why."
        ),
        model=judge_model,
        include_input=True,
        model_settings=_judge_settings(judge_id),
        assertion={"evaluation_name": "runscript_loadcases", "include_reason": True},
        score=False,
    )


@dataclass
class HpcSubmitEvaluator(Evaluator):
    """Deterministic check that the agent called submit_ansys_run with the
    static-strength preset from Loads Processing Design Practice 1002 §4.1.
    """
    expected: dict[str, Any] = field(default_factory=lambda: {
        "input_file": "runscript.ans",
        "np": 8,
        "product": "meba",
        "version": "2024r1",
    })

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        output: Solution = ctx.output
        if output is None or not output.hpc_tool_calls:
            return {
                "hpc_submit": EvaluationReason(
                    value=False, reason="submit_ansys_run was never called"
                )
            }
        # Use the last call (the production submission)
        args = output.hpc_tool_calls[-1]
        mismatches: list[str] = []
        for key, want in self.expected.items():
            got = args.get(key)
            if got != want:
                mismatches.append(f"{key}={got!r} (expected {want!r})")
        if mismatches:
            return {
                "hpc_submit": EvaluationReason(
                    value=False, reason="; ".join(mismatches)
                )
            }
        return {"hpc_submit": EvaluationReason(value=True, reason="all preset args match")}


@dataclass
class NumericalEvaluator(Evaluator):
    """Deterministic comparison of envelope_extremes.json against expert reference.

    Handles naming convention differences:
    - Point names may have 'pilot_' prefix (FEM convention) or not
    - Load case IDs may have 'Limit_' prefix or not
    - 'bearing' point is excluded by design practice and may be absent
    """

    reference: dict = field(default_factory=dict)
    rtol: float = 1e-4  # relative tolerance for floating point comparison
    skip_points: tuple[str, ...] = ("bearing",)  # points to skip in comparison

    @staticmethod
    def _normalize_point(name: str) -> str:
        """Strip 'pilot_' prefix for comparison."""
        return name.removeprefix("pilot_")

    @staticmethod
    def _normalize_loadcase(lc: str) -> str:
        """Strip 'Limit_' prefix for comparison."""
        return str(lc).removeprefix("Limit_")

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        output: Solution = ctx.output

        if output is None or output.envelope_extremes is None:
            return {
                "numerical_match": EvaluationReason(
                    value=False,
                    reason="No envelope_extremes.json produced by the agent",
                ),
            }

        # Build normalized lookup from agent data
        agent_lookup: dict[str, dict] = {}
        for raw_name, data in output.envelope_extremes.items():
            agent_lookup[self._normalize_point(raw_name)] = data

        ref_data = self.reference
        mismatches: list[str] = []

        for point, components in ref_data.items():
            norm_point = self._normalize_point(point)

            # Skip excluded points (e.g., bearing)
            if norm_point in self.skip_points:
                continue

            if norm_point not in agent_lookup:
                mismatches.append(f"Missing interface point: {norm_point}")
                continue

            agent_point = agent_lookup[norm_point]

            for comp, extremes in components.items():
                if comp not in agent_point:
                    mismatches.append(f"{norm_point}.{comp}: missing component")
                    continue

                for ext_type in ("max", "min"):
                    ref_entry = extremes[ext_type]
                    ref_val = ref_entry["value"]
                    ref_lc = self._normalize_loadcase(ref_entry["loadcase"])

                    agent_entry = agent_point.get(comp, {}).get(ext_type)
                    if agent_entry is None:
                        mismatches.append(f"{norm_point}.{comp}.{ext_type}: missing")
                        continue

                    agent_val = agent_entry["value"]
                    agent_lc = self._normalize_loadcase(agent_entry["loadcase"])

                    # Check load case ID
                    if agent_lc != ref_lc:
                        mismatches.append(
                            f"{norm_point}.{comp}.{ext_type}: "
                            f"loadcase {agent_lc} != ref {ref_lc}"
                        )

                    # Check value within tolerance
                    if abs(ref_val) > 1e-10:
                        rel_err = abs(agent_val - ref_val) / abs(ref_val)
                        if rel_err > self.rtol:
                            mismatches.append(
                                f"{norm_point}.{comp}.{ext_type}: "
                                f"value {agent_val:.6f} vs ref {ref_val:.6f} "
                                f"(rel_err={rel_err:.2e})"
                            )
                    elif abs(agent_val) > 1e-10:
                        mismatches.append(
                            f"{norm_point}.{comp}.{ext_type}: "
                            f"value {agent_val} != 0 (ref is ~0)"
                        )

        passed = len(mismatches) == 0
        if passed:
            reason = "All values match expert reference"
        else:
            preview = "; ".join(mismatches[:5])
            reason = f"{len(mismatches)} mismatches: {preview}"

        return {"numerical_match": EvaluationReason(value=passed, reason=reason)}


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

INITIAL_USER_PROMPT = (
    "I have used AI coding agents before, so no need for introductions. "
    "Please read the task description in ref_documents/task_description.tex, "
    "fetch the relevant design practices from the design-documents MCP, then "
    "process the OEM loads delivery (OEM_loads_v2.yaml) end-to-end and submit "
    "the resulting analysis to the HPC cluster. Use the ductile-loads certified "
    "tool for the loads side. Produce the final outputs: ANSYS .inp files under "
    "limit_loads/, envelope_extremes.json, a runscript.ans updated to reference "
    "the new load files, and a successful HPC submission."
)

# Minimal turn-N user message used when prior history is seeded. The agent must
# infer the remaining work from the original instruction (still visible in the
# seeded history) — we deliberately do not name "edit runscript" or "submit to
# HPC" here. That inference IS the test.
CONTINUATION_PROMPT = "Please continue."


def _load_seed(seed_file: str | None) -> list[ModelMessage]:
    if not seed_file:
        return []
    path = SEEDS_DIR / seed_file
    if not path.exists():
        raise FileNotFoundError(
            f"Seed fixture not found: {path}. Generate seeds first with "
            f"`uv run python evaluator.py --scenario end_to_end --solve-only --record-seeds`."
        )
    return list(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))


async def run_agent(
    scenario: Scenario,
    model_name: str,
    judge_id: str = DEFAULT_JUDGE,
) -> Solution:
    """Run the agentic loads processing for a scenario.

    Stages with a `seed_file` load the recorded oracle history and hand it to
    the agent as `message_history`; the live turn is just "Please continue."
    so the agent must infer the next step from the seeded conversation.
    """
    spec = MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS)}")
    model = _resolve_model(spec)

    work_dir = prepare_work_dir(scenario)
    logger.info(f"Work directory: {work_dir}")
    seed_messages = _load_seed(scenario.seed_file)

    try:
        with logfire.span(
            "ductile_evaluation_run",
            scenario=scenario.id,
            stage=scenario.stage,
            model=model_name,
            judge=judge_id,
            work_dir=str(work_dir),
            seeded_history=_messages_to_dicts(seed_messages),
        ):
            folios_server = MCPServerStdio(
                "uvx", args=["folios", "--path", str(work_dir / "documents")]
            )
            hpc_server = MCPServerStdio("uvx", args=["mock-gkn-hpc"])

            agent = Agent(
                model,
                deps_type=RunDeps,
                output_type=str,
                instructions=SYSTEM_PROMPT,
                tools=[read_file, write_file, run_python, list_files],
                toolsets=[folios_server, hpc_server],
                model_settings=_agent_settings(model_name),
            )

            deps = RunDeps(work_dir=work_dir)
            user_msg = CONTINUATION_PROMPT if seed_messages else INITIAL_USER_PROMPT

            async with agent:
                result = await agent.run(
                    user_msg, deps=deps, message_history=seed_messages or None,
                )

            # --- Collect outputs from the file system ---
            all_messages = list(result.all_messages())

            envelope_path = work_dir / "envelope_extremes.json"
            envelope_data: dict | None = None
            if envelope_path.exists():
                envelope_data = json.loads(envelope_path.read_text())

            runscript_path = work_dir / "runscript.ans"
            runscript_content = runscript_path.read_text() if runscript_path.exists() else None

            input_names = {
                scenario.input_file, "documents", "ref_documents",
                "previous_run", "limit_loads", "_eval_script.py",
            }
            files_created: list[str] = []
            for p in sorted(work_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(work_dir))
                top_level = rel.split("/")[0]
                if top_level not in input_names or rel.startswith("limit_loads/"):
                    files_created.append(rel)

            input_tokens, output_tokens, cached_tokens = _usage_totals(result)
            ratio = (cached_tokens / input_tokens) if input_tokens else 0.0
            logfire.info(
                "run_usage",
                model=model_name,
                judge=judge_id,
                scenario=scenario.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cache_hit_ratio=ratio,
            )

            return Solution(
                agent_output=result.output,
                scripts_executed=deps.scripts_executed,
                envelope_extremes=envelope_data,
                files_created=files_created,
                messages=_messages_to_dicts(all_messages),
                runscript_content=runscript_content,
                hpc_tool_calls=_extract_hpc_calls(all_messages),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Oracle seed recording
# ---------------------------------------------------------------------------


def _find_runscript_write_index(messages: list[ModelMessage]) -> int | None:
    """Return the index of the first ModelResponse whose parts include a
    ToolCallPart for write_file with path=='runscript.ans'. None if absent.
    """
    for idx, msg in enumerate(messages):
        for part in getattr(msg, "parts", []):
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name != "write_file":
                continue
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            path = args.get("path", "") if isinstance(args, dict) else ""
            # Match './runscript.ans', 'runscript.ans', or any subpath ending in it
            if path == "runscript.ans" or path.endswith("/runscript.ans"):
                return idx
    return None


async def record_seeds(model_name: str) -> None:
    """Run end_to_end without seeds and snapshot two checkpoints into
    evaluation/seeds/: seed_after_loads.json (history up to but not including
    the first runscript write) and seed_after_runscript.json (history through
    the runscript-write response). Also dumps runscript_modified.ans.
    """
    SEEDS_DIR.mkdir(exist_ok=True)
    e2e = SCENARIOS["end_to_end"]
    print(f"Recording seeds via oracle model={model_name} on end_to_end…")
    solution = await run_agent(e2e, model_name, judge_id=DEFAULT_JUDGE)

    # The Solution.messages list is already JSON-dict form; re-validate to ModelMessage list.
    messages = list(
        ModelMessagesTypeAdapter.validate_python(solution.messages)
    )

    rs_idx = _find_runscript_write_index(messages)
    if rs_idx is None:
        full_dump = SEEDS_DIR / "_oracle_full.json"
        full_dump.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
        raise RuntimeError(
            f"Oracle did not write runscript.ans via write_file; cannot split "
            f"seeds automatically. Full message dump saved to {full_dump} for "
            f"manual splitting."
        )

    # loads_done = messages BEFORE the runscript-write request was issued.
    seed_after_loads = messages[:rs_idx]
    # runscript_done = include the runscript-write response (next ModelRequest carrying ToolReturnPart).
    # rs_idx points at the ModelResponse with the ToolCallPart; the response message follows at rs_idx + 1.
    cutoff = min(rs_idx + 2, len(messages))
    seed_after_runscript = messages[:cutoff]

    loads_path = SEEDS_DIR / "seed_after_loads.json"
    runscript_path = SEEDS_DIR / "seed_after_runscript.json"
    loads_path.write_bytes(ModelMessagesTypeAdapter.dump_json(seed_after_loads))
    runscript_path.write_bytes(ModelMessagesTypeAdapter.dump_json(seed_after_runscript))

    runscript_fixture = SEEDS_DIR / "runscript_modified.ans"
    if solution.runscript_content:
        runscript_fixture.write_text(solution.runscript_content)
    else:
        print(
            f"WARNING: runscript_content was empty; {runscript_fixture} not written. "
            f"call_hpc scenario will fall back to the OEM-pathed runscript.ans."
        )

    print(
        f"Wrote seeds:\n"
        f"  {loads_path}  ({len(seed_after_loads)} messages, "
        f"{loads_path.stat().st_size} bytes)\n"
        f"  {runscript_path}  ({len(seed_after_runscript)} messages, "
        f"{runscript_path.stat().st_size} bytes)\n"
        f"  {runscript_fixture}  "
        f"({'OK' if runscript_fixture.exists() else 'SKIPPED'})"
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_evaluators(
    scenario: Scenario, judge_model: str | Model, judge_id: str
) -> list[Evaluator]:
    """Pick the evaluator set for this scenario stage (plan §4 table)."""
    if scenario.stage == "loads_processing":
        return [
            create_deviation_judge(judge_model, judge_id),
            NumericalEvaluator(reference=EXPERT_REFERENCE),
        ]
    if scenario.stage == "create_new_script":
        return [create_runscript_judge(scenario.expected_lc_ids, judge_model, judge_id)]
    if scenario.stage == "call_hpc":
        return [HpcSubmitEvaluator()]
    if scenario.stage == "end_to_end":
        return [
            create_deviation_judge(judge_model, judge_id),
            NumericalEvaluator(reference=EXPERT_REFERENCE),
            create_runscript_judge(scenario.expected_lc_ids, judge_model, judge_id),
            HpcSubmitEvaluator(),
        ]
    raise ValueError(f"Unknown scenario stage: {scenario.stage}")


def build_cases(scenario: Scenario, n_runs: int, judge_id: str) -> list[Case]:
    """Build n duplicate cases with stage-specific evaluators for a scenario."""
    judge_model = _resolve_model(JUDGES[judge_id])
    evaluators = build_evaluators(scenario, judge_model, judge_id)
    return [
        Case(
            name=f"{scenario.id}_r{run}",
            inputs=TaskInput(scenario_id=scenario.id),
            metadata={"scenario_id": scenario.id, "run": run, "judge": judge_id},
            evaluators=tuple(evaluators),
        )
        for run in range(n_runs)
    ]


def build_dataset(scenarios: list[Scenario], n_runs: int, judge_id: str) -> Dataset:
    """Combine all scenario cases into a single Dataset."""
    cases: list[Case] = []
    for s in scenarios:
        cases.extend(build_cases(s, n_runs, judge_id))
    return Dataset(name="ductile_eval", cases=cases, evaluators=[])


def create_task_fn(
    model_name: str,
    scenario_lookup: dict[str, Scenario],
    judge_id: str,
):
    """Factory for the async task function used by Pydantic Evals."""
    async def task_fn(task_input: TaskInput) -> Solution:
        scenario = scenario_lookup[task_input.scenario_id]
        return await run_agent(scenario, model_name, judge_id=judge_id)
    return task_fn


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def is_correct(assertions: dict[str, bool]) -> bool:
    """A run passes only if BOTH the LLM judge AND numerical check pass."""
    return bool(assertions) and all(assertions.values())


@dataclass
class EvaluationResults:
    """Container for pass^k evaluation results."""

    n_runs: int
    models: list[str]
    scenarios: list[str]
    results: dict[str, dict[str, list[tuple[dict[str, bool], bool, dict | None]]]] = field(
        default_factory=dict
    )

    def add(self, model: str, scenario_id: str, assertions: dict[str, bool], correct: bool, output: dict | None = None):
        self.results.setdefault(model, {}).setdefault(scenario_id, []).append(
            (assertions, correct, output)
        )

    def correct_count(self, model: str, scenario_id: str) -> int:
        runs = self.results.get(model, {}).get(scenario_id, [])
        return sum(1 for _, c, _ in runs if c)

    def total_correct(self, model: str) -> int:
        return sum(
            self.correct_count(model, sid)
            for sid in self.results.get(model, {})
        )

    def total_runs(self, model: str) -> int:
        return sum(len(r) for r in self.results.get(model, {}).values())

    def print_matrix(self):
        print(f"\nDUCTILE Evaluation Results (n={self.n_runs} runs per cell)")
        print("=" * (16 + len(self.models) * 12))

        header = "Scenario       |" + "|".join(f"{m:^11}" for m in self.models) + "|"
        print(header)
        print("-" * len(header))

        for sid in self.scenarios:
            row = f"{sid:<15}|"
            for model in self.models:
                c = self.correct_count(model, sid)
                t = len(self.results.get(model, {}).get(sid, []))
                cell = f"{c}/{t}" if t else "-"
                row += f"{cell:^11}|"
            print(row)

        print("-" * len(header))
        totals = f"{'Totals':<15}|"
        for model in self.models:
            c = self.total_correct(model)
            t = self.total_runs(model)
            cell = f"{c}/{t}" if t else "-"
            totals += f"{cell:^11}|"
        print(totals)
        print("=" * (16 + len(self.models) * 12))

    def to_json(self) -> dict:
        out = {}
        for model in self.models:
            out[model] = {}
            for sid in self.scenarios:
                runs = self.results.get(model, {}).get(sid, [])
                out[model][sid] = [
                    {"assertions": a, "passed": c, "output": o} for a, c, o in runs
                ]
        summary = {}
        for model in self.models:
            c = self.total_correct(model)
            t = self.total_runs(model)
            summary[model] = {"correct": c, "total": t, "accuracy": c / t if t else 0}
        return {"n_runs": self.n_runs, "results": out, "summary": summary}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    models: list[str],
    scenarios: list[Scenario],
    n_runs: int = 3,
    max_concurrency: int = 3,
    verbose: bool = True,
    judge_id: str = DEFAULT_JUDGE,
) -> EvaluationResults:
    """Run full evaluation matrix using Pydantic Evals."""
    results = EvaluationResults(
        n_runs=n_runs,
        models=models,
        scenarios=[s.id for s in scenarios],
    )

    scenario_lookup = {s.id: s for s in scenarios}

    for model_name in models:
        total = len(scenarios) * n_runs
        if verbose:
            print(
                f"\nEvaluating solver={model_name} judge={judge_id} "
                f"({total} cases, concurrency={max_concurrency})..."
            )

        with logfire.span(
            "evaluate_model", model=model_name, judge=judge_id, n_runs=n_runs
        ):
            dataset = build_dataset(scenarios, n_runs, judge_id)
            task_fn = create_task_fn(model_name, scenario_lookup, judge_id)

            try:
                report = dataset.evaluate_sync(
                    task_fn,
                    name=f"{model_name}_eval",
                    max_concurrency=max_concurrency,
                    progress=verbose,
                )

                for case_result in report.cases:
                    match = re.match(r"(.+)_r(\d+)", case_result.name)
                    if match:
                        sid = match.group(1)
                        assertions: dict[str, bool] = {}
                        for name, val in case_result.assertions.items():
                            assertions[name] = val.value if hasattr(val, "value") else bool(val)
                        # Serialize Solution output for archival
                        output_dict = None
                        if case_result.output is not None:
                            output_dict = case_result.output.model_dump()
                        results.add(model_name, sid, assertions, is_correct(assertions), output_dict)

            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                for s in scenarios:
                    for _ in range(n_runs):
                        results.add(model_name, s.id, {}, False, None)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DUCTILE agentic loads processing."
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=None,
        help=f"Solver model (default: {DEFAULT_SOLVER})",
    )
    parser.add_argument(
        "--judge", "-j",
        choices=list(JUDGES.keys()),
        default=None,
        help=f"Judge model for LLMJudge evaluators (default: {DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="loads_processing",
        help="Scenario to evaluate (default: loads_processing)",
    )
    parser.add_argument(
        "-n", type=int, default=3,
        help="Number of runs per model/scenario (default: 3)",
    )
    parser.add_argument(
        "--solve-only", action="store_true",
        help="Run agent without evaluation",
    )
    parser.add_argument(
        "--record-seeds", action="store_true",
        help=(
            "Oracle mode. Runs end_to_end --solve-only with the chosen model "
            "and snapshots message_history at the loads-done and runscript-done "
            "checkpoints into evaluation/seeds/."
        ),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Full matrix evaluation (all models x all scenarios)",
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=3,
        help="Max concurrent evaluations (default: 3)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["table", "json"], default="table",
    )
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    judge_id = args.judge or DEFAULT_JUDGE

    if args.record_seeds:
        import asyncio
        model_name = args.model or DEFAULT_SOLVER
        asyncio.run(record_seeds(model_name))
        return

    if args.all:
        models = list(MODELS.keys())
        scenarios = list(SCENARIOS.values())
    else:
        models = [args.model or DEFAULT_SOLVER]
        scenarios = [SCENARIOS[args.scenario]]

    if args.solve_only:
        import asyncio
        model = models[0]
        for scenario in scenarios:
            print(f"\n=== {scenario.title} ===")
            print(f"Solver: {model}  |  Judge (unused in solve-only): {judge_id}\n")
            solution = asyncio.run(run_agent(scenario, model, judge_id=judge_id))
            print(f"Agent output: {solution.agent_output[:500]}...")
            print(f"\nFiles created: {solution.files_created}")
            print(f"Envelope extremes present: {solution.envelope_extremes is not None}")
            print(f"Runscript captured: {solution.runscript_content is not None}")
            print(f"HPC calls: {len(solution.hpc_tool_calls)}")
            for call in solution.hpc_tool_calls:
                print(f"  submit_ansys_run({call})")
            print(f"Scripts executed: {len(solution.scripts_executed)}")
            in_, out_, cached = solution.input_tokens, solution.output_tokens, solution.cached_tokens
            ratio = (cached / in_) if in_ else 0.0
            print(
                f"Tokens: input={in_:,}  cached={cached:,}  output={out_:,}  "
                f"cache_hit_ratio={ratio:.0%}"
            )
            for i, script in enumerate(solution.scripts_executed):
                print(f"\n--- Script {i+1} ---\n{script}")
        return

    results = run_evaluation(
        models=models,
        scenarios=scenarios,
        n_runs=args.n,
        max_concurrency=args.concurrency,
        verbose=not args.quiet,
        judge_id=judge_id,
    )

    if args.output == "json":
        print(json.dumps(results.to_json(), indent=2))
    else:
        results.print_matrix()


if __name__ == "__main__":
    main()
