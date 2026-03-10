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
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, LLMJudge

load_dotenv()

# Configure Logfire
logfire.configure(service_name="ductile-evaluator", send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

logging.basicConfig(
    level=logging.INFO,
    handlers=[logfire.LogfireLoggingHandler(fallback=logging.StreamHandler())],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODELS = {
    "haiku": "anthropic:claude-3-5-haiku-latest",
    "sonnet": "anthropic:claude-sonnet-4-5",
    "opus": "anthropic:claude-opus-4-6",
}

# Fixed judge model for consistent grading (per paper Sec. 4.3)
JUDGE_MODEL = "anthropic:claude-opus-4-6"

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

@dataclass
class Scenario:
    """A loads processing scenario to evaluate."""
    id: str
    title: str
    input_file: str  # relative to agent/inputs/

SCENARIOS = {
    "v2": Scenario(
        id="v2",
        title="OEM loads delivery v2 processing",
        input_file="OEM_loads_v2.yaml",
    ),
    "v3": Scenario(
        id="v3",
        title="OEM loads delivery v3 processing",
        input_file="OEM_loads_v3.yaml",
    ),
}


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


@dataclass
class TaskInput:
    """Input to an evaluation case."""
    scenario_id: str


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

def prepare_work_dir(scenario: Scenario) -> Path:
    """Create an isolated temporary working directory with input files."""
    work_dir = Path(tempfile.mkdtemp(prefix="ductile_eval_")).resolve()

    # Input YAML at root (matches setup.sh)
    shutil.copy(
        AGENT_DIR / "inputs" / scenario.input_file,
        work_dir / scenario.input_file,
    )

    # Design practice document (replaces MCP folios in eval context)
    (work_dir / "documents").mkdir()
    shutil.copy(
        AGENT_DIR / "documents" / "1001_v1.md",
        work_dir / "documents" / "1001_v1.md",
    )

    # Task description and design practice as LaTeX (readable by agent)
    (work_dir / "ref_documents").mkdir()
    for tex_file in ["task_description.tex", "loads_processing_design_practice.tex"]:
        src = AGENT_DIR / "ref_documents" / tex_file
        if src.exists():
            shutil.copy(src, work_dir / "ref_documents" / tex_file)

    # Previous run reference data
    shutil.copytree(AGENT_DIR / "previous_run", work_dir / "previous_run")

    # Empty output directory
    (work_dir / "limit_loads").mkdir()

    return work_dir


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def create_deviation_judge() -> LLMJudge:
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
        model=JUDGE_MODEL,
        include_input=True,
        model_settings=ModelSettings(temperature=0.0),
        assertion={"evaluation_name": "deviations_handled", "include_reason": True},
        score=False,
    )


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

DELEGATION_PROMPT = (
    "I have used AI coding agents before, so no need for introductions. "
    "Please read the task description in ref_documents/task_description.tex "
    "and fetch the design practice document from the MCP tools. "
    "Then process the OEM loads delivery ({input_file}) end-to-end "
    "following the design practice methodology. Use the ductile-loads "
    "certified tool for all calculations. "
    "Produce the final outputs: ANSYS .inp files in limit_loads/, "
    "envelope_extremes.json, and an exceedance comparison against previous_run/."
)


async def run_agent(scenario: Scenario, model_name: str) -> Solution:
    """Run the agentic loads processing for a scenario."""
    model = MODELS.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")

    work_dir = prepare_work_dir(scenario)
    logger.info(f"Work directory: {work_dir}")

    try:
        with logfire.span(
            "ductile_evaluation_run",
            scenario=scenario.id,
            model=model_name,
            work_dir=str(work_dir),
        ):
            # Replicate engineer_1 environment: folios MCP serves design practice
            folios_server = MCPServerStdio(
                "uvx", args=["folios", "--path", str(work_dir / "documents")]
            )

            agent = Agent(
                model,
                deps_type=RunDeps,
                output_type=str,
                instructions=SYSTEM_PROMPT,
                tools=[read_file, write_file, run_python, list_files],
                toolsets=[folios_server],
            )

            deps = RunDeps(work_dir=work_dir)
            prompt = DELEGATION_PROMPT.format(input_file=scenario.input_file)

            async with agent:
                result = await agent.run(prompt, deps=deps)

            # --- Collect outputs from the file system ---

            # Read envelope_extremes.json if produced
            envelope_path = work_dir / "envelope_extremes.json"
            envelope_data = None
            if envelope_path.exists():
                envelope_data = json.loads(envelope_path.read_text())

            # List files created by the agent (exclude inputs we placed)
            input_names = {
                scenario.input_file, "documents", "ref_documents",
                "previous_run", "limit_loads", "_eval_script.py",
            }
            files_created = []
            for p in sorted(work_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(work_dir))
                top_level = rel.split("/")[0]
                # Include agent-created files + anything in limit_loads/
                if top_level not in input_names or rel.startswith("limit_loads/"):
                    files_created.append(rel)

            return Solution(
                agent_output=result.output,
                scripts_executed=deps.scripts_executed,
                envelope_extremes=envelope_data,
                files_created=files_created,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_cases(scenario: Scenario, n_runs: int) -> list[Case]:
    """Build n duplicate cases with evaluators for a scenario."""
    evaluators = [
        create_deviation_judge(),
        NumericalEvaluator(reference=EXPERT_REFERENCE),
    ]

    return [
        Case(
            name=f"{scenario.id}_r{run}",
            inputs=TaskInput(scenario_id=scenario.id),
            metadata={"scenario_id": scenario.id, "run": run},
            evaluators=tuple(evaluators),
        )
        for run in range(n_runs)
    ]


def build_dataset(scenarios: list[Scenario], n_runs: int) -> Dataset:
    """Combine all scenario cases into a single Dataset."""
    cases = []
    for s in scenarios:
        cases.extend(build_cases(s, n_runs))
    return Dataset(name="ductile_eval", cases=cases, evaluators=[])


def create_task_fn(model_name: str, scenario_lookup: dict[str, Scenario]):
    """Factory for the async task function used by Pydantic Evals."""
    async def task_fn(task_input: TaskInput) -> Solution:
        scenario = scenario_lookup[task_input.scenario_id]
        return await run_agent(scenario, model_name)
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
            print(f"\nEvaluating {model_name} ({total} cases, concurrency={max_concurrency})...")

        with logfire.span("evaluate_model", model=model_name, n_runs=n_runs):
            dataset = build_dataset(scenarios, n_runs)
            task_fn = create_task_fn(model_name, scenario_lookup)

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
        help="Model to evaluate (default: sonnet)",
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="v2",
        help="Scenario to evaluate (default: v2)",
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

    # Default model is opus unless --all
    if args.all:
        models = list(MODELS.keys())
        scenarios = list(SCENARIOS.values())
    else:
        models = [args.model or "opus"]
        scenarios = [SCENARIOS[args.scenario]]

    if args.solve_only:
        import asyncio
        model = models[0]
        for scenario in scenarios:
            print(f"\n=== {scenario.title} ===")
            print(f"Model: {model}\n")
            solution = asyncio.run(run_agent(scenario, model))
            print(f"Agent output: {solution.agent_output[:500]}...")
            print(f"\nFiles created: {solution.files_created}")
            print(f"Envelope extremes present: {solution.envelope_extremes is not None}")
            print(f"Scripts executed: {len(solution.scripts_executed)}")
            for i, script in enumerate(solution.scripts_executed):
                print(f"\n--- Script {i+1} ---\n{script}")
        return

    results = run_evaluation(
        models=models,
        scenarios=scenarios,
        n_runs=args.n,
        max_concurrency=args.concurrency,
        verbose=not args.quiet,
    )

    if args.output == "json":
        print(json.dumps(results.to_json(), indent=2))
    else:
        results.print_matrix()


if __name__ == "__main__":
    main()
