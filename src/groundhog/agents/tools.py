"""Default agent tools built from toolkit capabilities.

Uses agent_tool() factory to wrap toolkit methods as agent-callable tools.
The optimizer calls build_default_agent_tools() and puts the result on
toolkit.agent_tools. Users can extend the list with custom tools.

All tools are built at optimizer init time — no workspace binding needed.
The strategy controls which tools are available per phase via filtering.
"""

import copy
import json
from pathlib import Path

from groundhog.base.agent import agent_tool


def _format_eval_result(result):
    """Format a StageResult into a readable string for the agent."""
    lines = []

    # Score and key metrics
    if result.metrics:
        score = result.metrics.get("score")
        if score is not None:
            lines.append(f"Score: {score}")
        for key, val in result.metrics.items():
            if key != "score":
                lines.append(f"  {key}: {val}")

    # Errors
    if result.errors:
        lines.append("")
        lines.append("ERRORS (score = 0 if any present):")
        for key, msg in result.errors.items():
            lines.append(f"  {key}: {msg}")

    # Warnings
    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for key, msg in result.warnings.items():
            lines.append(f"  {key}: {msg}")

    # Artifacts — list paths so the agent knows they exist
    if result.artifacts:
        lines.append("")
        lines.append("Artifacts:")
        for name, path in result.artifacts.items():
            lines.append(f"  {path}")

    return "\n".join(lines)


def eval_to_dir(stage, path, output_dir):
    """Run eval stage on a file, write results + artifacts to output_dir.

    Returns a formatted string with metrics, errors, warnings, and paths
    to any generated artifacts.
    """
    code = Path(path).read_text(encoding="utf-8")
    result = stage.call(code)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Always write results.json with score + metrics
    results_data = {"score": result.score, "metrics": result.metrics}
    if result.errors:
        results_data["errors"] = result.errors
    if result.warnings:
        results_data["warnings"] = result.warnings
    (out / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str), encoding="utf-8")

    # Write artifacts, collect paths
    written = {}
    for name, content in result.artifacts.items():
        if name.startswith("_"):
            continue
        dest = out / name
        if isinstance(content, bytes):
            dest.write_bytes(content)
        elif isinstance(content, str):
            dest.write_text(content, encoding="utf-8")
        else:
            dest = out / (name if name.endswith(".json") else f"{name}.json")
            dest.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
        written[name] = str(dest)

    # Return copy with artifacts replaced by paths, and format for agent
    out_result = copy.copy(result)
    out_result.artifacts = written
    return _format_eval_result(out_result)


def build_default_agent_tools(toolkit) -> list:
    """Build agent tools from toolkit capabilities.

    Inspects what's on the toolkit and wraps available methods.
    Returns a plain list — caller puts it on toolkit.agent_tools.
    """
    tools = []

    if hasattr(toolkit, 'learnings'):
        tools.append(agent_tool(
            name="get-learnings",
            description=(
                "Read accumulated learnings from previous optimization runs. "
                "Returns notes about what worked, what didn't, dead-ends, "
                "and key thresholds."
            ),
            func=toolkit.learnings.get,
            params={
                "last": {"type": "int", "default": 20,
                         "description": "Number of recent entries to return"},
                "random": {"type": "int", "default": 10,
                           "description": "Number of random older entries to include"},
            },
        ))

    if hasattr(toolkit, 'task'):
        through = getattr(toolkit, 'agent_through', None) or getattr(toolkit, 'through', None)
        for stage in toolkit.task.evaluator.eval_stages(toolkit.task.data, through=through):
            description = (
                f"{stage.description}. "
                f"Returns score, metrics, errors/warnings, and paths to "
                f"any generated artifacts."
            )
            tools.append(agent_tool(
                name=stage.name,
                description=description,
                func=lambda path, output_dir=f"workspace/output/{stage.name}", s=stage: eval_to_dir(s, path, output_dir),
                params={
                    "path": {"type": "path",
                             "description": "Path to the .py file to evaluate"},
                    "output_dir": {"type": "path",
                                   "default": f"workspace/output/{stage.name}",
                                   "description": "Directory for results and artifacts"},
                },
            ))

    return tools
