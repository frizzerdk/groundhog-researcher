"""Result serialization utilities for strategies."""

import json
from pathlib import Path

from groundhog.base.types import EvaluationResult, StageResult

# Dedicated per-attempt metadata file (name, strategy, cost, flags, ...). Kept
# out of result.json, which is for evaluation output only.
METADATA_FILENAME = "metadata.json"


def write_result(path: Path, result: EvaluationResult, metadata: dict = None):
    """Write result.json to an attempt directory.

    Called by strategies before commit(). Serializes the EvaluationResult
    with metrics, errors, warnings, and optional metadata.
    """
    result_data = {
        "completed": result.completed,
        "failed_stage": result.failed_stage,
        "stages": {},
    }
    for stage_name, stage_result in result.stages.items():
        result_data["stages"][stage_name] = {
            "metrics": stage_result.metrics,
            "errors": stage_result.errors,
            "warnings": stage_result.warnings,
        }

    (Path(path) / "result.json").write_text(
        json.dumps(result_data, indent=2, default=str), encoding="utf-8")

    # result.json stays eval-only; attempt metadata lives in its own file.
    if metadata:
        write_metadata(path, metadata)


def read_result(data: dict) -> EvaluationResult:
    """Parse a result.json dict back into an EvaluationResult.

    Backend-agnostic: the folder backend reads the dict from disk, the git
    backend from the object store — both rebuild the result the same way.
    """
    stages = {}
    for name, stage_data in data.get("stages", {}).items():
        stages[name] = StageResult(
            metrics=stage_data.get("metrics", {}),
            errors=stage_data.get("errors", {}),
            warnings=stage_data.get("warnings", {}),
        )
    return EvaluationResult(
        stages=stages,
        completed=data.get("completed", True),
        failed_stage=data.get("failed_stage"),
    )


def write_metadata(path, metadata: dict) -> None:
    """Merge ``metadata`` into the attempt's metadata.json (creating it).

    Used by strategies for attempt-level metadata (strategy, cost, flags) and
    by the backend at commit to record the human-readable name. Merges, so
    repeated calls accumulate rather than overwrite.
    """
    target = Path(path) / METADATA_FILENAME
    existing = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
    existing.update(metadata)
    target.write_text(json.dumps(existing, indent=2, default=str),
                      encoding="utf-8")


def read_attempt_metadata(attempt) -> dict:
    """Read a committed attempt's metadata via its read_file API.

    Backend-agnostic (folder + git). Falls back to legacy metadata embedded in
    result.json for attempts written before metadata.json existed.
    """
    text = attempt.read_file(METADATA_FILENAME)
    if text:
        try:
            return json.loads(text)
        except ValueError:
            pass
    legacy = attempt.read_file("result.json")
    if legacy:
        try:
            return json.loads(legacy).get("metadata", {})
        except ValueError:
            pass
    return {}
