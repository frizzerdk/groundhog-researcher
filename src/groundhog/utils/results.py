"""Result serialization utilities for strategies."""

import json
from pathlib import Path

from groundhog.base.types import EvaluationResult


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

    if metadata:
        result_data["metadata"] = metadata

    (Path(path) / "result.json").write_text(
        json.dumps(result_data, indent=2, default=str), encoding="utf-8")
