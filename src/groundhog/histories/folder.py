"""Folder-based attempt history. Each attempt is a numbered directory.

Directory structure:
    TaskName/
        attempts/
            001_none/       ← first attempt (no parent)
                solution.py
                result.json
                conversation.json
                conversation.md
                TASK_CONTEXT.md
            002_1/          ← second attempt (parent=1)
                ...
        learnings.md        ← accumulated learnings (managed separately)
"""

import json
import shutil
from pathlib import Path
from typing import Callable, Optional, List

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.base.attempt_history import Attempt, Workspace, AttemptHistory


class FolderAttempt(Attempt):
    """Attempt stored as a folder on disk. Read-only."""

    def __init__(self, number: int, parent: Optional[int], path: Path):
        self.number = number
        self.parent = parent
        self.path = path

    @property
    def code(self) -> str:
        return (self.path / "solution.py").read_text(encoding="utf-8")

    @property
    def result(self) -> EvaluationResult:
        data = json.loads((self.path / "result.json").read_text(encoding="utf-8"))
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

    @property
    def metadata(self) -> dict:
        data = json.loads((self.path / "result.json").read_text(encoding="utf-8"))
        return data.get("metadata", {})

    def __repr__(self):
        return f"Attempt({self.number}, parent={self.parent})"


class FolderWorkspace(Workspace):
    """A working directory for one attempt. Write files, then commit or abort."""

    def __init__(self, number: int, parent: Optional[int], path: Path):
        self.number = number
        self.parent = parent
        self.path = path
        self.path.mkdir(parents=True)

    def commit(self, result: EvaluationResult, metadata: Optional[dict] = None) -> FolderAttempt:
        """Write result.json and finalize as an immutable attempt."""
        result_data = {
            "parent": self.parent,
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
            # Write artifacts as separate files
            for artifact_name, artifact_data in stage_result.artifacts.items():
                artifact_path = self.path / artifact_name
                if isinstance(artifact_data, bytes):
                    artifact_path.write_bytes(artifact_data)
                elif isinstance(artifact_data, str):
                    artifact_path.write_text(artifact_data, encoding="utf-8")
                else:
                    artifact_path.write_text(json.dumps(artifact_data, indent=2), encoding="utf-8")

        if metadata:
            result_data["metadata"] = metadata

        (self.path / "result.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
        return FolderAttempt(number=self.number, parent=self.parent, path=self.path)

    def abort(self):
        """Delete the workspace folder entirely."""
        if self.path.exists():
            shutil.rmtree(self.path)


class FolderAttemptHistory(AttemptHistory):
    """Each attempt is a directory: {number}_{parent}/"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path) / "attempts"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._count = self._scan_count()

    def _scan_count(self) -> int:
        if not self.base_path.exists():
            return 0
        max_num = 0
        for d in self.base_path.iterdir():
            if d.is_dir():
                try:
                    num = int(d.name.split("_", 1)[0])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return max_num

    def _folder_name(self, number: int, parent: Optional[int]) -> str:
        parent_str = str(parent) if parent is not None else "none"
        return f"{number:03d}_{parent_str}"

    def workspace(self, parent: Optional[int] = None) -> FolderWorkspace:
        """Create a new workspace folder. Strategy writes files here, then commits or aborts."""
        self._count += 1
        number = self._count
        path = self.base_path / self._folder_name(number, parent)
        return FolderWorkspace(number=number, parent=parent, path=path)

    def list(self) -> List[FolderAttempt]:
        attempts = []
        for d in sorted(self.base_path.iterdir()):
            if not d.is_dir():
                continue
            # Only list committed attempts (have result.json)
            if not (d / "result.json").exists():
                continue
            parts = d.name.split("_", 1)
            number = int(parts[0])
            parent = None if parts[1] == "none" else int(parts[1])
            attempts.append(FolderAttempt(number=number, parent=parent, path=d))
        return attempts

    def get(self, number: int) -> Optional[FolderAttempt]:
        for attempt in self.list():
            if attempt.number == number:
                return attempt
        return None

    def best(self, scorer: Callable[[StageResult], float]) -> Optional[FolderAttempt]:
        attempts = self.list()
        if not attempts:
            return None

        def score_attempt(attempt):
            result = attempt.result
            if not result.completed:
                return -1.0
            last_stage = list(result.stages.values())[-1]
            return scorer(last_stage)

        return max(attempts, key=score_attempt)

    def lineage(self, attempt: FolderAttempt) -> List[FolderAttempt]:
        chain = [attempt]
        current = attempt
        while current.parent is not None:
            current = self.get(current.parent)
            if current is None:
                break
            chain.append(current)
        chain.reverse()
        return chain
