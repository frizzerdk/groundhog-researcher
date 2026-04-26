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
import time
from pathlib import Path
from typing import Callable, Optional, List

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.base.attempt_history import Attempt, Workspace, AttemptHistory


# Stale claim sentinels (left behind by a crashed allocator) are reclaimed
# after this long. 5 minutes is well past any sane allocation latency.
_CLAIM_TTL_SECONDS = 300

# Defensive cap on retry contention — should never be hit in practice.
_ALLOC_MAX_RETRIES = 100


class FolderAttempt(Attempt):
    """Attempt stored as a folder on disk. Read-only."""

    def __init__(self, number: int, parent: Optional[int], path: Path):
        self.number = number
        self.parent = parent
        self.path = path

    # Binary file extensions that should not be read as text
    _BINARY_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".bmp", ".ico", ".pdf",
                    ".zip", ".gz", ".tar", ".bin", ".pkl", ".npy", ".npz",
                    ".whl", ".so", ".dll", ".exe", ".pyc"}

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

    def list_files(self) -> List[str]:
        """List all files in this attempt (relative paths)."""
        return sorted(
            str(f.relative_to(self.path)).replace("\\", "/")
            for f in self.path.rglob("*") if f.is_file()
        )

    def read_file(self, path: str) -> Optional[str]:
        """Read a text file from this attempt. Returns None if not found."""
        target = self.path / path
        if not target.exists():
            return None
        if target.suffix.lower() in self._BINARY_EXTS:
            size_kb = target.stat().st_size / 1024
            return f"[binary file: {size_kb:.0f}KB — use your file viewer to inspect]"
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            size_kb = target.stat().st_size / 1024
            return f"[binary file: {size_kb:.0f}KB — use your file viewer to inspect]"

    def __repr__(self):
        return f"Attempt({self.number}, parent={self.parent})"


class FolderWorkspace(Workspace):
    """A working directory for one attempt. Write files, then commit or abort."""

    def __init__(self, number: int, parent: Optional[int], path: Path):
        self.number = number
        self.parent = parent
        self.path = path
        # exist_ok=True so FolderAttemptHistory.workspace can pre-create the
        # directory as part of its atomic-claim handshake.
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "work").mkdir(exist_ok=True)

    def commit(self, success: bool = True) -> FolderAttempt:
        """Mark this workspace as done by renaming the folder.

        Strategy must write all files (solution.py, result.json, etc.)
        before calling commit(). This just flips the visibility flag.
        """
        suffix = "_done" if success else "_fail"
        new_path = self.path.parent / (self.path.name + suffix)
        self.path.rename(new_path)
        self.path = new_path
        return FolderAttempt(number=self.number, parent=self.parent, path=new_path)

    def abort(self):
        """Delete the workspace folder entirely."""
        if self.path.exists():
            shutil.rmtree(self.path)


class FolderAttemptHistory(AttemptHistory):
    """Each attempt is a directory: {number}_{parent}/"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path) / "attempts"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _used_numbers(self) -> set[int]:
        """Numbers currently held by attempts on disk. Cleans stale claim sentinels."""
        used: set[int] = set()
        now = time.time()
        for d in self.base_path.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith(".claim_"):
                try:
                    if (now - d.stat().st_mtime) > _CLAIM_TTL_SECONDS:
                        d.rmdir()
                except OSError:
                    pass
                continue
            try:
                used.add(int(d.name.split("_", 1)[0]))
            except ValueError:
                pass
        return used

    def _folder_name(self, number: int, parent: Optional[int]) -> str:
        parent_str = str(parent) if parent is not None else "none"
        return f"{number:03d}_{parent_str}"

    def workspace(self, parent: Optional[int] = None) -> FolderWorkspace:
        """Create a new workspace folder. Strategy writes files here, then commits or aborts.

        Allocation is atomic across concurrent processes: each call rescans the
        attempts directory, picks ``max + 1``, then claims the number with a
        hidden ``.claim_NNN`` sentinel directory. ``mkdir(exist_ok=False)`` is
        atomic on POSIX and NTFS; whichever process wins owns the number.
        Losers retry with a fresh scan.
        """
        for _ in range(_ALLOC_MAX_RETRIES):
            number = max(self._used_numbers(), default=0) + 1
            sentinel = self.base_path / f".claim_{number:03d}"
            try:
                sentinel.mkdir(exist_ok=False)
            except FileExistsError:
                # Another process is mid-claim at this number. Brief backoff
                # so we don't spin while they finish, then re-scan.
                time.sleep(0.05)
                continue

            path = self.base_path / self._folder_name(number, parent)
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                # Defensive: leftover from a previous run at the same path.
                # Drop the claim and let the next iteration pick a higher number.
                try:
                    sentinel.rmdir()
                except OSError:
                    pass
                continue

            try:
                sentinel.rmdir()
            except OSError:
                pass  # Best-effort; TTL handles leftovers.

            return FolderWorkspace(number=number, parent=parent, path=path)

        raise RuntimeError(
            f"Could not allocate a workspace number after {_ALLOC_MAX_RETRIES} retries"
        )

    def list(self, only_done: bool = True) -> List[FolderAttempt]:
        attempts = []
        for d in sorted(self.base_path.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name.endswith("_done"):
                base = name[:-5]
            elif name.endswith("_fail"):
                if only_done:
                    continue
                base = name[:-5]
            else:
                if only_done:
                    continue
                base = name  # in-progress
            parts = base.split("_", 1)
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
