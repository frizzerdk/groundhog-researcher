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
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, List

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.base.attempt_history import Attempt, Workspace, AttemptHistory
from groundhog.utils.results import read_result, write_metadata, read_attempt_metadata


# Stale claim sentinels (left behind by a crashed allocator) are reclaimed
# after this long. 5 minutes is well past any sane allocation latency.
_CLAIM_TTL_SECONDS = 300

# Defensive cap on retry contention — should never be hit in practice.
_ALLOC_MAX_RETRIES = 100


class FolderAttempt(Attempt):
    """Attempt stored as a folder on disk. Read-only."""

    def __init__(self, id: str, parent: Optional[str], path: Path):
        self.id = id
        self.parent = parent
        self.path = path
        # Creation time = the attempt directory's mtime (set when the workspace
        # last wrote into it). Frozen here so it stays a stable sort key.
        self.created_at = path.stat().st_mtime if path.exists() else 0.0

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
        return read_result(data)

    @property
    def metadata(self) -> dict:
        return read_attempt_metadata(self)

    @property
    def name(self) -> str:
        return self.metadata.get("name", "")

    @property
    def status(self) -> str:
        name = self.path.name
        if name.endswith("_done"):
            return "done"
        if name.endswith("_fail"):
            return "fail"
        return "in-progress"

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
        return f"Attempt({self.id}, parent={self.parent})"


class FolderWorkspace(Workspace):
    """A working directory for one attempt. Write files, then commit or abort."""

    def __init__(self, display_id: str, parent: Optional[str], path: Path, name: str = ""):
        self.display_id = display_id
        self.parent = parent
        self.path = path
        self.name = name
        # exist_ok=True so FolderAttemptHistory.workspace can pre-create the
        # directory as part of its atomic-claim handshake.
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "work").mkdir(exist_ok=True)

    def commit(self, success: bool = True) -> FolderAttempt:
        """Mark this workspace as done by renaming the folder.

        Strategy must write all files (solution.py, result.json, etc.)
        before calling commit(). This just flips the visibility flag.
        """
        if self.name:
            write_metadata(self.path, {"name": self.name})
        suffix = "_done" if success else "_fail"
        new_path = self.path.parent / (self.path.name + suffix)
        self.path.rename(new_path)
        self.path = new_path
        return FolderAttempt(id=self.display_id, parent=self.parent, path=new_path)

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

    def _folder_name(self, number: int, parent: Optional[str]) -> str:
        parent_str = parent if parent is not None else "none"
        return f"{number:03d}_{parent_str}"

    def workspace(self, parent: Optional[str] = None) -> FolderWorkspace:
        """Create a new workspace folder. Strategy writes files here, then commits or aborts.

        Allocation is atomic across concurrent processes: each call rescans the
        attempts directory, picks ``max + 1``, then claims the number with a
        hidden ``.claim_NNN`` sentinel directory. ``mkdir(exist_ok=False)`` is
        atomic on POSIX and NTFS; whichever process wins owns the number.
        Losers retry with a fresh scan.
        """
        # The contract is string ids; tolerate an int parent from older callers
        # by coercing to the canonical string form.
        parent = str(parent) if parent is not None else None
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

            return FolderWorkspace(display_id=str(number), parent=parent, path=path)

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
            id = str(int(parts[0]))
            parent = None if parts[1] == "none" else parts[1]
            attempts.append(FolderAttempt(id=id, parent=parent, path=d))
        return attempts

    def get(self, id: str) -> Optional[FolderAttempt]:
        # Resolve any COMMITTED attempt — failed ones included (a child's
        # recorded parent may be failed; treating it as missing would
        # misclassify the child as fresh). Open workspaces stay invisible,
        # matching the git backend, which rev-parses any commit.
        for attempt in self.list(only_done=False):
            if attempt.id == id and attempt.path.name.endswith(("_done", "_fail")):
                return attempt
        return None

    def set_note(self, attempt_or_id, key: str, value: str) -> None:
        """Mutable annotation in a ``notes.json`` sidecar beside the record.

        The attempt record (solution/result/metadata) stays immutable —
        notes are an explicitly mutable scratch channel (e.g. the latest
        computed score cache)."""
        if not re.match(r"^[a-z0-9_-]{1,64}$", key or ""):
            raise ValueError(f"invalid note key {key!r} (use [a-z0-9_-], max 64)")
        a = self.get(attempt_or_id) if isinstance(attempt_or_id, str) else attempt_or_id
        if a is None:
            raise KeyError(f"unknown attempt {attempt_or_id!r}")
        notes_path = Path(a.path) / "notes.json"
        try:
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            notes = {}
        notes[key] = str(value)
        notes_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")

    def get_note(self, attempt_or_id, key: str) -> Optional[str]:
        a = self.get(attempt_or_id) if isinstance(attempt_or_id, str) else attempt_or_id
        if a is None:
            return None
        try:
            notes = json.loads((Path(a.path) / "notes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        v = notes.get(key)
        return None if v is None else str(v)

    def best(self, scorer: Callable[[StageResult], float]) -> Optional[FolderAttempt]:
        attempts = self.list()
        if not attempts:
            return None

        def score_attempt(attempt):
            # A done attempt may carry no result.json (committed without
            # --eval) — unscored, never a crash for everyone else.
            try:
                result = attempt.result
            except (OSError, ValueError):
                return -1.0
            if not result.completed:
                return -1.0
            stages = list(result.stages.values())
            if not stages:
                return -1.0
            return scorer(stages[-1])

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

    # --- In-progress lifecycle (list / resume / reap) ------------------
    # In-progress = an un-suffixed ``NNN_<parent>`` dir (no _done/_fail). It
    # persists on disc through a crash, so it is listable and resumable.

    def _is_in_progress_dir(self, d) -> bool:
        name = d.name
        return (d.is_dir() and not name.startswith(".claim_")
                and not name.endswith("_done") and not name.endswith("_fail"))

    def list_in_progress(self):
        from groundhog.base.attempt_history import InProgress
        items = []
        for d in sorted(self.base_path.iterdir()):
            if not self._is_in_progress_dir(d):
                continue
            parts = d.name.split("_", 1)
            try:
                wsid = str(int(parts[0]))
            except ValueError:
                continue
            parent = None if (len(parts) < 2 or parts[1] == "none") else parts[1]
            items.append(InProgress(
                workspace_id=wsid, parent=parent,
                started_at=d.stat().st_mtime, path=d, live=True))
        items.sort(key=lambda ip: ip.started_at)
        return items

    def resume(self, workspace_id: str) -> FolderWorkspace:
        for d in self.base_path.iterdir():
            if not self._is_in_progress_dir(d):
                continue
            parts = d.name.split("_", 1)
            try:
                if str(int(parts[0])) != str(workspace_id):
                    continue
            except ValueError:
                continue
            parent = None if (len(parts) < 2 or parts[1] == "none") else parts[1]
            return FolderWorkspace(display_id=str(workspace_id), parent=parent, path=d)
        raise KeyError(f"no in-progress workspace {workspace_id!r}")

    def reap_in_progress(self, ttl_s: float = 300.0) -> int:
        now = time.time()
        reaped = 0
        for ip in self.list_in_progress():
            if (now - ip.started_at) <= ttl_s:
                continue
            try:
                shutil.rmtree(ip.path)
                reaped += 1
            except OSError:
                pass
        return reaped
