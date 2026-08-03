"""Folder-based attempt history. Each attempt is a numbered directory.

Directory structure:
    TaskName/
        attempts/
            001_none/       ← first attempt (no parent)
                solution.py
                result.json
                metadata.json
                notes.json          (mutable annotations, e.g. score cache)
                TASK_CONTEXT.md
            002_1/          ← second attempt (parent=1)
                ...
        learnings.md        ← derived learnings digest (ledger lives in attempts)
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
from groundhog.utils.liveness import (
    clear_heartbeat, pid_alive, read_heartbeat, write_heartbeat)


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

    def __init__(self, display_id: str, parent: Optional[str], path: Path,
                 name: str = "", history: "Optional[FolderAttemptHistory]" = None):
        self.display_id = display_id
        self.parent = parent
        self.path = path
        self.name = name
        self._history = history
        # exist_ok=True so FolderAttemptHistory.workspace can pre-create the
        # directory as part of its atomic-claim handshake.
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "work").mkdir(exist_ok=True)
        self.heartbeat()

    def commit(self, success: bool = True) -> FolderAttempt:
        """Record this workspace as an attempt by renaming the folder.
        ``success=False`` records it as FAILED — recorded work, never
        discarded (CLI: ``commit --fail``).

        Strategy must write all files (solution.py, result.json, etc.)
        before calling commit(). This just flips the visibility flag.
        """
        if self.name:
            write_metadata(self.path, {"name": self.name})
        suffix = "_done" if success else "_fail"
        new_path = self.path.parent / (self.path.name + suffix)
        self.path.rename(new_path)
        self.path = new_path
        self._clear_heartbeat()
        return FolderAttempt(id=self.display_id, parent=self.parent, path=new_path)

    def abort(self):
        """Discard: delete the workspace folder, leaving NO record. To
        record failed work instead, use ``commit(success=False)``."""
        if self.path.exists():
            shutil.rmtree(self.path)
        self._clear_heartbeat()

    def heartbeat(self):
        """Refresh the pid+timestamp liveness marker so ``reap`` can tell a
        working session from a crashed one."""
        if self._history is not None:
            write_heartbeat(self._history._heartbeat(self.display_id))

    def _clear_heartbeat(self):
        if self._history is not None:
            clear_heartbeat(self._history._heartbeat(self.display_id))


class FolderAttemptHistory(AttemptHistory):
    """Each attempt is a directory: {number}_{parent}/"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path) / "attempts"
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Heartbeats for open workspaces — a hidden sibling of the attempt
        # dirs, so it is never mistaken for one.
        self._wip_dir = self.base_path / ".wip"
        self._wip_dir.mkdir(exist_ok=True)

    def _heartbeat(self, wsid: str) -> Path:
        return self._wip_dir / str(wsid)

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
            except PermissionError:
                # Windows: a sentinel being rmdir'd by its winner is briefly
                # in a "delete pending" state — mkdir at the same path raises
                # PermissionError instead of FileExistsError. Same contention,
                # same treatment: back off and re-scan.
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

            return FolderWorkspace(display_id=str(number), parent=parent,
                                   path=path, history=self)

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
            try:
                id = str(int(parts[0]))
            except ValueError:
                continue  # .wip, .claim_ sentinels, foreign dirs
            parent = None if parts[1] == "none" else parts[1]
            attempts.append(FolderAttempt(id=id, parent=parent, path=d))
        return attempts

    def get(self, id: str) -> Optional[FolderAttempt]:
        # Resolve any COMMITTED attempt — failed ones included (a child's
        # recorded parent may be failed; treating it as missing would
        # misclassify the child as fresh). Open workspaces stay invisible,
        # matching the git backend, which rev-parses any commit. An exact
        # id wins; otherwise a UNIQUE prefix resolves (the git backend gets
        # the same via rev-parse's abbreviated shas).
        id = str(id)
        committed = [a for a in self.list(only_done=False)
                     if a.path.name.endswith(("_done", "_fail"))]
        for attempt in committed:
            if attempt.id == id:
                return attempt
        matches = [a for a in committed if a.id.startswith(id)]
        return matches[0] if len(matches) == 1 else None

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
        from groundhog.utils.fileio import atomic_write_text, locked
        notes_path = Path(a.path) / "notes.json"
        # Locked read-modify-write: writes to DIFFERENT keys must never lose
        # each other (the per-tag ``tag-<name>`` keys rely on it).
        with locked(notes_path):
            try:
                notes = json.loads(notes_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                notes = {}
            notes[key] = str(value)
            atomic_write_text(notes_path, json.dumps(notes, indent=2))

    def get_note(self, attempt_or_id, key: str) -> Optional[str]:
        v = self.list_notes(attempt_or_id).get(key)
        return None if v is None else str(v)

    def list_notes(self, attempt_or_id) -> dict:
        a = self.get(attempt_or_id) if isinstance(attempt_or_id, str) else attempt_or_id
        if a is None:
            return {}
        try:
            notes = json.loads((Path(a.path) / "notes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return notes if isinstance(notes, dict) else {}

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
        return (d.is_dir() and not name.startswith(".")
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
            hb = read_heartbeat(self._heartbeat(wsid))
            # No heartbeat at all (pre-heartbeat store): dir mtime stands in
            # and the pid reads dead, so the ttl grace still applies.
            started = float(hb.get("started_at", 0.0)) if hb else d.stat().st_mtime
            items.append(InProgress(
                workspace_id=wsid, parent=parent, started_at=started,
                path=d, live=pid_alive(hb.get("pid"))))
        items.sort(key=lambda ip: ip.started_at)
        return items

    def resume(self, workspace_id: str) -> FolderWorkspace:
        """Re-acquire an open workspace by exact id or any UNIQUE prefix."""
        wanted = str(workspace_id)
        open_dirs = {}
        for d in self.base_path.iterdir():
            if not self._is_in_progress_dir(d):
                continue
            parts = d.name.split("_", 1)
            try:
                open_dirs[str(int(parts[0]))] = d
            except ValueError:
                continue
        if wanted not in open_dirs:
            matches = [i for i in open_dirs if i.startswith(wanted)]
            if len(matches) > 1:
                raise KeyError(f"ambiguous workspace id {wanted!r} "
                               f"(matches {', '.join(sorted(matches))})")
            if not matches:
                raise KeyError(f"no in-progress workspace {wanted!r}")
            wanted = matches[0]
        d = open_dirs[wanted]
        parts = d.name.split("_", 1)
        parent = None if (len(parts) < 2 or parts[1] == "none") else parts[1]
        return FolderWorkspace(display_id=wanted, parent=parent, path=d,
                               history=self)

    def reap_in_progress(self, ttl_s: float = 300.0) -> int:
        # Same contract as the git backend: LIVE sessions (heartbeat pid
        # alive) are left alone regardless of age; the TTL is a grace period
        # for dead pids (a missing heartbeat reads as dead).
        now = time.time()
        reaped = 0
        for ip in self.list_in_progress():
            if ip.live:
                continue
            if (now - ip.started_at) <= ttl_s:
                continue
            try:
                shutil.rmtree(ip.path)
            except OSError:
                continue
            clear_heartbeat(self._heartbeat(ip.workspace_id))
            reaped += 1
        return reaped
