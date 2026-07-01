"""Attempt History — the optimizer's memory of every candidate.

An immutable tree of attempts. Each attempt stores code, raw results (metrics,
not scores), artifacts, and a parent pointer. Scores are never persisted —
scoring is always a read-side concern via the Evaluator's scorer, so scoring
criteria can change without re-evaluating.

Workspace pattern: history.workspace(parent) → Workspace with a path.
Strategy works in the path (writes files, runs eval), then calls commit() or
abort(). Commit makes it an immutable attempt. Abort deletes everything.

Properties:
- Immutable — once committed, an attempt never changes
- Atomic — committed fully or aborted entirely
- Complete — nothing discarded; failed attempts are recorded too
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from groundhog.base.types import EvaluationResult, StageResult


@dataclass
class InProgress:
    """A workspace that was opened but not yet committed — live or crashed.

    Surfaced by ``AttemptHistory.list_in_progress`` so a user, an agent, or the
    optimizer can ``resume`` it (keep working, then commit) or ``reap`` it
    (abort a crashed one). ``live`` is a best-effort liveness signal (a fresh
    heartbeat); a stale ``live=False`` marks a crashed workspace.
    """
    workspace_id: str
    parent: Optional[str]
    started_at: float
    path: Optional[Path] = None
    live: bool = True


@dataclass
class SeedSpec:
    """The parent→child copy convention, made explicit.

    A child workspace copies ONLY this set from its parent — the parent's
    solution (a starting point) and core direction — never the parent's
    ``result.json``/``metadata.json``. Mirrors what the strategies do, so manual
    and agent-driven attempt creation behave identically.
    """
    copy_solution: bool = True
    solution_dest: str = "solution.py"
    inherit_direction: bool = True


class Attempt(ABC):
    """A committed attempt in the history. Read-only.

    Identity is ``id`` — an opaque string assigned at commit (the commit
    hash for the git backend, the attempt number for the folder backend).
    ``parent`` is the id of the parent attempt, or None for a root.
    ``created_at`` is a unix-epoch float and the canonical sort key.
    ``name`` is a human-readable, display-only label — never an identity or
    a lookup key.
    """
    id: str
    parent: Optional[str]
    created_at: float
    name: str

    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def result(self) -> EvaluationResult: ...

    @property
    @abstractmethod
    def metadata(self) -> dict: ...

    @property
    @abstractmethod
    def status(self) -> str:
        """``"done"`` | ``"fail"`` | ``"in-progress"``. Sourced from the git
        ``Status:`` trailer or the folder ``_done``/``_fail`` suffix."""
        ...

    @abstractmethod
    def list_files(self) -> List[str]:
        """List all files in this attempt (relative paths)."""
        ...

    @abstractmethod
    def read_file(self, path: str) -> Optional[str]:
        """Read a text file from this attempt. Returns None if not found.
        Returns a placeholder for binary files."""
        ...


class Workspace(ABC):
    """A working directory for one attempt. Start → work → commit or abort.

    A workspace has no committed ``id`` until ``commit()`` returns an
    Attempt. ``display_id`` is a human-facing in-flight label (the folder
    number, or the git temp-dir id); ``name`` is a mutable display label the
    strategy may set before commit. ``parent`` is the parent attempt id.
    """
    display_id: str
    name: str
    parent: Optional[str]
    path: Path

    @abstractmethod
    def commit(self, success: bool = True) -> Attempt:
        """Mark this workspace as a completed attempt. No going back."""
        ...

    @abstractmethod
    def abort(self):
        """Discard this workspace. No trace left."""
        ...

    def checkpoint(self):
        """Snapshot in-flight state so a crashed run can be resumed from here.

        Default: a no-op, and both shipped backends rely on it — folder and
        git workspaces live on disk, so in-flight edits already survive a
        crash (``resume()`` re-binds them). A backend that buffers state in
        memory would override this to flush.
        """
        return None

    def heartbeat(self):
        """Refresh this workspace's liveness marker.

        Default: a no-op (the folder backend has no liveness state). The git
        backend rewrites its pid+timestamp heartbeat so ``reap`` can tell a
        working session from a crashed one. Called by
        ``WorkspaceHandle.set_attempt``; long-running holders may call it
        directly.
        """
        return None


class AttemptHistory(ABC):
    """Storage and retrieval for all attempts."""

    @abstractmethod
    def workspace(self, parent: Optional[str] = None) -> Workspace:
        """Create a new workspace to work in. Call commit() or abort() when done."""
        ...

    @abstractmethod
    def list(self, only_done: bool = True) -> List[Attempt]: ...

    @abstractmethod
    def get(self, id: str) -> Optional[Attempt]: ...

    @abstractmethod
    def best(self, scorer: Callable[[StageResult], float]) -> Optional[Attempt]: ...

    @abstractmethod
    def lineage(self, attempt: Attempt) -> List[Attempt]: ...

    # --- In-progress lifecycle (manual / agent / crash-recovery) --------
    # Default implementations so existing backends keep working; the git and
    # folder backends override list_in_progress/resume/reap where they can
    # enumerate open workspaces.

    def list_in_progress(self) -> List[InProgress]:
        """Open (uncommitted) workspaces — live or crashed. Default: none."""
        return []

    def resume(self, workspace_id: str) -> Workspace:
        """Re-acquire an in-progress workspace to keep working, then commit/abort."""
        raise NotImplementedError("resume() is not supported by this backend")

    def reap_in_progress(self, ttl_s: float = 300.0) -> int:
        """Abort crashed (stale) in-progress workspaces; leave live ones.
        Returns the number reaped. Default: nothing to reap."""
        return 0

    def materialize(self, attempt_or_id) -> Path:
        """Ensure the attempt's files exist as a folder on disk; return it.

        What's on disk is dynamic — folders may be pruned, or a synced store
        may arrive as git objects with no worktrees. Consumers that need a
        real directory (the CLI, workspace-relative tools) call this instead
        of assuming one exists. Idempotent: a folder already on disk is
        returned as-is.

        Default covers path-carrying backends (folder); the git backend
        overrides to check out a worktree from the object store on demand.
        """
        a = self.get(attempt_or_id) if isinstance(attempt_or_id, str) else attempt_or_id
        if a is None:
            raise KeyError(f"unknown attempt {attempt_or_id!r}")
        p = getattr(a, "path", None)
        if p is not None and Path(p).exists():
            return Path(p)
        raise NotImplementedError(
            f"{type(self).__name__} cannot materialize attempt {a.id!r} to disk"
        )

    def seed_from_parent(self, ws: Workspace, parent: Optional[Attempt],
                         spec: Optional[SeedSpec] = None) -> None:
        """Copy ONLY the convention set parent→child (backend-agnostic).

        The parent's solution (a starting point) and core direction — never the
        parent's ``result.json``/``metadata.json``. Reads come from the
        committed Attempt API (object store on git, disk on folder), so it
        behaves identically on both. Use for manual/agent attempt creation.
        """
        if parent is None:
            return
        spec = spec or SeedSpec()
        from groundhog.utils.direction import inherit_direction_from_attempt
        code = getattr(parent, "code", None)
        if spec.copy_solution and code:
            dest = ws.path / spec.solution_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code, encoding="utf-8")
        if spec.inherit_direction:
            inherit_direction_from_attempt(parent, ws.path)

    def derive_trunks(self, scorer: Callable[[StageResult], float]) -> List[List[Attempt]]:
        """Find improvement chains — trunks are derived, not stored.

        A trunk is a chain from a root where each step improved on its parent
        under the given scorer. Change the scorer, get different trunks.
        """
        attempts = self.list()
        if not attempts:
            return []

        children = {}
        roots = []
        for a in attempts:
            if a.parent is None:
                roots.append(a)
            else:
                children.setdefault(a.parent, []).append(a)

        def score_attempt(attempt):
            result = attempt.result
            if not result.completed:
                return -1.0
            last = list(result.stages.values())[-1]
            return scorer(last)

        trunks = []
        for root in roots:
            trunk = [root]
            current = root
            current_score = score_attempt(current)
            while current.id in children:
                best_child = None
                best_score = current_score
                for child in children[current.id]:
                    s = score_attempt(child)
                    if s > best_score:
                        best_child = child
                        best_score = s
                if best_child is None:
                    break
                trunk.append(best_child)
                current = best_child
                current_score = best_score
            trunks.append(trunk)
        return trunks

    def derive_families(self) -> List[List[Attempt]]:
        """Group attempts by their core direction (read-side derived view).

        A family = attempts sharing the same normalized ``core_direction.md``
        content. Falls back to legacy ``approach.md``. Attempts with no
        direction file land in a "no-direction" sentinel family (stable key
        ``None``). Families are returned sorted by creation time of the
        oldest member, so the seed family appears first.

        See ``Optimizer/Implementation Details/Family Identity.md``.
        """
        from groundhog.utils.direction import (
            read_direction_from_attempt,
            normalize_direction,
        )

        groups: dict = {}
        for a in self.list():
            text = read_direction_from_attempt(a)
            key = normalize_direction(text) if text else None
            groups.setdefault(key, []).append(a)
        # Sort each group by creation time; sort families by oldest member.
        # Tie-break on id so ordering stays deterministic when timestamps tie.
        for members in groups.values():
            members.sort(key=lambda x: (x.created_at, x.id))
        return sorted(groups.values(), key=lambda g: (g[0].created_at, g[0].id))
