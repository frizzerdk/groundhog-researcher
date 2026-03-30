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
from pathlib import Path
from typing import Callable, List, Optional

from groundhog.base.types import EvaluationResult, StageResult


class Attempt(ABC):
    """A committed attempt in the history. Read-only."""
    number: int
    parent: Optional[int]

    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def result(self) -> EvaluationResult: ...

    @property
    @abstractmethod
    def metadata(self) -> dict: ...


class Workspace(ABC):
    """A working directory for one attempt. Start → work → commit or abort."""
    number: int
    parent: Optional[int]
    path: Path

    @abstractmethod
    def commit(self, result: EvaluationResult, metadata: Optional[dict] = None) -> Attempt:
        """Finalize this workspace as an immutable attempt."""
        ...

    @abstractmethod
    def abort(self):
        """Discard this workspace. No trace left."""
        ...


class AttemptHistory(ABC):
    """Storage and retrieval for all attempts."""

    @abstractmethod
    def workspace(self, parent: Optional[int] = None) -> Workspace:
        """Create a new workspace to work in. Call commit() or abort() when done."""
        ...

    @abstractmethod
    def list(self) -> List[Attempt]: ...

    @abstractmethod
    def get(self, number: int) -> Optional[Attempt]: ...

    @abstractmethod
    def best(self, scorer: Callable[[StageResult], float]) -> Optional[Attempt]: ...

    @abstractmethod
    def lineage(self, attempt: Attempt) -> List[Attempt]: ...

    def derive_trunks(self, scorer: Callable[[StageResult], float]) -> List[List[Attempt]]:
        """Find improvement chains — trunks are derived, not stored.

        A trunk is a chain from a root where each step improved on its parent
        under the given scorer. Change the scorer, get different trunks.
        """
        attempts = self.list()
        if not attempts:
            return []

        by_number = {a.number: a for a in attempts}
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
            while current.number in children:
                best_child = None
                best_score = current_score
                for child in children[current.number]:
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
