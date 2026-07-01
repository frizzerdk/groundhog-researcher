"""pytest configuration for the groundhog test suite.

The ``e2e_mnist_agent`` directory holds recorded attempt artifacts, including
agent-written ``test_*.py`` files that match pytest's collection pattern but
are not part of this suite. Exclude the whole tree from collection.

Fixtures
--------
``history_factory`` — parametrized over [folder, git]; returns a zero-arg
callable that builds a fresh AttemptHistory of that backend in a tmp dir. The
git parameter is skipped when ``git`` is not on PATH.

``commit_attempt`` — runs the full workspace → write → commit cycle and
returns the committed Attempt. Backend-agnostic.
"""

import subprocess

import pytest

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.results import write_result
from groundhog.utils.direction import write_direction

collect_ignore = ["e2e_mnist_agent"]


def _git_available():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


@pytest.fixture(params=["folder", "git"])
def history_factory(request, tmp_path):
    """Build a fresh history of the parametrized backend, rooted in tmp."""
    if request.param == "git":
        if not _git_available():
            pytest.skip("git not on PATH")
        from groundhog.histories.git import GitAttemptHistory

        def make():
            return GitAttemptHistory(tmp_path / "store")
    else:
        def make():
            return FolderAttemptHistory(tmp_path / "store")

    return make


@pytest.fixture
def commit_attempt():
    """Helper: workspace → write solution/result/(direction) → commit."""

    def _commit(history, *, parent=None, code="def solve(): return 1",
                direction=None, metrics=None, completed=True, success=True,
                name=None):
        ws = history.workspace(parent=parent)
        (ws.path / "solution.py").write_text(code, encoding="utf-8")
        if direction is not None:
            write_direction(ws.path, direction)
        result = EvaluationResult(
            stages={"eval": StageResult(metrics=metrics or {"score": 1.0})},
            completed=completed,
        )
        write_result(ws.path, result)
        if name is not None:
            ws.name = name
        return ws.commit(success=success)

    return _commit
