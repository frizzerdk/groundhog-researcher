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

``attempt_without_result`` — committed done attempt with no result.json
(the ``groundhog attempt commit`` no-``--eval`` shape).

``failed_attempt_without_code`` — failed attempt lacking solution.py.

``open_workspace_alongside`` — a run with committed attempts plus one
open (in-progress) workspace; yields ``(history, committed, open_ws)``
as a namespace.
"""

import subprocess
from types import SimpleNamespace

import pytest

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.results import write_metadata, write_result
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
                name=None, metadata=None):
        ws = history.workspace(parent=parent)
        (ws.path / "solution.py").write_text(code, encoding="utf-8")
        if direction is not None:
            write_direction(ws.path, direction)
        result = EvaluationResult(
            stages={"eval": StageResult(metrics=metrics or {"score": 1.0})},
            completed=completed,
        )
        write_result(ws.path, result, metadata=metadata)
        if name is not None:
            ws.name = name
        return ws.commit(success=success)

    return _commit


@pytest.fixture
def attempt_without_result(history_factory):
    """Committed done attempt with no result.json — the shape `groundhog
    attempt commit` (no --eval) leaves: solution + metadata carrying the
    ``no_recorded_result`` flag, nothing else."""
    history = history_factory()
    ws = history.workspace()
    (ws.path / "solution.py").write_text("def solve(): return 1",
                                         encoding="utf-8")
    write_metadata(ws.path, {"strategy": "manual", "prior": None, "cost": 0.0,
                             "no_recorded_result": True})
    return ws.commit(success=True)


@pytest.fixture
def failed_attempt_without_code(history_factory):
    """Failed attempt lacking solution.py — generation crashed before any
    code existed; only the failed result was recorded."""
    history = history_factory()
    ws = history.workspace()
    result = EvaluationResult(
        stages={"generate": StageResult(errors={"crash": "generation failed"})},
        completed=False, failed_stage="generate")
    write_result(ws.path, result)
    return ws.commit(success=False)


@pytest.fixture
def open_workspace_alongside(history_factory, commit_attempt):
    """A run mid-flight: two committed attempts plus one open (in-progress)
    workspace parented on the latest."""
    history = history_factory()
    first = commit_attempt(history, metrics={"score": 0.3})
    second = commit_attempt(history, parent=first.id, metrics={"score": 0.6})
    ws = history.workspace(parent=second.id)
    (ws.path / "solution.py").write_text("def solve(): return 0",
                                         encoding="utf-8")
    return SimpleNamespace(history=history, committed=[first, second],
                           open_ws=ws)
