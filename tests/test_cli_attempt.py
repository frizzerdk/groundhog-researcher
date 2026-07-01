"""Tests for the `groundhog attempt` group, `groundhog eval`, and the run-dir
loader (src/groundhog/rundir.py).

These exercise the manual attempt lifecycle without any LLM. Each test builds a
self-contained tmp run dir with a minimal ``task.py`` defining a module-level
``task`` (a tiny real Task whose evaluator scores from ``solution.py``), then
calls the CLI handlers directly with arg lists and asserts exit codes + stdout.
"""

import os
import subprocess
from contextlib import contextmanager

import pytest

from groundhog import rundir
from groundhog.cli import attempt_group, cmd_eval


# --- Minimal, dependency-free task.py contents -----------------------------
#
# The task: write solve() returning a float; score = how close to a target.
# The evaluator reads solution.py from the workspace dir (Path) OR a code
# string, mirroring the real polymorphic Task.evaluate contract.

_TASK_BODY = '''
from pathlib import Path
from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult


class TinyData(Data):
    def get_train(self): return {"target": 50.0}
    def get_test(self): return {"target": 50.0}


class TinyContext(Context):
    def get_brief(self): return "Write solve() returning a float near 50."
    def get_extended(self): return "def solve() -> float"


def _read(code_or_path):
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


def _score(result):
    # Score read-side from metrics — result.score is NOT persisted (write_result
    # only stores metrics/errors/warnings), mirroring the real system where the
    # canonical scorer always interprets metrics, never the in-memory .score.
    if result.errors:
        return -1.0
    m = result.metrics
    if "value" not in m:
        return 0.0
    return max(0.0, 1.0 - abs(m["value"] - m["target"]) / 100.0)


class TinyEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read(code_or_path)
        ns = {}
        try:
            exec(code, ns)
            value = float(ns["solve"]())
        except Exception as e:
            return StageResult(errors={"crash": str(e)})
        target = data.get_test()["target"]
        score = max(0.0, 1.0 - abs(value - target) / 100.0)
        return StageResult(score=score, metrics={"value": value, "target": target})

    def get_stages(self, data):
        return [
            EvalStage("evaluate", "full",
                      lambda cp, d=data: self.evaluate(cp, d),
                      scorer=_score),
        ]


task = Task(data=TinyData(), context=TinyContext(), evaluator=TinyEvaluator(),
            name="TinyTask")
'''

# Unguarded variant: builds an optimizer and calls run()/status() at module
# top level, reading sys.argv — exactly like C:/repo/groundhog-runs/task.py.
# The loader must NOT let this fire a real optimization or print status. A real
# run() would seed an attempt into history (seed_strategy writes solution.py +
# commits), so an empty history after load proves run() was neutralized.
_UNGUARDED_TAIL = '''
import sys
from pathlib import Path
from groundhog import SimpleOptimizer, assemble_toolkit

from groundhog.strategies.fresh import FreshApproach

# A trivial seed strategy that commits one attempt — if run() actually executes
# its seeding path, history would become non-empty.
class _SeedOnce(FreshApproach):
    def __call__(self, toolkit, *a, **k):
        ws = toolkit.history.workspace(parent=None)
        (ws.path / "solution.py").write_text("def solve():\\n    return 50.0\\n", encoding="utf-8")
        from groundhog.base.types import EvaluationResult, StageResult
        from groundhog.utils.results import write_result
        write_result(ws.path, EvaluationResult(stages={"evaluate": StageResult(metrics={"value": 50.0})}))
        ws.commit(success=True)
        return {}

optimizer = SimpleOptimizer(assemble_toolkit(task, path=Path(__file__).parent),
                            seed_strategy=_SeedOnce())

if len(sys.argv) > 1 and sys.argv[1] == "status":
    optimizer.status()
else:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    optimizer.run(n)
'''


def _write_run_dir(tmp_path, *, unguarded=False, git=False):
    """Create a run dir with a task.py. Optionally make it unguarded and/or
    pre-seed a git attempt store so the loader picks the git backend."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = _TASK_BODY
    if unguarded:
        body = body + _UNGUARDED_TAIL
    (run_dir / "task.py").write_text(body, encoding="utf-8")

    if git:
        from groundhog.histories.git import GitAttemptHistory
        # Constructing it creates attempts/.git (a bare store).
        GitAttemptHistory(run_dir)
    return run_dir


@contextmanager
def _in_dir(path):
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def _git_available():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# --- Loader tests ----------------------------------------------------------

def test_load_run_folder(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    loaded = rundir.load_run(run_dir=run_dir)
    assert loaded.task is not None
    assert loaded.task.name == "TinyTask"
    assert loaded.toolkit is not None
    assert loaded.history is not None
    # Folder backend by default.
    from groundhog.histories.folder import FolderAttemptHistory
    assert isinstance(loaded.history, FolderAttemptHistory)
    assert loaded.history.list() == []


def test_load_run_git_backend(tmp_path):
    if not _git_available():
        pytest.skip("git not on PATH")
    run_dir = _write_run_dir(tmp_path, git=True)
    loaded = rundir.load_run(run_dir=run_dir)
    from groundhog.histories.git import GitAttemptHistory
    assert isinstance(loaded.history, GitAttemptHistory)


def test_loader_does_not_run_unguarded_optimizer(tmp_path):
    """An unguarded task.py calling optimizer.run(99) at module level must load
    without running — no attempts created, no sentinel written."""
    run_dir = _write_run_dir(tmp_path, unguarded=True)
    loaded = rundir.load_run(run_dir=run_dir)
    assert loaded.optimizer is not None
    assert loaded.task is not None
    # run() was neutralized to a no-op, so its seed strategy never committed
    # an attempt — history is empty.
    assert loaded.history.list() == []
    assert loaded.history.list(only_done=False) == []


def test_find_task_py_searches_parents(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    sub = run_dir / "nested" / "deeper"
    sub.mkdir(parents=True)
    with _in_dir(sub):
        found = rundir.find_task_py()
    assert found == (run_dir / "task.py").resolve()


def test_find_task_py_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        rundir.find_task_py(run_dir=empty)


def test_loader_restores_environment(tmp_path):
    """cwd, sys.argv, and SimpleOptimizer.run must be restored after load."""
    import sys
    from groundhog.optimizers.simple import SimpleOptimizer

    run_dir = _write_run_dir(tmp_path, unguarded=True)
    cwd_before = os.getcwd()
    argv_before = list(sys.argv)
    run_before = SimpleOptimizer.run
    rundir.load_run(run_dir=run_dir)
    assert os.getcwd() == cwd_before
    assert sys.argv == argv_before
    assert SimpleOptimizer.run is run_before


# --- attempt new / commit --eval / list / show ------------------------------

def test_attempt_new_and_commit_eval(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        # new
        rc = attempt_group(["new"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Opened workspace" in out
        # extract the workspace id from "Opened workspace <id>"
        wsid = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]

        # write a perfect solution into the workspace
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n",
                                             encoding="utf-8")

        # commit --eval
        rc = attempt_group(["commit", wsid, "--eval"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Evaluation:" in out
        assert "Committed attempt" in out

        # list shows it with a real score
        rc = attempt_group(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1.0000" in out  # perfect score

        # show prints details + files
        rc = attempt_group(["show", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "status:" in out
        assert "solution.py" in out

        # the result.json was written
        attempt = rundir.load_run(run_dir=run_dir).history.get("1")
        assert attempt is not None
        assert attempt.result.completed
        assert "result.json" in attempt.list_files()


def test_attempt_commit_eval_failing_solution(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        attempt_group(["new"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]

        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    raise ValueError('boom')\n",
                                             encoding="utf-8")

        rc = attempt_group(["commit", wsid, "--eval"])
        # commit itself succeeds (exit 0); the attempt is finalized as a fail.
        assert rc == 0
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "(fail)" in out

        # It shows up only under --all.
        capsys.readouterr()
        attempt_group(["list"])
        assert "No attempts yet." in capsys.readouterr().out
        attempt_group(["list", "--all"])
        assert "fail" in capsys.readouterr().out


def test_attempt_new_seeds_from_parent(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        # First attempt.
        attempt_group(["new"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n",
                                             encoding="utf-8")
        attempt_group(["commit", wsid, "--eval"])
        capsys.readouterr()

        # New child (default parent = best). Seed copies parent's solution.py.
        rc = attempt_group(["new"])
        assert rc == 0
        out = capsys.readouterr().out
        wsid2 = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]
        assert "parent: 1" in out

        loaded = rundir.load_run(run_dir=run_dir)
        ws2 = [ip.path for ip in loaded.history.list_in_progress()
               if ip.workspace_id == wsid2][0]
        assert (ws2 / "solution.py").read_text(encoding="utf-8").strip().endswith("return 50.0")


# --- in-progress / resume / abort / reap ------------------------------------

def test_in_progress_resume_abort(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        attempt_group(["new"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]

        rc = attempt_group(["in-progress"])
        assert rc == 0
        assert wsid in capsys.readouterr().out

        rc = attempt_group(["resume", wsid])
        assert rc == 0
        assert "Resumed workspace" in capsys.readouterr().out

        rc = attempt_group(["abort", wsid])
        assert rc == 0
        assert "aborted" in capsys.readouterr().out

        # Now gone.
        capsys.readouterr()
        attempt_group(["in-progress"])
        assert "No in-progress" in capsys.readouterr().out


def test_reap_empty(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = attempt_group(["reap"])
        assert rc == 0
        assert "reaped 0" in capsys.readouterr().out


def test_attempt_best_and_empty(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = attempt_group(["best"])
        assert rc == 0
        assert "No attempts yet." in capsys.readouterr().out


def test_attempt_show_missing(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = attempt_group(["show", "999"])
        assert rc == 1
        assert "No such attempt" in capsys.readouterr().out


def test_attempt_unknown_subcommand(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = attempt_group(["frobnicate"])
        assert rc == 1
        assert "Unknown attempt subcommand" in capsys.readouterr().out


def test_attempt_help():
    assert attempt_group([]) == 0
    assert attempt_group(["-h"]) == 0


# --- eval ------------------------------------------------------------------

def test_eval_directory(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    sol_dir = tmp_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "solution.py").write_text("def solve():\n    return 50.0\n", encoding="utf-8")
    with _in_dir(run_dir):
        rc = cmd_eval([str(sol_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "overall: 1.0000" in out
        assert "completed" in out


def test_eval_py_file(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    py = tmp_path / "candidate.py"
    py.write_text("def solve():\n    return 0.0\n", encoding="utf-8")
    with _in_dir(run_dir):
        rc = cmd_eval([str(py)])
        assert rc == 0
        out = capsys.readouterr().out
        # value=0, target=50 -> score 0.5
        assert "0.5000" in out


def test_eval_attempt_id(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        attempt_group(["new"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines() if l.startswith("Opened workspace")][0].split()[-1]
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n", encoding="utf-8")
        attempt_group(["commit", wsid, "--eval"])
        capsys.readouterr()

        rc = cmd_eval(["1"])
        assert rc == 0
        assert "overall: 1.0000" in capsys.readouterr().out


def test_eval_json(tmp_path, capsys):
    import json

    run_dir = _write_run_dir(tmp_path)
    py = tmp_path / "candidate.py"
    py.write_text("def solve():\n    return 50.0\n", encoding="utf-8")
    with _in_dir(run_dir):
        rc = cmd_eval([str(py), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["completed"] is True
        assert data["overall_score"] == 1.0
        assert "evaluate" in data["stages"]


def test_eval_failing_returns_2(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    py = tmp_path / "candidate.py"
    py.write_text("def solve():\n    raise ValueError('boom')\n", encoding="utf-8")
    with _in_dir(run_dir):
        rc = cmd_eval([str(py)])
        # Eval ran but failed the gate -> exit 2.
        assert rc == 2
        assert "FAILED" in capsys.readouterr().out


def test_eval_unresolvable_target(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = cmd_eval(["does-not-exist"])
        assert rc == 1
        assert "Cannot resolve eval target" in capsys.readouterr().out


def test_eval_help(capsys):
    assert cmd_eval(["-h"]) == 0
    assert cmd_eval([]) == 1
