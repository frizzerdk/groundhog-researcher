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


def build_toolkit():
    """The run-dir contract: assemble + configure the bench, never run."""
    from groundhog import assemble_toolkit
    here = Path(__file__).parent
    if (here / "attempts" / ".git").exists():
        from groundhog import GitAttemptHistory
        history = GitAttemptHistory(here)
    else:
        from groundhog import FolderAttemptHistory
        history = FolderAttemptHistory(here)
    return assemble_toolkit(task, history=history, path=here)
'''

def _write_run_dir(tmp_path, *, git=False, no_hook=False):
    """Create a run dir with a contract-shaped task.py. ``git=True`` pre-seeds
    a git attempt store so build_toolkit() picks the git backend;
    ``no_hook=True`` strips build_toolkit to exercise the contract error."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = _TASK_BODY
    if no_hook:
        body = body.split("def build_toolkit():")[0]
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


def test_loader_requires_build_toolkit(tmp_path):
    """The run-dir contract: no build_toolkit() -> a clear, actionable error
    (the old monkeypatch-and-scan fallback is gone by design)."""
    run_dir = _write_run_dir(tmp_path, no_hook=True)
    with pytest.raises(RuntimeError, match="build_toolkit"):
        rundir.load_run(run_dir=run_dir)


def test_loader_rejects_non_toolkit_return(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    task_py = run_dir / "task.py"
    body = task_py.read_text(encoding="utf-8")
    body = body.split("def build_toolkit():")[0] + (
        "def build_toolkit():\n    return 42\n"
    )
    task_py.write_text(body, encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected a Toolkit"):
        rundir.load_run(run_dir=run_dir)


def test_loaded_history_is_the_runs_real_store(tmp_path):
    """The toolkit's history IS the store the run uses — the loader never
    reconstructs its own (the old wrong-store risk)."""
    run_dir = _write_run_dir(tmp_path)
    loaded = rundir.load_run(run_dir=run_dir)
    assert loaded.history is loaded.toolkit.history


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
    """cwd and sys.path must be restored after load (the loader deliberately
    chdirs into the run dir around the build_toolkit() call)."""
    import sys

    run_dir = _write_run_dir(tmp_path)
    cwd_before = os.getcwd()
    path_before = list(sys.path)
    rundir.load_run(run_dir=run_dir)
    assert os.getcwd() == cwd_before
    assert sys.path == path_before


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


# --- groundhog tool list / run ------------------------------------------------

_TOOL_HOOK = '''

def agent_tools(toolkit):
    from groundhog import agent_tool
    ws = toolkit.ws  # stable handle, captured once — the task.py pattern

    def read_current() -> str:
        return (ws.path / "solution.py").read_text(encoding="utf-8")

    def greet(name: str = "world") -> str:
        return f"hello {name}"

    return [
        agent_tool(name="read-current", description="Read the current attempt's solution.",
                   func=read_current, params={}),
        agent_tool(name="greet", description="Say hello.", func=greet,
                   params={"name": {"type": "str", "default": "world"}}),
    ]
'''


def _write_tool_run_dir(tmp_path):
    """Contract task.py whose build_toolkit wires the agent_tools hook."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = _TASK_BODY.replace(
        "    return assemble_toolkit(task, history=history, path=here)",
        "    return assemble_toolkit(task, history=history, path=here, agent_tools=agent_tools)",
    ) + _TOOL_HOOK
    (run_dir / "task.py").write_text(body, encoding="utf-8")
    return run_dir


def test_tool_list(tmp_path, capsys):
    from groundhog.cli import tool_group
    run_dir = _write_tool_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = tool_group(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "read-current" in out and "greet" in out


def test_tool_run_with_params(tmp_path, capsys):
    from groundhog.cli import tool_group
    run_dir = _write_tool_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = tool_group(["run", "greet", "-p", "name=frederik"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hello frederik" in out


def test_tool_run_ws_relative_against_committed_attempt(tmp_path, capsys):
    """A build-time tool closing over toolkit.ws reads a CHOSEN attempt's
    files when the CLI points the handle via --attempt."""
    from groundhog.cli import attempt_group, tool_group
    run_dir = _write_tool_run_dir(tmp_path)
    with _in_dir(run_dir):
        assert attempt_group(["new"]) == 0
        wsid = capsys.readouterr().out.strip().splitlines()[0].split()[-1]
        # write a distinctive solution, commit it
        ws_dirs = [p for p in (run_dir / "attempts").iterdir() if p.is_dir()]
        (ws_dirs[0] / "solution.py").write_text("def solve():\n    return 41.5\n",
                                                encoding="utf-8")
        assert attempt_group(["commit", wsid]) == 0
        capsys.readouterr()

        from groundhog import rundir
        aid = rundir.load_run().history.list()[0].id

        rc = tool_group(["run", "read-current", "--attempt", aid])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "41.5" in out


def test_tool_run_unset_ws_fails_clean(tmp_path, capsys):
    from groundhog.cli import tool_group
    run_dir = _write_tool_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = tool_group(["run", "read-current"])   # no --attempt, nothing set
    out = capsys.readouterr().out
    assert rc == 2
    assert "no attempt in flight" in out


def test_tool_run_unknown_tool(tmp_path, capsys):
    from groundhog.cli import tool_group
    run_dir = _write_tool_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = tool_group(["run", "nope"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No tool named" in out
