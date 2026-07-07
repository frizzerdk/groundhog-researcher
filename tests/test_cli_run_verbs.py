"""Tests for `groundhog run [N]` and `groundhog status` — running a run dir's
optimizer through the run-dir contract, without task.py __main__ boilerplate.
"""

from groundhog import rundir
from groundhog.cli import _resolve_optimizer, attempt_group, cmd_run, cmd_status

from test_cli_attempt import _TASK_BODY, _in_dir, _write_run_dir


_MODULE_OPTIMIZER = '''

class _RecordingOptimizer:
    def run(self, n=10):
        print(f"RECORDED RUN n={n}")


optimizer = _RecordingOptimizer()
'''

_BUILD_OPTIMIZER_HOOK = '''

def build_optimizer(toolkit):
    class _HookOptimizer:
        def run(self, n=10):
            print(f"HOOK RUN n={n} task={toolkit.task.name}")
    return _HookOptimizer()
'''

_REAL_OPTIMIZER_HOOK = '''

def build_optimizer(toolkit):
    from groundhog import SimpleOptimizer

    class TinyStrategy:
        def __call__(self, tk, config=None):
            ws = tk.history.workspace(parent=None)
            (ws.path / "solution.py").write_text(
                "def solve():\\n    return 50.0\\n", encoding="utf-8")
            (ws.path / "core_direction.md").write_text(
                "constant baseline\\n", encoding="utf-8")
            result = tk.task.evaluate(ws.path)
            tk.finalize(ws, result, None, strategy="tiny")
            return {}

    return SimpleOptimizer(toolkit, strategy=TinyStrategy(), seed_strategy=None)
'''


def _append_to_task(run_dir, snippet):
    task_py = run_dir / "task.py"
    task_py.write_text(task_py.read_text(encoding="utf-8") + snippet,
                       encoding="utf-8")


def test_run_uses_module_level_optimizer(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _append_to_task(run_dir, _MODULE_OPTIMIZER)
    with _in_dir(run_dir):
        rc = cmd_run(["3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "module-level optimizer" in out
    assert "RECORDED RUN n=3" in out


def test_run_uses_build_optimizer_hook_with_toolkit(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _append_to_task(run_dir, _BUILD_OPTIMIZER_HOOK)
    with _in_dir(run_dir):
        rc = cmd_run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "build_optimizer()" in out
    assert "HOOK RUN n=10 task=TinyTask" in out


def test_run_default_n_and_bad_n(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _append_to_task(run_dir, _MODULE_OPTIMIZER)
    with _in_dir(run_dir):
        assert cmd_run(["ten"]) == 1
        assert "N must be an integer" in capsys.readouterr().out
        assert cmd_run([]) == 0
        assert "RECORDED RUN n=10" in capsys.readouterr().out


def test_run_falls_back_to_default_rotation(tmp_path):
    """No optimizer/build_optimizer in task.py -> the classic SimpleOptimizer
    rotation over the loaded toolkit, and the choice is announced."""
    from groundhog.optimizers.simple import SimpleOptimizer

    run_dir = _write_run_dir(tmp_path)
    run = rundir.load_run(run_dir=run_dir)
    optimizer, chosen = _resolve_optimizer(run)
    assert isinstance(optimizer, SimpleOptimizer)
    assert optimizer.toolkit is run.toolkit
    assert "default SimpleOptimizer" in chosen
    assert "Improve" in chosen


def test_run_end_to_end_commits_attempts(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _append_to_task(run_dir, _REAL_OPTIMIZER_HOOK)
    with _in_dir(run_dir):
        rc = cmd_run(["1"])
        assert rc == 0
    capsys.readouterr()
    history = rundir.load_run(run_dir=run_dir).history
    attempts = history.list()
    assert len(attempts) == 1
    assert attempts[0].metadata["strategy"] == "tiny"


def test_run_rejects_non_optimizer_hook(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _append_to_task(run_dir, "\n\ndef build_optimizer(toolkit):\n    return 42\n")
    with _in_dir(run_dir):
        rc = cmd_run(["1"])
    assert rc == 1
    assert "expected an optimizer with .run(n)" in capsys.readouterr().out


def test_status_empty_run(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = cmd_status([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "attempts: 0 (0 done, 0 failed)" in out
    assert "in-progress: none" in out


def test_status_shows_summary_and_in_progress(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        # One committed attempt.
        attempt_group(["new"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines()
                if l.startswith("Opened workspace")][0].split()[-1]
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n",
                                             encoding="utf-8")
        (ws_path / "core_direction.md").write_text("constant baseline\n",
                                                   encoding="utf-8")
        attempt_group(["commit", wsid, "--eval"])
        capsys.readouterr()

        # One open workspace.
        attempt_group(["new"])
        out = capsys.readouterr().out
        open_wsid = [l for l in out.splitlines()
                     if l.startswith("Opened workspace")][0].split()[-1]

        rc = cmd_status([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "attempts: 1 (1 done, 0 failed)" in out
        assert "best:" in out and "1.0000" in out
        assert "families: 1" in out
        assert "constant baseline" in out
        assert "in-progress:" in out
        assert open_wsid in out


def test_run_and_status_help(capsys):
    assert cmd_run(["-h"]) == 0
    assert cmd_status(["--help"]) == 0
