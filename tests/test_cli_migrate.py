"""Tests for `groundhog init --git` (task.py wired to GitAttemptHistory) and
`groundhog migrate-store <dest>` (folder store replayed through the git
backend, source untouched)."""

import json
from pathlib import Path

import pytest

from groundhog import rundir
from groundhog.cli import attempt_group, cmd_migrate_store, init
from groundhog.tools.migrate import wire_git_history

from test_cli_attempt import _TASK_BODY, _git_available, _in_dir

TEMPLATES_DIR = Path(__file__).parents[1] / "src" / "groundhog" / "templates"

needs_git = pytest.mark.skipif(not _git_available(), reason="git not on PATH")


# --- wire_git_history (the shared task.py patcher) ---------------------------

@pytest.mark.parametrize("template", ["basic.py", "llm_guide.py",
                                      "mnist_task.py", "mock_task.py"])
def test_wire_git_history_patches_every_template(tmp_path, template):
    task_py = tmp_path / "task.py"
    task_py.write_text((TEMPLATES_DIR / template).read_text(encoding="utf-8"),
                       encoding="utf-8")
    assert wire_git_history(task_py) is True
    text = task_py.read_text(encoding="utf-8")
    assert "assemble_toolkit(task, history=history" in text
    assert "history = GitAttemptHistory(_pathlib.Path(__file__).parent)" in text
    # The commented remote= line travels with the wiring.
    assert 'remote="git@github.com:you/attempts-store.git"' in text
    # Still valid Python (llm_guide has a commented assemble_toolkit call
    # that must not be the patch target).
    compile(text, str(task_py), "exec")
    assert "# history=history" not in text


def test_wire_git_history_is_idempotent(tmp_path):
    task_py = tmp_path / "task.py"
    task_py.write_text((TEMPLATES_DIR / "basic.py").read_text(encoding="utf-8"),
                       encoding="utf-8")
    wire_git_history(task_py)
    once = task_py.read_text(encoding="utf-8")
    assert wire_git_history(task_py) is True
    assert task_py.read_text(encoding="utf-8") == once


def test_wire_git_history_unpatchable(tmp_path):
    task_py = tmp_path / "task.py"
    task_py.write_text("task = None\n", encoding="utf-8")
    assert wire_git_history(task_py) is False


# --- init --git ---------------------------------------------------------------

@needs_git
def test_init_git_scaffolds_git_backed_run(tmp_path, capsys):
    target = tmp_path / "my_task"
    rc = init("init-mock", str(target), script_only=True, use_git=True)
    assert rc == 0
    assert "GitAttemptHistory" in capsys.readouterr().out

    loaded = rundir.load_run(run_dir=target)
    from groundhog.histories.git import GitAttemptHistory
    assert isinstance(loaded.history, GitAttemptHistory)
    assert (target / "attempts" / ".git").exists()


def test_init_without_git_flag_unchanged(tmp_path, capsys):
    target = tmp_path / "my_task"
    rc = init("init-mock", str(target), script_only=True)
    assert rc == 0
    text = (target / "task.py").read_text(encoding="utf-8")
    assert "GitAttemptHistory" not in text


# --- migrate-store -------------------------------------------------------------

# A folder-only build_toolkit, so migration exercises the task.py patch.
_FOLDER_TASK_BODY = _TASK_BODY.split("def build_toolkit():")[0] + '''
def build_toolkit():
    from groundhog import assemble_toolkit
    here = Path(__file__).parent
    return assemble_toolkit(task, path=here)
'''


def _write_folder_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(_FOLDER_TASK_BODY, encoding="utf-8")
    return run_dir


def _commit(run_dir, capsys, *, value="50.0", direction=None, parent=None,
            fail=False):
    cmd = ["new", "--no-seed"] + (["--parent", parent] if parent
                                  else ["--fresh"])
    attempt_group(cmd)
    out = capsys.readouterr().out
    wsid = [l for l in out.splitlines()
            if l.startswith("Opened workspace")][0].split()[-1]
    loaded = rundir.load_run(run_dir=run_dir)
    ws_path = [ip.path for ip in loaded.history.list_in_progress()
               if ip.workspace_id == wsid][0]
    body = ("def solve():\n    raise ValueError('boom')\n" if fail
            else f"def solve():\n    return {value}\n")
    (ws_path / "solution.py").write_text(body, encoding="utf-8")
    if direction:
        (ws_path / "core_direction.md").write_text(direction + "\n",
                                                   encoding="utf-8")
    attempt_group(["commit", wsid, "--eval"])
    capsys.readouterr()
    return wsid


def _populated_run(tmp_path, capsys):
    """Root (done) -> child (done), plus one fresh fail. Score notes exist
    (the CLI's standard finish caches them)."""
    run_dir = _write_folder_run(tmp_path)
    with _in_dir(run_dir):
        _commit(run_dir, capsys, value="40.0", direction="constant baseline")
        _commit(run_dir, capsys, value="50.0", parent="1")
        _commit(run_dir, capsys, direction="doomed family", fail=True)
    return run_dir


@needs_git
def test_migrate_store_replays_the_tree(tmp_path, capsys):
    run_dir = _populated_run(tmp_path, capsys)
    (run_dir / ".venv").mkdir()
    (run_dir / ".venv" / "junk.txt").write_text("x", encoding="utf-8")
    dest = tmp_path / "migrated"

    with _in_dir(run_dir):
        rc = cmd_migrate_store([str(dest)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "found 3 committed attempts (2 done, 1 fail), 3 score notes" in out
    assert "task.py wired to GitAttemptHistory" in out

    # Copy shape: run files travel, .venv and the old store don't.
    assert (dest / "task.py").exists()
    assert not (dest / ".venv").exists()
    assert (dest / "attempts" / ".git").exists()

    loaded = rundir.load_run(run_dir=dest)
    from groundhog.histories.git import GitAttemptHistory
    assert isinstance(loaded.history, GitAttemptHistory)

    migrated = loaded.history.list(only_done=False)
    assert len(migrated) == 3
    by_folder_id = {a.metadata["migrated_from_folder_id"]: a for a in migrated}
    assert set(by_folder_id) == {1, 2, 3}

    # Status + lineage preserved: child of 1 points at 1's new sha.
    assert by_folder_id[1].status == "done"
    assert by_folder_id[2].status == "done"
    assert by_folder_id[3].status == "fail"
    assert by_folder_id[2].parent == by_folder_id[1].id
    assert by_folder_id[1].parent is None

    # Score notes replayed; results readable through the git backend.
    src_history = rundir.load_run(run_dir=run_dir).history
    for folder_id in (1, 2):
        note = loaded.history.get_note(by_folder_id[folder_id], "score")
        assert note == src_history.get_note(str(folder_id), "score")
    assert by_folder_id[2].result.completed

    # Source untouched.
    assert len(src_history.list(only_done=False)) == 3


@needs_git
def test_migrate_store_refuses_in_progress(tmp_path, capsys):
    run_dir = _populated_run(tmp_path, capsys)
    with _in_dir(run_dir):
        attempt_group(["new", "--fresh", "--no-seed"])
        capsys.readouterr()
        rc = cmd_migrate_store([str(tmp_path / "migrated")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "in-progress workspaces exist" in out
    assert not (tmp_path / "migrated").exists()


def test_migrate_store_refuses_existing_dest_and_nested_dest(tmp_path, capsys):
    run_dir = _populated_run(tmp_path, capsys)
    existing = tmp_path / "existing"
    existing.mkdir()
    with _in_dir(run_dir):
        assert cmd_migrate_store([str(existing)]) == 1
        assert "refusing to overwrite" in capsys.readouterr().out
        assert cmd_migrate_store([str(run_dir / "inner")]) == 1
        assert "OUTSIDE the source" in capsys.readouterr().out


def test_migrate_store_dry_run_writes_nothing(tmp_path, capsys):
    run_dir = _populated_run(tmp_path, capsys)
    dest = tmp_path / "migrated"
    with _in_dir(run_dir):
        rc = cmd_migrate_store([str(dest), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run - nothing written" in out
    assert "found 3 committed attempts" in out
    assert not dest.exists()


def test_migrate_store_refuses_git_source(tmp_path, capsys):
    if not _git_available():
        pytest.skip("git not on PATH")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(_TASK_BODY, encoding="utf-8")
    from groundhog.histories.git import GitAttemptHistory
    GitAttemptHistory(run_dir)
    with _in_dir(run_dir):
        rc = cmd_migrate_store([str(tmp_path / "migrated")])
    assert rc == 1
    assert "already a git attempt store" in capsys.readouterr().out


def test_migrate_store_usage(capsys):
    assert cmd_migrate_store(["-h"]) == 0
    assert cmd_migrate_store([]) == 1
    assert cmd_migrate_store(["a", "b"]) == 1
