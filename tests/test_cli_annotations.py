"""Tests for attempt annotations: `groundhog attempt note/tag/untag` and the
`attempt list --tag` filter. Parametrized over both store backends — the CLI
reaches them through the same history get_note/set_note API.
"""

import pytest

from groundhog import rundir
from groundhog.cli import attempt_group

from test_cli_attempt import _git_available, _in_dir, _write_run_dir


@pytest.fixture(params=["folder", "git"])
def run_dir(request, tmp_path):
    if request.param == "git" and not _git_available():
        pytest.skip("git not on PATH")
    return _write_run_dir(tmp_path, git=(request.param == "git"))


def _commit_one(run_dir, capsys, value="50.0", direction="constant baseline"):
    attempt_group(["new", "--fresh", "--no-seed"])
    out = capsys.readouterr().out
    wsid = [l for l in out.splitlines()
            if l.startswith("Opened workspace")][0].split()[-1]
    loaded = rundir.load_run(run_dir=run_dir)
    ws_path = [ip.path for ip in loaded.history.list_in_progress()
               if ip.workspace_id == wsid][0]
    (ws_path / "solution.py").write_text(f"def solve():\n    return {value}\n",
                                         encoding="utf-8")
    (ws_path / "core_direction.md").write_text(direction + "\n",
                                               encoding="utf-8")
    attempt_group(["commit", wsid, "--eval"])
    capsys.readouterr()
    history = rundir.load_run(run_dir=run_dir).history
    return history.list()[-1].id


def test_note_set_and_get(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)

        rc = attempt_group(["note", aid, "verdict", "promising"])
        assert rc == 0
        assert "note[verdict] = promising" in capsys.readouterr().out

        rc = attempt_group(["note", aid, "verdict"])
        assert rc == 0
        assert capsys.readouterr().out.strip() == "promising"


def test_note_get_missing(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        rc = attempt_group(["note", aid, "verdict"])
        assert rc == 1
        assert "no note" in capsys.readouterr().out


def test_note_invalid_key_rejected(run_dir, capsys):
    """Key validation is the history's (folder + git share the charset)."""
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        rc = attempt_group(["note", aid, "Bad Key!", "x"])
        assert rc == 1
        assert "Could not set note" in capsys.readouterr().out


def test_note_unknown_attempt(run_dir, capsys):
    with _in_dir(run_dir):
        rc = attempt_group(["note", "nope", "verdict", "x"])
        assert rc == 1
        assert "No such attempt" in capsys.readouterr().out


def test_note_usage(run_dir, capsys):
    with _in_dir(run_dir):
        assert attempt_group(["note"]) == 1
        assert attempt_group(["note", "1"]) == 1


def test_tag_untag_roundtrip(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)

        rc = attempt_group(["tag", aid, "keeper"])
        assert rc == 0
        assert "keeper" in capsys.readouterr().out
        rc = attempt_group(["tag", aid, "baseline"])
        assert rc == 0
        assert "keeper, baseline" in capsys.readouterr().out

        # Stored comma-joined under the "tags" note key.
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note(aid, "tags") == "keeper,baseline"

        # Idempotent.
        rc = attempt_group(["tag", aid, "keeper"])
        assert rc == 0
        assert "already tagged" in capsys.readouterr().out
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note(aid, "tags") == "keeper,baseline"

        rc = attempt_group(["untag", aid, "keeper"])
        assert rc == 0
        capsys.readouterr()
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note(aid, "tags") == "baseline"

        rc = attempt_group(["untag", aid, "keeper"])
        assert rc == 1
        assert "not tagged" in capsys.readouterr().out


def test_tag_rejects_bad_tags(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        assert attempt_group(["tag", aid, "a,b"]) == 1
        assert attempt_group(["tag", aid, "has space"]) == 1
        assert attempt_group(["tag", "nope", "keeper"]) == 1


def test_list_tag_filter(run_dir, capsys):
    with _in_dir(run_dir):
        a1 = _commit_one(run_dir, capsys, value="50.0",
                         direction="constant baseline")
        a2 = _commit_one(run_dir, capsys, value="49.0",
                         direction="second family")
        attempt_group(["tag", a1, "keeper"])
        capsys.readouterr()

        rc = attempt_group(["list", "--tag", "keeper"])
        assert rc == 0
        out = capsys.readouterr().out
        rows = [l for l in out.splitlines()[1:] if l.strip()]
        assert len(rows) == 1
        assert rows[0].startswith(a1[:8])
        assert a2[:8] not in out

        rc = attempt_group(["list", "--tag", "nothing"])
        assert rc == 0
        assert "No attempts tagged 'nothing'." in capsys.readouterr().out


def test_list_tag_filter_includes_failed_with_all(run_dir, capsys):
    with _in_dir(run_dir):
        attempt_group(["new", "--fresh", "--no-seed"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines()
                if l.startswith("Opened workspace")][0].split()[-1]
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text(
            "def solve():\n    raise ValueError('boom')\n", encoding="utf-8")
        (ws_path / "core_direction.md").write_text("constant baseline\n",
                                                   encoding="utf-8")
        attempt_group(["commit", wsid, "--eval"])
        capsys.readouterr()
        history = rundir.load_run(run_dir=run_dir).history
        aid = history.list(only_done=False)[-1].id

        # Tagging a failed attempt goes through the object-accepting API.
        history.set_note(history.list(only_done=False)[-1], "tags", "flaky")

        rc = attempt_group(["list", "--all", "--tag", "flaky"])
        assert rc == 0
        out = capsys.readouterr().out
        assert aid[:8] in out
