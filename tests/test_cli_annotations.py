"""Tests for attempt annotations: `groundhog attempt note/tag/untag` and the
`attempt list --tag` filter. Parametrized over both store backends — the CLI
reaches them through the same history get_note/set_note/list_notes API.

Tags are stored one note key per tag (``tag-<name>``), so concurrent taggers
touch different keys and cannot lose each other's writes; the two-writer
subprocess test proves that for the folder backend's shared notes.json.
"""

import subprocess
import sys

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
        assert "baseline, keeper" in capsys.readouterr().out

        # One note key per tag — no shared comma-joined value to race on.
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note(aid, "tag-keeper") == "1"
        assert history.get_note(aid, "tag-baseline") == "1"
        assert history.get_note(aid, "tags") is None

        # Idempotent.
        rc = attempt_group(["tag", aid, "keeper"])
        assert rc == 0
        assert "already tagged" in capsys.readouterr().out

        rc = attempt_group(["untag", aid, "keeper"])
        assert rc == 0
        capsys.readouterr()
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note(aid, "tag-keeper") == "0"  # tombstone

        rc = attempt_group(["untag", aid, "keeper"])
        assert rc == 1
        assert "not tagged" in capsys.readouterr().out


def test_tags_read_legacy_comma_note(run_dir, capsys):
    """Read-both-shapes shim: comma-joined "tags" notes written by older
    versions still show, and a tombstone hides a legacy tag."""
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        history = rundir.load_run(run_dir=run_dir).history
        history.set_note(aid, "tags", "old-one,old-two")

        attempt_group(["tag", aid, "fresh"])
        out = capsys.readouterr().out
        assert "fresh, old-one, old-two" in out

        rc = attempt_group(["list", "--tag", "old-one"])
        assert rc == 0
        assert aid[:8] in capsys.readouterr().out

        rc = attempt_group(["untag", aid, "old-two"])
        assert rc == 0
        capsys.readouterr()
        rc = attempt_group(["list", "--tag", "old-two"])
        assert rc == 0
        assert "No attempts tagged" in capsys.readouterr().out


def test_show_lists_tags(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        attempt_group(["tag", aid, "keeper"])
        capsys.readouterr()
        rc = attempt_group(["show", aid])
        assert rc == 0
        out = capsys.readouterr().out
        assert "tags:    keeper" in out


def test_list_rejects_dangling_tag_option(run_dir, capsys):
    with _in_dir(run_dir):
        _commit_one(run_dir, capsys)
        rc = attempt_group(["list", "--tag"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "Usage" in out


def test_tag_rejects_bad_tags(run_dir, capsys):
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        assert attempt_group(["tag", aid, "a,b"]) == 1
        assert attempt_group(["tag", aid, "has space"]) == 1
        assert attempt_group(["tag", aid, "UPPER"]) == 1
        assert attempt_group(["tag", "nope", "keeper"]) == 1


def test_note_backend_failure_is_a_message_not_a_traceback(run_dir, capsys):
    """A note channel that raises (broken repo/sidecar) surfaces as a clean
    CLI message with exit 1."""
    with _in_dir(run_dir):
        aid = _commit_one(run_dir, capsys)
        run = rundir.load_run(run_dir=run_dir)

        def boom(*a, **k):
            raise OSError("notes channel unavailable")

        import groundhog.cli as cli_mod
        original = cli_mod._resolve_run

        class _BrokenHistory:
            def get(self, attempt_id):
                return run.history.get(attempt_id)
            set_note = boom
            def get_note(self, *a, **k):
                raise OSError("notes channel unavailable")
            def list_notes(self, *a, **k):
                raise OSError("notes channel unavailable")

        class _Run:
            history = _BrokenHistory()
            task = run.task
            toolkit = run.toolkit
            run_dir = run.run_dir

        cli_mod._resolve_run = lambda args=None: _Run()
        try:
            assert attempt_group(["note", aid, "verdict", "x"]) == 1
            assert "Could not set note" in capsys.readouterr().out
            assert attempt_group(["tag", aid, "keeper"]) == 1
            assert "Could not tag" in capsys.readouterr().out
        finally:
            cli_mod._resolve_run = original


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


NOTE_WORKER = """
import sys
from pathlib import Path
from groundhog.histories.folder import FolderAttemptHistory

store, aid, name, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
h = FolderAttemptHistory(Path(store))
for i in range(n):
    h.set_note(aid, "tag-" + name + "-" + str(i), "1")
"""

N_KEYS = 30


def test_concurrent_taggers_cannot_lose_each_other(tmp_path):
    """Two processes writing distinct tag keys interleave on one notes.json;
    set_note's locked read-modify-write must keep every key."""
    from groundhog.histories.folder import FolderAttemptHistory

    history = FolderAttemptHistory(tmp_path / "store")
    ws = history.workspace()
    (ws.path / "solution.py").write_text("def solve(): return 1",
                                         encoding="utf-8")
    attempt = ws.commit(success=True)

    script = tmp_path / "worker.py"
    script.write_text(NOTE_WORKER, encoding="utf-8")
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(tmp_path / "store"),
             attempt.id, name, str(N_KEYS)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for name in ("a", "b")
    ]
    for p in procs:
        _, err = p.communicate(timeout=180)
        assert p.returncode == 0, err.decode("utf-8", "replace")

    notes = history.list_notes(attempt.id)
    expected = {f"tag-{w}-{i}" for w in ("a", "b") for i in range(N_KEYS)}
    assert expected <= set(notes)
