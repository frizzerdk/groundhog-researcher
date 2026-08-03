"""Lifecycle edges — heartbeat liveness on the folder backend, unique-prefix
id resolution on both backends, uniform abort semantics, and CRLF-stable git
commits (integration-test findings)."""

import json
import os
import os.path
import subprocess

import pytest

from groundhog.histories.folder import FolderAttemptHistory


def _git_available():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_git = pytest.mark.skipif(not _git_available(), reason="git not on PATH")


# --- folder heartbeat liveness (ported from the git backend) -----------------

def _forge_heartbeat(history, wsid, *, pid, started_at):
    (history._wip_dir / wsid).write_text(
        json.dumps({"pid": pid, "started_at": started_at}), encoding="utf-8")


def test_folder_sigkilled_workspace_reads_dead_and_reaps(tmp_path):
    """A SIGKILLed session leaves a heartbeat with a dead pid: in-progress
    must show CRASHED (not live forever) and reap must collect it."""
    h = FolderAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    _forge_heartbeat(h, ws.display_id, pid=2 ** 31 - 1, started_at=0.0)

    ips = h.list_in_progress()
    assert [ip.workspace_id for ip in ips] == [ws.display_id]
    assert ips[0].live is False
    assert h.reap_in_progress(ttl_s=1.0) == 1
    assert h.list_in_progress() == []
    assert not ws.path.exists()


def test_folder_reap_leaves_live_session_regardless_of_age(tmp_path):
    h = FolderAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    _forge_heartbeat(h, ws.display_id, pid=os.getpid(), started_at=0.0)
    assert h.list_in_progress()[0].live is True
    assert h.reap_in_progress(ttl_s=1.0) == 0
    assert ws.path.exists()


def test_folder_dead_but_recent_gets_grace(tmp_path):
    import time
    h = FolderAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    _forge_heartbeat(h, ws.display_id, pid=2 ** 31 - 1,
                     started_at=time.time())
    assert h.reap_in_progress(ttl_s=300.0) == 0
    ws.abort()


def test_folder_resume_takes_ownership_of_heartbeat(tmp_path):
    h = FolderAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    _forge_heartbeat(h, ws.display_id, pid=2 ** 31 - 1, started_at=0.0)
    assert h.list_in_progress()[0].live is False
    h.resume(ws.display_id)
    assert h.list_in_progress()[0].live is True


# --- unique-prefix id resolution (both backends) ------------------------------

def test_resume_and_abort_accept_unique_id_prefix(history_factory):
    h = history_factory()
    ws = h.workspace()
    (ws.path / "solution.py").write_text("wip", encoding="utf-8")

    resumed = h.resume(ws.display_id[:8])
    assert resumed.display_id == ws.display_id
    assert (resumed.path / "solution.py").read_text(encoding="utf-8") == "wip"
    resumed.abort()
    assert h.list_in_progress() == []


def test_resume_ambiguous_prefix_raises(history_factory):
    h = history_factory()
    ws1 = h.workspace()
    ws2 = h.workspace()
    shared = os.path.commonprefix([ws1.display_id, ws2.display_id])
    with pytest.raises(KeyError):
        h.resume(shared)
    ws1.abort()
    ws2.abort()


def test_resume_unknown_id_raises(history_factory):
    h = history_factory()
    with pytest.raises(KeyError):
        h.resume("zzzzzzzz")


def test_get_and_notes_accept_unique_id_prefix(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h)
    short = a.id[:8]
    assert h.get(short) is not None
    assert h.get(short).id == a.id
    h.set_note(a, "score", "0.5000")
    assert h.get_note(h.get(short), "score") == "0.5000"


# --- abort semantics: abort = discard, commit --fail = record -----------------

def test_abort_discards_and_fail_commit_records(history_factory, commit_attempt):
    h = history_factory()
    ws = h.workspace()
    (ws.path / "solution.py").write_text("scrap", encoding="utf-8")
    ws.abort()
    assert h.list(only_done=False) == []
    assert h.list_in_progress() == []

    failed = commit_attempt(h, success=False, completed=False)
    assert failed.status == "fail"
    assert [a.id for a in h.list(only_done=False)] == [failed.id]


# --- EOL normalization at git commit ------------------------------------------

@needs_git
def test_git_commit_normalizes_crlf_for_duplicate_gate(tmp_path):
    """CRLF text is stored LF at commit, so a child byte-identical to its
    parent modulo line endings is caught by the solution-identical gate."""
    from groundhog.histories.git import GitAttemptHistory
    from groundhog.utils.direction import solution_matches_attempt

    h = GitAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    (ws.path / "solution.py").write_bytes(b"def solve():\r\n    return 1\r\n")
    parent = ws.commit()
    assert "\r" not in parent.code

    child = h.workspace(parent=parent.id)
    (child.path / "solution.py").write_bytes(b"def solve():\r\n    return 1\r\n")
    assert solution_matches_attempt(child.path, parent)
    child.abort()


@needs_git
def test_git_commit_leaves_binary_bytes_alone(tmp_path):
    from groundhog.histories.git import GitAttemptHistory

    h = GitAttemptHistory(tmp_path / "store")
    ws = h.workspace()
    payload = b"\x00\x01\r\nnot text"
    (ws.path / "blob.pkl").write_bytes(payload)
    (ws.path / "solution.py").write_text("x = 1", encoding="utf-8")
    a = ws.commit()
    assert h._git("show", f"{a.id}:blob.pkl").stdout == payload
