"""toolkit.ws — the attempt pointer — and history.materialize.

The three semantics-pinning tests (design sessions, 2026-07):
1. POINTER FOLLOWS: one tool, defined once at build time, reads a different
   attempt dir per attempt — the core reason set_attempt exists.
2. UNSET FAILS CLEAN: invoking with nothing in flight surfaces a readable
   tool error, never a stale read.
3. CROSS-THREAD VISIBILITY: agent tools run on an HTTP server thread; the
   plain-attribute slot must be visible there (a ContextVar would not be —
   that is why it was rejected).
Plus: bracket lifecycle, double-set guard, resolution of live/committed
targets on both backends, and the git reap-keeps-live-sessions regression.
"""

import json
import shutil as _shutil
import threading
import time
from pathlib import Path

import pytest

from groundhog import Task, agent_tool, assemble_toolkit
from groundhog.base.types import Data, Context, Evaluator, EvalStage, StageResult
from groundhog.base.workspace_handle import (
    ReadOnlyWorkspaceView, WorkspaceNotSetError, WorkspaceStateError,
)


class _Data(Data):
    def get_train(self): return None
    def get_test(self): return None


class _Ctx(Context):
    def get_brief(self): return "b"
    def get_extended(self): return "e"


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult()

    def get_stages(self, data):
        return [EvalStage("eval", "eval", lambda cp: StageResult(),
                          scorer=lambda r: r.metrics.get("score", 0.0))]


def _toolkit(history):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    return assemble_toolkit(task, history=history)


# --- 1 · pointer follows ------------------------------------------------------

def test_tool_defined_once_follows_the_current_attempt(history_factory):
    history = history_factory()
    tk = _toolkit(history)

    ws_handle = tk.ws  # captured ONCE, like a task.py agent_tools hook does

    def where() -> str:
        return str(ws_handle.path)

    tool = agent_tool(name="where", description="d", func=where, params={})

    seen = []
    for _ in range(2):
        ws = history.workspace()
        with tk.ws.attempt(ws):
            out = tool.execute().output
            assert out == str(ws.path)
            seen.append(out)
            (ws.path / "solution.py").write_text("x = 1", encoding="utf-8")
            ws.commit(success=True)
    assert seen[0] != seen[1], "tool did not follow the pointer across attempts"


# --- 2 · unset fails clean ----------------------------------------------------

def test_unset_read_is_a_clean_tool_error(history_factory):
    tk = _toolkit(history_factory())

    def where() -> str:
        return str(tk.ws.path)

    tool = agent_tool(name="where", description="d", func=where, params={})
    result = tool.execute()
    assert not result.success
    assert "no attempt in flight" in (result.error or "")


def test_unset_reads_raise_workspace_not_set(history_factory):
    tk = _toolkit(history_factory())
    with pytest.raises(WorkspaceNotSetError):
        _ = tk.ws.current
    with pytest.raises(WorkspaceNotSetError):
        _ = tk.ws.path
    assert tk.ws.is_set() is False


# --- 3 · cross-thread visibility ----------------------------------------------

def test_pointer_set_on_one_thread_is_visible_from_another(history_factory):
    """Tools are invoked over HTTP on a server thread — the slot must be a
    plain attribute so that thread sees it. Pins the ContextVar rejection."""
    history = history_factory()
    tk = _toolkit(history)
    ws = history.workspace()
    seen = {}

    def reader():
        try:
            seen["path"] = str(tk.ws.path)
        except Exception as e:  # noqa: BLE001 — recorded for the assert
            seen["error"] = repr(e)

    with tk.ws.attempt(ws):
        t = threading.Thread(target=reader)
        t.start()
        t.join()
    ws.abort()

    assert seen.get("path") == str(ws.path), f"other thread saw: {seen}"


# --- bracket + guard semantics --------------------------------------------------

def test_bracket_clears_on_exception(history_factory):
    history = history_factory()
    tk = _toolkit(history)
    ws = history.workspace()
    with pytest.raises(RuntimeError, match="boom"):
        with tk.ws.attempt(ws):
            raise RuntimeError("boom")
    assert tk.ws.is_set() is False
    ws.abort()


def test_double_set_without_clear_raises(history_factory):
    history = history_factory()
    tk = _toolkit(history)
    ws1, ws2 = history.workspace(), history.workspace()
    tk.ws.set_attempt(ws1)
    with pytest.raises(WorkspaceStateError):
        tk.ws.set_attempt(ws2)
    tk.ws.clear()
    ws1.abort()
    ws2.abort()


def test_handle_never_owns_lifecycle_readonly_view_refuses(history_factory, commit_attempt):
    history = history_factory()
    a = commit_attempt(history)
    tk = _toolkit(history)
    view = tk.ws.set_attempt(a.id)
    assert isinstance(view, ReadOnlyWorkspaceView)
    with pytest.raises(WorkspaceStateError, match="read-only"):
        view.commit()
    with pytest.raises(WorkspaceStateError, match="read-only"):
        view.abort()
    tk.ws.clear()


# --- resolution: live wsid + committed id, both backends -------------------------

def test_resolve_in_progress_wsid_rebinds_live(history_factory):
    history = history_factory()
    tk = _toolkit(history)
    ws = history.workspace()
    (ws.path / "solution.py").write_text("wip", encoding="utf-8")
    wsid = ws.display_id

    with tk.ws.attempt(wsid) as bound:
        assert (Path(bound.path) / "solution.py").read_text(encoding="utf-8") == "wip"
    bound.abort()


def test_resolve_committed_id_reads_files_via_path(history_factory, commit_attempt):
    history = history_factory()
    a = commit_attempt(history, code="def solve(): return 7")
    tk = _toolkit(history)
    with tk.ws.attempt(a.id):
        text = (tk.ws.path / "solution.py").read_text(encoding="utf-8")
    assert "return 7" in text


def test_resolve_garbage_raises(history_factory):
    tk = _toolkit(history_factory())
    with pytest.raises(WorkspaceStateError, match="cannot resolve"):
        tk.ws.set_attempt("no-such-attempt-anywhere")


# --- materialize -----------------------------------------------------------------

def test_materialize_is_idempotent(history_factory, commit_attempt):
    history = history_factory()
    a = commit_attempt(history)
    p1 = history.materialize(a.id)
    p2 = history.materialize(a.id)
    assert p1 == p2
    assert Path(p1).exists()


def test_materialize_unknown_id_raises(history_factory):
    history = history_factory()
    with pytest.raises(KeyError):
        history.materialize("definitely-not-an-attempt")


# --- git-specific: rematerialize + reap regression --------------------------------

_needs_git = pytest.mark.skipif(_shutil.which("git") is None,
                                reason="git not on PATH")


def _git_history(tmp_path):
    from groundhog.histories.git import GitAttemptHistory
    return GitAttemptHistory(tmp_path / "store")


@_needs_git
def test_git_materialize_recreates_pruned_worktree(tmp_path, commit_attempt):
    """The synced-clone / pruned-disk case: the folder is gone, the objects
    remain — materialize must bring the folder back."""
    import shutil
    history = _git_history(tmp_path)
    a = commit_attempt(history, code="def solve(): return 99", name="probe")

    p = Path(history.materialize(a.id))
    assert p.exists()
    shutil.rmtree(p)  # simulate pruned disk / fresh clone without worktrees
    assert not p.exists()

    p2 = Path(history.materialize(a.id))
    assert p2.exists()
    assert (p2 / "solution.py").read_text(encoding="utf-8") == "def solve(): return 99"


@_needs_git
def test_git_attempt_path_is_lazy_materialize(tmp_path, commit_attempt):
    history = _git_history(tmp_path)
    a = commit_attempt(history, code="x = 1")
    p = a.path  # property → materialize on demand
    assert Path(p).exists()
    assert (Path(p) / "solution.py").exists()


@_needs_git
def test_git_reap_leaves_live_session_regardless_of_age(tmp_path):
    """Audit 2026-07-01 bug #2: reap force-removed LIVE sessions older than
    the TTL because the heartbeat was written once and never refreshed. The
    contract is 'abort crashed workspaces; leave live ones' — a live pid now
    survives any age; the TTL is a grace period for dead pids only."""
    history = _git_history(tmp_path)
    ws = history.workspace()
    wsid = ws.display_id

    # Backdate the heartbeat far past the TTL, keeping OUR (live) pid.
    hb_path = history._heartbeat(wsid)
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    hb["started_at"] = time.time() - 3600
    hb_path.write_text(json.dumps(hb), encoding="utf-8")

    reaped = history.reap_in_progress(ttl_s=1.0)
    assert reaped == 0, "reap killed a LIVE session"
    assert wsid in [ip.workspace_id for ip in history.list_in_progress()]

    # A dead pid past the grace period IS reaped.
    hb["pid"] = 999999999
    hb_path.write_text(json.dumps(hb), encoding="utf-8")
    assert history.reap_in_progress(ttl_s=1.0) == 1


@_needs_git
def test_set_attempt_refreshes_heartbeat(tmp_path):
    history = _git_history(tmp_path)
    tk = _toolkit(history)
    ws = history.workspace()
    hb_path = history._heartbeat(ws.display_id)
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    hb["started_at"] = time.time() - 3600
    hb_path.write_text(json.dumps(hb), encoding="utf-8")

    with tk.ws.attempt(ws):
        refreshed = json.loads(hb_path.read_text(encoding="utf-8"))
        assert time.time() - refreshed["started_at"] < 60, \
            "set_attempt did not refresh the heartbeat"
    ws.abort()


# --- self-review regressions (2026-07-02 pre-merge review) ------------------

@_needs_git
def test_git_materialize_detached_is_idempotent(tmp_path, commit_attempt):
    """A SYNCED attempt has no local attempt/<sha> branch — materialize takes
    the detached path. It must return the SAME worktree on every call, never
    mint duplicates (review finding #1: duplicate checkouts, GitError on the
    4th call)."""
    import subprocess
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)],
                   check=True, capture_output=True)
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a_store = GitAttemptHistory(tmp_path / "A", remote=str(remote), policy=policy)
    b_store = GitAttemptHistory(tmp_path / "B", remote=str(remote), policy=policy)

    a1 = commit_attempt(a_store, code="def solve(): return 5")
    assert a1.id in [x.id for x in b_store.list()]  # synced, objects only

    p1 = Path(b_store.materialize(a1.id))
    p2 = Path(b_store.materialize(a1.id))
    p3 = Path(b_store.materialize(a1.id))
    p4 = Path(b_store.materialize(a1.id))
    assert p1 == p2 == p3 == p4, "detached materialize minted duplicates"
    assert (p1 / "solution.py").read_text(encoding="utf-8") == "def solve(): return 5"


@_needs_git
def test_resolve_never_steals_a_live_foreign_heartbeat(tmp_path):
    """Pointing the handle at a wsid owned by another LIVE process must not
    rewrite its heartbeat (review finding #3: a short-lived CLI read left a
    dead pid behind, making the owner reapable)."""
    import json
    import os

    from groundhog.base.workspace_handle import ForeignWorkspaceView, WorkspaceStateError

    history = _git_history(tmp_path)
    tk = _toolkit(history)
    ws = history.workspace()
    wsid = ws.display_id

    # Simulate a live FOREIGN owner: our parent process is alive and isn't us.
    hb_path = history._heartbeat(wsid)
    hb = json.loads(hb_path.read_text(encoding="utf-8"))
    hb["pid"] = os.getppid()
    hb_path.write_text(json.dumps(hb), encoding="utf-8")

    view = tk.ws.set_attempt(wsid)
    assert isinstance(view, ForeignWorkspaceView)
    assert Path(view.path) == Path(ws.path)
    with pytest.raises(WorkspaceStateError, match="another process"):
        view.commit()
    # The owner's heartbeat is untouched.
    hb_after = json.loads(hb_path.read_text(encoding="utf-8"))
    assert hb_after["pid"] == os.getppid(), "heartbeat was stolen"
    tk.ws.clear()
    ws.abort()


def test_load_run_history_survives_cwd_change(tmp_path, monkeypatch):
    """Template-style task.py (no explicit path) must yield ABSOLUTE store
    roots: the loader restores the caller's cwd after build_toolkit(), and a
    relative 'attempts' would re-root at whatever dir the CLI runs from
    (review finding #2)."""
    from groundhog import rundir

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(
        "from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult, assemble_toolkit\n"
        "class D(Data):\n"
        "    def get_train(self): return None\n"
        "    def get_test(self): return None\n"
        "class C(Context):\n"
        "    def get_brief(self): return 'b'\n"
        "    def get_extended(self): return 'e'\n"
        "class E(Evaluator):\n"
        "    def evaluate(self, cp, d): return StageResult()\n"
        "    def get_stages(self, d):\n"
        "        return [EvalStage('eval', 'e', lambda cp: StageResult())]\n"
        "task = Task(data=D(), context=C(), evaluator=E(), name='t')\n"
        "def build_toolkit():\n"
        "    return assemble_toolkit(task)   # NO explicit path — the template default\n",
        encoding="utf-8")

    loaded = rundir.load_run(run_dir=run_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert loaded.history.list() == []          # would raise FileNotFoundError before
    ws = loaded.history.workspace()
    assert Path(ws.path).is_absolute()
    assert str(ws.path).startswith(str(run_dir)), \
        f"store re-rooted outside the run dir: {ws.path}"
    ws.abort()
