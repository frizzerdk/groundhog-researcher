"""P3 sync — two git stores over one bare remote, conflict-free both ways.

Git-only (the folder backend has no sync). Sync is best-effort: a commit must
succeed even when the remote is unreachable. ``fetch_ttl_s=0`` here makes reads
fetch every time so the assertions are deterministic.
"""

import subprocess

import pytest

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.utils.results import write_result


def _git_available():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _git_available(), reason="git not on PATH")


def _commit(history, *, parent=None, code="x = 1"):
    ws = history.workspace(parent=parent)
    (ws.path / "solution.py").write_text(code, encoding="utf-8")
    write_result(ws.path, EvaluationResult(
        stages={"eval": StageResult(metrics={"score": 1.0})}))
    return ws.commit()


def _bare_remote(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)],
                   check=True, capture_output=True)
    return remote


def test_sync_two_stores_conflict_free(tmp_path):
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a_store = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    b_store = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)

    # A commits and pushes; B fetches on its next read and sees it.
    a1 = _commit(a_store, code="from A")
    assert a1.id in [x.id for x in b_store.list()]
    assert b_store.get(a1.id) is not None
    assert b_store.get(a1.id).code == "from A"

    # B commits and pushes; A sees it. Different origins ⇒ disjoint ref
    # namespaces ⇒ no conflict in either direction.
    b1 = _commit(b_store, code="from B")
    assert {a1.id, b1.id} <= {x.id for x in a_store.list()}
    assert {a1.id, b1.id} <= {x.id for x in b_store.list()}


def test_commit_succeeds_when_remote_unreachable(tmp_path):
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    bogus = str(tmp_path / "nope.git")  # never created
    policy = SyncPolicy(fetch_ttl_s=0.0, timeout_s=5.0)
    store = GitAttemptHistory(tmp_path / "A", remote=bogus, policy=policy)

    a = _commit(store, code="local only")  # push fails, swallowed
    assert a is not None
    # The read's fetch also fails and degrades to local.
    assert a.id in [x.id for x in store.list(only_done=False)]


def test_local_store_never_touches_remote(tmp_path):
    """remote=None is a pure no-op: no push, no fetch, fully local."""
    from groundhog.histories.git import GitAttemptHistory
    store = GitAttemptHistory(tmp_path / "A")  # no remote
    a = _commit(store)
    assert [x.id for x in store.list()] == [a.id]
