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


# --- Real-GitHub e2e (env-gated; never runs in CI) ---------------------------
#
# Two stores on one machine syncing through the standing private repo
# frizzerdk/groundhog-sync-test, plus the synced-clone self-heal: a THIRD
# fresh store fetches the refs and materializes a worktree it never created.
# Auth: token-in-URL from `gh auth token` at runtime (the hermetic _git
# chokepoint strips credential helpers by design). Enable with
# GROUNDHOG_GITHUB_SYNC=1.

import os
import shutil


def _github_remote_url():
    gh = shutil.which("gh")
    if not gh:
        return None
    tok = subprocess.run([gh, "auth", "token"], capture_output=True, text=True)
    token = (tok.stdout or "").strip()
    if tok.returncode != 0 or not token:
        return None
    return f"https://x-access-token:{token}@github.com/frizzerdk/groundhog-sync-test.git"


@pytest.mark.skipif(os.environ.get("GROUNDHOG_GITHUB_SYNC") != "1",
                    reason="real-GitHub e2e; set GROUNDHOG_GITHUB_SYNC=1")
def test_github_sync_e2e_two_stores_and_clone_self_heal(tmp_path):
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy

    url = _github_remote_url()
    if url is None:
        pytest.skip("gh CLI not authenticated")

    policy = SyncPolicy(fetch_ttl_s=0.0, timeout_s=60.0)
    a_store = GitAttemptHistory(tmp_path / "A", remote=url, policy=policy)
    b_store = GitAttemptHistory(tmp_path / "B", remote=url, policy=policy)

    try:
        # A commits -> pushes to GitHub; B fetches and reads it back.
        a1 = _commit(a_store, code="from A via github")
        assert a1.id in [x.id for x in b_store.list()]
        assert b_store.get(a1.id).code == "from A via github"

        # B commits; both see both (disjoint per-origin refs, conflict-free).
        b1 = _commit(b_store, code="from B via github")
        assert {a1.id, b1.id} <= {x.id for x in a_store.list()}
        assert {a1.id, b1.id} <= {x.id for x in b_store.list()}

        # The synced-clone self-heal: a FRESH store (no shared disk with A/B)
        # sees the attempts as objects only — materialize() checks out a
        # worktree it never had.
        c_store = GitAttemptHistory(tmp_path / "C", remote=url, policy=policy)
        assert a1.id in [x.id for x in c_store.list()]
        p = c_store.materialize(a1.id)
        assert (p / "solution.py").read_text(encoding="utf-8") == "from A via github"
    finally:
        # Leave the standing repo tidy: best-effort delete of this run's refs
        # (exact names — push --delete does not glob).
        ls = subprocess.run(["git", "ls-remote", url, "refs/attempts/*"],
                            capture_output=True, text=True, timeout=60)
        origins = {a_store.origin, b_store.origin}
        stale = [line.split("\t", 1)[1] for line in ls.stdout.splitlines()
                 if "\t" in line and line.split("\t", 1)[1].split("/")[2] in origins]
        if stale:
            subprocess.run(["git", "--git-dir", str(a_store._git_dir),
                            "push", url, "--delete", *stale],
                           capture_output=True, timeout=120)
