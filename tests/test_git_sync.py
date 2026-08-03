"""P3 sync — two git stores over one bare remote, conflict-free both ways.

Git-only (the folder backend has no sync). Sync is best-effort: a commit must
succeed even when the remote is unreachable. ``fetch_ttl_s=0`` here makes reads
fetch every time so the assertions are deterministic.
"""

import os
import shutil
import subprocess
import time

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


def test_notes_are_local_only_but_results_still_score(tmp_path):
    """Notes never sync — raw results travel with the attempt refs, and the
    other store recomputes scores read-side from the fetched attempt."""
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    from groundhog.utils.queries import safe_result
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    b = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)

    a1 = _commit(a, code="from A")
    a.set_note(a1, "score", "0.875")

    b.list()                 # triggers a fetch of the attempt refs
    fetched = b.get(a1.id)
    assert fetched is not None
    assert b.get_note(a1.id, "score") is None          # note stayed local
    assert a.get_note(a1.id, "score") == "0.875"

    result = safe_result(fetched)                       # raw results traveled
    assert result is not None
    scorer = lambda sr: sr.metrics.get("score", 0.0)    # noqa: E731
    assert scorer(list(result.stages.values())[-1]) == 1.0
    assert b.best(scorer).id == a1.id


def test_root_attempt_reads_as_root_across_stores(tmp_path):
    """A fresh (root) attempt branches off the ORIGINATING store's base commit,
    whose sha differs from the reader's base — it must still read parent None."""
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    time.sleep(1.1)   # cross a second boundary so the base commits differ
    b = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)
    assert a._base_sha != b._base_sha   # base commit date is not pinned

    a1 = _commit(a, code="root")   # parent=None -> root
    assert a1.parent is None
    b.list()   # fetch the attempt (and its base commit ancestry) into B
    fetched = b.get(a1.id)
    assert fetched is not None
    assert fetched.parent is None


def test_backfill_pushes_preexisting_attempts_on_attach(tmp_path):
    """A store used offline, then given a remote, backfills its refs once."""
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)

    a_local = GitAttemptHistory(tmp_path / "A")   # no remote
    a1 = _commit(a_local, code="one")
    a2 = _commit(a_local, parent=a1.id, code="two")

    # Re-open the SAME store with a remote: backfill pushes both refs.
    a_synced = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    assert a_synced._backfill_marker().exists()

    b = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)
    assert {a1.id, a2.id} <= {x.id for x in b.list()}


def test_first_contact_probe_reports_reachability(tmp_path, capsys):
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    GitAttemptHistory(tmp_path / "A", remote=remote,
                      policy=SyncPolicy(fetch_ttl_s=0.0))
    assert "sync: remote reachable" in capsys.readouterr().err


def test_reachable_probe_runs_once_per_store_and_remote(tmp_path, capsys):
    """A reachable probe writes the marker: reconstructing the same store
    against the same remote does zero first-contact network work."""
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    store = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    assert "sync: remote reachable" in capsys.readouterr().err
    assert store._backfill_marker().exists()

    GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    err = capsys.readouterr().err
    assert "sync:" not in err and "unreachable" not in err
    # A DIFFERENT remote is a fresh first contact.
    other = str(_bare_remote(tmp_path / "other"))
    GitAttemptHistory(tmp_path / "A", remote=other, policy=policy)
    assert "sync: remote reachable" in capsys.readouterr().err


def test_first_contact_probe_warns_when_unreachable(tmp_path, capsys):
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    bogus = str(tmp_path / "nope.git")   # never created
    store = GitAttemptHistory(tmp_path / "A", remote=bogus,
                              policy=SyncPolicy(timeout_s=5.0))
    assert "remote unreachable" in capsys.readouterr().err
    # No marker -> the probe is retried on the next construction.
    assert not store._backfill_marker().exists()
    GitAttemptHistory(tmp_path / "A", remote=bogus,
                      policy=SyncPolicy(timeout_s=5.0))
    assert "remote unreachable" in capsys.readouterr().err


def test_backfill_is_chunked(tmp_path, monkeypatch):
    """Backfill pushes refspecs in batches (Windows argv limit); the marker
    lands only after the LAST chunk."""
    import groundhog.histories.git as gitmod
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    monkeypatch.setattr(gitmod, "_BACKFILL_CHUNK", 2)

    a_local = GitAttemptHistory(tmp_path / "A")   # no remote
    attempts = [_commit(a_local, code=f"v{i}") for i in range(5)]

    pushes = []
    real_git = GitAttemptHistory._git

    def spy(self, *args, **kw):
        if args and str(args[0]) == "push":
            pushes.append(args)
        return real_git(self, *args, **kw)

    monkeypatch.setattr(GitAttemptHistory, "_git", spy)
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a_synced = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    assert len(pushes) == 3   # 5 refs in chunks of 2
    assert all(len(p) - 2 <= 2 for p in pushes)
    assert a_synced._backfill_marker().exists()

    b = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)
    assert {x.id for x in attempts} <= {x.id for x in b.list()}


def test_get_foreign_base_sha_is_not_an_attempt(tmp_path):
    """get(<other store's base sha>) must return None, not a phantom attempt —
    the base commit is the origin of the tree, never an attempt."""
    from groundhog.histories.git import GitAttemptHistory, SyncPolicy
    remote = str(_bare_remote(tmp_path))
    policy = SyncPolicy(fetch_ttl_s=0.0)
    a = GitAttemptHistory(tmp_path / "A", remote=remote, policy=policy)
    time.sleep(1.1)   # cross a second boundary so the base commits differ
    b = GitAttemptHistory(tmp_path / "B", remote=remote, policy=policy)
    assert a._base_sha != b._base_sha

    a1 = _commit(a, code="root")
    b.list()   # fetch A's attempt (and its base-commit ancestry) into B
    assert b.get(a1.id) is not None
    assert b.get(a._base_sha) is None
    assert b.get(b._base_sha) is None


# --- Real-GitHub e2e (env-gated; never runs in CI) ---------------------------
#
# Two stores on one machine syncing through the standing private repo
# frizzerdk/groundhog-sync-test, plus the synced-clone self-heal: a THIRD
# fresh store fetches the refs and materializes a worktree it never created.
# Auth: token-in-URL from `gh auth token` at runtime (the hermetic _git
# chokepoint strips credential helpers by design). Enable with
# GROUNDHOG_GITHUB_SYNC=1.

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
