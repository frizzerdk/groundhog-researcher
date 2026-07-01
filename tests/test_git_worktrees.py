"""The git backend is a browsable repo scoped to ``attempts/``: a bare repo
(``attempts/.git``) plus one slug-named worktree folder per attempt on an
``attempt/<sha>`` branch. The commit graph is the lineage; pruning a folder
keeps history; task/learnings live beside attempts/, outside git."""

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


def _commit(history, *, parent=None, name="", code="x = 1"):
    ws = history.workspace(parent=parent)
    if name:
        ws.name = name
    (ws.path / "solution.py").write_text(code, encoding="utf-8")
    write_result(ws.path, EvaluationResult(
        stages={"eval": StageResult(metrics={"score": 1.0})}))
    return ws.commit()


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo,
                          capture_output=True, text=True).stdout


def test_attempts_are_browsable_worktrees(tmp_path):
    from groundhog.histories.git import GitAttemptHistory
    run = tmp_path / "store"
    h = GitAttemptHistory(run)
    a = _commit(h, name="prototypical-net", code="A")
    b = _commit(h, parent=a.id, name="prototypical-net-v2", code="B")

    # Git is scoped to attempts/ (a bare repo); the run folder itself is plain.
    attempts = run / "attempts"
    assert (attempts / ".git").is_dir()
    assert not (run / ".git").exists()

    # attempts/ holds only .git + one slug-named worktree folder per attempt.
    folders = [p for p in attempts.iterdir() if p.is_dir() and p.name != ".git"]
    assert len(folders) == 2
    assert (attempts / "prototypical-net").is_dir()
    assert (attempts / "prototypical-net" / "solution.py").read_text() == "A"
    assert (attempts / "prototypical-net" / "metadata.json").exists()

    # Opening an attempt behaves like a worktree, on its hash branch.
    fa = attempts / "prototypical-net"
    assert _git(fa, "status", "-sb").startswith(f"## attempt/{a.id[:12]}")

    # cd attempts && git log --graph: the commit graph IS the lineage.
    log = _git(attempts, "log", "--graph", "--oneline", "--all")
    assert a.id[:7] in log and b.id[:7] in log

    # Branches are the hash handles, one per attempt.
    branches = _git(attempts, "branch")
    assert f"attempt/{a.id[:12]}" in branches
    assert f"attempt/{b.id[:12]}" in branches


def test_pruning_a_worktree_keeps_history(tmp_path):
    from groundhog.histories.git import GitAttemptHistory
    run = tmp_path / "store"
    h = GitAttemptHistory(run)
    a = _commit(h, name="cnn", code="A")

    attempts = run / "attempts"
    folder = attempts / "cnn"
    assert folder.is_dir()
    # "best+recent on disc" = remove the browsable folder; the commit stays.
    subprocess.run(["git", "worktree", "remove", str(folder)],
                   cwd=attempts, check=True, capture_output=True)
    assert not folder.exists()

    again = h.get(a.id)
    assert again is not None and again.code == "A"


def test_failed_attempt_is_kept_as_a_folder(tmp_path):
    from groundhog.histories.git import GitAttemptHistory
    run = tmp_path / "store"
    h = GitAttemptHistory(run)
    ws = h.workspace()
    ws.name = "genetic-pool"
    (ws.path / "solution.py").write_text("boom", encoding="utf-8")
    write_result(ws.path, EvaluationResult(
        stages={"eval": StageResult(errors={"eval": "x"})}, completed=False))
    a = ws.commit(success=False)

    assert a.status == "fail"
    # Kept as an ordinary slug folder; status lives in metadata + the trailer.
    folder = run / "attempts" / "genetic-pool"
    assert folder.is_dir()
    assert (folder / "solution.py").read_text() == "boom"


def test_in_progress_list_and_resume_recovers_edits(tmp_path):
    from groundhog.histories.git import GitAttemptHistory
    run = tmp_path / "store"
    h = GitAttemptHistory(run)
    a = _commit(h, name="seed", code="A")

    # Open a workspace and write some work, but DON'T commit (in-progress).
    ws = h.workspace(parent=a.id)
    (ws.path / "solution.py").write_text("WIP edits", encoding="utf-8")
    wsid = ws.display_id

    # It's listable as in-progress, with the right parent, and live.
    ips = h.list_in_progress()
    assert [ip.workspace_id for ip in ips] == [wsid]
    assert ips[0].parent == a.id and ips[0].live

    # Resume re-binds; the uncommitted edits are still there (crash recovery).
    ws2 = h.resume(wsid)
    assert (ws2.path / "solution.py").read_text() == "WIP edits"

    # Finishing it commits normally; it's then no longer in-progress.
    write_result(ws2.path, EvaluationResult(
        stages={"eval": StageResult(metrics={"score": 1.0})}))
    att = ws2.commit()
    assert att.parent == a.id
    assert h.list_in_progress() == []


def test_reap_removes_crashed_wip(tmp_path):
    import json
    from groundhog.histories.git import GitAttemptHistory
    run = tmp_path / "store"
    h = GitAttemptHistory(run)
    ws = h.workspace()
    wsid = ws.display_id
    wt = h.list_in_progress()[0].path

    # Simulate a crash: forge the heartbeat to a dead pid + ancient start time.
    (h._wip_dir / wsid).write_text(
        json.dumps({"pid": 2 ** 31 - 1, "started_at": 0.0}), encoding="utf-8")

    assert h.reap_in_progress(ttl_s=1.0) == 1
    assert h.list_in_progress() == []
    assert wt is not None and not wt.exists()
