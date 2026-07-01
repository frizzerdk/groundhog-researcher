"""Backend-agnostic contract tests — folder and git must behave identically.

Parametrized over ``history_factory`` (folder, git). Assertions are on the
*contract* only, never on id format: folder ids look like ``"1"``, git ids are
commit shas, so the same assertions must hold for both. If a test here passes
for folder but fails for git, the git backend has diverged from the contract.
"""

import json


def test_commit_roundtrip(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, code="def solve(): return 42")
    assert isinstance(a.id, str) and a.id
    assert a.parent is None
    assert isinstance(a.created_at, float)
    assert a.status == "done"

    again = h.get(a.id)
    assert again is not None
    assert again.id == a.id
    assert again.code == "def solve(): return 42"


def test_parent_edge(history_factory, commit_attempt):
    h = history_factory()
    a1 = commit_attempt(h)
    a2 = commit_attempt(h, parent=a1.id)
    assert a1.parent is None
    assert a2.parent == a1.id


def test_list_is_chronological(history_factory, commit_attempt):
    h = history_factory()
    a1 = commit_attempt(h)
    a2 = commit_attempt(h, parent=a1.id)
    a3 = commit_attempt(h, parent=a2.id)
    ids = [a.id for a in h.list()]
    assert ids == [a1.id, a2.id, a3.id]
    # simple.py relies on list()[-1] being the newest attempt.
    assert h.list()[-1].id == a3.id


def test_lineage(history_factory, commit_attempt):
    h = history_factory()
    a1 = commit_attempt(h)
    a2 = commit_attempt(h, parent=a1.id)
    a3 = commit_attempt(h, parent=a2.id)
    assert [a.id for a in h.lineage(a3)] == [a1.id, a2.id, a3.id]


def test_best_uses_scorer(history_factory, commit_attempt):
    h = history_factory()
    commit_attempt(h, metrics={"score": 0.1})
    hi = commit_attempt(h, metrics={"score": 0.9})
    commit_attempt(h, metrics={"score": 0.5})
    best = h.best(lambda sr: sr.metrics.get("score", 0.0))
    assert best.id == hi.id


def test_get_unknown_returns_none(history_factory, commit_attempt):
    h = history_factory()
    commit_attempt(h)
    assert h.get("does-not-exist") is None


def test_read_and_list_files(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, code="X = 1")
    files = a.list_files()
    assert "solution.py" in files
    assert "result.json" in files
    assert a.read_file("solution.py") == "X = 1"
    assert a.read_file("nope.py") is None


def test_fail_is_kept_but_filtered(history_factory, commit_attempt):
    h = history_factory()
    ok = commit_attempt(h)
    bad = commit_attempt(h, success=False, completed=False)
    assert bad.status == "fail"

    done_ids = [a.id for a in h.list(only_done=True)]
    all_ids = [a.id for a in h.list(only_done=False)]
    assert ok.id in done_ids
    assert bad.id not in done_ids
    assert bad.id in all_ids


def test_abort_leaves_no_trace(history_factory):
    h = history_factory()
    before = len(h.list(only_done=False))
    ws = h.workspace()
    (ws.path / "solution.py").write_text("x = 1", encoding="utf-8")
    ws.abort()
    assert len(h.list(only_done=False)) == before


def test_derive_trunks_single_chain(history_factory, commit_attempt):
    h = history_factory()
    a1 = commit_attempt(h, direction="rollout", metrics={"score": 0.1})
    a2 = commit_attempt(h, parent=a1.id, direction="rollout", metrics={"score": 0.5})
    commit_attempt(h, parent=a2.id, direction="rollout", metrics={"score": 0.9})
    trunks = h.derive_trunks(lambda sr: sr.metrics.get("score", 0.0))
    assert len(trunks) == 1
    assert len(trunks[0]) == 3


def test_derive_families_groups_by_direction(history_factory, commit_attempt):
    h = history_factory()
    commit_attempt(h, direction="rollout")
    commit_attempt(h, direction="mcts")
    families = h.derive_families()
    assert len(families) == 2


def test_name_round_trips_but_is_not_a_lookup_key(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, name="improve-relu-init")
    fetched = h.get(a.id)
    assert fetched.name == "improve-relu-init"
    # Name is display-only — it never resolves an attempt.
    assert h.get("improve-relu-init") is None


def test_metadata_lives_outside_result_json(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, name="some-name")
    raw = json.loads(a.read_file("result.json"))
    assert "metadata" not in raw  # result.json is eval-only
    meta = a.read_file("metadata.json")
    assert meta is not None and "some-name" in meta  # name lives in metadata.json


def test_child_workspace_starts_clean(history_factory, commit_attempt):
    """A child workspace does NOT inherit the parent's eval artifacts — both
    backends start a child empty of the parent's result.json/metadata.json so a
    crashed-before-write child can never read the parent's stale metrics."""
    h = history_factory()
    a = commit_attempt(h, code="X = 1", metrics={"score": 0.9}, name="parent")
    ws = h.workspace(parent=a.id)
    assert not (ws.path / "result.json").exists()
    assert not (ws.path / "metadata.json").exists()
    ws.abort()


def test_in_progress_list_resume_abort(history_factory):
    """An open workspace is listable as in-progress, resumable (its uncommitted
    edits survive), and gone once aborted — for both backends."""
    h = history_factory()
    ws = h.workspace()
    (ws.path / "solution.py").write_text("wip edits", encoding="utf-8")
    wsid = ws.display_id

    assert wsid in [ip.workspace_id for ip in h.list_in_progress()]
    ws2 = h.resume(wsid)
    assert (ws2.path / "solution.py").read_text() == "wip edits"
    ws2.abort()
    assert wsid not in [ip.workspace_id for ip in h.list_in_progress()]
