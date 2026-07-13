"""Tests for the read layer (utils/queries) — pure queries over an
AttemptHistory + scorer, exercised against both backends via the
parametrized ``history_factory`` fixture.

Ordering assertions are structural (parent chains, monotone trajectory)
rather than timestamp-based: the git backend's ``created_at`` has
second resolution, so attempts committed in quick succession may tie.
"""

import json

from groundhog.utils import queries


def scorer(stage):
    return float(stage.metrics.get("score", -1.0))


def _seed(history, commit_attempt):
    """Two-member family + one failed sibling. Returns (a1, a2, a3)."""
    a1 = commit_attempt(history, direction="# Alpha approach\n\nbody",
                        metrics={"score": 0.5},
                        metadata={"strategy": "agent", "cost": 0.25})
    a2 = commit_attempt(history, parent=a1.id,
                        direction="# Alpha approach\n\nbody",
                        metrics={"score": 0.8},
                        metadata={"strategy": "improve", "cost": "0.1"})
    a3 = commit_attempt(history, direction="beta search",
                        metrics={"score": 0.9}, completed=False, success=False,
                        metadata={"strategy": "manual"})
    return a1, a2, a3


def test_attempt_table_rows(history_factory, commit_attempt):
    h = history_factory()
    a1, a2, a3 = _seed(h, commit_attempt)

    table = queries.attempt_table(h, scorer)
    assert {r["id"] for r in table} == {a1.id, a2.id, a3.id}
    rows = {r["id"]: r for r in table}

    assert rows[a1.id]["parent"] is None
    assert rows[a1.id]["status"] == "done"
    assert rows[a1.id]["score"] == 0.5
    assert rows[a1.id]["strategy"] == "agent"
    assert rows[a1.id]["cost"] == 0.25
    assert isinstance(rows[a1.id]["created_at"], float)

    assert rows[a2.id]["parent"] == a1.id
    assert rows[a2.id]["cost"] == 0.1  # coerced from the string in metadata

    assert rows[a3.id]["status"] == "fail"
    assert rows[a3.id]["score"] is None  # failed results are never scored
    assert rows[a3.id]["cost"] is None

    json.dumps(table)  # every row is JSON-serializable as-is


def test_attempt_table_only_done_and_no_scorer(history_factory, commit_attempt):
    h = history_factory()
    a1, a2, a3 = _seed(h, commit_attempt)

    done = queries.attempt_table(h, scorer, only_done=True)
    assert {r["id"] for r in done} == {a1.id, a2.id}

    unscored = queries.attempt_table(h)
    assert all(r["score"] is None for r in unscored)


def test_families(history_factory, commit_attempt):
    h = history_factory()
    a1, a2, a3 = _seed(h, commit_attempt)
    c1 = commit_attempt(h, metrics={"score": 0.1})  # no direction -> sentinel

    fams = {f["family_name"]: f for f in queries.families(h, scorer)}
    assert set(fams) == {"Alpha approach", "beta search", "(no direction)"}

    alpha = fams["Alpha approach"]
    assert set(alpha["members"]) == {a1.id, a2.id}
    assert alpha["root_id"] in alpha["members"]
    assert alpha["best_id"] == a2.id
    assert alpha["best_score"] == 0.8
    assert alpha["latest_activity"] == max(a1.created_at, a2.created_at)

    beta = fams["beta search"]
    assert beta["members"] == [a3.id]
    assert beta["best_id"] is None  # its only member failed -> unscoreable
    assert beta["best_score"] is None

    assert fams["(no direction)"]["members"] == [c1.id]
    json.dumps(list(fams.values()))


def test_lineage(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, metrics={"score": 0.1})
    b = commit_attempt(h, parent=a.id, metrics={"score": 0.2})
    c = commit_attempt(h, parent=b.id, metrics={"score": 0.3})

    chain = queries.lineage(h, c.id, scorer)
    assert [r["id"] for r in chain] == [a.id, b.id, c.id]
    assert [r["score"] for r in chain] == [0.1, 0.2, 0.3]

    assert queries.lineage(h, "no-such-id") == []


def test_run_summary(history_factory, commit_attempt):
    h = history_factory()
    a1, a2, a3 = _seed(h, commit_attempt)

    s = queries.run_summary(h, scorer)
    assert s["n_attempts"] == 3
    assert s["n_done"] == 2
    assert s["n_failed"] == 1
    assert s["best"] == {"id": a2.id, "score": 0.8, "name": a2.name}
    assert s["n_families"] == 2
    assert s["total_cost"] == 0.35

    traj = s["score_trajectory"]
    assert len(traj) == 2  # one entry per SCORED attempt
    bests = [b for _, b in traj]
    assert bests == sorted(bests)  # best-so-far never decreases
    assert bests[-1] == 0.8
    json.dumps(s)


def test_reads_exclude_open_workspaces(open_workspace_alongside):
    """A run mid-flight: the open workspace is not a fact of the run yet, so
    no read-layer view may count it (the phantom-attempt regression)."""
    run = open_workspace_alongside
    committed_ids = {a.id for a in run.committed}

    table = queries.attempt_table(run.history, scorer)
    assert {r["id"] for r in table} == committed_ids
    assert all(r["status"] in ("done", "fail") for r in table)

    fams = queries.families(run.history, scorer)
    assert {m for f in fams for m in f["members"]} == committed_ids

    s = queries.run_summary(run.history, scorer)
    assert s["n_attempts"] == len(committed_ids)
    assert s["n_done"] + s["n_failed"] == s["n_attempts"]


def test_run_summary_empty(history_factory):
    h = history_factory()
    s = queries.run_summary(h, scorer)
    assert s["n_attempts"] == 0
    assert s["best"] is None
    assert s["score_trajectory"] == []
    assert s["total_cost"] == 0


def test_sub_results_convention(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, metrics={
        "score": 0.5,
        "accuracy_by_cell": {"0,0": 1.0, "0,1": 0.0},
        "cells": {"a": {"score": 1.0}},
        "n": 3,
        "notes_by_cell": "not a dict",
    })
    plain = commit_attempt(h, metrics={"score": 0.5})

    tables = queries.sub_results(h, a.id)
    assert tables == {"eval": {
        "accuracy_by_cell": {"0,0": 1.0, "0,1": 0.0},
        "cells": {"a": {"score": 1.0}},
    }}
    assert queries.sub_results(h, plain.id) == {}
    assert queries.sub_results(h, "no-such-id") == {}


def test_attempt_detail(history_factory, commit_attempt):
    h = history_factory()
    a = commit_attempt(h, metrics={"score": 0.5, "cells": {"a": 1}},
                       metadata={"strategy": "agent", "cost": 0.2})
    b = commit_attempt(h, parent=a.id, metrics={"score": 0.7})

    d = queries.attempt_detail(h, b.id, scorer)
    assert d["id"] == b.id
    assert d["parent"] == a.id
    assert d["score"] == 0.7
    assert d["stages"]["eval"]["score"] == 0.7
    assert d["stages"]["eval"]["metrics"]["score"] == 0.7
    assert "solution.py" in d["files"]
    assert d["lineage"] == [a.id, b.id]
    assert d["sub_results"] == {}
    json.dumps(d)

    d = queries.attempt_detail(h, a.id, scorer)
    assert d["metadata"]["strategy"] == "agent"
    assert d["sub_results"] == {"eval": {"cells": {"a": 1}}}

    assert queries.attempt_detail(h, "no-such-id", scorer) is None


def test_safe_result_and_safe_code_on_normal_attempt(history_factory,
                                                     commit_attempt):
    h = history_factory()
    a = commit_attempt(h, metrics={"score": 0.5})

    result = queries.safe_result(a)
    assert result is not None
    assert result.stages["eval"].metrics["score"] == 0.5
    assert queries.safe_code(a) == "def solve(): return 1"


def test_safe_result_none_without_result_json(history_factory):
    h = history_factory()
    ws = h.workspace()
    (ws.path / "solution.py").write_text("def solve(): return 1",
                                         encoding="utf-8")
    a = ws.commit(success=True)

    assert queries.safe_result(a) is None
    assert queries.safe_code(a) == "def solve(): return 1"


def test_safe_result_none_on_corrupt_result_json(history_factory):
    h = history_factory()
    ws = h.workspace()
    (ws.path / "solution.py").write_text("def solve(): return 1",
                                         encoding="utf-8")
    (ws.path / "result.json").write_text("{not json", encoding="utf-8")
    a = ws.commit(success=True)

    assert queries.safe_result(a) is None


def test_safe_code_none_without_solution(history_factory):
    h = history_factory()
    ws = h.workspace()
    (ws.path / "result.json").write_text(
        json.dumps({"completed": False, "failed_stage": "generate",
                    "stages": {"generate": {"metrics": {},
                                            "errors": {"crash": "boom"},
                                            "warnings": {}}}}),
        encoding="utf-8")
    a = ws.commit(success=False)

    assert queries.safe_code(a) is None
    result = queries.safe_result(a)
    assert result is not None and not result.completed
