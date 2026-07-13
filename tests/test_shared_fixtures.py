"""Smoke tests proving the shared conftest fixtures produce exactly the
states they describe, on both history backends (the fixtures ride the
parametrized ``history_factory``)."""

from groundhog.utils.queries import safe_code, safe_result


def test_attempt_without_result_state(attempt_without_result):
    a = attempt_without_result
    assert a.status == "done"
    assert "result.json" not in a.list_files()
    assert safe_result(a) is None
    assert safe_code(a) == "def solve(): return 1"
    assert a.metadata["no_recorded_result"] is True


def test_failed_attempt_without_code_state(failed_attempt_without_code):
    a = failed_attempt_without_code
    assert a.status == "fail"
    assert "solution.py" not in a.list_files()
    assert safe_code(a) is None
    result = safe_result(a)
    assert result is not None
    assert not result.completed
    assert result.failed_stage == "generate"


def test_open_workspace_alongside_state(open_workspace_alongside):
    run = open_workspace_alongside
    committed_ids = {a.id for a in run.committed}
    assert {a.id for a in run.history.list()} == committed_ids

    in_progress = run.history.list_in_progress()
    assert len(in_progress) == 1
    assert in_progress[0].parent == run.committed[-1].id

    assert run.open_ws.parent == run.committed[-1].id
    assert (run.open_ws.path / "solution.py").exists()
