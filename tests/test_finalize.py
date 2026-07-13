"""The standard finish: finalize_attempt owns promote→gates→record→commit→note.

Convention, not contract: the helper is bound as toolkit.finalize but the
Strategy contract never requires it; these tests pin what "the 95% finish"
does so every caller (AgentStrategy, the CLI, sessions) means the same
thing by it.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from groundhog import Task
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    EvaluationResult,
    Evaluator,
    StageResult,
)
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.direction import read_direction, write_direction
from groundhog.utils.finalize import finalize_attempt


class _Data(Data):
    def get_train(self):
        return None

    def get_test(self):
        return None


class _Ctx(Context):
    def get_brief(self):
        return "b"

    def get_extended(self):
        return "e"


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult()

    def get_stages(self, data):
        return [
            EvalStage(
                "eval",
                "eval",
                lambda cp: StageResult(),
                scorer=lambda r: r.metrics.get("score", 0.0),
            )
        ]


def _toolkit(tmp):
    history = FolderAttemptHistory(Path(tmp))
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    return SimpleNamespace(task=task, history=history)


def _ok_result(score=0.5):
    return EvaluationResult(
        stages={"eval": StageResult(metrics={"score": score})}
    )


def test_standard_finish_commits_names_and_caches_note():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "Rollout beam search\n\nGreedy with lookahead.")

        attempt = finalize_attempt(tk, ws, _ok_result(0.5), None)

        assert attempt.status == "done"
        assert attempt.name == "rollout-beam-search"
        meta = attempt.metadata
        assert meta["strategy"] == "manual"
        assert meta["cost"] == 0.0
        assert tk.history.get_note(attempt.id, "score") == "0.5000"


def test_fresh_without_direction_commits_as_failed():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")

        result = _ok_result()
        attempt = finalize_attempt(tk, ws, result, None)

        assert result.completed is False
        assert result.failed_stage == "core_direction"
        assert attempt.status == "fail"
        assert attempt.metadata["gate_failure"] == (
            "fresh attempt did not create core_direction.md"
        )
        # Note lookup via the OBJECT: failed attempts don't resolve by id
        # (history.get walks done attempts only).
        assert tk.history.get_note(attempt, "score") == "fail"


def test_fresh_duplicate_direction_commits_as_failed():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        first = tk.history.workspace()
        (first.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(first.path, "rollout")
        finalize_attempt(tk, first, _ok_result(), None)

        second = tk.history.workspace()
        (second.path / "solution.py").write_text("print(2)", encoding="utf-8")
        write_direction(second.path, "rollout")
        attempt = finalize_attempt(tk, second, _ok_result(), None)

        assert attempt.status == "fail"
        assert attempt.metadata["gate_failure"] == (
            "fresh attempt duplicated an existing core direction"
        )


def test_direction_promoted_from_work_before_gates():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        work = ws.path / "work"
        work.mkdir(exist_ok=True)
        (work / "core_direction.md").write_text("rollout\n", encoding="utf-8")

        attempt = finalize_attempt(tk, ws, _ok_result(), None)
        assert attempt.status == "done"
        assert attempt.name == "rollout"


def test_inherited_modified_direction_is_restored_and_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent_ws = tk.history.workspace()
        (parent_ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(parent_ws.path, "rollout")
        parent = finalize_attempt(tk, parent_ws, _ok_result(), None)

        child_ws = tk.history.workspace(parent=parent.id)
        (child_ws.path / "solution.py").write_text("print(2)", encoding="utf-8")
        write_direction(child_ws.path, "a totally different idea")

        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("direction_restored") is True
        assert read_direction(child_ws.path).strip() == "rollout"


def _inherited_child(tk, parent_direction, child_direction, child_code="print(2)"):
    parent_ws = tk.history.workspace()
    (parent_ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
    write_direction(parent_ws.path, parent_direction)
    parent = finalize_attempt(tk, parent_ws, _ok_result(), None)

    child_ws = tk.history.workspace(parent=parent.id)
    (child_ws.path / "solution.py").write_text(child_code, encoding="utf-8")
    write_direction(child_ws.path, child_direction)
    return parent, child_ws


def test_body_only_edit_restores_full_direction():
    # Directions are IMMUTABLE by default — a body edit is a modification
    # and the parent's FULL direction is restored (the body-refinable
    # premise was rejected; an explicit direction-change strategy is a
    # future feature).
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent, child_ws = _inherited_child(
            tk,
            "rollout\n\nGreedy lookahead.",
            "rollout\n\nGreedy lookahead with deeper search.",
        )
        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("direction_restored") is True
        kept = read_direction(child_ws.path)
        assert kept.strip() == "rollout\n\nGreedy lookahead."
        assert "deeper search" not in kept


def test_first_line_and_body_edit_restores_full_direction():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent, child_ws = _inherited_child(
            tk,
            "rollout\n\nGreedy lookahead.",
            "beam search\n\nGreedy lookahead, wider.",
        )
        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("direction_restored") is True
        kept = read_direction(child_ws.path)
        assert kept.strip() == "rollout\n\nGreedy lookahead."
        assert "wider" not in kept


def test_deleted_direction_records_a_plain_restore():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent, child_ws = _inherited_child(
            tk,
            "rollout\n\nGreedy lookahead.",
            "placeholder",
        )
        (child_ws.path / "core_direction.md").unlink()
        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("direction_restored") is True
        assert read_direction(child_ws.path).strip() == "rollout\n\nGreedy lookahead."


def test_unchanged_inherited_direction_is_not_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent, child_ws = _inherited_child(
            tk,
            "rollout\n\nGreedy lookahead.",
            "rollout\n\nGreedy lookahead.",
        )
        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("direction_restored") is None


def test_identical_solution_flags_non_promotable_but_commits_done():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        parent_ws = tk.history.workspace()
        (parent_ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(parent_ws.path, "rollout")
        parent = finalize_attempt(tk, parent_ws, _ok_result(), None)

        child_ws = tk.history.workspace(parent=parent.id)
        (child_ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(child_ws.path, "rollout")

        attempt = finalize_attempt(tk, child_ws, _ok_result(), parent)

        assert attempt.status == "done"
        assert attempt.metadata.get("non_promotable") is True
        assert attempt.metadata["non_promotable_reason"] == (
            "solution.py is byte-identical to parent"
        )


def test_caller_metadata_wins_over_strategy_and_cost_args():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout")

        attempt = finalize_attempt(
            tk,
            ws,
            _ok_result(),
            None,
            metadata={"strategy": "agent", "prior": None, "cost": 1.23},
            strategy="ignored",
            cost=9.9,
        )
        assert attempt.metadata["strategy"] == "agent"
        assert attempt.metadata["cost"] == 1.23


def test_strategy_label_and_explicit_name():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout")

        attempt = finalize_attempt(
            tk, ws, _ok_result(), None, strategy="session", name="My Plan"
        )
        assert attempt.metadata["strategy"] == "session"
        assert attempt.name == "my-plan"


def test_bench_binding_runs_the_same_finish():
    from groundhog.assemble import assemble_toolkit

    with tempfile.TemporaryDirectory() as tmp:
        task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
        tk = assemble_toolkit(task, path=Path(tmp))
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout")

        attempt = tk.finalize(ws, _ok_result(), strategy="session")
        assert attempt.status == "done"
        assert attempt.metadata["strategy"] == "session"


def test_scorer_exception_never_fails_a_finished_attempt():
    """The score note runs AFTER the commit and calls user scorer code —
    any exception it raises must be swallowed, or the strategy's except
    path would abort (delete) the attempt that just committed."""

    class _BoomEval(_Eval):
        def get_stages(self, data):
            return [
                EvalStage(
                    "eval",
                    "eval",
                    lambda cp: StageResult(),
                    scorer=lambda r: 1 / 0,  # user code: ZeroDivisionError
                )
            ]

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = Task(data=_Data(), context=_Ctx(), evaluator=_BoomEval(), name="t")
        tk = SimpleNamespace(task=task, history=history)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout")

        attempt = finalize_attempt(tk, ws, _ok_result(0.5), None)
        assert attempt.status == "done"
        # Note skipped, attempt intact.
        assert tk.history.get_note(attempt, "score") is None


def test_optimizer_registry_resolves_strategy_suffix_alias():
    """PlanApproaches queues under "fresh_agent"; FreshAgentStrategy must be
    resolvable under that conventional short name, or every queued item
    burns on "unknown strategy" (campaign finding)."""
    from groundhog import FreshAgentStrategy, Improve, SimpleOptimizer

    with tempfile.TemporaryDirectory() as tmp:
        tk = _toolkit(tmp)
        tk.path = Path(tmp)
        tk.log = type("L", (), {"info": lambda self, m: None,
                                "end": lambda self: None})()
        opt = SimpleOptimizer(
            tk, strategy=Improve(), extras=[FreshAgentStrategy()]
        )
        assert "fresh_agent" in opt._strategy_registry
        assert isinstance(
            opt._strategy_registry["fresh_agent"], FreshAgentStrategy
        )
