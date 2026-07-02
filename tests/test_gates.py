"""Legitimacy gates: pure facts, stable messages, bench binding.

The gate kit (utils/gates.py) must report exactly what AgentStrategy's
finish used to check inline — same conditions, same message strings —
while staying pure: no file is created, moved, or modified by evaluate.
"""

import tempfile
from pathlib import Path

from groundhog.base.types import EvaluationResult, StageResult
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.direction import write_direction
from groundhog.utils.gates import (
    DIRECTION_DUPLICATE,
    DIRECTION_MISSING,
    DIRECTION_MODIFIED,
    SOLUTION_IDENTICAL,
    GateKit,
    evaluate_gates,
)
from groundhog.utils.results import write_result


def _commit_attempt(history, direction, code="print(1)", parent=None):
    ws = history.workspace(parent=parent)
    (ws.path / "solution.py").write_text(code, encoding="utf-8")
    write_direction(ws.path, direction)
    write_result(
        ws.path,
        EvaluationResult(stages={"e": StageResult(metrics={"score": 0.5})}),
    )
    return ws.commit(success=True)


def _fake_parent(code="print(1)", direction="rollout"):
    """Attempt-like object exposing read_file/code, backend-agnostic."""
    files = {"core_direction.md": direction + "\n"}

    class P:
        def read_file(self, rel):
            return files.get(rel)

    p = P()
    p.code = code
    return p


def test_fresh_without_direction_fails():
    with tempfile.TemporaryDirectory() as tmp:
        violations = evaluate_gates(Path(tmp), None, history=None)
        assert [v.gate for v in violations] == [DIRECTION_MISSING]
        assert violations[0].severity == "fail"
        assert violations[0].message == (
            "fresh attempt did not create core_direction.md"
        )


def test_fresh_with_direction_at_root_passes():
    with tempfile.TemporaryDirectory() as tmp:
        write_direction(tmp, "rollout beam search")
        assert evaluate_gates(Path(tmp), None, history=None) == []


def test_fresh_direction_in_work_counts_and_evaluate_stays_pure():
    # A direction still sitting in work/ counts as present (the standard
    # finish promotes it) — and evaluate must NOT do the promoting.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        (work / "core_direction.md").write_text("rollout\n", encoding="utf-8")
        assert evaluate_gates(Path(tmp), None, history=None) == []
        assert not (Path(tmp) / "core_direction.md").exists()


def test_fresh_duplicate_direction_fails_and_exclude_spares_self():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        existing = _commit_attempt(history, "rollout beam search")

        ws = history.workspace()
        write_direction(ws.path, "rollout beam search")
        # The strategy always excludes the workspace's own display id —
        # with only_done=False the open workspace is itself visible in
        # history and would otherwise match its own direction.
        violations = evaluate_gates(
            ws.path, None, history=history, exclude=[ws.display_id]
        )
        assert [v.gate for v in violations] == [DIRECTION_DUPLICATE]
        assert violations[0].severity == "fail"
        assert violations[0].message == (
            "fresh attempt duplicated an existing core direction"
        )

        # Excluding the existing attempt as well makes the direction unique.
        spared = evaluate_gates(
            ws.path,
            None,
            history=history,
            exclude=[ws.display_id, existing.id],
        )
        assert spared == []


def test_fresh_unique_direction_passes_against_history():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit_attempt(history, "rollout beam search")

        ws = history.workspace()
        write_direction(ws.path, "genetic algorithm with tournament selection")
        assert (
            evaluate_gates(
                ws.path, None, history=history, exclude=[ws.display_id]
            )
            == []
        )


def test_inherited_modified_direction_flags():
    with tempfile.TemporaryDirectory() as tmp:
        parent = _fake_parent(direction="rollout")
        write_direction(tmp, "something else entirely")
        (Path(tmp) / "solution.py").write_text("print(2)", encoding="utf-8")
        violations = evaluate_gates(Path(tmp), parent)
        assert [v.gate for v in violations] == [DIRECTION_MODIFIED]
        assert violations[0].severity == "flag"


def test_inherited_unchanged_direction_passes():
    with tempfile.TemporaryDirectory() as tmp:
        parent = _fake_parent(direction="rollout")
        write_direction(tmp, "rollout")
        (Path(tmp) / "solution.py").write_text("print(2)", encoding="utf-8")
        assert evaluate_gates(Path(tmp), parent) == []


def test_identical_solution_flags_but_does_not_fail():
    with tempfile.TemporaryDirectory() as tmp:
        parent = _fake_parent(code="print(1)", direction="rollout")
        write_direction(tmp, "rollout")
        (Path(tmp) / "solution.py").write_text("print(1)", encoding="utf-8")
        violations = evaluate_gates(Path(tmp), parent)
        assert [v.gate for v in violations] == [SOLUTION_IDENTICAL]
        assert violations[0].severity == "flag"
        assert violations[0].message == "solution.py is byte-identical to parent"


def test_fresh_missing_direction_reports_only_missing():
    # No direction text means the duplicate gate cannot (and must not) fire.
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp) / "store")
        _commit_attempt(history, "rollout")
        ws_dir = Path(tmp) / "ws"
        ws_dir.mkdir()
        violations = evaluate_gates(ws_dir, None, history=history)
        assert [v.gate for v in violations] == [DIRECTION_MISSING]


def test_gatekit_binds_history_and_derives_exclude():
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        existing = _commit_attempt(history, "rollout beam search")

        toolkit = SimpleNamespace(history=history)
        kit = GateKit(toolkit)

        ws_dir = Path(tmp) / "candidate"
        ws_dir.mkdir()
        write_direction(ws_dir, "rollout beam search")

        # Plain path: duplicate found.
        found = kit.evaluate(ws_dir)
        assert [v.gate for v in found] == [DIRECTION_DUPLICATE]

        # Workspace-like object whose display id IS the existing attempt:
        # exclude derives automatically and the duplicate is spared.
        ws_like = SimpleNamespace(path=ws_dir, display_id=existing.id)
        assert kit.evaluate(ws_like) == []


def test_assemble_installs_gates():
    from groundhog import Task, assemble_toolkit
    from groundhog.base.types import (
        Context,
        Data,
        EvalStage,
        Evaluator,
    )

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

    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    with tempfile.TemporaryDirectory() as tmp:
        tk = assemble_toolkit(task, path=Path(tmp))
        assert isinstance(tk.gates, GateKit)
        ws_dir = Path(tmp) / "ws"
        ws_dir.mkdir()
        violations = tk.gates.evaluate(ws_dir)
        assert [v.gate for v in violations] == [DIRECTION_MISSING]
