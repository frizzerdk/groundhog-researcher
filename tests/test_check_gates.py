"""check-gates: the mid-work self-check button (first framework default tool).

The tool reads the attempt in flight via toolkit.ws and reports what the
commit-time gate would find. It must be present on every assembled
toolkit, bound to it (the defaults layer binds its own tools), degrade
gracefully when nothing is in flight, and never mutate the workspace.
"""

import tempfile
from pathlib import Path

from groundhog import Task, assemble_toolkit
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    EvaluationResult,
    Evaluator,
    StageResult,
)
from groundhog.utils.direction import write_direction
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


def _tk(tmp):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    return assemble_toolkit(task, path=Path(tmp))


def _tool(tk):
    tools = {t.name: t for t in tk.agent_tools}
    assert "check-gates" in tools, f"default tools: {sorted(tools)}"
    return tools["check-gates"]


def _ok_result(score=0.5):
    return EvaluationResult(
        stages={"eval": StageResult(metrics={"score": score})}
    )


def test_default_tool_present_and_bound():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        # Bound by the defaults builder itself: executing must not raise the
        # unbound-toolkit RuntimeError.
        res = tool.execute()
        assert res.success, res.error
        assert "No attempt is in flight" in res.output


def test_fresh_workspace_without_direction_reports_fail():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        with tk.ws.attempt(ws):
            res = tool.execute()
        assert res.success, res.error
        assert "[FAIL] direction-missing" in res.output
        assert "FAILED" in res.output
        # Read-only: the check must not promote or create anything.
        assert not (ws.path / "core_direction.md").exists()
        ws.abort()


def test_fresh_workspace_with_direction_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout beam search")
        with tk.ws.attempt(ws):
            res = tool.execute()
        assert res.success, res.error
        assert "All gates pass" in res.output
        ws.abort()


def test_child_workspace_flags_identical_solution():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)

        parent_ws = tk.history.workspace()
        (parent_ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(parent_ws.path, "rollout")
        parent = finalize_attempt(tk, parent_ws, _ok_result(), None)

        child = tk.history.workspace(parent=parent.id)
        (child.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(child.path, "rollout")
        with tk.ws.attempt(child):
            res = tool.execute()
        assert res.success, res.error
        assert "[FLAG] solution-identical" in res.output
        # A child never faces the fresh-direction gates.
        assert "direction-duplicate" not in res.output
        child.abort()


def test_committed_attempt_view_does_not_duplicate_itself():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)

        ws = tk.history.workspace()
        (ws.path / "solution.py").write_text("print(1)", encoding="utf-8")
        write_direction(ws.path, "rollout beam search")
        attempt = finalize_attempt(tk, ws, _ok_result(), None)

        # Point the handle at the committed record (what the CLI's
        # `tool run check-gates --attempt <id>` does).
        with tk.ws.attempt(attempt.id):
            res = tool.execute()
        assert res.success, res.error
        assert "All gates pass" in res.output
