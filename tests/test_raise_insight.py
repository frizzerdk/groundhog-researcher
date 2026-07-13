"""raise-insight: the message-out-of-the-sandbox default tool.

Appends a stamped markdown entry to the run-root ``insights.md`` (learnings
-style append) and, when an attempt is in flight with a started log, records
an event so the note shows up in the attempt log. It must be present on every
assembled toolkit, degrade gracefully with no open attempt, and never touch
the solution.
"""

import re
import tempfile
from pathlib import Path

from groundhog import Task, assemble_toolkit
from groundhog.base.types import Context, Data, EvalStage, Evaluator, StageResult
from groundhog.tools.attempt_logger import PhaseEvent

ISO = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"


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
        return [EvalStage("eval", "eval", lambda cp: StageResult(),
                          scorer=lambda r: r.metrics.get("score", 0.0))]


def _tk(tmp):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    return assemble_toolkit(task, path=Path(tmp))


def _tool(tk):
    tools = {t.name: t for t in tk.agent_tools}
    assert "raise-insight" in tools, f"default tools: {sorted(tools)}"
    return tools["raise-insight"]


def _insights(tk):
    return (tk.path / "insights.md").read_text(encoding="utf-8")


def test_default_tool_present_and_bound():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        res = tool.execute(text="hello")
        assert res.success, res.error
        assert "no open attempt" in res.output
        assert (tk.path / "insights.md").exists()


def test_append_format_and_stamping():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        tool.execute(kind="blocker", text="the eval command is too slow")
        content = _insights(tk)
        assert re.search(rf"^## {ISO} \| blocker$", content, re.MULTILINE)
        assert "the eval command is too slow" in content


def test_second_entry_gets_separator():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        tool.execute(kind="idea", text="first")
        tool.execute(kind="insight", text="second")
        content = _insights(tk)
        assert "\n\n---\n\n" in content
        assert content.count("## ") == 2


def test_empty_text_records_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        res = tool.execute(text="   ")
        assert res.success, res.error
        assert "nothing recorded" in res.output
        assert not (tk.path / "insights.md").exists()


def test_unknown_kind_folds_to_insight():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        tool.execute(kind="rambling", text="note")
        content = _insights(tk)
        assert re.search(rf"^## {ISO} \| insight$", content, re.MULTILINE)


def test_workspace_id_and_phase_stamped_and_logged():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        ws = tk.history.workspace()
        tk.attempt_logger.attempt_start(ws.path)
        tk.attempt_logger.log(PhaseEvent(phase="explore"))
        with tk.ws.attempt(ws):
            res = tool.execute(kind="tool-request", text="a profiler tool")
        assert res.success, res.error
        content = _insights(tk)
        assert re.search(
            rf"^## {ISO} \| tool-request \| attempt {ws.display_id} \| phase explore$",
            content, re.MULTILINE)
        # The note is recorded in the attempt log too.
        jsonl = (ws.path / "attemptlog.jsonl").read_text(encoding="utf-8")
        assert '"type": "insight"' in jsonl
        assert "a profiler tool" in jsonl
        ws.abort()


def test_no_started_log_skips_event_but_appends_file():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        ws = tk.history.workspace()
        # Attempt is in flight, but the logger was never started — the file
        # entry stands on its own; no attemptlog.jsonl is written.
        with tk.ws.attempt(ws):
            res = tool.execute(text="works without a started log")
        assert res.success, res.error
        content = _insights(tk)
        assert re.search(
            rf"^## {ISO} \| insight \| attempt {ws.display_id}$",
            content, re.MULTILINE)
        assert not (ws.path / "attemptlog.jsonl").exists()
        ws.abort()


def test_non_utf8_byte_does_not_kill_the_channel():
    """The append never reads the file back, so one stray non-UTF8 byte
    (e.g. from a crashed agent) can't permanently break raise-insight."""
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        (tk.path / "insights.md").write_bytes(b"## old entry\n\xff\xfe garbage\n")
        res = tool.execute(text="still works")
        assert res.success, res.error
        raw = (tk.path / "insights.md").read_bytes()
        assert b"\xff\xfe garbage" in raw  # prior bytes untouched
        assert b"still works" in raw
        assert b"---" in raw  # separator present (platform newlines vary)


def test_text_is_capped_at_the_tool_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        tk = _tk(tmp)
        tool = _tool(tk)
        res = tool.execute(text="x" * 100_000)
        assert res.success, res.error
        content = _insights(tk)
        assert len(content) < 10_000
        assert "[truncated]" in content
