"""The task.py agent_tools(toolkit) hook + the one merge rule.

Design (2026-07 sessions): per-task agent tools live in the task.py MODULE
(never on the pure Task object). assemble_toolkit calls the hook LAST against
the finished toolkit; precedence is strategy > task > default via one
name-keyed merge helper used at both merge points. The strategy-side merge
also fixes a latent bug: _get_tools previously concatenated without dedup.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from groundhog import Task, assemble_toolkit, agent_tool
from groundhog.base.types import (
    Data, Context, Evaluator, EvalStage, StageResult,
)
from groundhog.agents.tools import _merge_agent_tools, collect_task_tools


class _Data(Data):
    def get_train(self): return None
    def get_test(self): return None


class _Ctx(Context):
    def get_brief(self): return "b"
    def get_extended(self): return "e"


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult()

    def get_stages(self, data):
        return [EvalStage("eval", "eval", lambda cp: StageResult(),
                          scorer=lambda r: r.metrics.get("score", 0.0))]


def _task():
    return Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")


def _tool(name="my-tool", reply="ok"):
    return agent_tool(name=name, description="d", func=lambda: reply, params={})


# --- The hook ---------------------------------------------------------------

def test_hook_runs_last_against_finished_toolkit(tmp_path):
    seen = {}

    def agent_tools(tk):
        # The bench is complete when the hook runs: closures over these are valid.
        seen["history"] = hasattr(tk, "history")
        seen["rng"] = hasattr(tk, "rng")
        seen["selection"] = hasattr(tk, "selection")
        seen["get_prior"] = hasattr(tk, "get_prior")
        return [_tool()]

    tk = assemble_toolkit(_task(), path=tmp_path, agent_tools=agent_tools)
    assert seen == {"history": True, "rng": True, "selection": True, "get_prior": True}
    assert "my-tool" in [t.name for t in tk.agent_tools]


def test_hook_is_optional(tmp_path):
    """No hook -> exactly the framework defaults, nothing else."""
    from groundhog.agents.tools import build_default_agent_tools
    tk = assemble_toolkit(_task(), path=tmp_path)
    assert [t.name for t in tk.agent_tools] == \
        [t.name for t in build_default_agent_tools(tk)]


def test_hook_returning_none_means_no_task_tools(tmp_path):
    from groundhog.agents.tools import build_default_agent_tools
    tk = assemble_toolkit(_task(), path=tmp_path, agent_tools=lambda t: None)
    assert [t.name for t in tk.agent_tools] == \
        [t.name for t in build_default_agent_tools(tk)]


def test_non_agent_tool_return_fails_at_build_time(tmp_path):
    with pytest.raises(TypeError, match="AgentTool"):
        assemble_toolkit(_task(), path=tmp_path, agent_tools=lambda t: ["not a tool"])


def test_duplicate_task_tool_names_fail_at_build_time(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        assemble_toolkit(_task(), path=tmp_path,
                         agent_tools=lambda t: [_tool("x"), _tool("x")])


def test_collect_task_tools_none_hook():
    assert collect_task_tools(None, object()) == []


# --- The merge rule ----------------------------------------------------------

def test_merge_higher_layer_wins_and_shadow_is_logged():
    lower = [_tool("a", "lower-a"), _tool("b", "lower-b")]
    higher = [_tool("b", "higher-b"), _tool("c", "higher-c")]
    messages = []
    log = SimpleNamespace(info=messages.append)

    merged = _merge_agent_tools(lower, higher, layer="strategy", log=log)

    by_name = {t.name: t for t in merged}
    assert set(by_name) == {"a", "b", "c"}
    assert by_name["b"].execute().output == "higher-b"
    assert len(messages) == 1 and "'b'" in messages[0] and "strategy" in messages[0]


def test_merge_is_idempotent():
    tools = [_tool("a"), _tool("b")]
    once = _merge_agent_tools(tools, tools, layer="task", log=SimpleNamespace(info=lambda m: None))
    assert sorted(t.name for t in once) == ["a", "b"]


# --- Strategy-side merge (the latent dup-concat bug) --------------------------

def test_get_tools_strategy_layer_shadows_task_layer():
    """A task tool named like a strategy tool must NOT reach the agent twice —
    the strategy's version wins and the base one is shadowed (was: plain
    concat, both went to the agent)."""
    from groundhog.strategies.agent import AgentStrategy

    def agent_tools(tk):
        return [_tool("get-learnings", "task-version")]

    with tempfile.TemporaryDirectory() as tmp:
        tk = assemble_toolkit(_task(), path=Path(tmp), agent_tools=agent_tools)
        strat = AgentStrategy()
        strat.through = None  # normally set by __call__ before _get_tools
        ws = SimpleNamespace(path=Path(tmp))
        tools = strat._get_tools(tk, ws, prior=None, phase="explore")

    names = [t.name for t in tools]
    assert names.count("get-learnings") == 1, f"duplicate tool reached the agent: {names}"
    winner = next(t for t in tools if t.name == "get-learnings")
    # The strategy's version wraps toolkit.learnings.get — executing it must
    # NOT return the task stub's payload.
    assert winner.execute().output != "task-version", \
        "task-layer tool was not shadowed by the strategy layer"
