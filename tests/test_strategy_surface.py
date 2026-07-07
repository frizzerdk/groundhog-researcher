"""The strategy surface: Strategy.name, scan discovery, attribution.

One name per strategy — attempt attribution (metadata["strategy"]), the
optimizer's queue registry, and discovery all read the same class
attribute. Discovery is a scan, not a registry: nothing registers itself,
``discover_strategies`` reads what exists at call time.
"""

import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

from groundhog import (
    Analyse,
    CrossPollinate,
    CrossPollinateAgent,
    FreshApproach,
    FreshAgentStrategy,
    Improve,
    PlanApproaches,
    SimpleOptimizer,
    Strategy,
    Task,
    Toolkit,
    discover_strategies,
)
from groundhog.backends.mock import MockBackend
from groundhog.base.backend import BackendRegistry
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    Evaluator,
    StageResult,
)
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.strategies.agent import AgentStrategy


# --- Fixtures ---------------------------------------------------------------

class _Data(Data):
    def get_train(self):
        return None

    def get_test(self):
        return None


class _Ctx(Context):
    def get_brief(self):
        return "Write solve() returning a float."

    def get_extended(self):
        return "def solve() -> float"


def _read(code_or_path):
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read(code_or_path)
        ns = {}
        try:
            exec(code, ns)
            value = float(ns["solve"]())
        except Exception as e:
            return StageResult(errors={"crash": str(e)})
        return StageResult(metrics={"value": value})

    def get_stages(self, data):
        return [
            EvalStage("eval", "eval",
                      lambda cp, d=data: self.evaluate(cp, d),
                      scorer=lambda r: -1.0 if r.errors
                      else r.metrics.get("value", 0.0) / 100.0),
        ]


def _task():
    return Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")


def _code(value):
    return f"def solve():\n    return {value}"


def _code_block(value):
    return f"```python\n{_code(value)}\n```"


def _commit_root(history, task, value, direction=None):
    from groundhog.utils.direction import write_direction
    from groundhog.utils.results import write_result
    ws = history.workspace()
    (ws.path / "solution.py").write_text(_code(value), encoding="utf-8")
    if direction:
        write_direction(ws.path, direction)
    write_result(ws.path, task.evaluate(_code(value)))
    return ws.commit(success=True)


# --- Strategy.name ----------------------------------------------------------

def test_strategy_name_defaults_derive_from_class_name():
    assert Improve.name == "improve"
    assert FreshApproach.name == "fresh_approach"
    assert CrossPollinate.name == "cross_pollinate"
    assert Analyse.name == "analyse"
    assert PlanApproaches.name == "plan_approaches"
    assert AgentStrategy.name == "agent"
    assert FreshAgentStrategy.name == "fresh_agent"
    assert CrossPollinateAgent.name == "cross_pollinate_agent"


def test_strategy_name_override_and_instance_access():
    class MyCustomStrategy(Strategy):
        def __call__(self, toolkit, config=None):
            return {}

    class Renamed(Strategy):
        name = "special"

        def __call__(self, toolkit, config=None):
            return {}

    assert MyCustomStrategy.name == "my_custom"
    assert MyCustomStrategy().name == "my_custom"
    assert Renamed.name == "special"
    assert Renamed().name == "special"


def test_optimizer_registry_resolves_declared_name():
    """The queue registry answers to Strategy.name plus the old aliases."""
    class WeirdStrategy(Strategy):
        name = "my_alias"

        def __call__(self, toolkit, config=None):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        tk = SimpleNamespace(task=_task(), history=history, path=Path(tmp))
        strat = WeirdStrategy()
        opt = SimpleOptimizer(tk, strategy=strat, seed_strategy=None)
        assert opt._strategy_registry["my_alias"] is strat
        assert opt._strategy_registry["weird_strategy"] is strat
        assert opt._strategy_registry["weird"] is strat


# --- Discovery --------------------------------------------------------------

def test_discover_finds_builtins():
    entries = discover_strategies()
    by_name = {e["name"]: e for e in entries}
    assert {
        "improve", "fresh_approach", "cross_pollinate", "analyse",
        "plan_approaches", "agent", "fresh_agent", "cross_pollinate_agent",
    } <= set(by_name)
    assert all(e["source"] == "builtin" for e in entries)

    improve = by_name["improve"]
    assert improve["cls"] is Improve
    assert improve["doc"] == "Refine existing code via LLM-generated diffs."
    assert "max_retries" in improve["params"]
    assert improve["params"]["max_retries"]["default"] == 3
    assert improve["params"]["max_retries"]["description"]


def test_discover_includes_task_module_strategies():
    mod = types.ModuleType("fake_task")

    class EchoStrategy(Strategy):
        """Echoes for tests."""

        def __call__(self, toolkit, config=None):
            return {}

    mod.EchoStrategy = EchoStrategy
    mod.Improve = Improve  # a re-exported builtin stays builtin

    entries = discover_strategies(module=mod)
    by_name = {e["name"]: e for e in entries}
    assert by_name["echo"]["source"] == "task"
    assert by_name["echo"]["cls"] is EchoStrategy
    assert by_name["echo"]["doc"] == "Echoes for tests."
    assert by_name["improve"]["source"] == "builtin"
    # Builtins sort first.
    sources = [e["source"] for e in entries]
    assert sources.index("task") > sources.index("builtin")


# --- Attribution round-trip -------------------------------------------------

def test_improve_attribution_matches_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        prior = _commit_root(history, task, 50.0)

        toolkit = Toolkit(task=task, history=history)
        toolkit.llm = BackendRegistry(default=MockBackend([_code_block(60.0)]))
        Improve()(toolkit)

        attempt = history.list()[-1]
        assert attempt.parent == prior.id
        assert attempt.metadata["strategy"] == Improve.name
        # The standard finish, not the optimizer, cached the score note.
        assert history.get_note(attempt.id, "score") == "0.6000"


def test_fresh_approach_attribution_matches_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        toolkit = Toolkit(task=task, history=history)
        # First response generates the code, second the fresh direction.
        toolkit.llm = BackendRegistry(default=MockBackend([
            _code_block(40.0), "novel direction",
        ]))
        FreshApproach()(toolkit)

        attempt = history.list()[-1]
        assert attempt.status == "done"
        assert attempt.metadata["strategy"] == FreshApproach.name
        assert history.get_note(attempt.id, "score") == "0.4000"


def test_cross_pollinate_attribution_matches_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        best = _commit_root(history, task, 70.0, direction="rollout")
        other = _commit_root(history, task, 30.0, direction="mcts")

        toolkit = Toolkit(task=task, history=history)
        toolkit.llm = BackendRegistry(default=MockBackend([_code_block(80.0)]))
        CrossPollinate()(toolkit)

        attempt = history.list()[-1]
        assert attempt.parent == best.id
        assert attempt.metadata["strategy"] == CrossPollinate.name
        assert attempt.metadata["inspiration"] == other.id
        assert history.get_note(attempt.id, "score") == "0.8000"


def test_analyse_attribution_matches_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        toolkit = Toolkit(task=task, history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(["compressed"]))
        from groundhog import MarkdownLearnings
        toolkit.learnings = MarkdownLearnings(Path(tmp))
        toolkit.learnings.add("something learned")

        out = Analyse()(toolkit)
        assert out["strategy"] == Analyse.name
