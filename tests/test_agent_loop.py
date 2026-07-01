"""AgentStrategy driven end-to-end by a scripted fake backend.

The agent phase loop (the largest module in src) previously never executed in
the suite (audit 2026-07-01 gap #5). These tests drive real attempts through
AgentStrategy.__call__ with a backend that "acts" by writing files into the
workspace — no HTTP, no LLM, fully deterministic.
"""

from pathlib import Path

from groundhog import Task, assemble_toolkit
from groundhog.base.agent import AgentBackend, AgentRegistry, AgentResult
from groundhog.base.types import Data, Context, Evaluator, EvalStage, StageResult
from groundhog.strategies.agent import AgentStrategy


GOOD = "def solve():\n    return 50.0\n"
BAD = "def solve():\n    raise RuntimeError('broken')\n"


class _Data(Data):
    def get_train(self): return {"target": 50.0}
    def get_test(self): return {"target": 50.0}


class _Ctx(Context):
    def get_brief(self): return "Write solve() returning a float near 50."
    def get_extended(self): return "def solve() -> float"


def _read(code_or_path):
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        ns = {}
        try:
            exec(_read(code_or_path), ns)
            value = float(ns["solve"]())
        except Exception as e:  # noqa: BLE001 — the task contract: errors -> StageResult
            return StageResult(errors={"crash": str(e)})
        target = data.get_test()["target"]
        return StageResult(metrics={"value": value, "target": target})

    def get_stages(self, data):
        return [EvalStage("evaluate", "full",
                          lambda cp, d=data: self.evaluate(cp, d),
                          scorer=lambda r: -1.0 if r.errors else 1.0)]


class ScriptedAgent(AgentBackend):
    """Each run() pops the next scripted solution and writes it where the
    agent is told to work (work/solution.py under spec.workspace_path)."""
    cost_model = "per_request"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def run(self, spec):
        self.calls.append(spec.goal[:60])
        solution = self.script.pop(0) if self.script else GOOD
        ws = Path(spec.workspace_path)
        work = ws / "work"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solution.py").write_text(solution, encoding="utf-8")
        # Real agents are prompted to record the attempt's core direction;
        # the finalize gate fails a fresh attempt without one.
        (ws / "core_direction.md").write_text("scripted: return the target\n",
                                              encoding="utf-8")
        return AgentResult(success=True, output="done", cost=0.01, session_id="s1")


def _toolkit(history, backend):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="tiny")
    tk = assemble_toolkit(task, history=history)
    tk.agent = AgentRegistry(default=backend)
    return tk


def test_full_attempt_happy_path(history_factory):
    history = history_factory()
    backend = ScriptedAgent([GOOD])
    tk = _toolkit(history, backend)

    log = AgentStrategy()(tk)

    assert not log.get("skipped"), f"strategy skipped: {log}"
    attempts = history.list()
    assert len(attempts) == 1
    a = attempts[0]
    assert a.result.completed, f"attempt failed: {a.result.failed_stage}"
    assert "return 50.0" in a.code
    assert len(backend.calls) == 1
    # The attempt pointer was bracketed and cleared.
    assert tk.ws.is_set() is False


def test_fix_loop_recovers_from_failing_solution(history_factory):
    """First scripted solution crashes at eval; the fix-phase agent call
    writes a good one — the attempt must complete, with 2 backend calls."""
    history = history_factory()
    backend = ScriptedAgent([BAD, GOOD])
    tk = _toolkit(history, backend)

    log = AgentStrategy()(tk)

    assert not log.get("skipped"), f"strategy skipped: {log}"
    attempts = history.list()
    assert len(attempts) == 1
    a = attempts[0]
    assert a.result.completed, "fix loop did not recover"
    assert "return 50.0" in a.code
    assert len(backend.calls) == 2, f"expected explore+fix, got {backend.calls}"
    assert tk.ws.is_set() is False


def test_pointer_is_live_during_the_attempt(history_factory):
    """toolkit.ws points at the in-flight workspace WHILE the agent runs —
    the whole reason build-time tools can read the current attempt."""
    history = history_factory()
    seen = {}

    class PeekingAgent(ScriptedAgent):
        def __init__(self, tk_ref):
            super().__init__([GOOD])
            self._tk = tk_ref

        def run(self, spec):
            seen["is_set"] = self._tk["tk"].ws.is_set()
            seen["path_matches"] = str(self._tk["tk"].ws.path) == str(spec.workspace_path)
            return super().run(spec)

    ref = {}
    backend = PeekingAgent(ref)
    tk = _toolkit(history, backend)
    ref["tk"] = tk

    AgentStrategy()(tk)

    assert seen == {"is_set": True, "path_matches": True}
