"""Optimizer console totals account every commit and attempt-less spend.

Integration finding: the serial loop counted only each call's LAST
commit (an ABTest pair showed one arm; a real run reported $0.18 while
$2.29 was spent), and Analyse's LLM cost attached to no attempt at all.
"""

from pathlib import Path
from types import SimpleNamespace

from groundhog import SimpleOptimizer, Strategy, Task
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    EvaluationResult,
    Evaluator,
    StageResult,
)
from groundhog.histories.folder import FolderAttemptHistory
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
            EvalStage("eval", "eval", lambda cp: StageResult(),
                      scorer=lambda r: r.metrics.get("score", 0.0))
        ]


class _Log:
    def info(self, text):
        pass

    def end(self):
        pass


def _toolkit(tmp_path):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    return SimpleNamespace(task=task,
                           history=FolderAttemptHistory(tmp_path / "store"),
                           log=_Log(), path=tmp_path)


class TwoCommits(Strategy):
    """ABTest-shaped: two commits per call, each carrying its own cost."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __call__(self, toolkit, config=None):
        for cost in (0.10, 0.25):
            self.n += 1
            ws = toolkit.history.workspace()
            (ws.path / "solution.py").write_text(f"# {self.n}",
                                                 encoding="utf-8")
            write_direction(ws.path, f"direction {self.n}")
            result = EvaluationResult(
                stages={"eval": StageResult(metrics={"score": 0.5})})
            finalize_attempt(toolkit, ws, result, None, metadata={
                "strategy": self.name, "prior": None, "cost": cost})
        return {"attempt": "many"}


class AttemptlessSpend(Strategy):
    """Analyse-shaped: commits nothing, reports its LLM cost."""

    def __call__(self, toolkit, config=None):
        return {"strategy": self.name, "cost": 0.5}


def test_serial_total_accounts_every_commit(tmp_path, capsys):
    tk = _toolkit(tmp_path)
    opt = SimpleOptimizer(tk, strategy=TwoCommits(), seed_strategy=None)
    opt.run(n=1)

    out = capsys.readouterr().out
    assert "Total cost: $0.3500" in out


def test_serial_total_includes_attemptless_cost(tmp_path, capsys):
    tk = _toolkit(tmp_path)
    opt = SimpleOptimizer(tk, strategy=AttemptlessSpend(), seed_strategy=None)
    opt.run(n=2)

    out = capsys.readouterr().out
    assert "cost $0.5000" in out
    assert "Total cost: $1.0000" in out


def test_parallel_total_includes_attemptless_cost(tmp_path, capsys):
    from groundhog import assemble_toolkit

    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    tk = assemble_toolkit(task, history=FolderAttemptHistory(tmp_path / "s"),
                          path=tmp_path, seed=42)
    opt = SimpleOptimizer(tk, strategy=AttemptlessSpend(), seed_strategy=None,
                          concurrency=2)
    opt.run(n=3)

    out = capsys.readouterr().out
    assert "Total cost: $1.5000" in out


def test_analyse_logs_its_cost_line(tmp_path):
    from groundhog import Analyse, MarkdownLearnings, Toolkit

    class _FakeLLM:
        def get(self, tier):
            return self

        def generate(self, prompt, system_prompt=""):
            return SimpleNamespace(text="narrative", cost=0.02, usage={},
                                   model="fake")

    class _Recorder:
        def __init__(self):
            self.lines = []

        def start(self, text):
            self.lines.append(text)

        def inline(self, text):
            pass

        def tock(self):
            pass

        def info(self, text):
            self.lines.append(text)

        def end(self):
            pass

    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    log = _Recorder()
    toolkit = Toolkit(task=task,
                      history=FolderAttemptHistory(tmp_path / "store"),
                      path=tmp_path, log=log)
    toolkit.llm = _FakeLLM()
    toolkit.learnings = MarkdownLearnings(tmp_path)
    toolkit.learnings.add("something learned")

    out = Analyse()(toolkit)
    assert out["cost"] > 0
    assert any("analyse cost" in line for line in log.lines)
