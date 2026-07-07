"""Stress tests for SimpleOptimizer(concurrency=N) — the opt-in parallel mode.

Hammers the racy sections (prior selection + workspace allocation,
commit/finalize, score notes) with 4 workers and asserts the invariants the
lock exists to protect: no duplicate attempt ids, no lost attempts, every
attempt committed exactly once. Parametrized over both history backends via
the ``history_factory`` fixture (folder gets the full 30-attempt run; git is
subprocess-bound, so it runs a shorter one).
"""

import random
import threading
import time
import uuid

from groundhog import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult,
    SimpleOptimizer, assemble_toolkit,
)
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.direction import write_direction


class StressData(Data):
    def get_train(self):
        return {"target": 50.0}

    def get_test(self):
        return {"target": 50.0}


class StressContext(Context):
    def get_brief(self):
        return "Return a number close to the target."

    def get_extended(self):
        return "def solve() -> float"


class StressEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        return self.get_stages(data)[0].call(code_or_path)

    @staticmethod
    def _scorer(result):
        if result.errors:
            return -1.0
        return max(0.0, 1.0 - result.metrics.get("distance", 100.0) / 100.0)

    def get_stages(self, data):
        def run(code_or_path, d=data):
            code = (code_or_path / "solution.py").read_text(encoding="utf-8")
            namespace = {}
            exec(code, namespace)
            value = namespace["solve"]()
            distance = abs(value - d.get_test()["target"])
            return StageResult(metrics={"distance": distance, "value": value})

        return [EvalStage("evaluate", "Full evaluation", run, scorer=self._scorer)]


class StressTask(Task):
    def __init__(self):
        super().__init__(data=StressData(), context=StressContext(),
                         evaluator=StressEvaluator(), name="StressTask")


class MockStrategy:
    """Full strategy loop, thread-safe: prior -> workspace -> work -> finalize.

    Tracks how many calls run at once so the test can prove the run actually
    overlapped instead of degenerating to serial. The gauge is a shared dict
    (the optimizer shallow-copies the strategy per parallel dispatch, so a
    plain instance attribute would track each copy separately)."""

    name = "mock"

    def __init__(self):
        self.gauge = {"lock": threading.Lock(), "active": 0, "max": 0}

    def __call__(self, toolkit, config=None):
        with self.gauge["lock"]:
            self.gauge["active"] += 1
            self.gauge["max"] = max(self.gauge["max"], self.gauge["active"])
        try:
            prior = toolkit.get_prior(toolkit)
            ws = toolkit.history.workspace(parent=prior.id if prior else None)
            value = random.uniform(0.0, 100.0)
            (ws.path / "solution.py").write_text(
                f"def solve():\n    return {value!r}\n", encoding="utf-8")
            if prior is None:
                write_direction(ws.path, f"direction {uuid.uuid4().hex}")
            time.sleep(random.uniform(0.005, 0.015))
            result = toolkit.task.evaluate(ws.path)
            attempt = toolkit.finalize(ws, result, prior=prior, strategy="mock")
            return {"attempt": attempt.id}
        finally:
            with self.gauge["lock"]:
                self.gauge["active"] -= 1


def test_concurrency_defaults_to_serial(tmp_path):
    tk = assemble_toolkit(StressTask(), history=FolderAttemptHistory(tmp_path),
                          path=tmp_path, seed=42)
    opt = SimpleOptimizer(tk, strategy=MockStrategy(), seed_strategy=None)
    assert opt.concurrency == 1


def test_parallel_stress_no_lost_or_duplicate_attempts(history_factory, tmp_path):
    history = history_factory()
    is_folder = isinstance(history, FolderAttemptHistory)
    n = 30 if is_folder else 10

    strategy = MockStrategy()
    tk = assemble_toolkit(StressTask(), history=history, path=tmp_path, seed=42)
    opt = SimpleOptimizer(tk, strategy=strategy, seed_strategy=None, concurrency=4)
    opt.run(n=n)

    everything = history.list(only_done=False)
    ids = [a.id for a in everything]
    assert len(ids) == n, f"lost attempts: expected {n}, got {len(ids)}"
    assert len(set(ids)) == n, "duplicate attempt ids"

    done = history.list()
    assert len(done) == n, "every attempt should commit exactly once as done"
    assert history.list_in_progress() == []

    for a in done:
        assert a.result.completed
        if a.parent is not None:
            assert history.get(a.parent) is not None, \
                f"attempt {a.id} points at unknown parent {a.parent}"

    if is_folder:
        # Numbering is dense — the scan-and-claim allocation never skipped
        # or reused a number — and no claim sentinels leaked.
        assert sorted(int(i) for i in ids) == list(range(1, n + 1))
        assert not any(p.name.startswith(".claim_")
                       for p in history.base_path.iterdir())

    assert strategy.gauge["max"] >= 2, "run never actually overlapped"
