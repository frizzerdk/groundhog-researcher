"""Tests that verify the core concepts are correctly implemented.

Each test section maps to a concept from the vault:
    GroundhogResearcher/Optimizer/

Concept → Test mapping:
    StageResult (base.py)
        - Flexible metrics (Dict[str, Any]), dict errors/warnings
    EvaluationResult (base.py)
        - Collects per-stage results, cascade stops on error
    EvalStage + Scorer (base.py, Scorer.md)
        - Per-stage scorers, scoring from metrics not stored scores
    Attempt History (Attempt History.md)
        - No scores persisted, immutable, atomic, complete (keeps failures)
        - Tree structure with parent pointers, lineage traversal
        - best() takes a scorer — reinterpretable
    Workspace (Strategy — Workspace.md)
        - Provides isolated path, commit-or-abort lifecycle
    Toolkit (Toolkit.md)
        - Dynamic attributes, override tracking, missing = AttributeError
    Strategy (Strategy.md, Strategy — Role)
        - Owns the full loop, return value is debug-only
    Optimizer (Optimizer.md)
        - Deterministic with seed, doesn't depend on strategy return
"""

import json
import os
import random
import shutil
import time
import tempfile
from pathlib import Path

from groundhog import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult, EvaluationResult,
    Toolkit, SimpleOptimizer, FolderAttemptHistory,
)
from groundhog.utils.results import write_result


# === Test fixtures ===

class FixtureData(Data):
    def get_train(self):
        return {"target": 50, "range": (0, 100)}
    def get_test(self):
        return {"target": 50.0, "range": (0, 100)}


class FixtureContext(Context):
    def get_brief(self):
        return "Return a number close to the target."
    def get_extended(self):
        return "def solve() -> float"


def _read_code(code_or_path):
    """Helper: accept string or Path, return code string."""
    from pathlib import Path
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text()


class FixtureEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read_code(code_or_path)
        namespace = {}
        exec(code, namespace)
        value = namespace["solve"]()
        target = data.get_test()["target"]
        distance = abs(value - target)
        return StageResult(
            score=max(0.0, 1.0 - distance / 100.0),
            metrics={"distance": distance, "value": value, "target": target},
        )

    @staticmethod
    def _scorer(result):
        if result.errors:
            return -1.0
        return max(0.0, 1.0 - result.metrics.get("distance", 100) / 100.0)

    @staticmethod
    def _smoke_scorer(result):
        return -1.0 if result.errors else 1.0

    def get_stages(self, data):
        return [
            EvalStage("smoke", "Syntax check",
                      lambda code_or_path: self._smoke(code_or_path),
                      scorer=self._smoke_scorer),
            EvalStage("evaluate", "Full evaluation",
                      lambda code_or_path, d=data: self.evaluate(code_or_path, d),
                      scorer=self._scorer),
        ]

    def _smoke(self, code_or_path):
        code = _read_code(code_or_path)
        try:
            namespace = {}
            exec(code, namespace)
            if "solve" not in namespace:
                return StageResult(errors={"missing": "No solve()"})
            namespace["solve"]()
            return StageResult(score=1.0)
        except Exception as e:
            return StageResult(errors={"syntax": str(e)})


class FixtureTask(Task):
    def __init__(self):
        super().__init__(
            data=FixtureData(),
            context=FixtureContext(),
            evaluator=FixtureEvaluator(),
            name="FixtureTask",
        )


def make_code(value):
    return f"def solve():\n    return {value}"


# === StageResult (base.py) ===
# Vault: flexible metrics, dict errors/warnings, no required fields

def test_stage_result_has_no_required_fields():
    """StageResult should work with all defaults."""
    r = StageResult()
    assert r.score == 0.0
    assert r.metrics == {}
    assert r.artifacts == {}
    assert r.errors == {}
    assert r.warnings == {}

def test_stage_result_metrics_are_flexible():
    """Metrics can hold any type — scalars, lists, dicts."""
    r = StageResult(metrics={
        "accuracy": 0.85,
        "timeseries": [0.1, 0.5, 0.8, 0.85],
        "config": {"lr": 0.01},
    })
    assert isinstance(r.metrics["timeseries"], list)
    assert isinstance(r.metrics["config"], dict)

def test_stage_result_errors_are_dict():
    """Errors are a dict — keyed by type, consumer decides how to handle."""
    r = StageResult(errors={"syntax": "invalid syntax line 3", "traceback": "..."})
    assert "syntax" in r.errors
    assert bool(r.errors) is True

def test_stage_result_no_errors_is_falsy():
    """Empty errors dict is falsy — easy to check `if result.errors`."""
    r = StageResult()
    assert not r.errors


# === EvaluationResult (base.py) ===
# Vault: Staged Evaluation.md — cascade through stages, stop on error

def test_evaluation_result_collects_stages():
    """EvaluationResult is a dict of stage name → StageResult."""
    task = FixtureTask()
    result = task.evaluate(make_code(50.0))
    assert "smoke" in result.stages
    assert "evaluate" in result.stages
    assert result.completed is True

def test_evaluation_stops_on_error():
    """Cascade stops at first stage with errors."""
    task = FixtureTask()
    result = task.evaluate("not valid python")
    assert result.completed is False
    assert result.failed_stage == "smoke"
    assert "evaluate" not in result.stages

def test_evaluation_through_limits_stages():
    """The 'through' parameter limits which stages run."""
    task = FixtureTask()
    result = task.evaluate(make_code(50.0), through="smoke")
    assert "smoke" in result.stages
    assert "evaluate" not in result.stages


# === EvalStage + Scorer (base.py) ===
# Vault: Scorer.md — per-stage scorers, scoring from metrics, reinterpretable

def test_each_stage_has_its_own_scorer():
    """Scorers are per-stage, not global."""
    task = FixtureTask()
    stages = task.evaluator.get_stages(task.data)
    smoke_stage = stages[0]
    eval_stage = stages[1]
    assert smoke_stage.scorer is not eval_stage.scorer

def test_scorer_reinterprets_from_metrics():
    """Scorer works from metrics, not from stored score."""
    result = StageResult(metrics={"distance": 10.0})
    score = FixtureEvaluator._scorer(result)
    assert score == 0.9

def test_default_scorer_returns_result_score():
    """Default scorer passes through result.score."""
    stage = EvalStage("test", "test", lambda code: StageResult(score=0.42))
    result = StageResult(score=0.42)
    assert stage.score(result) == 0.42


# === Attempt History (attempt_history.py) ===
# Vault: Attempt History.md — immutable, atomic, complete, no scores stored
# Vault: Attempt History — Derived Views.md — tree structure, lineage, best via scorer

def test_attempt_history_stores_no_scores():
    """Attempts on disk must not contain score fields — only metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = FixtureTask()

        ws = history.workspace()
        (ws.path / "solution.py").write_text(make_code(50.0))
        result = task.evaluate(make_code(50.0))
        write_result(ws.path, result)

        attempt = ws.commit(success=result.completed)

        # Read raw JSON — no "score" key at stage level
        raw = json.loads((attempt.path / "result.json").read_text())
        for stage_name, stage_data in raw["stages"].items():
            assert "score" not in stage_data, f"Stage '{stage_name}' has a score field — should only have metrics"

def test_attempt_history_is_immutable():
    """Once committed, an attempt's files should not change."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        ws = history.workspace()
        (ws.path / "solution.py").write_text(make_code(50.0))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})
        write_result(ws.path, result)

        attempt = ws.commit(success=result.completed)

        code_before = attempt.code
        result_before = attempt.result

        # Read again — should be identical
        attempt2 = history.get(attempt.number)
        assert attempt2.code == code_before

def test_attempt_history_tree_structure():
    """Each attempt has at most one parent, forming a tree."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})

        ws1 = history.workspace(parent=None)
        (ws1.path / "solution.py").write_text("v1")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.number)
        (ws2.path / "solution.py").write_text("v2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a1.number)
        (ws3.path / "solution.py").write_text("v3")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        assert a1.parent is None
        assert a2.parent == a1.number
        assert a3.parent == a1.number

def test_attempt_history_lineage():
    """Lineage walks from attempt back to root."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})

        ws1 = history.workspace()
        (ws1.path / "solution.py").write_text("v1")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.number)
        (ws2.path / "solution.py").write_text("v2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a2.number)
        (ws3.path / "solution.py").write_text("v3")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        chain = history.lineage(a3)
        assert [a.number for a in chain] == [1, 2, 3]

def test_attempt_history_best_uses_scorer():
    """best() takes a scorer function — different scorers give different bests."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        for val in [10.0, 90.0, 50.0]:
            ws = history.workspace()
            (ws.path / "solution.py").write_text(make_code(val))
            result = EvaluationResult(stages={
                "eval": StageResult(metrics={"value": val, "distance": abs(val - 50)})
            })
            write_result(ws.path, result)

            ws.commit(success=result.completed)

        # Scorer: closest to 50
        best_close = history.best(lambda r: -r.metrics.get("distance", 100))
        assert best_close.result.stages["eval"].metrics["value"] == 50.0

        # Scorer: highest value
        best_high = history.best(lambda r: r.metrics.get("value", 0))
        assert best_high.result.stages["eval"].metrics["value"] == 90.0

def test_attempt_history_records_failed_attempts():
    """Failed attempts are stored too — nothing discarded."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        ws = history.workspace()
        (ws.path / "solution.py").write_text("bad code")
        result = EvaluationResult(
            stages={"smoke": StageResult(errors={"syntax": "invalid"})},
            completed=False,
            failed_stage="smoke",
        )
        write_result(ws.path, result)

        ws.commit(success=result.completed)

        assert len(history.list(only_done=False)) == 1
        assert history.list(only_done=False)[0].result.completed is False


# === Workspace (attempt_history.py) ===
# Vault: Strategy — Workspace.md — isolated path, transactional commit/abort

def test_workspace_provides_path():
    """Workspace gives a path to write files to."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        ws = history.workspace()
        assert ws.path.exists()
        assert ws.path.is_dir()

def test_workspace_commit_finalizes():
    """After commit, workspace becomes a listed attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        assert len(history.list()) == 0

        ws = history.workspace()
        (ws.path / "solution.py").write_text(make_code(42))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})
        write_result(ws.path, result)

        ws.commit(success=result.completed)

        assert len(history.list()) == 1

def test_workspace_abort_leaves_no_trace():
    """After abort, nothing is listed in history."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        ws = history.workspace()
        (ws.path / "solution.py").write_text("something")
        ws.abort()

        assert len(history.list()) == 0
        assert not ws.path.exists()


def test_workspace_allocation_is_atomic_under_concurrency():
    """Concurrent allocators against the same attempts dir produce unique numbers.

    The bug being guarded against: pre-0.2.11, FolderAttemptHistory cached
    `_count` at construction and incremented it in-memory. Two histories
    pointing at the same directory would each return number 1, then 2, then
    3 — a parallel optimizer setup silently produced duplicate attempt IDs.

    We simulate parallel optimizers by spawning multiple threads, each with
    its own history instance backed by the same directory. The fix must
    handshake through the disk (scan + sentinel mkdir) to stay consistent.
    """
    import threading

    with tempfile.TemporaryDirectory() as tmp:
        results: list[list[int]] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def alloc(count: int):
            try:
                h = FolderAttemptHistory(Path(tmp))  # one per "optimizer"
                barrier.wait(timeout=10)             # release them together
                nums = [h.workspace().number for _ in range(count)]
                results.append(nums)
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=alloc, args=(5,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"allocator raised: {errors}"
        all_nums = [n for batch in results for n in batch]
        assert len(all_nums) == 20, f"expected 20 allocations, got {len(all_nums)}"
        assert len(set(all_nums)) == 20, f"duplicate numbers: {sorted(all_nums)}"
        assert sorted(all_nums) == list(range(1, 21)), \
            f"expected dense 1..20, got {sorted(all_nums)}"


def test_workspace_stale_claim_is_reclaimed():
    """A leftover claim sentinel doesn't permanently lock its number."""
    import os

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        # Plant a stale claim sentinel and age it past the TTL.
        from groundhog.histories import folder as folder_mod
        stale = history.base_path / ".claim_001"
        stale.mkdir()
        old = time.time() - (folder_mod._CLAIM_TTL_SECONDS + 60)
        os.utime(stale, (old, old))

        # First allocation should reclaim 001 (cleaning up the stale sentinel).
        ws = history.workspace()
        assert ws.number == 1
        assert not stale.exists()


# === Toolkit (toolkit.py) ===
# Vault: Toolkit.md — dynamic attributes, override tracking, strategy handles missing

def test_toolkit_attributes_are_dynamic():
    """Toolkit accepts any attributes."""
    tk = Toolkit(foo="bar", num=42)
    assert tk.foo == "bar"
    assert tk.num == 42

def test_toolkit_override_is_tracked(capsys):
    """Overriding a toolkit attribute prints a message."""
    tk = Toolkit(rng=random.Random(1))
    tk.rng = random.Random(2)
    captured = capsys.readouterr()
    assert "overriding" in captured.out
    assert "rng" in captured.out

def test_toolkit_missing_attribute_raises():
    """Accessing missing attribute raises AttributeError — strategy handles it."""
    tk = Toolkit(task="something")
    try:
        _ = tk.nonexistent
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# === Strategy (strategy.py) ===
# Vault: Strategy.md — owns full loop, return is debug only
# Vault: Strategy — Role — business logic layer, unconstrained

def test_strategy_return_is_not_depended_on():
    """Optimizer doesn't depend on strategy return value for correctness."""
    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))

        class ReturnsNothing:
            def __call__(self, toolkit, config=None):
                ws = toolkit.history.workspace()
                (ws.path / "solution.py").write_text(make_code(50.0))
                result = toolkit.task.evaluate(make_code(50.0))
                write_result(ws.path, result)

                ws.commit(success=result.completed)
                return {}  # empty dict — optimizer should still work

        optimizer = SimpleOptimizer(task, strategy=ReturnsNothing(), seed=42, history=history, seed_strategy=None)
        optimizer.run(n=3)
        assert len(history.list()) == 3

def test_strategy_owns_evaluation_and_recording():
    """Strategy evaluates and records — optimizer doesn't."""
    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))
        recorded = []

        class TrackingStrategy:
            def __call__(self, toolkit, config=None):
                ws = toolkit.history.workspace()
                code = make_code(42.0)
                (ws.path / "solution.py").write_text(code)
                result = toolkit.task.evaluate(code)
                write_result(ws.path, result)

                attempt = ws.commit(success=result.completed)
                recorded.append(attempt.number)
                return {"attempt": attempt.number}

        optimizer = SimpleOptimizer(task, strategy=TrackingStrategy(), seed=42, history=history, seed_strategy=None)
        optimizer.run(n=5)

        # Strategy recorded all 5
        assert len(recorded) == 5
        assert len(history.list()) == 5


# === Optimizer (optimizer.py) ===
# Vault: Optimizer.md — deterministic with seed, doesn't own eval or recording

def test_optimizer_is_deterministic_with_seed():
    """Same seed → same results."""
    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            task = FixtureTask()
            history = FolderAttemptHistory(Path(tmp))
            rng_values = []

            class DeterministicStrategy:
                def __call__(self, toolkit, config=None):
                    val = toolkit.rng.uniform(0, 100)
                    rng_values.append(val)
                    ws = toolkit.history.workspace()
                    code = make_code(val)
                    (ws.path / "solution.py").write_text(code)
                    result = toolkit.task.evaluate(code)
                    write_result(ws.path, result)

                    ws.commit(success=result.completed)
                    return {"value": val}

            optimizer = SimpleOptimizer(task, strategy=DeterministicStrategy(), seed=42, history=history, seed_strategy=None)
            optimizer.run(n=5)
            results.append(rng_values[:])

    assert results[0] == results[1]


def test_optimizer_extras_registers_strategy_for_queue():
    """extras=[...] makes a strategy reachable via the queue without rotating it."""
    from groundhog.tools.queue import add as queue_add

    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))
        called = {"rotation": 0, "extra": 0}

        class RotationStrat:
            def __call__(self, toolkit, config=None):
                called["rotation"] += 1
                ws = toolkit.history.workspace()
                (ws.path / "solution.py").write_text(make_code(1.0))
                result = toolkit.task.evaluate(make_code(1.0))
                write_result(ws.path, result)
                ws.commit(success=result.completed)

        class ExtraStrat:
            def __call__(self, toolkit, config=None):
                called["extra"] += 1
                ws = toolkit.history.workspace()
                (ws.path / "solution.py").write_text(make_code(2.0))
                result = toolkit.task.evaluate(make_code(2.0))
                write_result(ws.path, result)
                ws.commit(success=result.completed)

        opt = SimpleOptimizer(
            task, strategy=RotationStrat(), extras=[ExtraStrat()],
            seed=42, path=Path(tmp), history=history, seed_strategy=None,
        )

        # Queue the extra by name; first iteration should pick it up.
        queue_add(Path(tmp), "extra_strat")
        opt.run(n=2)

        assert called["extra"] == 1, "extra was not invoked from queue"
        assert called["rotation"] == 1, "rotation should still cover the second iteration"


def test_optimizer_extras_does_not_overwrite_rotation():
    """If an extras name collides with a rotation strategy, rotation wins."""
    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))

        class Improve:
            def __call__(self, toolkit, config=None):
                pass

        rotation = Improve()
        # Same class name, different instance — should not displace rotation's slot.
        extra = Improve()

        opt = SimpleOptimizer(
            task, strategy=rotation, extras=[extra],
            seed=42, path=Path(tmp), history=history, seed_strategy=None,
        )

        # Rotation registered "improve" first; extras must not clobber it.
        assert opt._strategy_registry["improve"] is rotation


def test_fresh_agent_strategy_returns_no_prior():
    """FreshAgentStrategy._select_prior always returns None regardless of history."""
    from groundhog import FreshAgentStrategy

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})

        # Seed the history so a real selector would pick *something*.
        ws = history.workspace()
        (ws.path / "solution.py").write_text("x = 1")
        write_result(ws.path, result)
        ws.commit(success=True)

        # Mock toolkit: anything _select_prior touches is fine to leave undefined,
        # since the override returns None before consulting any selector.
        toolkit = Toolkit(history=history)
        prior = FreshAgentStrategy()._select_prior(toolkit)
        assert prior is None


def test_agent_config_tier_default_and_override():
    """AgentConfig.tier defaults to 'default'; explicit value flows through."""
    from groundhog import AgentStrategy
    s_default = AgentStrategy()
    assert s_default.config.tier == "default"

    s_budget = AgentStrategy(tier="budget")
    assert s_budget.config.tier == "budget"


def test_compacted_learnings_manual_distill_mode():
    """No compactor → entries queue until distill() is called manually."""
    from groundhog import Compacted, MarkdownLearnings

    with tempfile.TemporaryDirectory() as tmp:
        inner = MarkdownLearnings(Path(tmp))
        c = Compacted(
            inner,
            current_path=Path(tmp) / "current.md",
        )
        c.add("first observation")
        c.add("second observation")

        # Inner store has both.
        assert "first observation" in inner.get()
        assert "second observation" in inner.get()
        # Queue holds both, current view is empty.
        assert len(c.queued()) == 2
        assert not (Path(tmp) / "current.md").exists() or \
            not (Path(tmp) / "current.md").read_text().strip()

        # get() falls back to inner when no current view exists.
        assert "first observation" in c.get()

        # Manually distill via a deterministic compactor.
        def compactor(current, queue):
            return "DIGEST: " + " | ".join(queue)

        ok = c.distill(compactor)
        assert ok
        assert (Path(tmp) / "current.md").read_text().strip().startswith("DIGEST:")
        assert c.queued() == []
        # get() now returns the compacted view.
        assert c.get().startswith("DIGEST:")


def test_compacted_learnings_auto_compaction_on_add():
    """compactor passed to ctor → every add() distills."""
    from groundhog import Compacted, MarkdownLearnings

    calls = []

    def compactor(current, queue):
        calls.append((current, list(queue)))
        return f"v{len(calls)}: " + " | ".join(queue)

    with tempfile.TemporaryDirectory() as tmp:
        c = Compacted(
            MarkdownLearnings(Path(tmp)),
            current_path=Path(tmp) / "current.md",
            compactor=compactor,
        )
        c.add("first")
        c.add("second")

        # Compactor invoked once per add.
        assert len(calls) == 2
        # Queue cleared on success.
        assert c.queued() == []
        # Current reflects the most recent compaction.
        assert c.get().startswith("v2:")


def test_promote_best_snapshots_on_improvement():
    """promote_best wraps a stage so its eval tool snapshots on score gain."""
    from groundhog.agents.tools import promote_best
    from groundhog import EvalStage, StageResult

    with tempfile.TemporaryDirectory() as tmp:
        # Stage that returns a score equal to int(code).
        def call(code):
            return StageResult(metrics={"score": float(int(code.strip()))})

        stage = EvalStage(
            name="evaluate", description="dummy",
            call=call, scorer=lambda r: r.metrics["score"],
        )
        ws = Path(tmp)
        (ws / "work").mkdir()
        dest = ws / "solution.py"

        tool = promote_best(stage, dest_path=dest)

        # First eval: score 5. Should promote.
        src = ws / "work" / "solution.py"
        src.write_text("5")
        tool.execute(path=str(src))
        assert dest.exists()
        assert dest.read_text() == "5"

        # Lower score: should NOT promote.
        src.write_text("3")
        tool.execute(path=str(src))
        assert dest.read_text() == "5"

        # Higher score: should promote.
        src.write_text("9")
        tool.execute(path=str(src))
        assert dest.read_text() == "9"


def test_build_eval_tools_uses_promote_dest():
    """promote_dest argument wraps the final stage; cheaper stages stay plain."""
    from groundhog.agents.tools import build_eval_tools
    from groundhog import Task, EvalStage, StageResult

    class _Eval(Evaluator):
        def get_stages(self, data):
            return [
                EvalStage(name="smoke", description="smoke",
                          call=lambda c: StageResult(metrics={"score": 0.0}),
                          scorer=lambda r: r.metrics["score"]),
                EvalStage(name="evaluate", description="evaluate",
                          call=lambda c: StageResult(metrics={"score": 1.0}),
                          scorer=lambda r: r.metrics["score"]),
            ]
        def evaluate(self, code_or_path, data):
            return StageResult(metrics={"score": 1.0})

    with tempfile.TemporaryDirectory() as tmp:
        toolkit = Toolkit(task=Task(
            data=FixtureData(), context=FixtureContext(),
            evaluator=_Eval(), name="t",
        ))
        tools = build_eval_tools(
            toolkit, ws_path=Path(tmp), promote_dest=Path(tmp) / "solution.py"
        )
        # Both stages produce tools; the final one's description mentions
        # the snapshot — that's how we tell promote_best wrapped it.
        assert len(tools) == 2
        assert "Snapshots" not in tools[0].description, "smoke should not promote"
        assert "Snapshots" in tools[1].description, "evaluate should promote"


def test_build_prior_tools_three_tool_progressive_disclosure():
    """get-priors lists, list-prior shows files, get-prior-file reads."""
    from groundhog.agents.tools import build_prior_tools

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 1.0})})

        # Lineage: 1 -> 2 -> 3
        ws1 = history.workspace()
        (ws1.path / "solution.py").write_text("v1")
        (ws1.path / "work" / "learnings.md").write_text("note from 1")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.number)
        (ws2.path / "solution.py").write_text("v2")
        (ws2.path / "work" / "learnings.md").write_text("note from 2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a2.number)
        (ws3.path / "solution.py").write_text("v3")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        # build_prior_tools called with prior=a3 (the immediate parent of a
        # hypothetical fourth attempt).
        tools = build_prior_tools(
            a3, history=history,
            scorer=lambda r: r.metrics.get("score", 0.0),
        )
        by_name = {t.name: t for t in tools}

        # get-priors lists the chain in distance order.
        listing = by_name["get-priors"].execute(n=10).output
        # Distance 1 = a3 (the prior itself), 2 = a2, 3 = a1.
        assert "attempt_3" in listing and "distance=1" in listing
        assert "attempt_2" in listing and "distance=2" in listing
        assert "attempt_1" in listing and "distance=3" in listing

        # list-prior with default 'parent' lists a3's files.
        files = by_name["list-prior"].execute().output
        assert "solution.py" in files

        # get-prior-file by id.
        contents = by_name["get-prior-file"].execute(
            attempt="2", file="work/learnings.md"
        ).output
        assert contents == "note from 2"

        # Default 'parent' resolves to a3.
        contents_parent = by_name["get-prior-file"].execute(
            attempt="parent", file="solution.py"
        ).output
        assert contents_parent == "v3"


def test_compacted_learnings_retains_queue_on_failure():
    """Compactor exception → queue is preserved for the next add()."""
    from groundhog import Compacted, MarkdownLearnings

    fail_calls = [0]

    def flaky(current, queue):
        fail_calls[0] += 1
        if fail_calls[0] == 1:
            raise RuntimeError("backend transient error")
        return "OK: " + " | ".join(queue)

    with tempfile.TemporaryDirectory() as tmp:
        c = Compacted(
            MarkdownLearnings(Path(tmp)),
            current_path=Path(tmp) / "current.md",
            compactor=flaky,
            quiet=True,
        )
        c.add("first")
        # First compaction failed; queue still holds entry.
        assert "first" in c.queued()
        assert not (Path(tmp) / "current.md").read_text().strip() if \
            (Path(tmp) / "current.md").exists() else True

        c.add("second")
        # Second compaction succeeds and folds in BOTH entries.
        assert c.queued() == []
        result = c.get()
        assert result.startswith("OK:")
        assert "first" in result and "second" in result


def test_user_agent_through_hook_is_respected():
    """User-assigned toolkit.agent_through survives .run() across iterations."""
    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))
        seen = []

        class WatchingStrategy:
            def __call__(self, toolkit, config=None):
                seen.append(getattr(toolkit, "agent_through", None))
                ws = toolkit.history.workspace()
                (ws.path / "solution.py").write_text(make_code(1.0))
                result = toolkit.task.evaluate(make_code(1.0))
                write_result(ws.path, result)
                ws.commit(success=result.completed)

        opt = SimpleOptimizer(
            task, strategy=WatchingStrategy(),
            seed=42, history=history, seed_strategy=None,
        )
        opt.toolkit.agent_through = "validate"
        opt.run(n=3)

        assert seen == ["validate", "validate", "validate"]


def test_queue_wait_learnings_serializes_concurrent_writes():
    """Concurrent add() calls don't drop entries when wrapped in QueueWaitLearnings."""
    import threading
    from groundhog import QueueWaitLearnings, MarkdownLearnings

    with tempfile.TemporaryDirectory() as tmp:
        inner = MarkdownLearnings(Path(tmp))
        l = QueueWaitLearnings(inner)

        N = 20
        barrier = threading.Barrier(N)
        errors = []

        def adder(i: int):
            try:
                barrier.wait(timeout=10)
                l.add(f"entry-{i:02d}")
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=adder, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"adders raised: {errors}"
        text = l.get()
        for i in range(N):
            assert f"entry-{i:02d}" in text, f"missing entry-{i:02d}"


def test_queue_wait_reclaims_stale_markers():
    """Markers older than the TTL are removed and don't block new claims."""
    from groundhog import QueueWaitLearnings, MarkdownLearnings

    with tempfile.TemporaryDirectory() as tmp:
        inner = MarkdownLearnings(Path(tmp))
        l = QueueWaitLearnings(inner, ttl_seconds=0.5)

        # Plant a stale marker as if a previous writer crashed.
        stale = l.queue_dir / f"{0:020d}_999_deadbeef.intent"
        stale.mkdir()
        old = time.time() - 60
        os.utime(stale, (old, old))

        # add() should reclaim the stale marker and proceed.
        l.add("after-recovery")
        assert "after-recovery" in l.get()
        assert not stale.exists()


def test_user_get_prior_hook_is_respected():
    """User-assigned toolkit.get_prior survives .run() and is invoked."""
    with tempfile.TemporaryDirectory() as tmp:
        task = FixtureTask()
        history = FolderAttemptHistory(Path(tmp))
        called = []

        class RecordingStrategy:
            def __call__(self, toolkit, config=None):
                toolkit.get_prior(toolkit)
                ws = toolkit.history.workspace()
                (ws.path / "solution.py").write_text(make_code(1.0))
                result = toolkit.task.evaluate(make_code(1.0))
                write_result(ws.path, result)
                ws.commit(success=result.completed)
                return {}

        def my_hook(tk):
            called.append(True)
            return None

        optimizer = SimpleOptimizer(task, strategy=RecordingStrategy(),
                                    seed=42, history=history, seed_strategy=None)
        optimizer.toolkit.get_prior = my_hook
        optimizer.run(n=2)

        assert optimizer.toolkit.get_prior is my_hook, "hook was replaced by run()"
        assert len(called) == 2, f"hook invoked {len(called)} times, expected 2"


# === Strategy Config tests ===

def test_strategy_config_from_dict_ignores_unknown():
    from groundhog.base.strategy import StrategyConfig, param
    from dataclasses import dataclass

    @dataclass
    class MyConfig(StrategyConfig):
        x: int = param(1, "test")
        y: str = param("a", "test")

    cfg = MyConfig.from_dict({"x": 5, "y": "b", "z": "unknown"})
    assert cfg.x == 5
    assert cfg.y == "b"

def test_strategy_config_describe():
    from groundhog.base.strategy import StrategyConfig, param
    from dataclasses import dataclass

    @dataclass
    class MyConfig(StrategyConfig):
        retries: int = param(3, "Max retries")

    desc = MyConfig().describe()
    assert "retries" in desc
    assert desc["retries"]["default"] == 3
    assert desc["retries"]["description"] == "Max retries"
    assert desc["retries"]["value"] == 3

def test_strategy_config_resolve():
    from groundhog.strategies.improve import Improve
    s = Improve(learnings_last=5)
    assert s.config.learnings_last == 5
    assert s.config.learnings_random == 10  # default

    resolved = s._resolve_config({"learnings_last": 0})
    assert resolved.learnings_last == 0
    assert resolved.learnings_random == 10  # still default


# === FolderAttemptHistory tests ===

def test_folder_history_workspace_commit():
    import tempfile
    from pathlib import Path
    from groundhog.histories.folder import FolderAttemptHistory
    from groundhog.utils.results import write_result
    from groundhog.base.types import EvaluationResult, StageResult

    with tempfile.TemporaryDirectory() as d:
        h = FolderAttemptHistory(Path(d) / "attempts")
        ws = h.workspace(parent=None)
        (ws.path / "solution.py").write_text("def solve(): return 42")
        result = EvaluationResult(stages={"test": StageResult(metrics={"score": 1.0})})
        write_result(ws.path, result)

        attempt = ws.commit(success=result.completed)
        assert attempt.number == 1
        assert attempt.parent is None
        assert attempt.code == "def solve(): return 42"
        assert attempt.result.completed

def test_folder_history_list_and_best():
    import tempfile
    from pathlib import Path
    from groundhog.histories.folder import FolderAttemptHistory
    from groundhog.utils.results import write_result
    from groundhog.base.types import EvaluationResult, StageResult

    with tempfile.TemporaryDirectory() as d:
        h = FolderAttemptHistory(Path(d) / "attempts")

        # Add two attempts with different scores
        ws1 = h.workspace(parent=None)
        (ws1.path / "solution.py").write_text("v1")
        r1 = EvaluationResult(stages={"test": StageResult(metrics={"score": 0.5})})
        write_result(ws1.path, r1)
        ws1.commit(success=True)

        ws2 = h.workspace(parent=1)
        (ws2.path / "solution.py").write_text("v2")
        r2 = EvaluationResult(stages={"test": StageResult(metrics={"score": 0.9})})
        write_result(ws2.path, r2)
        ws2.commit(success=True)

        assert len(h.list()) == 2
        best = h.best(lambda sr: sr.metrics.get("score", 0))
        assert best.number == 2

def test_folder_history_lineage():
    import tempfile
    from pathlib import Path
    from groundhog.histories.folder import FolderAttemptHistory
    from groundhog.utils.results import write_result
    from groundhog.base.types import EvaluationResult, StageResult

    with tempfile.TemporaryDirectory() as d:
        h = FolderAttemptHistory(Path(d) / "attempts")
        r = lambda s: EvaluationResult(stages={"t": StageResult(metrics={"score": s})})

        ws1 = h.workspace(parent=None)
        (ws1.path / "solution.py").write_text("v1")
        r1 = r(0.5)
        write_result(ws1.path, r1)
        ws1.commit(success=True)

        ws2 = h.workspace(parent=1)
        (ws2.path / "solution.py").write_text("v2")
        r2 = r(0.8)
        write_result(ws2.path, r2)
        ws2.commit(success=True)

        attempt2 = h.list()[-1]
        lineage = h.lineage(attempt2)
        assert [a.number for a in lineage] == [1, 2]


# === Queue (tools/queue.py) ===

def test_queue_preserves_file_when_drained():
    """queue.json stays as `[]` after the last item is consumed.

    Tools that treat the queue as visible persistent state (CLIs, editors)
    can rely on the file always existing once it has been used.
    """
    from groundhog.tools.queue import add, read_next

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        add(path, "fresh_approach", config={"mode": "blank"})
        add(path, "analyse")

        item1 = read_next(path)
        assert item1["strategy"] == "fresh_approach"
        item2 = read_next(path)
        assert item2["strategy"] == "analyse"

        # Queue is drained but file still exists.
        queue_file = path / "queue.json"
        assert queue_file.exists()
        assert json.loads(queue_file.read_text()) == []

        # Subsequent reads return None without recreating or deleting the file.
        assert read_next(path) is None
        assert queue_file.exists()


def test_queue_returns_none_when_file_absent():
    """read_next is a no-op when no queue file has ever been created."""
    from groundhog.tools.queue import read_next

    with tempfile.TemporaryDirectory() as tmp:
        assert read_next(Path(tmp)) is None
        assert not (Path(tmp) / "queue.json").exists()


# === MarkdownLearnings tests ===

def test_learnings_add_get():
    import tempfile
    from pathlib import Path
    from groundhog.learnings.markdown import MarkdownLearnings

    with tempfile.TemporaryDirectory() as d:
        l = MarkdownLearnings(Path(d))
        l.add("First observation")
        l.add("Second observation")
        assert l.count() == 2
        text = l.get()
        assert "First" in text
        assert "Second" in text

def test_learnings_sampling():
    import tempfile
    from pathlib import Path
    from groundhog.learnings.markdown import MarkdownLearnings

    with tempfile.TemporaryDirectory() as d:
        l = MarkdownLearnings(Path(d))
        for i in range(20):
            l.add(f"Learning #{i}")

        # last=3 should return 3 recent entries
        sampled = l.get(last=3, random=0)
        entries = [e.strip() for e in sampled.split("---") if e.strip()]
        assert len(entries) == 3

        # last=3 + random=2 should return 5
        sampled = l.get(last=3, random=2)
        entries = [e.strip() for e in sampled.split("---") if e.strip()]
        assert len(entries) == 5

def test_learnings_edit():
    import tempfile
    from pathlib import Path
    from groundhog.learnings.markdown import MarkdownLearnings

    with tempfile.TemporaryDirectory() as d:
        l = MarkdownLearnings(Path(d))
        l.add("KNN works well")
        l.edit("KNN works well", "KNN works okay")
        assert "okay" in l.get()


# === BackendRegistry fallback tests ===

def test_backend_registry_fallback():
    from groundhog.base.backend import BackendRegistry, LLMResponse, LLMBackend

    class DummyBackend(LLMBackend):
        def __init__(self, name):
            self.name = name
        def generate(self, prompt, system_prompt=None):
            return LLMResponse(text=self.name, model=self.name)

    reg = BackendRegistry(default=DummyBackend("default"), high=DummyBackend("high"))
    assert reg.get("high").name == "high"
    assert reg.get("default").name == "default"
    assert reg.get("missing_tier").name == "default"  # falls back


# === End-to-end mock task test ===

def test_mock_task_end_to_end():
    """Run MockTask with MockStrategy for 5 iterations — full loop including path-based evaluation."""
    import sys
    import tempfile
    from pathlib import Path

    # Import mock task and strategy from templates
    from groundhog.templates.mock_task import MockTask
    from groundhog.templates.mock_strategy import MockStrategy
    from groundhog import SimpleOptimizer

    with tempfile.TemporaryDirectory() as d:
        task = MockTask(seed=42)
        strategy = MockStrategy()
        optimizer = SimpleOptimizer(task, strategy=strategy, seed=69, seed_strategy=None, path=Path(d))
        optimizer.run(n=5)

        # Verify attempts were created
        attempts = optimizer.history.list()
        assert len(attempts) == 5

        # Verify scoring works (all should have valid scores)
        scorer = optimizer._get_scorer()
        for a in attempts:
            assert a.result.completed
            score = optimizer._score_attempt(a, scorer)
            assert 0.0 <= score <= 1.0

        # Verify best is deterministic with seed
        best = optimizer.history.best(scorer)
        assert best is not None

        # Verify trunks can be derived
        trunks = optimizer.history.derive_trunks(scorer)
        assert len(trunks) > 0

        # Verify solution.py exists in each attempt
        for a in attempts:
            assert (a.path / "solution.py").exists()


# === Run all tests ===

if __name__ == "__main__":
    import sys

    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []

    for test in tests:
        name = test.__name__
        try:
            if "capsys" in test.__code__.co_varnames:
                print(f"  SKIP {name} (needs pytest capsys)")
                continue
            test()
            print(f"  PASS {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed.append(name)

    print(f"\n{len(tests)} tests, {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
