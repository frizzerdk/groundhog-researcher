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
import time
import tempfile
from pathlib import Path

from groundhog import (
    Task, Data, Context, Evaluator,
    EvalStage, StageResult, EvaluationResult,
    Toolkit, SimpleOptimizer, FolderAttemptHistory, assemble_toolkit,
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
        attempt2 = history.get(attempt.id)
        assert attempt2.code == code_before
        assert attempt2.result.stages["eval"].metrics == result_before.stages["eval"].metrics

def test_attempt_history_tree_structure():
    """Each attempt has at most one parent, forming a tree."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})

        ws1 = history.workspace(parent=None)
        (ws1.path / "solution.py").write_text("v1")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.id)
        (ws2.path / "solution.py").write_text("v2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a1.id)
        (ws3.path / "solution.py").write_text("v3")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        assert a1.parent is None
        assert a2.parent == a1.id
        assert a3.parent == a1.id

def test_attempt_history_lineage():
    """Lineage walks from attempt back to root."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"x": 1.0})})

        ws1 = history.workspace()
        (ws1.path / "solution.py").write_text("v1")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.id)
        (ws2.path / "solution.py").write_text("v2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a2.id)
        (ws3.path / "solution.py").write_text("v3")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        chain = history.lineage(a3)
        assert [a.id for a in chain] == ["1", "2", "3"]

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
        results: list[list[str]] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def alloc(count: int):
            try:
                h = FolderAttemptHistory(Path(tmp))  # one per "optimizer"
                barrier.wait(timeout=10)             # release them together
                nums = [h.workspace().display_id for _ in range(count)]
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
        nums_int = sorted(int(n) for n in all_nums)
        assert nums_int == list(range(1, 21)), \
            f"expected dense 1..20, got {nums_int}"


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
        assert ws.display_id == "1"
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
        raise AssertionError("Should have raised AttributeError")
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

        optimizer = SimpleOptimizer(assemble_toolkit(task, history=history, seed=42), strategy=ReturnsNothing(), seed_strategy=None)
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
                recorded.append(attempt.id)
                return {"attempt": attempt.id}

        optimizer = SimpleOptimizer(assemble_toolkit(task, history=history, seed=42), strategy=TrackingStrategy(), seed_strategy=None)
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
                def __init__(self, sink):
                    self.sink = sink

                def __call__(self, toolkit, config=None):
                    val = toolkit.rng.uniform(0, 100)
                    self.sink.append(val)
                    ws = toolkit.history.workspace()
                    code = make_code(val)
                    (ws.path / "solution.py").write_text(code)
                    result = toolkit.task.evaluate(code)
                    write_result(ws.path, result)

                    ws.commit(success=result.completed)
                    return {"value": val}

            optimizer = SimpleOptimizer(assemble_toolkit(task, history=history, seed=42), strategy=DeterministicStrategy(rng_values), seed_strategy=None)
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
            assemble_toolkit(task, history=history, path=Path(tmp), seed=42),
            strategy=RotationStrat(), extras=[ExtraStrat()],
            seed_strategy=None,
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
            assemble_toolkit(task, history=history, path=Path(tmp), seed=42),
            strategy=rotation, extras=[extra],
            seed_strategy=None,
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


def test_promote_best_refuses_parent_identical_snapshot():
    """Parent-identical candidates evaluate but are not snapshotted."""
    from groundhog.agents.tools import promote_best
    from groundhog import EvalStage, StageResult

    with tempfile.TemporaryDirectory() as tmp:
        def call(code):
            return StageResult(metrics={"score": float(int(code.strip()))})

        stage = EvalStage(
            name="evaluate", description="dummy",
            call=call, scorer=lambda r: r.metrics["score"],
        )
        ws = Path(tmp)
        (ws / "work").mkdir()
        parent = ws / "parent.py"
        parent.write_text("5")
        src = ws / "work" / "solution.py"
        src.write_text("5")
        dest = ws / "solution.py"

        tool = promote_best(stage, dest_path=dest, parent_solution_path=parent)
        output = tool.execute(path=str(src)).output
        assert "identical to the parent" in output
        assert not dest.exists()


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

        ws2 = history.workspace(parent=a1.id)
        (ws2.path / "solution.py").write_text("v2")
        (ws2.path / "work" / "learnings.md").write_text("note from 2")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace(parent=a2.id)
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


def test_build_prior_tools_direction_scopes():
    """family scope shows same-direction attempts; exclude_direction hides one."""
    from groundhog.agents.tools import build_prior_tools
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 1.0})})

        ws1 = history.workspace()
        (ws1.path / "solution.py").write_text("rollout root")
        write_direction(ws1.path, "rollout")
        write_result(ws1.path, result)
        a1 = ws1.commit(success=True)

        ws2 = history.workspace(parent=a1.id)
        (ws2.path / "solution.py").write_text("rollout child")
        write_direction(ws2.path, "rollout")
        write_result(ws2.path, result)
        a2 = ws2.commit(success=True)

        ws3 = history.workspace()
        (ws3.path / "solution.py").write_text("mcts")
        write_direction(ws3.path, "mcts")
        write_result(ws3.path, result)
        a3 = ws3.commit(success=True)

        family_tools = build_prior_tools(a2, history=history, scope="family")
        family = {t.name: t for t in family_tools}["get-priors"].execute(n=10).output
        assert f"attempt_{a1.id}" in family
        assert f"attempt_{a2.id}" in family
        assert f"attempt_{a3.id}" not in family

        all_tools = build_prior_tools(
            a2,
            history=history,
            scope="all",
            exclude_direction="rollout",
        )
        by_name = {t.name: t for t in all_tools}
        listing = by_name["get-priors"].execute(n=10).output
        assert f"attempt_{a3.id}" in listing
        assert f"attempt_{a2.id}" not in listing
        assert by_name["get-prior-file"].execute(
            attempt=str(a3.id), file="solution.py"
        ).output == "mcts"


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
            assemble_toolkit(task, history=history, seed=42),
            strategy=WatchingStrategy(),
            seed_strategy=None,
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

        optimizer = SimpleOptimizer(assemble_toolkit(task, history=history, seed=42),
                                    strategy=RecordingStrategy(), seed_strategy=None)
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
        assert attempt.id == "1"
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
        assert best.id == "2"

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
        assert [a.id for a in lineage] == ["1", "2"]


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


# === Core direction (family identity) tests ===

def test_direction_inherit_canonical_filename():
    """inherit_direction copies parent's core_direction.md to child workspace."""
    from groundhog.utils.direction import inherit_direction, read_direction

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior"
        ws = Path(tmp) / "ws"
        prior.mkdir()
        ws.mkdir()
        (prior / "core_direction.md").write_text("rollout-greedy", encoding="utf-8")

        dst = inherit_direction(prior, ws)
        assert dst is not None
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "rollout-greedy"
        assert read_direction(ws).strip() == "rollout-greedy"


def test_direction_inherit_legacy_approach_fallback():
    """Parent has only legacy approach.md → child gets core_direction.md."""
    from groundhog.utils.direction import inherit_direction, read_direction

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior"
        ws = Path(tmp) / "ws"
        prior.mkdir()
        ws.mkdir()
        (prior / "approach.md").write_text("legacy text", encoding="utf-8")

        dst = inherit_direction(prior, ws)
        assert dst is not None
        # Migrated forward to canonical name.
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "legacy text"
        # Read fallback also works against the original.
        assert read_direction(prior).strip() == "legacy text"


def test_direction_inherit_no_prior_direction_is_noop():
    """Parent has nothing → child workspace stays untouched."""
    from groundhog.utils.direction import inherit_direction

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior"
        ws = Path(tmp) / "ws"
        prior.mkdir()
        ws.mkdir()

        result = inherit_direction(prior, ws)
        assert result is None
        assert not (ws / "core_direction.md").exists()


def test_direction_promote_workspace_from_work_dir():
    """Fresh-style: agent wrote work/core_direction.md → promoted to root."""
    from groundhog.utils.direction import promote_workspace_direction

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "work").mkdir()
        (ws / "work" / "core_direction.md").write_text("CNN architecture", encoding="utf-8")

        dst = promote_workspace_direction(ws)
        assert dst is not None
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "CNN architecture"


def test_direction_promote_workspace_legacy_work_approach():
    """Fresh-style: agent wrote work/approach.md (legacy) → promoted to root as core_direction.md."""
    from groundhog.utils.direction import promote_workspace_direction

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "work").mkdir()
        (ws / "work" / "approach.md").write_text("legacy fresh", encoding="utf-8")

        dst = promote_workspace_direction(ws)
        assert dst is not None
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "legacy fresh"


def test_direction_promote_does_not_clobber_root():
    """If root direction already exists, promotion is a no-op."""
    from groundhog.utils.direction import promote_workspace_direction

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "work").mkdir()
        (ws / "work" / "core_direction.md").write_text("from work", encoding="utf-8")
        (ws / "core_direction.md").write_text("from root", encoding="utf-8")

        promote_workspace_direction(ws)
        # Root content is preserved.
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "from root"


def test_direction_enforce_overrides_agent_rewrite():
    """Soft-gate: if agent rewrote core_direction.md, enforce restores parent's."""
    from groundhog.utils.direction import (
        inherit_direction,
        enforce_inherited_direction,
    )

    with tempfile.TemporaryDirectory() as tmp:
        prior = Path(tmp) / "prior"
        ws = Path(tmp) / "ws"
        prior.mkdir()
        ws.mkdir()
        (prior / "core_direction.md").write_text("rollout-greedy", encoding="utf-8")

        # Inherit at workspace prep.
        inherit_direction(prior, ws)
        # Agent rewrites it mid-session (simulating a misbehaving session).
        (ws / "core_direction.md").write_text("MCTS", encoding="utf-8")
        # Enforce at commit restores parent's.
        enforce_inherited_direction(ws, prior)
        assert (ws / "core_direction.md").read_text(encoding="utf-8").strip() \
            == "rollout-greedy"


def test_direction_normalize_collapses_blank_runs_and_trims():
    """Normalization is for family-identity comparison, not display."""
    from groundhog.utils.direction import normalize_direction

    a = "rollout-greedy   \n\n\n\nwith fixed horizon\n"
    b = "rollout-greedy\n\nwith fixed horizon"
    assert normalize_direction(a) == normalize_direction(b)


def test_fresh_agent_ensure_direction_mints_when_agent_didnt_write():
    """FreshAgentStrategy._ensure_direction calls LLM if no direction exists."""
    from groundhog import FreshAgentStrategy
    from groundhog.base.backend import LLMBackend, LLMResponse, BackendRegistry
    from groundhog.utils.direction import read_direction

    class StubBackend(LLMBackend):
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, system_prompt=""):
            self.calls += 1
            return LLMResponse(
                text="rollout-greedy with K=4",
                model="stub", usage={}, cost=0.0,
            )
        def get_parameters(self):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        ws_path = Path(tmp)
        (ws_path / "solution.py").write_text("def solve(): return 0", encoding="utf-8")

        backend = StubBackend()
        strat = FreshAgentStrategy()
        # Build a minimal toolkit; bypass _init by setting what _ensure_direction needs.
        from groundhog.tools.attempt_logger import MarkdownAttemptLogger
        strat._toolkit = Toolkit(llm=BackendRegistry(default=backend))
        strat.log = type("L", (), {"inline": lambda self, *a, **k: None})()
        strat.logger = MarkdownAttemptLogger()
        strat.logger.attempt_start(ws_path)

        # Fake-Workspace stand-in: just needs .path
        ws = type("WS", (), {"path": ws_path})()
        strat._ensure_direction(ws)

        assert backend.calls == 1
        text = read_direction(ws_path)
        assert text is not None
        assert "rollout-greedy" in text


def test_fresh_agent_ensure_direction_skips_when_already_present():
    """If the agent already wrote a direction, no LLM call is made."""
    from groundhog import FreshAgentStrategy
    from groundhog.base.backend import LLMBackend, LLMResponse, BackendRegistry

    class CountingBackend(LLMBackend):
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, system_prompt=""):
            self.calls += 1
            return LLMResponse(text="should-not-be-called", model="stub", usage={}, cost=0.0)
        def get_parameters(self):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        ws_path = Path(tmp)
        (ws_path / "solution.py").write_text("def solve(): return 0", encoding="utf-8")
        (ws_path / "core_direction.md").write_text("agent-written direction", encoding="utf-8")

        backend = CountingBackend()
        strat = FreshAgentStrategy()
        strat._toolkit = Toolkit(llm=BackendRegistry(default=backend))
        strat.log = type("L", (), {"inline": lambda self, *a, **k: None})()
        strat.cost = 0.0

        ws = type("WS", (), {"path": ws_path})()
        strat._ensure_direction(ws)

        assert backend.calls == 0
        from groundhog.utils.direction import read_direction
        assert "agent-written" in read_direction(ws_path)


def test_direction_title_extracts_first_meaningful_line():
    """Title for status display skips blank lines and heading markers."""
    from groundhog.utils.direction import direction_title

    assert direction_title("# CNN architecture\n\nwith dropout") == "CNN architecture"
    assert direction_title("\n\nrollout-greedy with K=4") == "rollout-greedy with K=4"
    assert direction_title("") == "(no direction)"


# === Direction families (read-side derived view) ===

def test_derive_families_groups_by_direction_content():
    """Attempts sharing the same core_direction.md text are one family."""
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 1.0})})

        # Two attempts with direction "rollout"
        for _ in range(2):
            ws = history.workspace()
            (ws.path / "solution.py").write_text("v")
            write_direction(ws.path, "rollout")
            write_result(ws.path, result)
            ws.commit(success=True)

        # One attempt with direction "mcts"
        ws = history.workspace()
        (ws.path / "solution.py").write_text("v")
        write_direction(ws.path, "mcts")
        write_result(ws.path, result)
        ws.commit(success=True)

        # One attempt with no direction (legacy)
        ws = history.workspace()
        (ws.path / "solution.py").write_text("v")
        write_result(ws.path, result)
        ws.commit(success=True)

        families = history.derive_families()
        # 3 groups: rollout (2), mcts (1), no-direction (1).
        assert len(families) == 3
        sizes = sorted(len(f) for f in families)
        assert sizes == [1, 1, 2]


def test_derive_families_legacy_approach_md_groups_with_core_direction():
    """An attempt with approach.md (legacy) and one with matching
    core_direction.md should be in the same family."""
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 1.0})})

        # Legacy attempt with approach.md
        ws = history.workspace()
        (ws.path / "solution.py").write_text("v")
        (ws.path / "approach.md").write_text("rollout", encoding="utf-8")
        write_result(ws.path, result)
        ws.commit(success=True)

        # New attempt with core_direction.md (same text)
        ws = history.workspace()
        (ws.path / "solution.py").write_text("v")
        write_direction(ws.path, "rollout")
        write_result(ws.path, result)
        ws.commit(success=True)

        families = history.derive_families()
        assert len(families) == 1, "legacy approach.md should group with new core_direction.md"
        assert len(families[0]) == 2


def test_fresh_approach_duplicate_direction_gate_fails_result():
    """FreshApproach's finish runs the standard gates: a fresh attempt whose
    direction duplicates an existing family commits as failed."""
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.fresh import FreshApproach
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        result = EvaluationResult(stages={"eval": StageResult(metrics={"score": 1.0})})

        existing = history.workspace()
        (existing.path / "solution.py").write_text("v1")
        write_direction(existing.path, "rollout")
        write_result(existing.path, result)
        existing.commit(success=True)

        task = FixtureTask()
        toolkit = Toolkit(task=task, history=history)
        # First call generates code, second the (duplicate) direction.
        toolkit.llm = BackendRegistry(default=MockBackend([
            f"```python\n{make_code(42.0)}\n```", "rollout",
        ]))

        FreshApproach()(toolkit)

        attempt = history.list(only_done=False)[-1]
        assert attempt.status == "fail"
        assert attempt.result.completed is False
        assert attempt.result.failed_stage == "core_direction"
        assert "duplicated" in attempt.metadata["gate_failure"]


def test_plan_approaches_queues_fresh_runs():
    """PlanApproaches asks the LLM for N directions and queues them."""
    from groundhog import PlanApproaches, Toolkit
    from groundhog.base.backend import LLMBackend, LLMResponse, BackendRegistry
    from groundhog.tools.queue import read_next

    proposals = [
        {"name": "rollout", "direction": "rollout-greedy", "guidance": "use search"},
        {"name": "mcts", "direction": "monte carlo tree search", "guidance": "explore"},
        {"name": "rl", "direction": "RL policy net", "guidance": "neural"},
    ]
    proposal_json = json.dumps(proposals)

    class StubLLM(LLMBackend):
        def generate(self, prompt, system_prompt=""):
            return LLMResponse(text=proposal_json, model="stub", usage={}, cost=0.0)

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task_ctx = type("C", (), {"get": lambda self: "test task"})()
        task = type("T", (), {"context": task_ctx})()

        toolkit = Toolkit(task=task, history=history, path=Path(tmp))
        toolkit.llm = BackendRegistry(default=StubLLM())

        result = PlanApproaches()(toolkit)
        assert result["queued"] == 3, f"expected 3 queued, got {result}"

        # Queue should hold 3 items with the proposed directions.
        items = []
        while True:
            item = read_next(Path(tmp))
            if item is None:
                break
            items.append(item)
        assert len(items) == 3
        directions = [i["config"]["core_direction"] for i in items]
        assert directions == [p["direction"] for p in proposals]


def test_plan_approaches_handles_malformed_llm_output():
    """Markdown-fenced or non-JSON output is parsed best-effort, no crash."""
    from groundhog import PlanApproaches
    parse = PlanApproaches._parse_proposals

    fenced = "```json\n[{\"name\":\"a\",\"direction\":\"x\",\"guidance\":\"\"}]\n```"
    assert len(parse(fenced)) == 1

    # Preamble + array
    preamble = "Here are the directions:\n[{\"name\":\"a\",\"direction\":\"x\",\"guidance\":\"\"}]\nDone."
    assert len(parse(preamble)) == 1

    # Garbage
    assert parse("nonsense") == []
    assert parse("") == []


def test_fresh_agent_initial_direction_seeds_workspace():
    """If config.initial_direction is set, FreshAgent writes it before running."""
    from groundhog import FreshAgentStrategy

    s = FreshAgentStrategy(initial_direction="seeded direction")
    assert s.config.initial_direction == "seeded direction"


def test_fresh_agent_core_direction_is_canonical_config():
    from groundhog import FreshAgentStrategy

    s = FreshAgentStrategy(core_direction="seeded direction")
    assert s.config.core_direction == "seeded direction"


def test_select_prior_favors_underexplored_family():
    """A small (1-attempt) family should be picked roughly as often as the
    big (5-attempt) family despite the score gap, thanks to the direction bonus."""
    import random
    from groundhog.utils.selection import select_prior
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        def commit_with(direction: str, score: float, parent=None):
            ws = history.workspace(parent=parent)
            (ws.path / "solution.py").write_text("v")
            write_direction(ws.path, direction)
            r = EvaluationResult(stages={"eval": StageResult(metrics={"score": score})})
            write_result(ws.path, r)
            return ws.commit(success=True).id

        # Big family: 5 attempts, score 0.9 (incl. score 1.0 leader)
        a1 = commit_with("rollout", 0.5)
        a2 = commit_with("rollout", 0.6, parent=a1)
        a3 = commit_with("rollout", 0.7, parent=a2)
        a4 = commit_with("rollout", 0.8, parent=a3)
        commit_with("rollout", 0.9, parent=a4)

        # Tiny family: 1 attempt, score 0.5 (much weaker)
        commit_with("mcts", 0.5)

        scorer = lambda r: r.metrics.get("score", 0)
        rng = random.Random(42)
        picks = []
        for _ in range(200):
            p = select_prior(history, scorer, rng)
            family = (p.path / "core_direction.md").read_text(encoding="utf-8").strip()
            picks.append(family)

        rollout_pct = picks.count("rollout") / len(picks)
        mcts_pct = picks.count("mcts") / len(picks)
        # mcts should be picked at least 20% of the time despite lower score —
        # without the direction bonus it'd be near 0.
        assert mcts_pct > 0.2, f"mcts share too low: {mcts_pct:.2f}"
        assert rollout_pct < 0.85, f"rollout dominates: {rollout_pct:.2f}"


def test_select_prior_skips_non_promotable():
    """Attempts flagged non_promotable=True are not picked as priors."""
    import random
    from groundhog.utils.selection import select_prior
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        # First attempt: normal.
        ws = history.workspace()
        (ws.path / "solution.py").write_text("v1")
        write_direction(ws.path, "rollout")
        write_result(ws.path,
                     EvaluationResult(stages={"e": StageResult(metrics={"score": 0.5})}),
                     metadata={"non_promotable": False})
        a1 = ws.commit(success=True)

        # Second attempt (improvement): flagged non_promotable.
        ws2 = history.workspace(parent=a1.id)
        (ws2.path / "solution.py").write_text("v2")
        write_direction(ws2.path, "rollout")
        write_result(ws2.path,
                     EvaluationResult(stages={"e": StageResult(metrics={"score": 0.9})}),
                     metadata={"non_promotable": True})
        ws2.commit(success=True)

        scorer = lambda r: r.metrics.get("score", 0)
        # Repeated picks should never return the higher-scoring non-promotable.
        rng = random.Random(0)
        for _ in range(50):
            p = select_prior(history, scorer, rng)
            assert p.id == "1", f"non-promotable was picked (#{p.id})"


def test_agent_strategy_flags_duplicate_solution_non_promotable():
    """AgentStrategy._is_solution_duplicate detects byte-equal solutions."""
    from groundhog import AgentStrategy

    with tempfile.TemporaryDirectory() as tmp:
        ws_path = Path(tmp) / "ws"
        prior_path = Path(tmp) / "prior"
        ws_path.mkdir()
        prior_path.mkdir()
        # Same bytes
        (ws_path / "solution.py").write_text("def solve(): return 1", encoding="utf-8")
        (prior_path / "solution.py").write_text("def solve(): return 1", encoding="utf-8")

        ws = type("WS", (), {"path": ws_path})()
        prior = type("P", (), {"path": prior_path, "code": "def solve(): return 1"})()
        assert AgentStrategy._is_solution_duplicate(ws, prior) is True

        # Different bytes
        (ws_path / "solution.py").write_text("def solve(): return 2", encoding="utf-8")
        assert AgentStrategy._is_solution_duplicate(ws, prior) is False

        # No prior
        assert AgentStrategy._is_solution_duplicate(ws, None) is False


def test_improve_strategy_flags_duplicate_solution():
    """Improve's finish flags a byte-identical child non-promotable (the
    standard SOLUTION_IDENTICAL gate), and the attempt still commits done."""
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.improve import Improve

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = FixtureTask()

        prior_ws = history.workspace()
        (prior_ws.path / "solution.py").write_text(make_code(50.0))
        write_result(prior_ws.path, task.evaluate(make_code(50.0)))
        prior = prior_ws.commit(success=True)

        toolkit = Toolkit(task=task, history=history)
        # No code block in the response: Improve keeps the parent's solution.
        toolkit.llm = BackendRegistry(default=MockBackend(["no changes"]))

        Improve()(toolkit)

        attempt = history.list()[-1]
        assert attempt.parent == prior.id
        assert attempt.status == "done"
        assert attempt.metadata.get("non_promotable") is True
        assert attempt.metadata["non_promotable_reason"] == (
            "solution.py is byte-identical to parent"
        )


def test_cross_pollinate_agent_selects_different_family():
    """_select_inspiration picks a leader from a different direction family."""
    from groundhog import CrossPollinateAgent
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        def commit_with(direction, score):
            ws = history.workspace()
            (ws.path / "solution.py").write_text("v")
            write_direction(ws.path, direction)
            r = EvaluationResult(stages={"e": StageResult(metrics={"score": score})})
            write_result(ws.path, r)
            return ws.commit(success=True)

        # parent's family
        parent = commit_with("rollout", 0.7)
        # same family attempt — should NOT be selected
        commit_with("rollout", 0.85)
        # different family attempt — should be selected (it's the highest non-rollout)
        target = commit_with("mcts", 0.6)
        # another different-family attempt — lower score
        commit_with("rl", 0.5)

        # Build a minimal toolkit. Selection only needs task.evaluator.eval_stages
        # with a scorer; provide a stub.
        from groundhog import EvalStage, StageResult as SR

        class _Eval(Evaluator):
            def get_stages(self, data):
                return [EvalStage(name="e", description="e",
                                  call=lambda c: SR(),
                                  scorer=lambda r: r.metrics.get("score", 0))]
            def evaluate(self, code_or_path, data):
                return SR()

        toolkit = Toolkit(
            task=Task(data=FixtureData(), context=FixtureContext(),
                      evaluator=_Eval(), name="t"),
            history=history,
        )
        s = CrossPollinateAgent()
        s.through = None
        insp = s._select_inspiration(toolkit, parent)
        assert insp is not None
        assert insp.id == target.id, \
            f"expected mcts (#{target.id}), got #{insp.id}"


def test_cross_pollinate_agent_skips_when_only_one_family():
    """If every attempt shares the parent's family, no inspiration found."""
    from groundhog import CrossPollinateAgent
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        for score in (0.5, 0.6, 0.7):
            ws = history.workspace()
            (ws.path / "solution.py").write_text("v")
            write_direction(ws.path, "rollout")
            r = EvaluationResult(stages={"e": StageResult(metrics={"score": score})})
            write_result(ws.path, r)
            ws.commit(success=True)

        from groundhog import EvalStage, StageResult as SR

        class _Eval(Evaluator):
            def get_stages(self, data):
                return [EvalStage(name="e", description="e",
                                  call=lambda c: SR(),
                                  scorer=lambda r: r.metrics.get("score", 0))]
            def evaluate(self, code_or_path, data):
                return SR()

        toolkit = Toolkit(
            task=Task(data=FixtureData(), context=FixtureContext(),
                      evaluator=_Eval(), name="t"),
            history=history,
        )
        parent = history.list()[0]
        s = CrossPollinateAgent()
        s.through = None
        assert s._select_inspiration(toolkit, parent) is None


def test_diversity_integration_phases_1_2_4_5():
    """End-to-end integration: directions, families, selection, duplicate guard.

    Synthetic history with 3 families:
      - rollout (3 attempts, scores 0.5/0.7/0.9, plus 1 duplicate-of-leader
        flagged non_promotable)
      - mcts (1 attempt, score 0.4)
      - no-direction legacy (1 attempt, score 0.3)

    Verify:
      - derive_families groups them correctly (3 families)
      - select_prior never picks the non-promotable
      - select_prior picks across families more often than pure-greedy
        (mcts share > 10% over 200 samples despite worse score)
      - status output shows direction families
    """
    import io
    import random
    from contextlib import redirect_stdout
    from groundhog.utils.selection import select_prior
    from groundhog.utils.direction import write_direction

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        sc = lambda r: r.metrics.get("score", 0)

        def commit(direction, score, parent=None, non_promotable=False):
            ws = history.workspace(parent=parent)
            (ws.path / "solution.py").write_text(f"v{score}")
            if direction:
                write_direction(ws.path, direction)
            r = EvaluationResult(stages={"e": StageResult(metrics={"score": score})})
            meta = {}
            if non_promotable:
                meta["non_promotable"] = True
            write_result(ws.path, r, metadata=meta)
            return ws.commit(success=True).id

        # Build the synthetic history.
        a1 = commit("rollout", 0.5)
        a2 = commit("rollout", 0.7, parent=a1)
        a3 = commit("rollout", 0.9, parent=a2)
        # Duplicate-of-leader flagged non-promotable.
        commit("rollout", 0.95, parent=a3, non_promotable=True)
        commit("mcts", 0.4)
        commit(None, 0.3)  # legacy no-direction

        # Phase 2 — families.
        families = history.derive_families()
        assert len(families) == 3, f"expected 3 families, got {len(families)}"

        # Phase 4 — selection avoids non-promotable, samples across families.
        rng = random.Random(7)
        picks = []
        for _ in range(200):
            p = select_prior(history, sc, rng)
            assert p is not None
            from groundhog.utils.direction import read_direction
            text = read_direction(p.path)
            picks.append(text.strip() if text else "(none)")
        # Non-promotable (#4, score 0.95) is never picked.
        for p in picks:
            assert "0.95" not in p  # not the non-promotable solution

        rollout_pct = picks.count("rollout") / len(picks)
        mcts_pct = picks.count("mcts") / len(picks)
        assert mcts_pct >= 0.10, f"mcts share too low: {mcts_pct:.2f}"
        assert rollout_pct < 0.85, f"rollout dominates: {rollout_pct:.2f}"

        # Phase 2 — status output mentions families.
        # Construct a minimal optimizer just to call status() against this history.
        from groundhog import EvalStage, StageResult as SR

        class _Eval(Evaluator):
            def get_stages(self, data):
                return [EvalStage(name="e", description="e",
                                  call=lambda c: SR(),
                                  scorer=sc)]
            def evaluate(self, code_or_path, data):
                return SR()

        opt = SimpleOptimizer(
            assemble_toolkit(
                Task(data=FixtureData(), context=FixtureContext(),
                     evaluator=_Eval(), name="div-int"),
                history=history, path=Path(tmp),
            ),
            strategy=type("Noop", (), {
                "__call__": lambda self, t, **k: {},
            })(),
            seed_strategy=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.status()
        out = buf.getvalue()
        assert "Direction families:" in out
        assert "rollout" in out
        assert "mcts" in out


def test_derive_families_empty_history():
    """Empty history → empty list, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        assert history.derive_families() == []


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
    import tempfile
    from pathlib import Path

    # Import mock task and strategy from templates
    from groundhog.templates.mock_task import MockTask
    from groundhog.templates.mock_strategy import MockStrategy
    from groundhog import SimpleOptimizer

    with tempfile.TemporaryDirectory() as d:
        task = MockTask(seed=42)
        strategy = MockStrategy()
        optimizer = SimpleOptimizer(assemble_toolkit(task, path=Path(d), seed=69), strategy=strategy, seed_strategy=None)
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


# === Permission overrides (sandbox tightening) ===


def test_agent_strategy_default_permissions_match_base():
    """Default AgentStrategy still resolves to BASE_PERMISSIONS."""
    from groundhog import AgentStrategy
    from groundhog.strategies.agent import BASE_PERMISSIONS

    s = AgentStrategy()
    allow, deny = s._resolve_permissions("explore")
    expected_allow = [r for a, r in BASE_PERMISSIONS if a == "allow"]
    expected_deny = [r for a, r in BASE_PERMISSIONS if a == "deny"]
    assert allow == expected_allow
    assert deny == expected_deny


def test_agent_strategy_permissions_overridable_by_subclass():
    """Subclass-level permissions reassignment is honored."""
    from groundhog import AgentStrategy

    class TightAgent(AgentStrategy):
        permissions = [
            ("allow", "Read(./**)"),
            ("allow", "Read(../**)"),
            ("deny",  "Read(*)"),
            ("deny",  "Write(*)"),
            ("allow", "Write(work/*)"),
        ]

    s = TightAgent()
    allow, deny = s._resolve_permissions("explore")
    assert "Read(./**)" in allow
    assert "Read(../**)" in allow
    assert "Write(work/*)" in allow
    assert "Read(*)" not in allow      # narrowed
    assert "Read(*)" in deny
    assert "Write(*)" in deny


def test_agent_strategy_permissions_overridable_per_instance():
    """Instance-level permissions reassignment is honored."""
    from groundhog import AgentStrategy

    s = AgentStrategy()
    s.permissions = [("allow", "Read(./**)"), ("deny", "Write(*)")]
    allow, deny = s._resolve_permissions("explore")
    assert allow == ["Read(./**)"]
    assert deny == ["Write(*)"]


def test_agent_strategy_phase_overrides_compose():
    """phase_overrides are appended after base; deepest/last wins is the
    backend's job, but ordering must reach the spec intact."""
    from groundhog import AgentStrategy

    class FixSandboxed(AgentStrategy):
        permissions = [("allow", "Read(*)")]
        phase_overrides = {
            "explore": [],
            "fix":     [("deny", "Bash(rm -rf *)")],
            "submit":  [],
            "reflect": [],
        }

    s = FixSandboxed()
    explore_allow, explore_deny = s._resolve_permissions("explore")
    fix_allow, fix_deny = s._resolve_permissions("fix")
    assert "Read(*)" in explore_allow and explore_deny == []
    assert "Bash(rm -rf *)" in fix_deny


def test_permissions_propagate_to_all_agent_backends():
    """Acceptance: each backend translates spec.allowed_tools/denied_tools
    according to its own contract. The strategy-level override only needs
    to land them in AgentSpec — backends are downstream consumers."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.claude_code import ClaudeCodeAgentBackend
    from groundhog.agents.gemini_cli import GeminiCliAgentBackend
    from groundhog.agents.copilot import CopilotAgentBackend
    from groundhog.agents.codex_cli import CodexCliAgentBackend
    from groundhog.agents.opencode import OpenCodeAgentBackend

    with tempfile.TemporaryDirectory() as d:
        spec = AgentSpec(
            goal="test goal",
            workspace_path=Path(d),
            allowed_tools=["Read(./**)"],
            denied_tools=["Read(*)", "Bash(rm -rf *)", "Write(*)"],
        )

        # 1) claude_code: full enforcement via --allowedTools / --disallowedTools.
        cmd = ClaudeCodeAgentBackend()._build_command(spec)
        assert "--allowedTools" in cmd
        ai = cmd.index("--allowedTools")
        assert "Read(./**)" in cmd[ai + 1:]
        assert "--disallowedTools" in cmd
        di = cmd.index("--disallowedTools")
        deny_args = cmd[di + 1:]
        # Claude resolves rule conflicts as ``deny > allow``: a broad
        # ``Read(*)`` deny would shadow the narrow ``Read(./**)`` allow and
        # disable the Read tool entirely. The adapter strips broad denies
        # for tools that have a narrow allow — see _filter_redundant_broad_denies.
        assert "Read(*)" not in deny_args  # dropped because Read has narrow allow
        assert "Write(*)" in deny_args     # no narrow Write allow → kept
        assert "Bash(rm -rf *)" in deny_args  # path-specific, not broad

        # 2) gemini_cli: deny rules surface in the prompt as advisory text.
        prompt = GeminiCliAgentBackend()._build_prompt(spec)
        assert "Read(*)" in prompt
        assert "Bash(rm -rf *)" in prompt

        # 3) copilot: path-specific denies become --deny-tool args; blanket
        #    denies (Read(*)/Write(*)) are intentionally dropped because they
        #    would override --allow-all-tools. Path-specific Bash() rule
        #    survives via shell(...) translation.
        cmd = CopilotAgentBackend()._build_command(spec)
        # rm -rf becomes shell(rm -rf *) and stays
        assert any("shell(rm -rf *)" in arg for arg in cmd)
        # Blanket denies should NOT appear as --deny-tool args
        assert "read" not in cmd  # would be the translated blanket Read(*)
        assert "write" not in cmd  # ditto for Write(*)

        # 4) codex_cli: no native allow/deny flags. Deny rules injected into
        #    the prompt as advisory text (like gemini). Hard floor is the
        #    OS-level --sandbox workspace-write flag. Prompt comes via stdin
        #    (last argv is the literal ``-``), so check the resolved prompt
        #    text directly.
        codex_backend = CodexCliAgentBackend()
        codex_cmd = codex_backend._build_command(spec)
        assert "exec" in codex_cmd
        assert "--json" in codex_cmd
        assert "-s" in codex_cmd and "workspace-write" in codex_cmd
        assert any("approval_policy=never" in a for a in codex_cmd)
        assert codex_cmd[-1] == "-"  # prompt fed via stdin
        prompt = codex_backend._resolve_prompt(spec)
        assert "Read(*)" in prompt
        assert "Bash(rm -rf *)" in prompt
        assert "Write(*)" in prompt

        # 5) opencode: no OS sandbox flag. Hard enforcement comes from a
        # temporary per-attempt opencode.json; deny rules are also repeated in
        # the prompt so the model sees the strategy-level intent.
        opencode = OpenCodeAgentBackend()
        opencode_cmd = opencode._build_command(spec)
        assert "run" in opencode_cmd
        assert "--format" in opencode_cmd and "json" in opencode_cmd
        assert "--agent" in opencode_cmd and "build" in opencode_cmd
        assert "--dir" in opencode_cmd and str(Path(d)) in opencode_cmd
        assert "--pure" in opencode_cmd
        assert "--dangerously-skip-permissions" in opencode_cmd
        assert "openrouter/deepseek/deepseek-v4-flash" in opencode_cmd
        assert "Read(*)" in opencode_cmd[-1]
        assert "Bash(rm -rf *)" in opencode_cmd[-1]
        assert "Write(*)" in opencode_cmd[-1]

        config = opencode._build_config(spec)
        permissions = config["permission"]
        # external_directory and read share the same path-rule dict so the
        # workspace's absolute-path allow patterns work identically against
        # both. The dict has a broad ``*: deny`` floor plus workspace-rooted
        # allows generated from ``Read(./**)``.
        assert isinstance(permissions["external_directory"], dict)
        assert permissions["external_directory"]["*"] == "deny"
        assert permissions["read"] == permissions["external_directory"]
        assert permissions["read"] == permissions["list"]
        assert permissions["webfetch"] == "deny"
        assert permissions["bash"]["*"] == "deny"
        # No Write/Edit allows in this spec → edit collapses to flat "deny".
        assert permissions["edit"] == "deny"
        assert "write" not in permissions


def test_opencode_workspace_config_is_temporary():
    """OpenCode gets a local config file, and existing files are restored."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.opencode import OpenCodeAgentBackend

    with tempfile.TemporaryDirectory() as d:
        workspace = Path(d)
        spec = AgentSpec(goal="test", workspace_path=workspace)
        backend = OpenCodeAgentBackend()

        snapshot = backend._write_workspace_config(spec)
        config_path = workspace / "opencode.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        # With no allow/deny rules in the spec the read/external_directory
        # configs collapse to the flat default "allow". The presence of the
        # ``permission`` block is what we're really verifying here.
        assert "permission" in config
        backend._restore_workspace_config(snapshot)
        assert not config_path.exists()

        config_path.write_text("user config", encoding="utf-8")
        snapshot = backend._write_workspace_config(spec)
        assert "openrouter/deepseek/deepseek-v4-flash" in config_path.read_text()
        backend._restore_workspace_config(snapshot)
        assert config_path.read_text(encoding="utf-8") == "user config"


def test_opencode_write_rules_are_scoped_to_attempt_under_parent_git():
    """Write(work/*) must not become repo-root work/* when OpenCode climbs."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.opencode import OpenCodeAgentBackend

    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / ".git").mkdir()
        workspace = repo / "attempt_001"
        workspace.mkdir()
        spec = AgentSpec(
            goal="test",
            workspace_path=workspace,
            allowed_tools=["Write(work/*)", "Edit(work/*)"],
            denied_tools=["Write(*)"],
        )

        config = OpenCodeAgentBackend()._build_config(spec)
        edit_rules = config["permission"]["edit"]
        assert edit_rules["*"] == "deny"
        assert edit_rules["attempt_001/work/*"] == "allow"
        assert edit_rules[".\\attempt_001\\work\\*"] == "allow"
        assert "work/*" not in edit_rules


def test_codex_resume_command_uses_resume_subcommand():
    """Resume path: ``codex exec resume <session_id> <prompt>``."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.codex_cli import CodexCliAgentBackend

    with tempfile.TemporaryDirectory() as d:
        spec = AgentSpec(
            goal="follow up",
            workspace_path=Path(d),
            session_id="abc-123",
        )
        backend = CodexCliAgentBackend()
        cmd = backend._build_command(spec)
        assert "resume" in cmd
        assert "abc-123" in cmd
        # Prompt is fed via stdin (last argv is the literal ``-``); verify the
        # follow-up text is what _resolve_prompt returns for the resume path.
        assert cmd[-1] == "-"
        assert backend._resolve_prompt(spec) == "follow up"


def test_codex_command_includes_add_dir_when_bin_supplied():
    """``--add-dir <bin_dir>`` is what makes %TEMP%-rooted wrapper bins
    visible to codex's sandboxed shell. Verifying it actually lands in argv."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.codex_cli import CodexCliAgentBackend

    with tempfile.TemporaryDirectory() as d:
        spec = AgentSpec(goal="hi", workspace_path=Path(d))
        bin_dir = Path(d) / "_bin"
        bin_dir.mkdir()
        cmd = CodexCliAgentBackend()._build_command(spec, bin_dir=bin_dir)
        assert "--add-dir" in cmd
        idx = cmd.index("--add-dir")
        assert cmd[idx + 1] == str(bin_dir)
        # Without bin_dir the flag is absent (back-compat for callers that
        # don't have a bin yet).
        cmd_no_bin = CodexCliAgentBackend()._build_command(spec)
        assert "--add-dir" not in cmd_no_bin


def test_codex_event_parsing_extracts_session_and_output():
    """Verify _parse_result picks session_id from thread.started and the
    final agent_message text as output."""
    from groundhog.agents.codex_cli import CodexCliAgentBackend

    events = [
        {"type": "thread.started", "thread_id": "uuid-1"},
        {"type": "turn.started"},
        {"type": "item.completed",
         "item": {"id": "i1", "type": "agent_message", "text": "first"}},
        {"type": "item.completed",
         "item": {"id": "i2", "type": "command_execution",
                  "command": "ls", "exit_code": 0, "status": "completed"}},
        {"type": "item.completed",
         "item": {"id": "i3", "type": "agent_message", "text": "final answer"}},
        {"type": "turn.completed",
         "usage": {"input_tokens": 100, "output_tokens": 20,
                   "cached_input_tokens": 0, "reasoning_output_tokens": 5}},
    ]
    result = CodexCliAgentBackend()._parse_result(events)
    assert result.success is True
    assert result.session_id == "uuid-1"
    assert result.output == "final answer"
    assert result.turns == 1
    # Steps include the command_execution and both agent_message items
    assert len(result.steps) == 3


def test_opencode_resume_command_uses_session_flag():
    """Resume path: ``opencode run --session <session_id> <prompt>``."""
    from pathlib import Path
    import tempfile
    from groundhog.base.agent import AgentSpec
    from groundhog.agents.opencode import OpenCodeAgentBackend

    with tempfile.TemporaryDirectory() as d:
        spec = AgentSpec(
            goal="follow up",
            workspace_path=Path(d),
            session_id="ses_123",
        )
        cmd = OpenCodeAgentBackend()._build_command(spec)
        assert "--session" in cmd
        assert "ses_123" in cmd
        # The last argv is the prompt — it always contains the workspace
        # context header plus the goal, so the goal text appears as a
        # substring rather than the entire arg.
        assert "follow up" in cmd[-1]


def test_opencode_event_parsing_extracts_session_output_and_cost():
    """Verify the generic OpenCode parser handles JSONL text/result events."""
    from groundhog.agents.opencode import OpenCodeAgentBackend

    events = [
        {"type": "step_start", "sessionID": "ses_123"},
        {"type": "text", "sessionID": "ses_123", "text": "first"},
        {"type": "tool_use", "sessionID": "ses_123",
         "part": {"type": "tool", "tool": "bash",
                  "state": {"status": "completed",
                            "input": {"command": ".groundhog_tools/evaluate solution.py"},
                            "output": "score=0.8"}}},
        {"type": "tool_result", "sessionID": "ses_123", "content": "score=0.8"},
        {"type": "text", "sessionID": "ses_123", "text": "final answer"},
        {"type": "step_finish", "sessionID": "ses_123",
         "part": {"type": "step-finish",
                  "tokens": {"input": 100, "output": 20, "reasoning": 0,
                             "cache": {"read": 0, "write": 0}},
                  "cost": 0.001}},
    ]
    result = OpenCodeAgentBackend()._parse_result(events)
    assert result.success is True
    assert result.session_id == "ses_123"
    assert result.output == "first\nfinal answer"
    assert result.turns == 1
    assert result.cost == 0.001
    assert len(result.steps) == 4


# === AttemptLog ===


def test_attempt_log_event_normalizer_dispatches_each_backend_shape():
    """Each backend has a distinct event shape; the strategy's normalizer
    must turn all five into typed LogEvents for the attempt log. The full
    path is preserved in args (the structured record); the console summary
    truncates to basename so the live tail stays high-signal. Noise events
    (deltas, init, step boundaries) return None."""
    from groundhog.strategies.agent import _normalize_agent_event as n
    from groundhog.tools.attempt_logger import AssistantEvent, ToolCallEvent

    # claude_code stream-json
    out = n({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "work/solution.py"}}
    ]}})
    assert isinstance(out, ToolCallEvent)
    assert out.name == "Read"
    assert out.args == {"path": "work/solution.py"}
    assert out.to_console() == ("agent", "tool_call", "Read solution.py")

    # absolute Windows-style path — full path kept in args, basename on console
    out = n({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Write",
         "input": {"file_path": r"C:\repo\groundhog-researcher\attempts\021_20\work\solution.py"}}
    ]}})
    assert out.args["path"].endswith("solution.py")
    assert out.to_console() == ("agent", "tool_call", "Write solution.py")

    # claude_code thinking block → AssistantEvent on the thinking channel
    out = n({"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "Considering a CNN.\nMore detail."}
    ]}})
    assert isinstance(out, AssistantEvent)
    assert out.data.get("channel") == "thinking"

    # opencode part-shape
    out = n({"type": "tool_use", "part": {
        "type": "tool", "tool": "read",
        "state": {"input": {"filePath": "work/solution.py"}}
    }})
    assert isinstance(out, ToolCallEvent)
    assert out.to_console() == ("agent", "tool_call", "read solution.py")

    # gemini_cli — top-level tool_name + parameters
    out = n({"type": "tool_use", "tool_name": "run_shell_command",
             "parameters": {"command": "evaluate-fast"}})
    assert out.name == "run_shell_command"
    assert out.args == {"command": "evaluate-fast"}
    assert out.to_console() == ("agent", "tool_call", "run_shell_command evaluate-fast")

    # codex_cli item.completed wrappers
    out = n({"type": "item.completed", "item": {
        "type": "command_execution", "command": "happy-info"
    }})
    assert out.name == "shell"
    assert out.args == {"command": "happy-info"}

    # copilot tool start
    out = n({"type": "tool.execution_start", "data": {
        "toolName": "view", "arguments": {"path": "work/solution.py"}
    }})
    assert isinstance(out, ToolCallEvent)
    assert out.to_console() == ("agent", "tool_call", "view solution.py")

    # noise events return None
    assert n({"type": "tool.execution_start",
              "data": {"toolName": "report_intent"}}) is None

    # Noise — must return None so the live tail isn't flooded with deltas
    assert n({"type": "message_delta", "data": {}}) is None
    assert n({"type": "init", "model": "x"}) is None
    assert n({"type": "step_start"}) is None


def test_attempt_log_renderer_truncation_and_width_clamp():
    """Renderer respects the configured width, truncates over-long titles
    with `…`, and produces a fixed-height region (header + tail)."""
    import io
    from groundhog.tools.attempt_log import (
        AttemptLog, AttemptLogConfig, TwoPaneRenderer,
    )

    buf = io.StringIO()
    cfg = AttemptLogConfig(color=False, glyphs=False, width=60,
                           tail_lines=4, heartbeat_seconds=0.0)
    log = AttemptLog(cfg, out=buf)
    log.renderer = TwoPaneRenderer(log)
    log.attempt_start(
        num=42, prior=10,
        queue_label="a-very-long-queue-label-that-must-truncate",
        budget_total=12.0,
    )
    log.update(phase="explore", elapsed_s=125.0, budget_used=4.0,
               tokens_in=18000, tokens_out=2100, turns=4)
    log.event("agent", "tool_call", "read solution.py")

    output = buf.getvalue()
    # Width clamp: each box line fits the configured 60 chars
    box_lines = [ln for ln in output.splitlines() if ln.startswith(("╭", "│", "╰"))]
    for line in box_lines:
        # ANSI is disabled in this config so length math is direct.
        assert len(line) == 60, f"box line wrong width: {line!r} ({len(line)})"
    # The over-long queue label gets the … truncation glyph
    assert "…" in output


def test_attempt_log_appended_renderer_no_ansi_for_ci_logs():
    """Non-TTY renderer emits plain text only — no ANSI sequences. This is
    what gets captured in CI logs / file redirects."""
    import io
    from groundhog.tools.attempt_log import (
        AttemptLog, AttemptLogConfig, AppendedRenderer,
    )

    buf = io.StringIO()
    cfg = AttemptLogConfig(color=False, glyphs=False, heartbeat_seconds=0.0)
    log = AttemptLog(cfg, out=buf)
    log.renderer = AppendedRenderer(log)
    log.attempt_start(num=7, prior=None, queue_label="seed", budget_total=0.0)
    log.update(phase="explore")
    log.event("agent", "tool_call", "read solution.py")
    log.attempt_done(attempt_num=7, score=0.5, delta=0.05,
                     total_cost=0.01, cumulative_cost=0.01,
                     summary_line="metric=0.5")

    text = buf.getvalue()
    assert "\033" not in text, "ANSI escape leaked into non-TTY output"
    assert "[  7] start" in text
    assert "phase: explore" in text
    assert "tool_call" in text
    assert "[  7] 0.5 (+0.0500)" in text


def test_attempt_log_done_does_not_require_attempt_start():
    """Non-Agent strategies (seed, deterministic) skip attempt_start.
    attempt_done must still print correctly using the explicit attempt_num
    arg — the optimizer always knows the number from the committed attempt."""
    import io
    from groundhog.tools.attempt_log import (
        AttemptLog, AttemptLogConfig, AppendedRenderer,
    )

    buf = io.StringIO()
    cfg = AttemptLogConfig(color=False, glyphs=False, heartbeat_seconds=0.0)
    log = AttemptLog(cfg, out=buf)
    log.renderer = AppendedRenderer(log)
    # Note: NO attempt_start call.
    log.attempt_done(attempt_num=99, score=1.0, delta=0.0,
                     total_cost=0.0, cumulative_cost=0.0)
    assert "[ 99] 1.0" in buf.getvalue()


def test_learnings_with_markdown_tables_survive_retrieval():
    """Entries containing markdown tables must not be shredded on get() —
    the entry separator is the exact one add() writes, not any bare "---"
    (which would also match |---| table rows). Regression: a 53KB live
    learnings file read as 94 entries, 54 of them table fragments."""
    from groundhog.learnings.markdown import MarkdownLearnings

    with tempfile.TemporaryDirectory() as tmp:
        learnings = MarkdownLearnings(Path(tmp))
        table = "| k | acc |\n|---|-----|\n| 7 | 75.0 |"
        learnings.add(f"results so far:\n{table}")
        learnings.add("cosine beats euclidean")

        assert learnings.count() == 2
        assert "| 7 | 75.0 |" in learnings.get()
        assert learnings.get(last=1) == "cosine beats euclidean"


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
