"""Challenge strategy: falsification probes against blocking assumptions.

Target selection (failed attempts + stale families, reason-bearing first),
diagnosis metadata round-trip, and the end-to-end loop with a mock LLM.
"""

import tempfile
from pathlib import Path

from groundhog import Challenge, MarkdownLearnings, Task, Toolkit
from groundhog.backends.mock import MockBackend
from groundhog.base.backend import BackendRegistry
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
from groundhog.utils.results import write_result


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


DIAGNOSIS = (
    "BLOCKER: values above 60 crash the harness\n"
    "EVIDENCE: inherited - claimed in one failure, never directly tested\n"
    "PLAN: return 80 and observe whether the harness actually rejects it"
)


def _commit(history, task, value, direction=None, parent=None, metadata=None,
            success=True, fail_errors=None):
    ws = history.workspace(parent=parent)
    (ws.path / "solution.py").write_text(_code(value), encoding="utf-8")
    if direction:
        write_direction(ws.path, direction)
    if fail_errors:
        result = EvaluationResult(
            stages={"eval": StageResult(errors=fail_errors)},
            completed=False, failed_stage="eval",
        )
    else:
        result = task.evaluate(_code(value))
    write_result(ws.path, result, metadata=metadata)
    return ws.commit(success=success)


def _select(toolkit, **config):
    strategy = Challenge(**config)
    strategy._init(toolkit, None)
    return strategy._select_target(toolkit)


# --- Target selection -------------------------------------------------------

def test_select_target_picks_failed_attempt():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        _commit(history, task, 50.0, direction="rollout")
        failed = _commit(history, task, 0.0, direction="mcts", success=False,
                         fail_errors={"crash": "division by zero"})

        toolkit = Toolkit(task=task, history=history)
        target = _select(toolkit)
        assert target is not None
        assert target.kind == "attempt"
        assert target.attempt.id == failed.id
        assert "division by zero" in target.reason


def test_select_target_prefers_stated_reason():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        _commit(history, task, 0.0, direction="a", success=False)
        with_reason = _commit(history, task, 0.0, direction="b", success=False,
                              metadata={"reason": "GPU memory ceiling makes this impossible"})

        toolkit = Toolkit(task=task, history=history)
        target = _select(toolkit)
        assert target.attempt.id == with_reason.id
        assert "GPU memory ceiling" in target.reason


def test_select_target_stale_family():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        a1 = _commit(history, task, 50.0, direction="rollout")
        a2 = _commit(history, task, 60.0, direction="rollout", parent=a1.id)
        a3 = _commit(history, task, 55.0, direction="rollout", parent=a2.id)
        _commit(history, task, 50.0, direction="rollout", parent=a3.id)
        _commit(history, task, 40.0, direction="mcts")  # small family, not stale

        toolkit = Toolkit(task=task, history=history)
        target = _select(toolkit, staleness_window=2)
        assert target is not None
        assert target.kind == "family"
        assert target.attempt.id == a2.id, "family leader is the best-scoring member"


def test_select_target_improving_family_not_stale():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        a1 = _commit(history, task, 50.0, direction="rollout")
        a2 = _commit(history, task, 60.0, direction="rollout", parent=a1.id)
        a3 = _commit(history, task, 70.0, direction="rollout", parent=a2.id)
        _commit(history, task, 80.0, direction="rollout", parent=a3.id)

        toolkit = Toolkit(task=task, history=history)
        assert _select(toolkit, staleness_window=2) is None


def test_select_target_skips_already_challenged():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        failed = _commit(history, task, 0.0, direction="mcts", success=False)
        _commit(history, task, 30.0, direction="Challenge: mcts blocker",
                metadata={"strategy": "challenge",
                          "challenge_target": {"kind": "attempt", "attempt": failed.id}})

        toolkit = Toolkit(task=task, history=history)
        assert _select(toolkit) is None


def test_select_target_skips_failed_challenges():
    """A challenge that itself failed is not re-challenged (no challenge loops)."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        _commit(history, task, 0.0, direction="Challenge: something", success=False,
                metadata={"strategy": "challenge",
                          "challenge_target": {"kind": "family", "attempt": "1"}})

        toolkit = Toolkit(task=task, history=history)
        assert _select(toolkit) is None


def test_select_target_explicit_id():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        done = _commit(history, task, 50.0, direction="rollout")
        _commit(history, task, 0.0, direction="mcts", success=False)

        toolkit = Toolkit(task=task, history=history)
        target = _select(toolkit, target=done.id)
        assert target.kind == "attempt"
        assert target.attempt.id == done.id


def test_no_target_skips():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        toolkit = Toolkit(task=_task(), history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(["unused"]))
        out = Challenge()(toolkit)
        assert out == {"skipped": "no challengeable target"}


# --- Diagnosis parsing --------------------------------------------------------

def test_parse_diagnosis_structured():
    d = Challenge._parse_diagnosis(DIAGNOSIS)
    assert d["blocker"] == "values above 60 crash the harness"
    assert d["evidence"].startswith("inherited")
    assert "return 80" in d["plan"]


def test_parse_diagnosis_multiline_and_fallback():
    d = Challenge._parse_diagnosis("BLOCKER: first\ncontinued line\nPLAN: do x\nstep two")
    assert d["blocker"] == "first\ncontinued line"
    assert d["plan"] == "do x\nstep two"

    d = Challenge._parse_diagnosis("just prose, no fields\nsecond line")
    assert d["blocker"] == "just prose, no fields"
    assert d["evidence"] == ""
    assert d["plan"] == ""


# --- End-to-end with mock LLM -------------------------------------------------

def test_challenge_end_to_end_metadata_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        failed = _commit(history, task, 0.0, direction="push values high",
                         success=False, fail_errors={"crash": "value too high"})

        toolkit = Toolkit(task=task, history=history)
        toolkit.learnings = MarkdownLearnings(Path(tmp))
        # diagnosis -> attack code -> learnings
        toolkit.llm = BackendRegistry(default=MockBackend([
            DIAGNOSIS, _code_block(80.0),
            "- The assumption FELL: 80 evaluated fine, the harness does not reject high values.",
        ]))

        out = Challenge()(toolkit)
        assert out["target"] == failed.id
        assert out["kind"] == "attempt"
        assert out["score"] == 0.8

        attempt = history.list()[-1]
        assert attempt.status == "done"
        assert attempt.parent == failed.id
        meta = attempt.metadata
        assert meta["strategy"] == Challenge.name == "challenge"
        assert meta["prior"] == failed.id
        assert meta["challenge_target"] == {
            "kind": "attempt", "attempt": failed.id,
            "reason": "failed at 'eval': {'crash': 'value too high'}",
        }
        assert meta["challenge_assumption"]["blocker"] == "values above 60 crash the harness"
        assert meta["challenge_assumption"]["evidence"].startswith("inherited")
        assert "return 80" in meta["challenge_assumption"]["plan"]

        direction = attempt.read_file("core_direction.md")
        assert direction.splitlines()[0] == "Challenge: values above 60 crash the harness"

        assert "FELL" in toolkit.learnings.get()
        assert history.get_note(attempt.id, "score") == "0.8000"


def test_challenge_stale_family_commits_fresh_rooted():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        a1 = _commit(history, task, 50.0, direction="rollout")
        a2 = _commit(history, task, 40.0, direction="rollout", parent=a1.id)
        _commit(history, task, 30.0, direction="rollout", parent=a2.id)

        toolkit = Toolkit(task=task, history=history)
        toolkit.llm = BackendRegistry(default=MockBackend([
            DIAGNOSIS, _code_block(70.0),
        ]))

        out = Challenge(staleness_window=2)(toolkit)
        assert out["kind"] == "family"
        assert out["target"] == a1.id

        attempt = history.list()[-1]
        assert attempt.parent is None
        assert attempt.metadata["prior"] is None
        assert attempt.metadata["challenge_target"]["kind"] == "family"


def test_failed_challenge_is_recorded_with_survival_learnings():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        task = _task()
        failed = _commit(history, task, 0.0, direction="mcts", success=False)

        toolkit = Toolkit(task=task, history=history)
        toolkit.learnings = MarkdownLearnings(Path(tmp))
        bad_code = "```python\ndef solve():\n    raise ValueError('still blocked')\n```"
        toolkit.llm = BackendRegistry(default=MockBackend([
            DIAGNOSIS, bad_code,
            "- The assumption SURVIVED: the probe crashed the same way.",
        ]))

        Challenge(max_retries=0)(toolkit)

        attempt = history.list(only_done=False)[-1]
        assert attempt.status == "fail"
        assert attempt.metadata["strategy"] == "challenge"
        assert attempt.metadata["challenge_target"]["attempt"] == failed.id
        assert "SURVIVED" in toolkit.learnings.get()
