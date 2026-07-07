"""ABTest meta-strategy: paired trials, ab metadata, derived scoreboard.

The arms here are stub strategies that run the real full loop shape —
select prior via the toolkit, workspace, write solution+result, finish
through the standard ``finalize_attempt`` — so the tests exercise the
actual metadata pass-through, not a mock of it.
"""

import tempfile
from pathlib import Path

from groundhog import ABTest, Strategy, Task, Toolkit
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
from groundhog.utils.results import write_result


# --- Fixtures ---------------------------------------------------------------

class _Data(Data):
    def get_train(self):
        return None

    def get_test(self):
        return None


class _Ctx(Context):
    def get_brief(self):
        return "brief"

    def get_extended(self):
        return "extended"


class _Eval(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult()

    def get_stages(self, data):
        return [
            EvalStage("evaluate", "final",
                      lambda cp: StageResult(),
                      scorer=lambda r: r.metrics.get("score", -1.0)),
        ]


def _task():
    return Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")


class StubArm(Strategy):
    """Full-loop stub: prior via toolkit, workspace, standard finalize."""

    def __init__(self, scores, fresh=False, **kwargs):
        super().__init__(**kwargs)
        self.scores = list(scores)
        self.fresh = fresh
        self.calls = 0

    def __call__(self, toolkit, config=None):
        self.calls += 1
        prior = None
        if not self.fresh and hasattr(toolkit, "get_prior"):
            prior = toolkit.get_prior(toolkit)
        ws = toolkit.history.workspace(parent=prior.id if prior else None)
        (ws.path / "solution.py").write_text(
            f"# {self.name} call {self.calls}", encoding="utf-8")
        if prior is None:
            write_direction(ws.path, f"{self.name} direction {self.calls}")
        score = self.scores[(self.calls - 1) % len(self.scores)]
        result = EvaluationResult(
            stages={"evaluate": StageResult(metrics={"score": score})})
        metadata = {"strategy": self.name,
                    "prior": prior.id if prior else None}
        attempt = finalize_attempt(toolkit, ws, result, prior, metadata=metadata)
        return {"attempt": attempt.id, "strategy": self.name}


class ArmA(StubArm):
    pass


class ArmB(StubArm):
    pass


class _LogRecorder:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(text)

    def end(self):
        pass


def _seed(history, direction="seed direction", score=0.5):
    ws = history.workspace()
    (ws.path / "solution.py").write_text("# seed", encoding="utf-8")
    write_direction(ws.path, direction)
    result = EvaluationResult(
        stages={"evaluate": StageResult(metrics={"score": score})})
    write_result(ws.path, result)
    return ws.commit(success=True)


def _toolkit(history, **extra):
    return Toolkit(task=_task(), history=history, log=_LogRecorder(), **extra)


def _ab_attempts(history, test="arm_a-vs-arm_b"):
    return [a for a in history.list(only_done=False)
            if a.status != "in-progress" and a.metadata.get("ab_test") == test]


def _commit_trial(history, test, arm, pair, score, completed=True):
    ws = history.workspace()
    (ws.path / "solution.py").write_text(f"# {arm} {pair}", encoding="utf-8")
    write_direction(ws.path, f"{arm} direction {pair}")
    result = EvaluationResult(
        stages={"evaluate": StageResult(metrics={"score": score})},
        completed=completed,
        failed_stage=None if completed else "evaluate",
    )
    write_result(ws.path, result, metadata={
        "strategy": f"arm_{arm}", "ab_test": test,
        "ab_arm": arm, "ab_pair": pair,
    })
    return ws.commit(success=completed)


# --- Paired trials ----------------------------------------------------------

def test_paired_trial_stamps_metadata_and_shares_prior():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        seed = _seed(history)
        # A selector that returns the NEWEST attempt: without pinning, arm B
        # would build on arm A's fresh commit instead of the shared prior.
        def newest(tk):
            return max(tk.history.list(), key=lambda a: (a.created_at, a.id))
        toolkit = _toolkit(history, get_prior=newest)

        ab = ABTest(ArmA(scores=[0.8]), ArmB(scores=[0.6]))
        out = ab(toolkit)

        attempts = _ab_attempts(history)
        assert len(attempts) == 2
        by_arm = {a.metadata["ab_arm"]: a for a in attempts}
        assert by_arm["a"].metadata["strategy"] == "arm_a"
        assert by_arm["b"].metadata["strategy"] == "arm_b"
        assert by_arm["a"].metadata["ab_pair"] == 1
        assert by_arm["b"].metadata["ab_pair"] == 1
        assert by_arm["a"].parent == seed.id
        assert by_arm["b"].parent == seed.id

        assert toolkit.get_prior is newest
        assert toolkit._extra_attempt_metadata is None
        assert out["verdict"] is None  # min_pairs default 5, only 1 pair
        assert out["summary"]["pairs"] == 1


def test_pair_id_increments_across_calls():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _seed(history)
        toolkit = _toolkit(history, get_prior=lambda tk: tk.history.list()[0])

        ab = ABTest(ArmA(scores=[0.8]), ArmB(scores=[0.6]))
        ab(toolkit)
        ab(toolkit)

        pairs = sorted(a.metadata["ab_pair"] for a in _ab_attempts(history))
        assert pairs == [1, 1, 2, 2]


def test_fresh_arm_ignores_pinned_prior_pairing_degrades_to_same_call():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        seed = _seed(history)
        toolkit = _toolkit(history, get_prior=lambda tk: seed)

        ab = ABTest(ArmA(scores=[0.8]), ArmB(scores=[0.6], fresh=True))
        ab(toolkit)

        by_arm = {a.metadata["ab_arm"]: a for a in _ab_attempts(history)}
        assert by_arm["a"].parent == seed.id
        assert by_arm["b"].parent is None
        assert by_arm["a"].metadata["ab_pair"] == by_arm["b"].metadata["ab_pair"] == 1


# --- Unpaired mode ----------------------------------------------------------

def test_unpaired_alternates_arms_without_pinning():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        toolkit = _toolkit(history)

        a, b = ArmA(scores=[0.8], fresh=True), ArmB(scores=[0.6], fresh=True)
        ab = ABTest(a, b, paired=False)
        for _ in range(4):
            ab(toolkit)

        assert not hasattr(toolkit, "get_prior")
        attempts = _ab_attempts(history)
        assert [a.metadata["ab_arm"] for a in attempts] == ["a", "b", "a", "b"]
        assert [a.metadata["ab_pair"] for a in attempts] == [1, 1, 2, 2]
        assert a.calls == 2 and b.calls == 2


# --- Scoreboard (derived from history) --------------------------------------

def test_summary_math_on_synthetic_history():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        test = "arm_a-vs-arm_b"
        _commit_trial(history, test, "a", 1, 0.8)
        _commit_trial(history, test, "b", 1, 0.6)
        _commit_trial(history, test, "a", 2, 0.5)
        _commit_trial(history, test, "b", 2, 0.7)
        _commit_trial(history, test, "a", 3, 0.9)
        _commit_trial(history, test, "b", 3, 0.0, completed=False)  # b failed
        _commit_trial(history, test, "a", 4, 0.4)  # unmatched trial
        _commit_trial(history, "other-test", "a", 1, 1.0)  # different test

        toolkit = _toolkit(history)
        ab = ABTest(ArmA(scores=[0.0]), ArmB(scores=[0.0]))
        s = ab.summary(toolkit)

        assert s["test"] == test
        assert s["arm_a"] == "arm_a" and s["arm_b"] == "arm_b"
        assert s["trials_a"] == 4 and s["trials_b"] == 3
        assert s["pairs"] == 3
        assert s["wins_a"] == 2  # pair 1 on score, pair 3 because b failed
        assert s["wins_b"] == 1
        assert s["ties"] == 0
        assert abs(s["mean_a"] - (0.8 + 0.5 + 0.9) / 3) < 1e-9
        assert abs(s["mean_b"] - (0.6 + 0.7) / 2) < 1e-9
        assert abs(s["mean_delta"] - 0.0) < 1e-9  # (+0.2 - 0.2) / 2


def test_verdict_logged_after_min_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        test = "arm_a-vs-arm_b"
        for pair, (sa, sb) in enumerate([(0.8, 0.6), (0.7, 0.5)], start=1):
            _commit_trial(history, test, "a", pair, sa)
            _commit_trial(history, test, "b", pair, sb)
        toolkit = _toolkit(history)

        ab = ABTest(ArmA(scores=[0.9], fresh=True), ArmB(scores=[0.6], fresh=True),
                    min_pairs=3)
        out = ab(toolkit)

        assert out["summary"]["pairs"] == 3
        assert out["verdict"] == "ABTest arm_a-vs-arm_b: A better on 3/3 pairs, mean +0.233"
        assert out["verdict"] in toolkit.log.lines


def test_explicit_test_name_scopes_the_scoreboard():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit_trial(history, "my-test", "a", 1, 0.9)
        _commit_trial(history, "my-test", "b", 1, 0.1)
        _commit_trial(history, "arm_a-vs-arm_b", "a", 1, 0.2)

        toolkit = _toolkit(history)
        ab = ABTest(ArmA(scores=[0.0]), ArmB(scores=[0.0]), test_name="my-test")
        s = ab.summary(toolkit)
        assert s["test"] == "my-test"
        assert s["pairs"] == 1 and s["wins_a"] == 1
