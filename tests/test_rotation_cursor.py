"""The rotation cursor persists across optimizer invocations.

Store-derive rule: the schedule position matters (a chunked campaign of
`groundhog run -n 1` calls must continue the weighted rotation, not
restart it at slot 0), so it is STORED — rotation_state.json in the run
dir, beside queue.json, advanced under the shared file lock. Deleting
the file resets the rotation to the schedule start.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from groundhog import SimpleOptimizer, Strategy, Task
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    Evaluator,
    StageResult,
)
from groundhog.histories.folder import FolderAttemptHistory


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


class Recorder(Strategy):
    """No-op strategy that records its label into a shared list."""

    def __init__(self, label, calls):
        super().__init__()
        self.label = label
        self.calls = calls

    def __call__(self, toolkit, config=None):
        self.calls.append(self.label)
        return {}


def _toolkit(tmp_path):
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    history = FolderAttemptHistory(tmp_path / "store")
    return SimpleNamespace(task=task, history=history, log=_Log(),
                           path=tmp_path)


def _optimizer(tk, calls):
    return SimpleOptimizer(
        tk,
        strategies=[(Recorder("A", calls), 2), (Recorder("B", calls), 1)],
        seed_strategy=None,
    )


def test_rotation_continues_across_invocations(tmp_path):
    calls = []
    tk = _toolkit(tmp_path)

    _optimizer(tk, calls).run(n=2)   # slots 0,1 -> A, A
    _optimizer(tk, calls).run(n=2)   # slots 2,0 -> B, A
    _optimizer(tk, calls).run(n=1)   # slot 1    -> A

    assert calls == ["A", "A", "B", "A", "A"]
    state = json.loads(
        (tmp_path / "rotation_state.json").read_text(encoding="utf-8"))
    assert state == {"position": 2}


def test_deleting_the_state_file_resets_the_rotation(tmp_path):
    calls = []
    tk = _toolkit(tmp_path)

    _optimizer(tk, calls).run(n=2)   # A, A
    (tmp_path / "rotation_state.json").unlink()
    _optimizer(tk, calls).run(n=2)   # reset -> A, A

    assert calls == ["A", "A", "A", "A"]


def test_pathless_toolkit_keeps_in_memory_rotation(tmp_path):
    calls = []
    task = Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")
    tk = SimpleNamespace(task=task,
                         history=FolderAttemptHistory(tmp_path / "store"),
                         log=_Log(), path=None)

    _optimizer(tk, calls).run(n=3)
    _optimizer(tk, calls).run(n=3)

    # Each invocation restarts at the schedule start — the documented
    # fallback when there is no run dir to persist into.
    assert calls == ["A", "A", "B", "A", "A", "B"]
    assert not Path("rotation_state.json").exists()


def test_queue_overrides_do_not_advance_the_cursor(tmp_path):
    from groundhog.tools.queue import add as queue_add

    calls = []
    tk = _toolkit(tmp_path)
    queued = Recorder("Q", calls)
    queued.name = "queued"

    opt = SimpleOptimizer(
        tk,
        strategies=[(Recorder("A", calls), 1), (Recorder("B", calls), 1)],
        extras=[queued],
        seed_strategy=None,
    )
    queue_add(tmp_path, "queued", source="user")
    opt.run(n=2)  # queue item first (no advance), then rotation slot 0

    assert calls == ["Q", "A"]
    state = json.loads(
        (tmp_path / "rotation_state.json").read_text(encoding="utf-8"))
    assert state == {"position": 1}
