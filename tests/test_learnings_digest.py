"""Learnings as ledger + lens.

The per-attempt learning record is the LEDGER (immutable, travels with
the attempt); the run-root learnings.md is a DERIVED digest rebuildable
from the ledger. These tests pin: the standard strategy finish records
the entry in the attempt, the rebuild is deterministic without an LLM,
respects dedupe + caps, reads agent scratchpads, and MarkdownLearnings
keeps reading the digest unchanged.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from groundhog import Task, Toolkit
from groundhog.base.types import (
    Context,
    Data,
    EvalStage,
    EvaluationResult,
    Evaluator,
    StageResult,
)
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.learnings.markdown import MarkdownLearnings
from groundhog.utils.direction import write_direction
from groundhog.utils.learnings_digest import (
    DIGEST_HEADER,
    SEPARATOR,
    attempt_learnings,
    rebuild_digest,
    record_attempt_learning,
)
from groundhog.utils.results import write_result


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
            EvalStage(
                "eval",
                "eval",
                lambda cp: StageResult(metrics={"score": 0.5}),
                scorer=lambda r: r.metrics.get("score", 0.0),
            )
        ]


def _task():
    return Task(data=_Data(), context=_Ctx(), evaluator=_Eval(), name="t")


def _ok_result(score=0.5):
    return EvaluationResult(stages={"eval": StageResult(metrics={"score": score})})


def _commit(history, *, learnings=None, work_learnings=None, direction=None,
            parent=None, success=True, code="print(1)"):
    ws = history.workspace(parent=parent)
    (ws.path / "solution.py").write_text(code, encoding="utf-8")
    if direction:
        write_direction(ws.path, direction)
    if learnings:
        for entry in learnings:
            record_attempt_learning(ws.path, entry)
    if work_learnings is not None:
        (ws.path / "work" / "learnings.md").write_text(
            work_learnings, encoding="utf-8")
    write_result(ws.path, _ok_result(), metadata={"strategy": "test"})
    return ws.commit(success=success)


# === Ledger: per-attempt records =========================================

def test_record_attempt_learning_appends_entries():
    with tempfile.TemporaryDirectory() as tmp:
        record_attempt_learning(tmp, "first note")
        record_attempt_learning(tmp, "second note")
        record_attempt_learning(tmp, "   ")  # blank entries are dropped

        text = (Path(tmp) / "learnings.md").read_text(encoding="utf-8")
        assert text.split(SEPARATOR) == ["first note", "second note\n"]


def test_attempt_learnings_reads_root_and_work_files():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        attempt = _commit(history, learnings=["root note"],
                          work_learnings="agent note", direction="rollout")

        assert attempt_learnings(attempt) == ["root note", "agent note"]


def test_attempt_learnings_strips_agent_seed():
    from groundhog.utils.learnings_digest import LEARNINGS_SEED

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        untouched = _commit(history, work_learnings=LEARNINGS_SEED,
                            direction="rollout")
        appended = _commit(history, work_learnings=LEARNINGS_SEED + "\nreal note",
                           direction="mcts")

        assert attempt_learnings(untouched) == []
        assert attempt_learnings(appended) == ["real note"]


def test_improve_records_learning_in_attempt_and_digest():
    """The standard Improve finish leaves the learning both in the committed
    attempt (the ledger) and in the run-root file (the compat digest)."""
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.improve import Improve

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, direction="rollout")

        toolkit = Toolkit(task=_task(), history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(
            ["no changes", "- learned: smaller is better"]))
        toolkit.learnings = MarkdownLearnings(Path(tmp))

        Improve()(toolkit)

        attempt = history.list()[-1]
        assert attempt.metadata["strategy"] == "improve"
        assert attempt_learnings(attempt) == ["- learned: smaller is better"]
        assert "- learned: smaller is better" in toolkit.learnings.get()


def test_improve_records_learnings_used_ledger():
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.improve import Improve
    from groundhog.utils.learnings_digest import learnings_used

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, direction="rollout")

        toolkit = Toolkit(task=_task(), history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(["no changes"]))
        toolkit.learnings = MarkdownLearnings(Path(tmp))
        toolkit.learnings.add("prefer small kernels")

        Improve()(toolkit)

        attempt = history.list()[-1]
        assert learnings_used(attempt) == ["prefer small kernels"]


def test_fresh_records_learnings_used_ledger():
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.fresh import FreshApproach
    from groundhog.utils.learnings_digest import learnings_used

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))

        toolkit = Toolkit(task=_task(), history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(
            ["```python\nprint(1)\n```", "greedy heuristic"]))
        toolkit.learnings = MarkdownLearnings(Path(tmp))
        toolkit.learnings.add("avoid deep nets")

        FreshApproach()(toolkit, {"mode": "different"})

        attempt = history.list()[-1]
        assert learnings_used(attempt) == ["avoid deep nets"]


def test_no_learnings_consumed_leaves_ledger_absent():
    from groundhog.backends.mock import MockBackend
    from groundhog.base.backend import BackendRegistry
    from groundhog.strategies.improve import Improve
    from groundhog.utils.learnings_digest import (
        LEARNINGS_USED_FILENAME, learnings_used,
    )

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, direction="rollout")

        toolkit = Toolkit(task=_task(), history=history)
        toolkit.llm = BackendRegistry(default=MockBackend(["no changes"]))
        toolkit.learnings = MarkdownLearnings(Path(tmp))

        Improve()(toolkit)

        attempt = history.list()[-1]
        assert attempt.read_file(LEARNINGS_USED_FILENAME) is None
        assert learnings_used(attempt) == []


def test_seed_carries_no_fabricated_observations():
    """The seed's examples used to embed concrete fake metrics (0.90->0.86,
    lr 1e-2 -> NaN) that promotion surfaced as real observations
    (remediation seed): placeholder-shaped examples only."""
    from groundhog.utils.learnings_digest import LEARNINGS_SEED

    assert "0.90" not in LEARNINGS_SEED
    assert "1e-2" not in LEARNINGS_SEED
    assert "[specific change X]" in LEARNINGS_SEED


def test_collect_learnings_strips_seed_block_on_promotion():
    from groundhog.strategies.agent import AgentStrategy
    from groundhog.utils.learnings_digest import LEARNINGS_SEED

    with tempfile.TemporaryDirectory() as tmp:
        ws_dir = Path(tmp) / "ws"
        (ws_dir / "work").mkdir(parents=True)
        (ws_dir / "work" / "learnings.md").write_text(
            LEARNINGS_SEED + "\n- tried dropout 0.5 -> acc up 2% -> keep it\n",
            encoding="utf-8")

        toolkit = SimpleNamespace(
            learnings=MarkdownLearnings(Path(tmp) / "store"))
        ws = SimpleNamespace(path=ws_dir)
        AgentStrategy()._collect_learnings(toolkit, ws)

        promoted = toolkit.learnings.get()
        assert "tried dropout 0.5" in promoted
        assert "[specific change X]" not in promoted
        assert "# Learnings" not in promoted

        # A file left as the untouched seed promotes nothing.
        untouched_tk = SimpleNamespace(
            learnings=MarkdownLearnings(Path(tmp) / "store2"))
        (ws_dir / "work" / "learnings.md").write_text(
            LEARNINGS_SEED, encoding="utf-8")
        AgentStrategy()._collect_learnings(untouched_tk, ws)
        assert untouched_tk.learnings.get() == ""


# === Lens: rebuild_digest =================================================

def test_rebuild_digest_is_deterministic_and_marked_derived():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, learnings=["note a"], direction="rollout")
        _commit(history, learnings=["note b"], direction="rollout")

        path = Path(tmp) / "learnings.md"
        first = rebuild_digest(history, path)
        second = rebuild_digest(history, path)

        assert first == second
        assert first.startswith(DIGEST_HEADER)
        assert path.read_text(encoding="utf-8") == first
        # Newest first within the family.
        assert first.index("note b") < first.index("note a")


def test_rebuild_digest_dedupes_exact_matches_and_caps():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, learnings=["same note"], direction="rollout")
        _commit(history, learnings=["same  note"], direction="rollout")  # ws-equal
        _commit(history, learnings=["note 1", "note 2", "note 3"],
                direction="mcts")

        path = Path(tmp) / "learnings.md"

        text = rebuild_digest(history, path)
        assert text.count("same") == 1  # whitespace-insensitive exact dedupe

        capped = rebuild_digest(history, path, max_entries=2)
        body = capped[len(DIGEST_HEADER):].strip()
        entries = [e for e in body.split(SEPARATOR) if e.strip()]
        assert len(entries) == 2


def test_rebuild_digest_groups_by_family_and_keeps_failures():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        a1 = _commit(history, learnings=["rollout insight"], direction="rollout")
        _commit(history, learnings=["mcts insight"], direction="mcts")
        _commit(history, learnings=["rollout follow-up"], direction="rollout",
                parent=a1.id)
        _commit(history, learnings=["failed but informative"],
                direction="rollout", success=False)

        text = rebuild_digest(history, Path(tmp) / "learnings.md")

        assert "failed but informative" in text
        assert "[attempt" in text and "| rollout]" in text and "| mcts]" in text
        # Grouped: both rollout entries are contiguous (mcts not between them).
        rollout_positions = [text.index("rollout insight"),
                             text.index("rollout follow-up")]
        mcts_position = text.index("mcts insight")
        assert not (min(rollout_positions) < mcts_position < max(rollout_positions))


def test_rebuild_digest_header_never_glues_into_entries():
    """The header used to join the first entry with a bare blank line, so
    every prompt embedding the digest carried the comment (remediation):
    the header now joins with the standard separator and the reader drops
    it as file furniture."""
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, learnings=["note a"], direction="rollout")

        text = rebuild_digest(history, Path(tmp) / "learnings.md")
        assert text.startswith(DIGEST_HEADER + SEPARATOR)

        learnings = MarkdownLearnings(Path(tmp))
        assert learnings.count() == 1
        assert DIGEST_HEADER not in learnings.get()
        assert DIGEST_HEADER not in learnings.get(last=1)


def test_markdown_learnings_reads_rebuilt_digest_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, learnings=["note a"], direction="rollout")
        _commit(history, learnings=["note b"], direction="mcts")

        learnings = MarkdownLearnings(Path(tmp))
        rebuild_digest(history, learnings._path)

        assert learnings.count() == 2
        assert "note a" in learnings.get()
        assert "note b" in learnings.get(last=2)
        # add() keeps working on top of the rebuilt digest.
        learnings.add("post-rebuild note")
        assert learnings.count() == 3


def test_rebuild_digest_with_llm_is_one_merge_pass():
    calls = []

    class _FakeLLM:
        def generate(self, prompt, system_prompt=""):
            calls.append(prompt)
            return SimpleNamespace(text="strong directive one\n---\nstrong directive two")

    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        _commit(history, learnings=["note a"], direction="rollout")
        _commit(history, learnings=["note b"], direction="mcts")

        path = Path(tmp) / "learnings.md"
        text = rebuild_digest(history, path, max_entries=2, llm=_FakeLLM())

        assert len(calls) == 1
        assert "note a" in calls[0] and "note b" in calls[0]
        assert text.startswith(DIGEST_HEADER)
        # LLM entries are normalized to the standard separator convention.
        assert MarkdownLearnings(Path(tmp)).count() == 2


def test_rebuild_digest_empty_history_writes_header_only():
    with tempfile.TemporaryDirectory() as tmp:
        history = FolderAttemptHistory(Path(tmp))
        path = Path(tmp) / "learnings.md"
        text = rebuild_digest(history, path)
        assert text == DIGEST_HEADER + "\n"


# === CLI: groundhog learnings =============================================

_TASK_BODY = '''
from pathlib import Path
from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult


class TinyData(Data):
    def get_train(self): return None
    def get_test(self): return None


class TinyContext(Context):
    def get_brief(self): return "b"
    def get_extended(self): return "e"


class TinyEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        return StageResult(metrics={"score": 0.5})

    def get_stages(self, data):
        return [EvalStage("eval", "eval", lambda cp: self.evaluate(cp, data),
                          scorer=lambda r: r.metrics.get("score", 0.0))]


task = Task(data=TinyData(), context=TinyContext(), evaluator=TinyEvaluator(),
            name="TinyTask")


def build_toolkit():
    from groundhog import FolderAttemptHistory, assemble_toolkit
    here = Path(__file__).parent
    return assemble_toolkit(task, history=FolderAttemptHistory(here), path=here)
'''


@contextmanager
def _in_dir(path):
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


def _cli_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(_TASK_BODY, encoding="utf-8")
    history = FolderAttemptHistory(run_dir)
    _commit(history, learnings=["target is 50"], direction="rollout")
    return run_dir


def test_cli_learnings_rebuild_and_list(tmp_path, capsys):
    from groundhog.cli import learnings_group

    run_dir = _cli_run_dir(tmp_path)
    with _in_dir(run_dir):
        assert learnings_group(["rebuild"]) == 0
        out = capsys.readouterr().out
        assert "Rebuilt digest: 1 entries" in out
        digest = (run_dir / "learnings.md").read_text(encoding="utf-8")
        assert digest.startswith(DIGEST_HEADER)
        assert "target is 50" in digest

        assert learnings_group(["list"]) == 0
        out = capsys.readouterr().out
        assert "target is 50" in out

        assert learnings_group(["list", "--attempt", "1"]) == 0
        out = capsys.readouterr().out
        assert "target is 50" in out


def test_cli_learnings_list_missing_attempt_and_empty(tmp_path, capsys):
    from groundhog.cli import learnings_group

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(_TASK_BODY, encoding="utf-8")
    with _in_dir(run_dir):
        assert learnings_group(["list"]) == 0
        assert "No per-attempt learnings" in capsys.readouterr().out

        assert learnings_group(["list", "--attempt", "99"]) == 1
        assert "No such attempt" in capsys.readouterr().out

        assert learnings_group(["bogus"]) == 1
        assert "Unknown learnings subcommand" in capsys.readouterr().out
