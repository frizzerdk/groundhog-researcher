"""Attempt logger — event round-trip, file outputs, console forwarding.

Vault: Attempt Log.md, Implementation Details/Markdown Attempt Logger.md.
Runnable directly (CI style): python tests/test_attempt_logger.py
"""

import json
import tempfile
from pathlib import Path

from groundhog.tools.attempt_logger import (
    AssistantEvent,
    EvalEvent,
    LogEvent,
    MarkdownAttemptLogger,
    PhaseEvent,
    SystemEvent,
    ToolCallEvent,
    UserEvent,
    eval_event,
    load_event,
)


def test_jsonl_is_append_only_truth():
    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger()
        log.attempt_start(tmp)
        log.log(UserEvent(content="improve the solution"))
        log.log(AssistantEvent(content="trying a CNN", role="claude", cost=0.05))

        lines = (Path(tmp) / "attemptlog.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "user"
        assert first["content"] == "improve the solution"
        assert first["timestamp"]


def test_events_round_trip_to_right_classes():
    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger()
        log.attempt_start(tmp)
        log.log(ToolCallEvent(name="evaluate", args={"path": "work/solution.py"}))
        log.log(EvalEvent(stage="validate", score=0.71, metrics={"acc": 0.71}))
        log.log(PhaseEvent(phase="explore"))

        events = log.events()
        assert [type(e) for e in events] == [ToolCallEvent, EvalEvent, PhaseEvent]
        assert events[0].name == "evaluate"
        assert events[1].score == 0.71


def test_unknown_event_type_falls_back_to_base():
    event = load_event({"type": "telemetry", "cost": 0.1, "gpu_seconds": 42})
    assert type(event) is LogEvent
    assert event.type == "telemetry"
    assert event.cost == 0.1
    assert event.data["gpu_seconds"] == 42
    assert event.to_markdown()  # base rendering still works
    assert event.to_console() is not None


def test_markdown_view_rendered_from_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger()
        log.attempt_start(tmp)
        log.log(UserEvent(content="the prompt text"))
        log.log(AssistantEvent(content="the response", role="gemini", cost=0.01))

        md = (Path(tmp) / "attemptlog.md").read_text(encoding="utf-8")
        assert "the prompt text" in md
        assert "gemini" in md
        assert "$0.0100" in md


def test_total_cost_sums_event_costs():
    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger()
        log.attempt_start(tmp)
        log.log(AssistantEvent(content="a", cost=0.02))
        log.log(AssistantEvent(content="b", cost=0.03))
        log.log(UserEvent(content="free"))
        assert abs(log.total_cost() - 0.05) < 1e-9


def test_render_marks_thinking_and_costs():
    think = AssistantEvent(content="pondering options", data={"channel": "thinking"})
    assert "(thinking)" in think.to_markdown()
    plain = AssistantEvent(content="final answer")
    assert "(thinking)" not in plain.to_markdown()
    run = LogEvent(type="agent_run", cost=0.4575, data={"turns": 3})
    assert "$0.4575" in run.to_markdown()


def test_eval_event_builder():
    from groundhog.base.types import EvaluationResult, StageResult

    ok = EvaluationResult(stages={"validate": StageResult(score=0.7, metrics={"acc": 0.7})})
    e = eval_event(ok, 0.71)
    assert (e.stage, e.score, e.metrics) == ("validate", 0.71, {"acc": 0.7})
    assert eval_event(ok).score == 0.7  # falls back to the stage's own score

    failed = EvaluationResult(stages={"smoke": StageResult(errors={"smoke": "boom"})},
                              completed=False, failed_stage="smoke")
    assert eval_event(failed).stage == "smoke"
    assert eval_event(failed).score == -1.0


def test_estimator_honors_embedded_cost():
    """OpenRouter usage carries the actual charge; table lookup is the
    fallback for providers that don't."""
    from groundhog.tools.cost_estimate import estimate_cost

    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger()
        log.attempt_start(tmp)
        log.log(AssistantEvent(content="a", role="google/gemini-3-flash-preview",
                               usage={"prompt_tokens": 100, "completion_tokens": 50,
                                      "cost": 0.0123}))
        log.log(AssistantEvent(content="b", role="gemini-2.5-flash",
                               usage={"promptTokenCount": 1000, "candidatesTokenCount": 1000}))

        est = estimate_cost(Path(tmp))
        table_priced = (1000 * 0.30 + 1000 * 2.50) / 1_000_000
        assert abs(est["total_cost"] - (0.0123 + table_priced)) < 1e-9
        assert est["unknown_models"] == []


def test_console_forwarding():
    calls = []

    class FakeConsole:
        def attempt_start(self, **kw):
            calls.append(("attempt_start", kw))

        def event(self, **kw):
            calls.append(("event", kw))

        def attempt_done(self, **kw):
            calls.append(("attempt_done", kw))

    with tempfile.TemporaryDirectory() as tmp:
        log = MarkdownAttemptLogger(console=FakeConsole())
        log.attempt_start(tmp, num=7, prior=3)
        log.log(PhaseEvent(phase="explore"))
        log.log(SystemEvent(content="hidden"))
        log.attempt_done(score=1.0)

    kinds = [c[0] for c in calls]
    assert kinds == ["attempt_start", "event", "attempt_done"]  # SystemEvent skips console
    assert calls[1][1] == {"source": "host", "kind": "phase", "summary": "explore"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
