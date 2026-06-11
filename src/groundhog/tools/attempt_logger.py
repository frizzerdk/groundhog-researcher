"""Attempt logger — the per-attempt event stream.

Vault: Attempt Log.md (concept), Markdown Attempt Logger.md (this design).

Everything that happens during an attempt becomes a LogEvent. Events know
how to render themselves (to_markdown / to_console); the logger fans each
event out to three consumers: attemptlog.jsonl (append-only source of
truth), attemptlog.md (human view, re-rendered from the jsonl), and an
optional live console renderer (AttemptLog).

Unknown event types degrade gracefully: load_event falls back to the base
LogEvent so old readers can parse files written by newer code.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class LogEvent:
    type: str
    timestamp: str = ""
    cost: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        head = f"**{self.type}**"
        if self.cost:
            head += f" *(${self.cost:.4f})*"
        body = json.dumps(self.data, ensure_ascii=False) if self.data else ""
        return f"{head}:\n\n{body}\n"

    def to_console(self) -> Optional[Tuple[str, str, str]]:
        """(source, kind, summary) for the live renderer, or None to skip."""
        summary = json.dumps(self.data, ensure_ascii=False)[:80] if self.data else self.type
        return ("host", "info", summary)


@dataclass
class UserEvent(LogEvent):
    type: str = "user"
    content: str = ""

    def to_markdown(self):
        return f"**User**:\n\n{self.content}\n"

    def to_console(self):
        first = self.content.strip().split("\n")[0][:80]
        return ("host", "info", f"prompt: {first}")


@dataclass
class SystemEvent(LogEvent):
    type: str = "system"
    content: str = ""

    def to_markdown(self):
        return f"**System**:\n\n{self.content}\n"

    def to_console(self):
        return None


@dataclass
class AssistantEvent(LogEvent):
    type: str = "assistant"
    content: str = ""
    role: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self):
        head = f"**{self.role or 'Assistant'}**"
        if self.data.get("channel") == "thinking":
            head += " *(thinking)*"
        if self.cost:
            head += f" *(${self.cost:.4f})*"
        return f"{head}:\n\n{self.content}\n"

    def to_console(self):
        first = self.content.strip().split("\n")[0][:120]
        return ("agent", "thinking", first) if first else None


@dataclass
class ToolCallEvent(LogEvent):
    type: str = "tool_call"
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self):
        args = json.dumps(self.args, ensure_ascii=False) if self.args else ""
        return f"> tool: **{self.name}** {args}\n"

    def to_console(self):
        # Long absolute paths bury the filename in the live tail — show the
        # basename; the full path stays in args for the record.
        path = self.args.get("path", "")
        if path:
            detail = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        else:
            detail = str(self.args.get("command", ""))[:60]
        summary = f"{self.name} {detail}".strip()
        return ("agent", "tool_call", summary)


@dataclass
class ToolResultEvent(LogEvent):
    type: str = "tool_result"
    name: str = ""
    output: str = ""

    def to_markdown(self):
        first = self.output.strip().split("\n")[0][:200]
        return f"> result **{self.name}**: {first}\n"

    def to_console(self):
        return None


@dataclass
class PhaseEvent(LogEvent):
    type: str = "phase"
    phase: str = ""

    def to_markdown(self):
        return f"---\n\n### Phase: {self.phase}\n"

    def to_console(self):
        return ("host", "phase", self.phase)


@dataclass
class EvalEvent(LogEvent):
    type: str = "eval"
    stage: str = ""
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self):
        pairs = " | ".join(f"{k}={v}" for k, v in list(self.metrics.items())[:5])
        line = f"> eval **{self.stage}**: score={self.score:.4f}"
        return f"{line} {pairs}\n" if pairs else f"{line}\n"

    def to_console(self):
        return ("host", "info", f"eval {self.stage}: {self.score:.4f}")


EVENT_TYPES = {
    "user": UserEvent,
    "system": SystemEvent,
    "assistant": AssistantEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "phase": PhaseEvent,
    "eval": EvalEvent,
}


def eval_event(result, score: Optional[float] = None) -> EvalEvent:
    """Build an EvalEvent from an EvaluationResult.

    ``score`` is the authoritative scorer-derived value; when omitted it
    falls back to the last stage's own ``.score``. Failed results always
    score -1.0.
    """
    stages = list(result.stages.values())
    name = result.failed_stage or (
        list(result.stages.keys())[-1] if result.stages else "evaluate"
    )
    if not result.completed:
        score = -1.0
    elif score is None:
        score = stages[-1].score if stages else 0.0
    return EvalEvent(stage=name, score=score,
                     metrics=stages[-1].metrics if stages else {})


def load_event(d: dict) -> LogEvent:
    """Reconstruct the right event class from a jsonl record.

    Unknown types fall back to LogEvent; unknown keys land in .data so
    files written by newer code stay readable."""
    cls = EVENT_TYPES.get(d.get("type", ""), LogEvent)
    names = {f.name for f in fields(cls)}
    known = {k: v for k, v in d.items() if k in names}
    extra = {k: v for k, v in d.items() if k not in names}
    if cls is LogEvent and "type" not in known:
        known["type"] = d.get("type", "unknown")
    event = cls(**known)
    if extra:
        event.data = {**event.data, **extra}
    return event


class MarkdownAttemptLogger:
    """Realizes the Attempt Log as attemptlog.jsonl + attemptlog.md.

    Scoped to one attempt at a time: attempt_start points it at the
    workspace; log() appends + re-renders + forwards to the console
    renderer (AttemptLog) when one is attached.
    """

    def __init__(self, console=None):
        self._console = console
        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def attempt_start(self, ws_path, **console_kwargs):
        self._path = Path(ws_path)
        self._path.mkdir(parents=True, exist_ok=True)
        if self._console is not None and console_kwargs:
            self._console.attempt_start(**console_kwargs)

    def log(self, event: LogEvent):
        if self._path is None:
            raise RuntimeError("attempt_start() must be called before log()")
        if not event.timestamp:
            event.timestamp = datetime.now().isoformat(timespec="seconds")
        with (self._path / "attemptlog.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._render_markdown()
        if self._console is not None:
            ev = event.to_console()
            if ev is not None:
                self._console.event(source=ev[0], kind=ev[1], summary=ev[2])

    def events(self) -> list:
        if self._path is None:
            return []
        jsonl = self._path / "attemptlog.jsonl"
        if not jsonl.exists():
            return []
        out = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(load_event(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def total_cost(self) -> float:
        return sum(e.cost for e in self.events())

    def _render_markdown(self):
        lines = [e.to_markdown() for e in self.events()]
        (self._path / "attemptlog.md").write_text("\n".join(lines), encoding="utf-8")

    # Console lifecycle passthroughs — no-ops without a console.

    def update(self, **kwargs):
        if self._console is not None:
            self._console.update(**kwargs)

    def attempt_done(self, **kwargs):
        if self._console is not None:
            self._console.attempt_done(**kwargs)

    def attempt_failed(self, **kwargs):
        if self._console is not None:
            self._console.attempt_failed(**kwargs)
