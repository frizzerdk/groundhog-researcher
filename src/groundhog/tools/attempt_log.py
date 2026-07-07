"""Per-attempt event log.

Anything that happens during one attempt — agent tool calls, host-side
events, evaluator phases, manual notes — flows through ``AttemptLog``.
The log owns the rendering strategy so callers don't care whether output
is going to a live terminal (in-place two-pane region) or a CI log file
(plain appended text).

Public surface is small and stable:

    log = AttemptLog(cfg)
    log.attempt_start(num=271, prior=230, queue_label="...")
    log.update(phase="explore", elapsed_s=124.5, budget_used=0.42, ...)
    log.event(source="agent", kind="tool_call", summary="read solution.py")
    log.attempt_done(score=1544.1, delta=-1019.5,
                     total_cost=3.66, cumulative_cost=7.90,
                     summary_line="win=0.54 Δ-0.05 ...")

Anyone holding a toolkit gets to ``toolkit.attempt_log`` and emits.

Renderer selection
------------------
- ``TwoPaneRenderer`` — used when stdout is a TTY. Persistent box at the
  top, fixed-height tail underneath, both updated in place via ANSI
  cursor moves. A daemon heartbeat refreshes the elapsed/budget lines
  during silent stretches (model thinking, network stall, eval running).
- ``AppendedRenderer`` — used when stdout is not a TTY (CI logs, file
  redirects). Same data, no ANSI; events scroll naturally.

Both renderers consume the same event stream so the only thing that
varies is presentation.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import IO, Optional


# --- Honest cost formatting ----------------------------------------------

def format_attempt_cost(cost: float, cost_model: str = "per_token") -> str:
    """Render an attempt's cost honestly, per how its backend charges.

    Backends differ: an API-billed run reports real dollars, a subscription
    run reports request-credit "value", and some report nothing at all. A
    single ``$0.00`` across all three lies about the last two — this maps
    ``cost_model`` (recorded in attempt metadata) to a truthful string.
    """
    if cost_model == "none":
        return "unreported (subscription)"
    if cost_model == "per_request":
        return f"${cost:.4f} (plan value)"
    return f"${cost:.4f}"


# --- ANSI helpers ---------------------------------------------------------

class _ANSI:
    R = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    MAGENTA = "\033[35m"
    CLEAR_LINE = "\033[2K"

    @staticmethod
    def up(n: int) -> str:
        return f"\033[{n}A" if n > 0 else ""


_GLYPH = {
    "tool_call":   "→",
    "tool_result": "✓",
    "thinking":    "↻",
    "edit":        "✎",
    "phase":       "▸",
    "error":       "⚠",
    "info":        "·",
}
_ASCII = {
    "tool_call":   "->",
    "tool_result": "<-",
    "thinking":    "..",
    "edit":        " +",
    "phase":       " >",
    "error":       "!!",
    "info":        " i",
}
_SOURCE_COLOR = {
    "agent": _ANSI.CYAN,
    "eval":  _ANSI.GREEN,
    "host":  _ANSI.YELLOW,
}


# --- Config + Event ------------------------------------------------------

@dataclass
class AttemptLogConfig:
    """Knobs for the AttemptLog. Defaults auto-disable on non-TTY."""

    color: bool = True
    glyphs: bool = True
    tail_lines: int = 6
    timestamp_style: str = "elapsed"   # "elapsed" | "delta"
    sources: tuple = ("agent", "eval", "host")
    show_thinking: bool = True
    width: int = 0                     # 0 = auto-fit terminal, clamped [60, 120]
    heartbeat_seconds: float = 10.0    # 0 disables
    summary_max_chars: int = 200       # for the post-attempt 1-line summary

    @classmethod
    def auto(cls, out: IO = sys.stdout) -> "AttemptLogConfig":
        """Pick defaults that match the output stream — no color/glyphs/heartbeat
        when stdout isn't a TTY."""
        is_tty = bool(getattr(out, "isatty", lambda: False)())
        if not is_tty:
            return cls(color=False, glyphs=False, heartbeat_seconds=0.0)
        # GROUNDHOG_NO_COLOR overrides like NO_COLOR (https://no-color.org/)
        if os.environ.get("NO_COLOR") or os.environ.get("GROUNDHOG_NO_COLOR"):
            return cls(color=False)
        return cls()


@dataclass
class AttemptEvent:
    source: str       # "agent" | "eval" | "host"
    kind: str         # "tool_call" | "tool_result" | "thinking" | "edit"
                      # | "phase" | "error" | "info"
    summary: str
    t: float = 0.0    # elapsed seconds since attempt_start


# --- AttemptLog (the public façade) --------------------------------------

class AttemptLog:
    """Per-attempt log. Holds state + delegates rendering to a Renderer."""

    def __init__(self, cfg: Optional[AttemptLogConfig] = None,
                 out: IO = sys.stdout, renderer: Optional["Renderer"] = None):
        self.cfg = cfg or AttemptLogConfig.auto(out)
        self.out = out
        self.events: deque[AttemptEvent] = deque(maxlen=self.cfg.tail_lines)
        self.state = _initial_state()
        self._t0: Optional[float] = None
        self._heartbeat: Optional[_Heartbeat] = None
        self._lock = threading.Lock()

        if renderer is None:
            is_tty = bool(getattr(out, "isatty", lambda: False)())
            renderer = TwoPaneRenderer(self) if is_tty else AppendedRenderer(self)
        self.renderer = renderer

    # ---- public API ----

    def attempt_start(self, num: int, prior: Optional[int] = None,
                      queue_label: str = "", budget_total: float = 0.0):
        with self._lock:
            self.state = _initial_state()
            self.state.update(attempt_num=num, prior_num=prior,
                              queue_label=queue_label,
                              phase="starting", budget_total=budget_total)
            self.events.clear()
            self._t0 = time.monotonic()
            self.renderer.attempt_start()
            self._start_heartbeat()

    def update(self, **kwargs):
        with self._lock:
            self.state.update(kwargs)
            self.renderer.refresh()

    def event(self, source: str, kind: str, summary: str,
              t: Optional[float] = None):
        if source not in self.cfg.sources:
            return
        if kind == "thinking" and not self.cfg.show_thinking:
            return
        with self._lock:
            ev = AttemptEvent(
                source=source, kind=kind, summary=summary,
                t=t if t is not None else self._elapsed(),
            )
            self.events.append(ev)
            if kind in ("tool_call", "edit"):
                self.state["tools"] = self.state.get("tools", 0) + 1
            self.renderer.refresh()

    def attempt_done(self, attempt_num: int, score: float, delta: float,
                     total_cost: float, cumulative_cost: float,
                     summary_line: str = "", cost_model: str = "per_token"):
        with self._lock:
            self._stop_heartbeat()
            # Strategies that don't open the live box (seed, deterministic,
            # any non-Agent strategy) skip attempt_start, so populate state
            # from the args here. Idempotent for AgentStrategy.
            self.state["attempt_num"] = attempt_num
            self.renderer.attempt_done(
                score=score, delta=delta,
                total_cost=total_cost, cumulative_cost=cumulative_cost,
                summary_line=summary_line[: self.cfg.summary_max_chars],
                cost_model=cost_model,
            )

    def attempt_failed(self, attempt_num: int, stage: str, errors: str,
                       total_cost: float, cumulative_cost: float,
                       cost_model: str = "per_token"):
        with self._lock:
            self._stop_heartbeat()
            self.state["attempt_num"] = attempt_num
            self.renderer.attempt_failed(
                stage=stage, errors=errors,
                total_cost=total_cost, cumulative_cost=cumulative_cost,
                cost_model=cost_model,
            )

    # ---- internal ----

    def _elapsed(self) -> float:
        return (time.monotonic() - self._t0) if self._t0 else 0.0

    def _heartbeat_tick(self):
        with self._lock:
            self.state["elapsed_s"] = self._elapsed()
            self.renderer.refresh()

    def _start_heartbeat(self):
        if self.cfg.heartbeat_seconds <= 0:
            return
        if self._heartbeat is not None:
            self._heartbeat.stop()
        self._heartbeat = _Heartbeat(
            interval=self.cfg.heartbeat_seconds, callback=self._heartbeat_tick)
        self._heartbeat.start()

    def _stop_heartbeat(self):
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None


# --- Renderer base + implementations -------------------------------------

class Renderer:
    """Pluggable rendering strategy. Subclasses implement the four hooks."""

    def attempt_start(self): ...
    def refresh(self): ...
    def attempt_done(self, score: float, delta: float,
                     total_cost: float, cumulative_cost: float,
                     summary_line: str, cost_model: str = "per_token"): ...
    def attempt_failed(self, stage: str, errors: str,
                       total_cost: float, cumulative_cost: float,
                       cost_model: str = "per_token"): ...


class TwoPaneRenderer(Renderer):
    """TTY: persistent header box + fixed-height in-place tail."""

    def __init__(self, log: AttemptLog):
        self.log = log
        self._lines_drawn = 0

    def attempt_start(self):
        self._lines_drawn = 0
        self._draw(initial=True)

    def refresh(self):
        self._draw()

    def attempt_done(self, score, delta, total_cost, cumulative_cost,
                     summary_line, cost_model="per_token"):
        # Freeze the live region — leave the box+tail in scrollback as history.
        self._lines_drawn = 0
        out = self.log.out
        c = self._color
        sign = "+" if delta >= 0 else ""
        marker = c(_ANSI.GREEN, " ★") if delta > 0 else ""
        score_str = c(_ANSI.BOLD, f"{score:.1f}")
        n = self.log.state.get("attempt_num")
        cost_str = format_attempt_cost(total_cost, cost_model)
        out.write(f"  [{n:>3}] {score_str} ({sign}{delta:.4f}){marker}  "
                  f"{cost_str} ({c(_ANSI.DIM, f'${cumulative_cost:.4f}')})\n")
        if summary_line:
            out.write(f"         {c(_ANSI.DIM, summary_line)}\n")
        out.write("\n")
        out.flush()

    def attempt_failed(self, stage, errors, total_cost, cumulative_cost,
                       cost_model="per_token"):
        self._lines_drawn = 0
        cfg = self.log.cfg
        out = self.log.out
        c = self._color
        n = self.log.state.get("attempt_num")
        msg = errors[: cfg.summary_max_chars]
        cost_str = format_attempt_cost(total_cost, cost_model)
        out.write(f"  [{n:>3}] {c(_ANSI.RED, 'FAIL')}  {stage}: {msg}  "
                  f"{cost_str} ({c(_ANSI.DIM, f'${cumulative_cost:.4f}')})\n\n")
        out.flush()

    # ---- drawing ----

    def _draw(self, initial: bool = False):
        log = self.log
        cfg = log.cfg
        out = log.out
        if not initial and self._lines_drawn:
            out.write(_ANSI.up(self._lines_drawn))

        c = self._color
        w = self._effective_width()
        inner = w - 2
        lines: list[str] = []

        # Title row
        s = log.state
        title = f" #{s['attempt_num']}"
        if s["prior_num"] is not None:
            title += f" prior=#{s['prior_num']}"
        if s["queue_label"]:
            title += f" · {s['queue_label']}"
        # Box-line char count = ╭ + ─ + title + rule + ╮ = 3 + |title| + |rule|.
        # Total must equal w, so |title| + |rule| = w - 3 = inner - 1.
        title = self._truncate(title + " ", inner - 1)
        rule = "─" * (inner - 1 - len(title))
        lines.append(f"╭─{c(_ANSI.BOLD, title)}{rule}╮")

        # Status rows
        elapsed = self._fmt_time(s["elapsed_s"])
        l1_left = f" phase: {c(_ANSI.MAGENTA, str(s['phase']))}"
        l1_right = f"elapsed: {c(_ANSI.BOLD, elapsed)} "
        lines.append(self._row(l1_left, l1_right, inner))

        used = s.get("budget_used", 0.0)
        total = s.get("budget_total", 0.0) or 0.0
        pct = (used / total * 100) if total else 0.0
        bcolor = _ANSI.RED if pct > 80 else (_ANSI.YELLOW if pct > 50 else _ANSI.GREEN)
        used_str = c(bcolor, f"${used:.2f}")
        l2_left = f" budget: {used_str}/${total:.2f}"
        l2_right = f"turns: {s.get('turns', 0)} "
        lines.append(self._row(l2_left, l2_right, inner))

        l3_left = f" tokens: {self._k(s.get('tokens_in', 0))} → {self._k(s.get('tokens_out', 0))}"
        l3_right = f"tools: {s.get('tools', 0)} "
        lines.append(self._row(l3_left, l3_right, inner))

        lines.append(f"╰{'─' * inner}╯")

        # Tail (always exactly cfg.tail_lines rows)
        max_summary = w - 18
        tail: list[str] = []
        for ev in log.events:
            ts = c(_ANSI.GRAY, f"+{self._fmt_time(ev.t)}")
            src = c(_SOURCE_COLOR.get(ev.source, _ANSI.GRAY), f"{ev.source:<5}")
            glyph = self._glyph(ev.kind)
            summary = self._truncate(ev.summary, max_summary)
            tail.append(f"  {ts} {src} {glyph} {summary}")
        while len(tail) < cfg.tail_lines:
            tail.append("")
        lines.extend(tail[: cfg.tail_lines])

        prefix = _ANSI.CLEAR_LINE if not initial else ""
        for line in lines:
            out.write(prefix + line + "\n")
        out.flush()
        self._lines_drawn = len(lines)

    # ---- helpers ----

    def _color(self, code: str, text: str) -> str:
        return f"{code}{text}{_ANSI.R}" if self.log.cfg.color else text

    def _glyph(self, kind: str) -> str:
        table = _GLYPH if self.log.cfg.glyphs else _ASCII
        return table.get(kind, "·" if self.log.cfg.glyphs else " .")

    def _effective_width(self) -> int:
        if self.log.cfg.width:
            return self.log.cfg.width
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            cols = 80
        return max(60, min(120, cols))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:>4.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m:>2}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars == 1:
            return "…"
        return text[: max_chars - 1] + "…"

    @staticmethod
    def _row(left: str, right: str, inner: int) -> str:
        bare_left = _strip_ansi(left)
        bare_right = _strip_ansi(right)
        pad = max(0, inner - len(bare_left) - len(bare_right))
        return f"│{left}{' ' * pad}{right}│"

    @staticmethod
    def _k(n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 1_000_000:
            return f"{n/1000:.0f}k"
        return f"{n/1_000_000:.1f}M"


class AppendedRenderer(Renderer):
    """Non-TTY: append events as they arrive, no ANSI, no in-place magic.

    Same event stream → flat log lines, plus phase-boundary markers and a
    final summary line. Best for CI logs / file redirects.
    """

    def __init__(self, log: AttemptLog):
        self.log = log
        self._last_phase = ""
        self._last_event_idx = 0  # how many events we've already printed

    def attempt_start(self):
        s = self.log.state
        out = self.log.out
        # Match the TwoPaneRenderer's `[{n:>3}]` padding so attempt
        # numbers align between the live and CI output styles.
        title = f"[{s['attempt_num']:>3}] start"
        if s["prior_num"] is not None:
            title += f" prior=#{s['prior_num']}"
        if s["queue_label"]:
            title += f" · {s['queue_label']}"
        out.write(f"\n{title}\n")
        if s.get("budget_total"):
            out.write(f"  budget=${s['budget_total']:.2f}\n")
        out.flush()
        self._last_phase = ""
        self._last_event_idx = 0

    def refresh(self):
        s = self.log.state
        out = self.log.out
        if s["phase"] != self._last_phase:
            out.write(f"  phase: {s['phase']}\n")
            self._last_phase = s["phase"]
        # Print only new events (those past _last_event_idx).
        events = list(self.log.events)
        # Note: events is bounded by tail_lines; once trimmed we lose old ones,
        # so for AppendedRenderer we accept that we only print as long as the
        # tail can hold them. Strategies emit events as they happen, not in
        # batches, so the deque almost never overflows mid-tick.
        for ev in events[self._last_event_idx :]:
            ts = TwoPaneRenderer._fmt_time(ev.t)
            out.write(f"    +{ts}  {ev.source:<5}  {ev.kind:<11} {ev.summary}\n")
        self._last_event_idx = len(events)
        out.flush()

    def attempt_done(self, score, delta, total_cost, cumulative_cost,
                     summary_line, cost_model="per_token"):
        out = self.log.out
        sign = "+" if delta >= 0 else ""
        n = self.log.state.get("attempt_num")
        cost_str = format_attempt_cost(total_cost, cost_model)
        out.write(f"  [{n:>3}] {score:.1f} ({sign}{delta:.4f})  "
                  f"{cost_str} (${cumulative_cost:.4f})\n")
        if summary_line:
            out.write(f"         {summary_line}\n")
        out.write("\n")
        out.flush()

    def attempt_failed(self, stage, errors, total_cost, cumulative_cost,
                       cost_model="per_token"):
        out = self.log.out
        n = self.log.state.get("attempt_num")
        msg = errors[: self.log.cfg.summary_max_chars]
        cost_str = format_attempt_cost(total_cost, cost_model)
        out.write(f"  [{n:>3}] FAIL  {stage}: {msg}  "
                  f"{cost_str} (${cumulative_cost:.4f})\n\n")
        out.flush()


# --- Heartbeat ------------------------------------------------------------

class _Heartbeat:
    """Daemon thread that fires ``callback`` every ``interval`` seconds."""

    def __init__(self, interval: float, callback):
        self.interval = interval
        self.callback = callback
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.callback()
            except Exception:
                # Heartbeat errors must not propagate or kill the run.
                pass


# --- Helpers --------------------------------------------------------------

def _initial_state() -> dict:
    return dict(
        attempt_num=None, prior_num=None, queue_label="",
        phase="-", elapsed_s=0.0,
        budget_used=0.0, budget_total=0.0,
        tokens_in=0, tokens_out=0,
        turns=0, tools=0,
    )


def _strip_ansi(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)
