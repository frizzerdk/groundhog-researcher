"""Mock renderer for the two-pane AttemptLog (Concept 4).

Run modes:
    uv run tools/mock_attempt_log.py snapshots   # static frames, easy to diff
    uv run tools/mock_attempt_log.py live        # animated, ANSI cursor moves
    uv run tools/mock_attempt_log.py ascii       # snapshots, no glyphs/color

Snapshots show 6 representative moments in an attempt's life so you can
judge spacing/density without actually waiting for an explore phase.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# --- ANSI ------------------------------------------------------------------

class A:
    R = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"

    @staticmethod
    def up(n: int) -> str:
        return f"\033[{n}A" if n > 0 else ""

    CLEAR_LINE = "\033[2K"
    CR = "\r"


# --- Config ----------------------------------------------------------------

@dataclass
class AttemptLogConfig:
    color: bool = True
    glyphs: bool = True
    tail_lines: int = 6
    timestamp_style: str = "elapsed"  # "elapsed" | "wall"
    sources: tuple = ("agent", "eval", "host")
    show_thinking: bool = True
    # Width: 0/None = auto-fit to terminal (clamped to [60, 120]).
    # Set to a positive int to force a specific width (handy for tests).
    width: int = 0


GLYPH = {
    "tool_call":   "→",
    "tool_result": "✓",
    "thinking":    "↻",
    "edit":        "✎",
    "phase":       "▸",
    "error":       "⚠",
    "info":        "·",
}
ASCII = {
    "tool_call":   "->",
    "tool_result": "<-",
    "thinking":    "..",
    "edit":        " +",
    "phase":       " >",
    "error":       "!!",
    "info":        " i",
}

SOURCE_COLOR = {
    "agent": A.CYAN,
    "eval":  A.GREEN,
    "host":  A.YELLOW,
}


# --- Event -----------------------------------------------------------------

@dataclass
class AttemptEvent:
    source: str       # "agent" | "eval" | "host"
    kind: str         # "tool_call" | "thinking" | "edit" | ...
    summary: str
    t: float = 0.0    # elapsed seconds since attempt start


# --- Renderer --------------------------------------------------------------

class TwoPaneRenderer:
    """Two-pane: persistent status box on top, scrolling event tail below."""

    def __init__(self, cfg: AttemptLogConfig, out=sys.stdout):
        self.cfg = cfg
        self.out = out
        self.events: deque[AttemptEvent] = deque(maxlen=cfg.tail_lines)
        self.state = dict(
            attempt_num=None, prior_num=None, queue_label="",
            phase="-", elapsed_s=0.0,
            budget_used=0.0, budget_total=0.0,
            tokens_in=0, tokens_out=0,
            turns=0, tools=0, last_action="",
        )
        self._lines_drawn = 0  # so we know how far to scroll back next frame

    # ---- public API ----

    def attempt_start(self, num: int, prior: Optional[int], queue_label: str = ""):
        self.state.update(attempt_num=num, prior_num=prior, queue_label=queue_label,
                          phase="starting", elapsed_s=0.0,
                          budget_used=0.0, tokens_in=0, tokens_out=0,
                          turns=0, tools=0, last_action="")
        self.events.clear()
        self._lines_drawn = 0
        if not self._suppress_redraw:
            self._render(initial=True)

    def update(self, **kw):
        self.state.update(kw)
        if not self._suppress_redraw:
            self._render()

    def event(self, source: str, kind: str, summary: str, t: Optional[float] = None):
        if source not in self.cfg.sources:
            return
        if kind == "thinking" and not self.cfg.show_thinking:
            return
        e = AttemptEvent(source=source, kind=kind, summary=summary,
                         t=t if t is not None else self.state["elapsed_s"])
        self.events.append(e)
        if kind in ("tool_call", "edit"):
            self.state["tools"] = self.state.get("tools", 0) + 1
        if not self._suppress_redraw:
            self._render()

    # Snapshot-mode toggle: silence intermediate redraws so we can replay
    # an event sequence and only print the final state once.
    _suppress_redraw: bool = False

    def begin_quiet(self):
        self._suppress_redraw = True

    def commit_static(self):
        """Render once with no cursor-move prefix; safe for static dump."""
        self._suppress_redraw = False
        self._lines_drawn = 0
        self._render(initial=True)

    def attempt_done(self, score: float, delta: float, total_cost: float,
                     cumulative_cost: float, summary_line: str = ""):
        self._freeze()  # convert the live region to scrolled-history
        c = self._color
        sign = "+" if delta >= 0 else ""
        marker = c(A.GREEN, " ★") if delta > 0 else ""
        print(f"  [{self.state['attempt_num']:>3}] "
              f"{c(A.BOLD, f'{score:.1f}')} ({sign}{delta:.1f}){marker}  "
              f"${total_cost:.4f} ({c(A.DIM, f'${cumulative_cost:.4f}')})",
              file=self.out)
        if summary_line:
            print(f"         {c(A.DIM, summary_line)}", file=self.out)
        print(file=self.out)

    # ---- internals ----

    def _color(self, code: str, text: str) -> str:
        return f"{code}{text}{A.R}" if self.cfg.color else text

    def _glyph(self, kind: str) -> str:
        table = GLYPH if self.cfg.glyphs else ASCII
        return table.get(kind, "·" if self.cfg.glyphs else " .")

    def _effective_width(self) -> int:
        """Resolve config width: 0 = auto-fit terminal, clamped to [60, 120]."""
        if self.cfg.width:
            return self.cfg.width
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            cols = 80
        return max(60, min(120, cols))

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate to max_chars, replacing the tail with `…` if needed."""
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars == 1:
            return "…"
        return text[: max_chars - 1] + "…"

    def _fmt_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:>4.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m:>2}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def _render(self, initial: bool = False):
        if not initial and self._lines_drawn:
            self.out.write(A.up(self._lines_drawn))

        c = self._color
        w = self._effective_width()
        inner = w - 2  # chars between │ │
        lines = []

        # ---- Header box (5 lines: top + 3 rows + bottom) ----

        # Title row: auto-truncates with … if it would overflow
        title = f" #{self.state['attempt_num']}"
        if self.state["prior_num"] is not None:
            title += f" prior=#{self.state['prior_num']}"
        if self.state["queue_label"]:
            title += f" · {self.state['queue_label']}"
        # 2 chars of ╭─ on the left, 1 char of ╮ on the right
        title = self._truncate(title + " ", inner - 2)
        rule = "─" * (inner - 2 - len(title))
        lines.append(f"╭─{c(A.BOLD, title)}{rule}╮")

        # Row 1: phase | elapsed
        elapsed = self._fmt_time(self.state["elapsed_s"])
        l1_left = f" phase: {c(A.MAGENTA, self.state['phase'])}"
        l1_right = f"elapsed: {c(A.BOLD, elapsed)} "
        lines.append(self._row(l1_left, l1_right, inner))

        # Row 2: budget bar | turns
        used = self.state["budget_used"]
        total = self.state["budget_total"]
        budget_pct = (used / total * 100) if total else 0.0
        bcolor = A.RED if budget_pct > 80 else (A.YELLOW if budget_pct > 50 else A.GREEN)
        used_str = c(bcolor, f"${used:.2f}")
        l2_left = f" budget: {used_str}/${total:.2f}"
        l2_right = f"turns: {self.state['turns']} "
        lines.append(self._row(l2_left, l2_right, inner))

        # Row 3: tokens | tools
        tk_in = self._k(self.state["tokens_in"])
        tk_out = self._k(self.state["tokens_out"])
        l3_left = f" tokens: {tk_in} → {tk_out}"
        l3_right = f"tools: {self.state['tools']} "
        lines.append(self._row(l3_left, l3_right, inner))

        lines.append(f"╰{'─' * inner}╯")

        # ---- Tail (fixed cfg.tail_lines rows, padded with blanks) ----

        # Reserved width for left-side timestamp + source + glyph
        # "  +<time> <src>   <glyph> " — varies a bit; budget 18
        max_summary = w - 18
        tail_rows: list[str] = []
        for ev in self.events:
            ts = c(A.GRAY, f"+{self._fmt_time(ev.t)}")
            src = c(SOURCE_COLOR.get(ev.source, A.GRAY), f"{ev.source:<5}")
            glyph = self._glyph(ev.kind)
            summary = self._truncate(ev.summary, max_summary)
            tail_rows.append(f"  {ts} {src} {glyph} {summary}")
        while len(tail_rows) < self.cfg.tail_lines:
            tail_rows.append("")
        lines.extend(tail_rows[: self.cfg.tail_lines])

        prefix = A.CLEAR_LINE if not initial else ""
        for line in lines:
            self.out.write(prefix + line + "\n")
        self.out.flush()
        self._lines_drawn = len(lines)

    def _freeze(self):
        """Stop tracking lines so the next print scrolls naturally below."""
        self._lines_drawn = 0

    @staticmethod
    def _row(left: str, right: str, inner: int) -> str:
        # Strips ANSI for length math; assumes left/right already include codes
        bare_left = _strip_ansi(left)
        bare_right = _strip_ansi(right)
        pad = inner - len(bare_left) - len(bare_right)
        if pad < 0:
            pad = 0
        return f"│{left}{' ' * pad}{right}│"

    @staticmethod
    def _k(n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 1_000_000:
            return f"{n/1000:.0f}k"
        return f"{n/1_000_000:.1f}M"


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


# --- Demo data -------------------------------------------------------------

SCRIPT = [
    # (delay_seconds, callable on renderer)
    (0.0,  lambda r: r.attempt_start(271, prior=230, queue_label="agent_refine_bold from frontline-1:distance-bands-retry")),
    (0.5,  lambda r: r.update(phase="explore", budget_total=12.00)),
    (0.8,  lambda r: r.event("agent", "tool_call", "read solution.py")),
    (1.6,  lambda r: r.event("agent", "tool_call", "get-learnings (last=10)")),
    (2.4,  lambda r: r.event("agent", "tool_call", "evaluate-fast")),
    (4.5,  lambda r: r.event("eval",  "info",      "fast: score=2563.6 win=0.54")),
    (5.0,  lambda r: r.update(elapsed_s=124.5, budget_used=0.42, tokens_in=18432, tokens_out=2105, turns=4)),
    (5.6,  lambda r: r.event("agent", "thinking",  "comet exclusion zone seems too tight")),
    (6.5,  lambda r: r.event("agent", "edit",      "work/solution.py (3 hunks)")),
    (7.2,  lambda r: r.event("agent", "tool_call", "evaluate-fast")),
    (7.5,  lambda r: r.update(elapsed_s=312.5, budget_used=1.24, tokens_in=42100, tokens_out=4210, turns=8, phase="explore")),
    (9.0,  lambda r: r.event("eval",  "info",      "fast: score=2566.8 (Δ+3.2) win=0.55")),
    (9.6,  lambda r: r.event("agent", "thinking",  "marginal gain — try widening fleet spread")),
    (10.5, lambda r: r.event("agent", "edit",      "work/solution.py (1 hunk)")),
    (11.0, lambda r: r.update(elapsed_s=478.0, budget_used=2.18, tokens_in=78021, tokens_out=8104, turns=12, phase="explore")),
    (11.6, lambda r: r.event("host",  "error",     "tool blocked: Write(probe_at_root.txt)")),
    (12.4, lambda r: r.event("agent", "tool_call", "evaluate-full")),
    (12.6, lambda r: r.update(elapsed_s=2141.4, budget_used=3.42, tokens_in=187_000, tokens_out=23_120, turns=14, phase="evaluate")),
    (13.4, lambda r: r.event("eval",  "phase",     "full: 1/8 games")),
    (14.2, lambda r: r.event("eval",  "phase",     "full: 5/8 games")),
    (15.0, lambda r: r.event("eval",  "phase",     "full: 8/8 games complete")),
    (15.5, lambda r: r.update(elapsed_s=3741.4, phase="reflect")),
    (15.8, lambda r: r.event("agent", "tool_call", "edit work/learnings.md")),
    (16.4, lambda r: r.update(elapsed_s=3787.5, phase="commit", turns=16, budget_used=3.66, tokens_out=24_512)),
    (17.0, lambda r: r.attempt_done(
        score=1544.1, delta=-1019.5,
        total_cost=3.66, cumulative_cost=7.90,
        summary_line="win_rate=0.54 Δ-0.05 · BT=1544.1±45.2 (n=226) · cost $3.66")),
    (17.5, lambda r: r.attempt_start(272, prior=228, queue_label="agent_refine_bold from frontline-2:threat-gradient")),
    (18.0, lambda r: r.update(phase="explore", budget_total=13.00)),
    (18.4, lambda r: r.event("agent", "tool_call", "read solution.py")),
    (19.0, lambda r: r.event("agent", "tool_call", "get-priors (target=#228)")),
]

# Snapshot moments — indices into SCRIPT to print as static frames
SNAPSHOTS = [
    ("After first 4 events (~2 min in)", 4),
    ("Mid-explore, edits + token pressure", 11),
    ("Hit a permission block; turning to full eval", 17),
    ("Reflect phase, almost done", 21),
    ("Attempt #271 done, #272 starting", 28),
]


def render_live(cfg: AttemptLogConfig):
    r = TwoPaneRenderer(cfg)
    last = 0.0
    for delay, action in SCRIPT:
        time.sleep(max(0.0, (delay - last) * 0.4))  # 0.4× speed
        last = delay
        action(r)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "snapshots"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "ascii":
        cfg = AttemptLogConfig(color=False, glyphs=False)
    else:
        cfg = AttemptLogConfig()

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for label, idx in SNAPSHOTS:
                _render_to(label, idx, cfg, f)
        print(f"wrote {out_path}", file=sys.stderr)
        return

    if mode == "live":
        render_live(cfg)
    else:
        for label, idx in SNAPSHOTS:
            render_snapshot(label, idx, cfg)


def _render_to(label, end_idx, cfg, out):
    print(f"\n=== {label} ===\n", file=out)
    r = TwoPaneRenderer(cfg, out=out)
    r.begin_quiet()
    for _, action in SCRIPT[: end_idx + 1]:
        action(r)
    r.commit_static()
    print(file=out)


def render_snapshot(label: str, end_idx: int, cfg: AttemptLogConfig):
    """Snapshot to stdout — replay events silently, render once at end."""
    print(f"\n=== {label} ===\n")
    r = TwoPaneRenderer(cfg)
    r.begin_quiet()
    for _, action in SCRIPT[: end_idx + 1]:
        action(r)
    r.commit_static()
    print()


if __name__ == "__main__":
    main()
