"""Strategy logger — inline and newline terminal output with timing.

Handles two modes:
  log.inline("generating...")   → appends to current line
  log.info("diff applied")      → newline with indent
  log.start("prior=#1 (0.697) | retries=0")  → first line of a strategy run
  log.end()                     → flush any pending inline text

Timer:
  log.tick()                    → start/reset the timer
  log.tock("generating")        → inline print with elapsed time: "generating (1.2s)... "
"""

import sys
import time


class StrategyLog:
    """Default strategy logger. Writes indented output under optimizer's attempt line."""

    INDENT = "         "  # 9 spaces — aligns under attempt metrics

    def __init__(self):
        self._inline_dirty = False
        self._line_started = False
        self._tick = None

    def start(self, text):
        """First line of strategy output — config summary."""
        self._flush_inline()
        print(f"{self.INDENT}{text}")
        self._line_started = False
        self.tick()

    def inline(self, text):
        """Append to current line — progress updates."""
        if not self._line_started:
            sys.stdout.write(self.INDENT)
            self._line_started = True
        sys.stdout.write(text)
        sys.stdout.flush()
        self._inline_dirty = True

    def info(self, text):
        """New line — important information."""
        self._flush_inline()
        print(f"{self.INDENT}{text}")

    def end(self):
        """Flush any pending inline output."""
        self._flush_inline()

    def tick(self):
        """Start/reset the timer."""
        self._tick = time.monotonic()

    def tock(self, label=""):
        """Inline print with elapsed time since last tick, then reset.

        Example: log.tock("generating") → "generating (1.2s)... "
        """
        elapsed = time.monotonic() - self._tick if self._tick else 0
        text = f"{label} ({elapsed:.1f}s)... " if label else f"({elapsed:.1f}s)... "
        self.inline(text)
        self.tick()

    def _flush_inline(self):
        if self._inline_dirty:
            print()
            self._inline_dirty = False
            self._line_started = False
