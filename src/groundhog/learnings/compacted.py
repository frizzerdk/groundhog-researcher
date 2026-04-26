"""Compacted — wrap any Learnings backend with a queued + condensed view.

Two related concerns this module solves:

1. **Unbounded growth.** Plain :class:`MarkdownLearnings` appends entries
   forever; over a long optimization run the file grows past anything
   useful for a prompt.
2. **Manual vs automatic distillation.** Some workflows want a human (or a
   scheduled job) to compress accumulated entries deliberately; others
   want every ``add()`` to fold the new entry into the condensed view via
   an LLM call.

Both behaviors share one mechanism: a queue file that buffers raw entries
and a separate "current" file that holds the condensed view returned to
``get()``. The optional ``compactor`` callable processes the queue:

    Compacted(MarkdownLearnings(path), current_path=path / "learnings_current.md")
        → manual mode. add() appends to vault + queue; get() returns the
          current view (or the inner store if no view exists yet).
          Call distill(compactor) explicitly to compress.

    Compacted(MarkdownLearnings(path),
              current_path=path / "learnings_current.md",
              compactor=make_llm_compactor(backend))
        → automatic mode. Every add() invokes the compactor; the queue
          is drained on success, retained on failure for the next add().

The ``inner`` Learnings is the append-only history of record. ``current``
is a derived view that can always be rebuilt from the inner store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

from groundhog.base.learnings import Learnings


# (current_text, queue_entries) -> rewritten_current
Compactor = Callable[[str, List[str]], str]


_SEPARATOR = "\n\n---\n\n"


def _split_entries(text: str) -> List[str]:
    """Parse markdown text separated by ``---`` lines into a list of entries."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split("\n---\n")]
    return [p for p in parts if p]


def _append(file: Path, text: str) -> None:
    """Append ``text`` to ``file`` using the standard separator. Creates
    the file if missing."""
    text = text.strip()
    if not text:
        return
    needs_sep = file.exists() and file.stat().st_size > 0
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        if needs_sep:
            f.write(_SEPARATOR)
        f.write(text)


class Compacted(Learnings):
    """Wraps a Learnings backend with a queue + optional auto-compaction.

    The wrapped ``inner`` backend is the append-only history. ``current_path``
    holds the condensed view; ``queue_path`` buffers entries that have not
    yet been folded in.

    Args:
        inner: Underlying Learnings backend (e.g. ``MarkdownLearnings``).
            Receives every ``add()`` for permanent retention.
        current_path: File path for the condensed view. Returned by ``get()``
            when present and non-empty; otherwise ``get()`` falls back to
            ``inner.get(...)``.
        queue_path: Optional path for the queue buffer. Defaults to
            ``current_path`` with a ``.queue.md`` suffix.
        compactor: Optional callable ``(current, queue_entries) -> new_current``.
            When set, ``add()`` invokes it after appending; on success the
            queue is cleared, on failure it is retained for the next ``add()``.
            When ``None``, entries accumulate in the queue until ``distill()``
            is called explicitly.
        seed_text: Optional initial content for ``current_path`` (only used
            on first init when the file is missing or empty).
        quiet: Suppress merge-failure stderr warnings.
    """

    def __init__(
        self,
        inner: Learnings,
        current_path: Path | str,
        queue_path: Path | str | None = None,
        compactor: Optional[Compactor] = None,
        seed_text: Optional[str] = None,
        quiet: bool = False,
    ):
        self.inner = inner
        self.current_path = Path(current_path)
        if queue_path is None:
            self.queue_path = self.current_path.with_suffix(
                self.current_path.suffix + ".queue"
            )
        else:
            self.queue_path = Path(queue_path)
        self.compactor = compactor
        self.quiet = quiet

        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        if seed_text and seed_text.strip():
            self._seed_if_empty(seed_text)

    def _seed_if_empty(self, seed_text: str) -> None:
        if not self.current_path.exists() or self.current_path.stat().st_size == 0:
            self.current_path.write_text(seed_text.strip() + "\n", encoding="utf-8")

    # --- Learnings interface ----------------------------------------------

    def add(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.inner.add(text)
        _append(self.queue_path, text)
        if self.compactor is not None:
            self.distill(self.compactor)

    def get(self, last: int = 0, random: int = 0) -> str:
        """Return the condensed view if it exists, else the inner store.

        ``last`` and ``random`` are ignored when the condensed view is in
        use — it's already condensed; sampling on top would be lossy.
        Sampling args still propagate to the inner store when there is no
        current view (e.g. before any compaction has happened).
        """
        if self.current_path.exists():
            current = self.current_path.read_text(encoding="utf-8").strip()
            if current:
                return current
        return self.inner.get(last=last, random=random)

    def edit(self, search: str, replace: str) -> None:
        """Edit the condensed view if it exists, otherwise the inner store."""
        if self.current_path.exists() and self.current_path.read_text(
            encoding="utf-8"
        ).strip():
            content = self.current_path.read_text(encoding="utf-8")
            if search not in content:
                raise ValueError(f"'{search}' not found in current view")
            self.current_path.write_text(
                content.replace(search, replace), encoding="utf-8"
            )
        else:
            self.inner.edit(search, replace)

    # --- Compaction --------------------------------------------------------

    def queued(self) -> List[str]:
        """Return the list of queued entries that have not yet been compacted."""
        if not self.queue_path.exists():
            return []
        return _split_entries(self.queue_path.read_text(encoding="utf-8"))

    def distill(self, compactor: Optional[Compactor] = None) -> bool:
        """Run the compactor against the current view + queued entries.

        On success, replaces the current file and clears the queue.
        On failure, leaves the queue intact so the next ``add()`` retries.

        Returns ``True`` if the queue was processed (even if empty);
        ``False`` if the compactor raised or returned an empty result.
        """
        fn = compactor if compactor is not None else self.compactor
        if fn is None:
            raise ValueError("distill requires a compactor (none configured)")

        queue_entries = self.queued()
        if not queue_entries:
            return True

        current = (
            self.current_path.read_text(encoding="utf-8").strip()
            if self.current_path.exists()
            else ""
        )

        try:
            new_current = fn(current, queue_entries)
        except Exception as e:
            if not self.quiet:
                print(
                    f"[Compacted] compactor failed: {type(e).__name__}: "
                    f"{str(e)[:200]} — keeping queue for retry",
                    file=sys.stderr,
                    flush=True,
                )
            return False

        if not new_current or not new_current.strip():
            if not self.quiet:
                print(
                    "[Compacted] compactor returned empty — keeping queue",
                    file=sys.stderr,
                    flush=True,
                )
            return False

        self.current_path.write_text(new_current.strip() + "\n", encoding="utf-8")
        self.queue_path.write_text("", encoding="utf-8")
        return True


# --- LLM-backed compactor factory -----------------------------------------

DEFAULT_COMPACTOR_SYSTEM_PROMPT = (
    "Maintain a learning log in observational voice: what is being seen "
    "about the problem, and which kinds of changes tend to produce results. "
    "Return only the rewritten document — no preamble, no commentary."
)

DEFAULT_COMPACTOR_PROMPT_TEMPLATE = """\
Rewrite the CURRENT learnings to fold in the NEW observations.

The output should read as a guide for someone starting fresh on this
problem — not a changelog for a codebase.

CURRENT:
<<<
{current}
>>>

NEW observations:
<<<
{queue}
>>>

Rules:

1. Observation, not directive. Describe what is seen. Avoid absolute
   commands like "always" or "never".

2. Tag claims by evidence strength when relevant: *observed once*,
   *observed repeatedly*, *confirmed across runs*. Inherent mechanics
   need no tag.

3. Domain-general where possible — write so a future approach using a
   different algorithm can still find it useful.

4. Dedupe. If two entries make the same point, keep the more specific.

5. Flag contradictions rather than silently dropping either side.

6. Target length: under {max_lines} lines. Compress older/weaker
   evidence first.

7. Return ONLY the rewritten learnings as markdown. No preamble.
"""


def make_llm_compactor(
    backend,
    prompt_template: str = DEFAULT_COMPACTOR_PROMPT_TEMPLATE,
    system_prompt: str = DEFAULT_COMPACTOR_SYSTEM_PROMPT,
    max_lines: int = 500,
) -> Compactor:
    """Wrap a groundhog ``LLMBackend`` as a :data:`Compactor`.

    The backend receives a prompt with the current document + queued
    entries and is asked to return the rewritten current view.

    Args:
        backend: Any groundhog ``LLMBackend`` (e.g. ``toolkit.llm.get("budget")``).
        prompt_template: Format string with ``{current}``, ``{queue}``,
            ``{max_lines}`` placeholders.
        system_prompt: System-level instruction handed to the backend.
        max_lines: Soft cap passed to the prompt to bound output size.
    """

    def compact(current: str, queue_entries: List[str]) -> str:
        queue_text = _SEPARATOR.join(queue_entries)
        prompt = prompt_template.format(
            current=current or "(empty)",
            queue=queue_text,
            max_lines=max_lines,
        )
        resp = backend.generate(prompt, system_prompt=system_prompt)
        return resp.text

    return compact
