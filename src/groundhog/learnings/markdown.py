"""Markdown-backed learnings. Entries separated by --- for clean splitting.

Stores entries in a single markdown file. Each add() appends a new entry
separated by '\\n\\n---\\n\\n'. get() supports sampling: last=N most recent +
random=M from older entries. This prevents learnings from growing unbounded
in prompts while keeping recent context and older diversity.
"""

import random as rand_module
from pathlib import Path

from groundhog.base.learnings import Learnings

SEPARATOR = "\n\n---\n\n"


class MarkdownLearnings(Learnings):
    """Knowledge stored as a markdown file with separated entries."""

    def __init__(self, path: Path):
        self._path = Path(path) / "learnings.md"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, text: str):
        existing = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        if existing.strip():
            self._path.write_text(existing.rstrip() + SEPARATOR + text.strip() + "\n", encoding="utf-8")
        else:
            self._path.write_text(text.strip() + "\n", encoding="utf-8")

    def get(self, last: int = 0, random: int = 0) -> str:
        if not self._path.exists():
            return ""
        entries = self._entries()
        if not entries:
            return ""
        if last <= 0 and random <= 0:
            return SEPARATOR.join(entries)
        return self._sample(entries, last, random)

    def count(self) -> int:
        """Total number of entries."""
        if not self._path.exists():
            return 0
        return len(self._entries())

    def edit(self, search: str, replace: str):
        content = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        if search not in content:
            raise ValueError(f"'{search}' not found in learnings")
        self._path.write_text(content.replace(search, replace), encoding="utf-8")

    def _entries(self):
        content = self._path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return [e.strip() for e in content.split("---") if e.strip()]

    def _sample(self, entries, last, random):
        if last >= len(entries):
            return SEPARATOR.join(entries)

        recent = entries[-last:] if last > 0 else []
        older = entries[:-last] if last > 0 else entries

        sampled = []
        if random > 0 and older:
            k = min(random, len(older))
            sampled = rand_module.sample(older, k)

        # sampled first (context), then recent (most relevant)
        combined = sampled + recent
        return SEPARATOR.join(combined)
