"""QueueWait — opt-in concurrent-write coordination for any Learnings backend.

Concern: :class:`MarkdownLearnings.add` and similar backends do
read-modify-write on a single file. Two processes calling ``add()``
simultaneously can drop one writer's text. Single-process users don't
hit this; users running parallel optimizers against the same project do.

Mechanism: a tiny queue of intent markers in a sibling directory. Each
writer atomically allocates a sequence number via ``mkdir(exist_ok=False)``
collision-retry (the same primitive ``FolderAttemptHistory`` uses for
attempt-number allocation), then waits until its marker is the lowest
sequence in the queue, performs the wrapped ``add()``, and removes its
marker. Ordering is established at claim-time, not by reading wall-clock
timestamps off the directory listing — so two writers can never both
think they're first.

Why not OS file locks: portability (fcntl on POSIX, msvcrt on Windows),
no extra dep, and learnings writes are rare enough that brief polling
is tolerable. Stale markers from crashed writers are reclaimed after
``ttl_seconds``.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

from groundhog.base.learnings import Learnings


_DEFAULT_TTL_SECONDS = 600   # 10 minutes
_POLL_SECONDS = 0.2
_CLAIM_MAX_RETRIES = 1000   # Defensive cap on mkdir-collision retry.


def _remove_marker(marker: Path) -> None:
    """Best-effort removal. Marker dirs may contain an ``owner`` tag file."""
    try:
        shutil.rmtree(marker)
    except OSError:
        pass


class QueueWaitLearnings(Learnings):
    """Wrap any Learnings backend to serialize ``add()`` calls across processes.

    Each ``add()`` claims a slot in a queue directory, waits until it holds
    the oldest unclaimed slot, then delegates to the inner backend.
    ``get()`` and ``edit()`` pass through unchanged — concurrent reads are
    safe (Python file IO is atomic for small files), and ``edit()`` is
    typically a single-writer interactive operation.

    Args:
        inner: the underlying ``Learnings`` (e.g. ``MarkdownLearnings``).
        queue_dir: directory for intent markers. Defaults to a sibling
            ``.queue_wait/`` next to the inner store's path attribute, or
            a temp-dir fallback if the inner has no ``_path``.
        ttl_seconds: stale-marker reclaim threshold. Markers older than
            this are treated as crashed-writer leftovers and removed.
        poll_seconds: how often to recheck the queue while waiting.
    """

    def __init__(
        self,
        inner: Learnings,
        queue_dir: Path | str | None = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        poll_seconds: float = _POLL_SECONDS,
    ):
        self.inner = inner
        if queue_dir is None:
            base = getattr(inner, "_path", None) or getattr(inner, "path", None)
            base = Path(base) if base is not None else Path(".")
            # If the inner store points at a file (typical: .md), the queue
            # directory is a sibling of the file. We can't rely on
            # ``is_file()`` because the file may not exist yet on first init.
            if base.suffix:
                base = base.parent
            queue_dir = base / ".queue_wait"
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.poll_seconds = poll_seconds

    # --- Learnings interface ----------------------------------------------

    def add(self, text: str) -> None:
        marker = self._claim()
        try:
            self._wait_my_turn(marker)
            self.inner.add(text)
        finally:
            _remove_marker(marker)

    def get(self, last: int = 0, random: int = 0) -> str:
        return self.inner.get(last=last, random=random)

    def edit(self, search: str, replace: str) -> None:
        # Coordinated like add() — edit is also read-modify-write.
        marker = self._claim()
        try:
            self._wait_my_turn(marker)
            self.inner.edit(search, replace)
        finally:
            _remove_marker(marker)

    # --- Internals ---------------------------------------------------------

    def _claim(self) -> Path:
        """Atomically allocate the next sequence number and create the marker.

        Mirrors the primitive in ``FolderAttemptHistory.workspace``: scan
        existing markers, take ``max + 1``, claim via ``mkdir(exist_ok=False)``,
        retry on collision. The atomic mkdir is the only place ordering is
        decided — once a writer's marker exists, no later writer can take a
        smaller number. This is what wall-clock timestamps couldn't guarantee.
        """
        for _ in range(_CLAIM_MAX_RETRIES):
            seq = self._next_seq()
            # The marker name is JUST the sequence number — two writers
            # racing on the same seq must collide on the same filename so
            # mkdir(exist_ok=False) can pick exactly one winner. Pid/nonce
            # for debuggability lives inside the marker as a tag file.
            marker = self.queue_dir / f"{seq:010d}.intent"
            try:
                marker.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            try:
                tag = f"pid={os.getpid()} nonce={uuid.uuid4().hex[:8]}"
                (marker / "owner").write_text(tag, encoding="utf-8")
            except OSError:
                pass  # Best-effort; the marker itself is what matters.
            return marker
        raise RuntimeError(
            f"Could not claim a queue slot after {_CLAIM_MAX_RETRIES} retries"
        )

    def _next_seq(self) -> int:
        """Highest in-use sequence + 1. Stale markers are skipped (they'll be
        reclaimed in ``_wait_my_turn``)."""
        max_seq = 0
        for entry in self.queue_dir.iterdir():
            if entry.suffix != ".intent":
                continue
            try:
                seq = int(entry.stem)
            except ValueError:
                continue
            if seq > max_seq:
                max_seq = seq
        return max_seq + 1

    def _wait_my_turn(self, marker: Path) -> None:
        """Spin until ``marker`` is the lexicographically smallest unexpired
        intent in the queue. Reclaims stale markers as a side effect."""
        while True:
            now = time.time()
            entries = []
            for entry in self.queue_dir.iterdir():
                if entry.suffix != ".intent":
                    continue
                try:
                    age = now - entry.stat().st_mtime
                except OSError:
                    continue
                if age > self.ttl_seconds:
                    _remove_marker(entry)
                    continue
                entries.append(entry.name)

            if not entries:
                # We hold the only marker (or it was just reclaimed).
                return
            entries.sort()
            if entries[0] == marker.name:
                return

            # Someone earlier is ahead of us; back off and recheck.
            time.sleep(self.poll_seconds)
