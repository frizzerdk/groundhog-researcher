"""Atomic writes and cross-process file locks for shared state files.

Lock discipline: acquire -> read -> mutate in memory -> atomic write ->
release. Never compute (evaluate, call an LLM, spawn a subprocess) while
holding the lock; the critical section is read+write only.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def atomic_write_text(path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (POSIX and NTFS).

    Writes to a pid-suffixed temp sibling, flushes+fsyncs, then
    ``os.replace``s over the target, so a reader sees either the old or
    the new content — never a partial file.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def locked(path):
    """Exclusive cross-process lock scoped to ``path``; blocks until held.

    Locks a ``.lock`` sibling (created if missing), never ``path`` itself,
    so the payload can be atomically replaced while the lock is held.
    Released and closed on exit. Follow the module's lock discipline:
    read, mutate in memory, ``atomic_write_text``, release.
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    f = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            f.seek(0)
            # LK_LOCK retries at a fixed 1s interval and gives up after 10
            # tries; LK_NBLCK in a short-sleep loop gives true blocking
            # semantics without the 1s contention penalty.
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.001)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()
