"""Strategy queue — file-based strategy override.

A JSON file the optimizer checks before each iteration. If there's an item,
it runs that strategy+config instead of the next in rotation. Items are
consumed on read (popped from the front).

The CLI and the optimizer may touch ``queue.json`` concurrently, so every
mutation runs under ``locked(queue_path)`` and lands via
``atomic_write_text`` — read, touch-edit in memory, write; no compute
inside the lock.

Usage:
    add(path, "fresh_approach", {"mode": "blank"}, source="user")
    item = read_next(path)  # pops first item, returns None if empty
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from groundhog.utils.fileio import atomic_write_text, locked


class QueueCorrupt(ValueError):
    """queue.json exists but cannot be read as a queue."""


def read_items(queue_path: Path) -> List[Dict[str, Any]]:
    """Parse ``queue_path`` into a list of well-formed queue items.

    A missing file is an empty queue. A file that does not parse, or whose
    top level is not a list, raises :class:`QueueCorrupt` — a corrupt
    queue must never be mistaken for an empty one (or silently clobbered
    by the next write). Non-dict elements are skipped with a warning.
    """
    if not queue_path.exists():
        return []
    try:
        raw = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QueueCorrupt(f"could not parse {queue_path}: {e}") from e
    if not isinstance(raw, list):
        raise QueueCorrupt(
            f"{queue_path} must hold a JSON list, got {type(raw).__name__}")
    items = []
    for element in raw:
        if isinstance(element, dict):
            items.append(element)
        else:
            print(f"WARNING: skipping malformed queue item: {element!r}")
    return items


def read_next(path: Path) -> Optional[Dict[str, Any]]:
    """Pop and return the first queue item. Returns None if queue is empty.

    The queue file is preserved as ``[]`` after the last item is consumed
    (rather than being unlinked), so tools that treat ``queue.json`` as
    visible persistent state can rely on it always existing once the queue
    has been used. A corrupt file is left untouched and reported as empty
    (with a warning) — never clobbered.
    """
    queue_path = Path(path) / "queue.json"
    if not queue_path.exists():
        return None

    with locked(queue_path):
        try:
            items = read_items(queue_path)
        except QueueCorrupt as e:
            print(f"WARNING: strategy queue unreadable, ignoring it ({e})")
            return None
        if not items:
            return None
        item = items.pop(0)
        atomic_write_text(queue_path, json.dumps(items, indent=2))
    return item


def add(path: Path, strategy: str, config: Optional[Dict] = None,
        source: str = "user") -> int:
    """Append a strategy override to the queue; returns its 1-based position.

    Raises :class:`QueueCorrupt` instead of overwriting a file it could
    not read — the previous read-parse-clobber wiped pending items.
    """
    queue_path = Path(path) / "queue.json"
    with locked(queue_path):
        items = read_items(queue_path)
        items.append({
            "strategy": strategy,
            "config": config or {},
            "source": source,
        })
        atomic_write_text(queue_path, json.dumps(items, indent=2))
    return len(items)


def clear(path: Path) -> int:
    """Drop all pending items (file preserved as ``[]``, the read_next
    convention). Returns the number of items dropped; a corrupt file is
    reset and reported as 0."""
    queue_path = Path(path) / "queue.json"
    if not queue_path.exists():
        return 0
    with locked(queue_path):
        try:
            n = len(read_items(queue_path))
        except QueueCorrupt:
            print(f"WARNING: {queue_path} was corrupt - resetting it")
            n = 0
        atomic_write_text(queue_path, "[]")
    return n
