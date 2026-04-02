"""Strategy queue — file-based strategy override.

A JSON file the optimizer checks before each iteration. If there's an item,
it runs that strategy+config instead of the next in rotation. Items are
consumed on read (popped from the front).

Usage:
    add(path, "fresh_approach", {"mode": "blank"}, source="user")
    item = read_next(path)  # pops first item, returns None if empty
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def read_next(path: Path) -> Optional[Dict[str, Any]]:
    """Pop and return the first queue item. Returns None if queue is empty."""
    queue_path = Path(path) / "queue.json"
    if not queue_path.exists():
        return None

    try:
        items = json.loads(queue_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None

    if not items:
        return None

    item = items.pop(0)
    if items:
        queue_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    else:
        queue_path.unlink()

    return item


def add(path: Path, strategy: str, config: Optional[Dict] = None, source: str = "user"):
    """Append a strategy override to the queue."""
    queue_path = Path(path) / "queue.json"

    items = []
    if queue_path.exists():
        try:
            items = json.loads(queue_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            items = []

    items.append({
        "strategy": strategy,
        "config": config or {},
        "source": source,
    })

    queue_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
