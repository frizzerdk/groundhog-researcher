"""Conversation log — JSON-first per-workspace logging.

JSON is the source of truth. Markdown is a human-readable view rendered from it.

Usage:
    conversation_log(ws.path, prompt, role="User")
    conversation_log(ws.path, system_prompt, role="System")
    conversation_log(ws.path, response)                      # LLMResponse
    conversation_log(ws.path, response, label="Learnings")   # tagged segment
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union

from groundhog.base.backend import LLMResponse


def conversation_log(ws_path: Path, message: Union[str, LLMResponse], *,
                     role: str = "User", label: str = ""):
    """Append a message to the conversation log.

    Args:
        ws_path: workspace directory
        message: plain string or LLMResponse object
        role: speaker name (used when message is a string)
        label: optional tag for the entry (e.g. "Learnings", "Retry 2")
    """
    if isinstance(message, LLMResponse):
        entry = {
            "timestamp": _now(),
            "role": message.model or "LLM",
            "message": message.text,
            "label": label,
            "usage": message.usage or {},
        }
    else:
        entry = {
            "timestamp": _now(),
            "role": role,
            "message": str(message),
            "label": label,
        }

    _append_json(ws_path, entry)
    _render_markdown(ws_path)


# --- JSON (source of truth) ---

def _now():
    return datetime.now().isoformat(timespec="seconds")


def _json_path(ws_path):
    return Path(ws_path) / "conversation.json"


def _read_json(ws_path):
    path = _json_path(ws_path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []


def _append_json(ws_path, entry):
    entries = _read_json(ws_path)
    entries.append(entry)
    _json_path(ws_path).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Markdown (rendered view) ---

def _render_markdown(ws_path):
    entries = _read_json(ws_path)
    lines = []
    for entry in entries:
        header = f"**{entry['role']}**"
        if entry.get("label"):
            header += f" [{entry['label']}]"
        lines.append(f"{header}:\n")
        lines.append(f"{entry['message']}\n")
        if entry.get("usage"):
            parts = [f"{k}={v}" for k, v in entry["usage"].items()
                     if isinstance(v, (int, float))]
            if parts:
                lines.append(f"*{', '.join(parts)}*\n")
        lines.append("---\n")
    (Path(ws_path) / "conversation.md").write_text("\n".join(lines), encoding="utf-8")
