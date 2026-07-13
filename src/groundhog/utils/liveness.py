"""Pid+timestamp heartbeats — tell a live workspace holder from a crashed one.

A heartbeat is a tiny json file ``{"pid": ..., "started_at": ...}`` kept
outside any committed tree. Both history backends write one per open
workspace, so ``list_in_progress`` can report liveness and ``reap`` can
tell a working session from a crashed one: a dead pid (past a grace TTL)
is reapable; a live pid is left alone regardless of age.
"""

import json
import os
import time
from pathlib import Path


def write_heartbeat(path) -> None:
    try:
        Path(path).write_text(
            json.dumps({"pid": os.getpid(), "started_at": time.time()}),
            encoding="utf-8")
    except OSError:
        pass


def read_heartbeat(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clear_heartbeat(path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False
