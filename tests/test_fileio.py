"""Stress tests for utils/fileio — cross-process lock + atomic write.

Two subprocess writers run interleaved read-modify-append cycles on one
JSON list file. With ``locked`` + ``atomic_write_text`` nothing may be
lost or duplicated and every read must parse (each worker json-loads the
file inside every cycle, so a single torn write fails the run). The
unlocked control demonstrates the lost-update race; it is skipped when
the race happens not to bite — the locked assertion is the contract.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

CYCLES = 100

WORKER = """
import json, sys
from pathlib import Path
from groundhog.utils.fileio import atomic_write_text, locked

target = Path(sys.argv[1])
writer = sys.argv[2]
cycles = int(sys.argv[3])
use_lock = sys.argv[4] == "locked"

def cycle(i):
    items = json.loads(target.read_text(encoding="utf-8"))
    items.append(writer + "-" + str(i))
    atomic_write_text(target, json.dumps(items))

for i in range(cycles):
    if use_lock:
        with locked(target):
            cycle(i)
    else:
        try:
            cycle(i)
        except OSError:
            pass  # control mode: a raced cycle is simply a lost item
"""


def _run_two_writers(tmp_path: Path, mode: str) -> list:
    target = tmp_path / "items.json"
    target.write_text("[]", encoding="utf-8")
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8")
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(target), name, str(CYCLES), mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for name in ("a", "b")
    ]
    for p in procs:
        _, err = p.communicate(timeout=180)
        assert p.returncode == 0, err.decode("utf-8", "replace")
    return json.loads(target.read_text(encoding="utf-8"))


def test_locked_two_writers_lose_nothing(tmp_path):
    items = _run_two_writers(tmp_path, "locked")
    expected = {f"{w}-{i}" for w in ("a", "b") for i in range(CYCLES)}
    assert len(items) == 2 * CYCLES  # no duplicates
    assert set(items) == expected    # nothing lost


def test_unlocked_control_loses_updates(tmp_path):
    items = _run_two_writers(tmp_path, "control")
    if len(items) == 2 * CYCLES:
        pytest.skip("control run happened not to race (timing-dependent)")
    assert len(items) < 2 * CYCLES


def test_atomic_write_text_replaces_content(tmp_path):
    target = tmp_path / "state.json"
    from groundhog.utils.fileio import atomic_write_text

    atomic_write_text(target, "[1]")
    assert json.loads(target.read_text(encoding="utf-8")) == [1]
    atomic_write_text(target, "[1, 2]")
    assert json.loads(target.read_text(encoding="utf-8")) == [1, 2]
    assert list(tmp_path.iterdir()) == [target]  # no temp files left behind
