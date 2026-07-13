"""Tests for `groundhog queue add/list/clear` — the CLI over tools/queue.py.

Reuses the tmp run-dir harness from test_cli_attempt: a contract-shaped
task.py whose build_toolkit() roots the store (and so queue.json) at the
run dir. The concurrency tests reuse the two-writer subprocess harness
pattern from test_fileio: every mutation of queue.json runs under
``locked`` + ``atomic_write_text``, so nothing may be lost or duplicated.
"""

import json
import subprocess
import sys
from pathlib import Path

from groundhog.cli import queue_group
from groundhog.tools.queue import add as queue_add, read_next

from test_cli_attempt import _in_dir, _write_run_dir


def _queue_file(run_dir):
    return run_dir / "queue.json"


def test_queue_add_writes_user_item(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = queue_group(["add", "fresh_approach"])
    assert rc == 0
    assert "Queued fresh_approach at position 1" in capsys.readouterr().out
    items = json.loads(_queue_file(run_dir).read_text(encoding="utf-8"))
    assert items == [{"strategy": "fresh_approach", "config": {}, "source": "user"}]


def test_queue_add_with_set_pairs(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = queue_group(["add", "improve", "--set", "mode=blank",
                          "--set", "tier=cheap"])
    assert rc == 0
    items = json.loads(_queue_file(run_dir).read_text(encoding="utf-8"))
    assert items[0]["config"] == {"mode": "blank", "tier": "cheap"}
    # The optimizer consumes exactly what the CLI wrote.
    item = read_next(run_dir)
    assert item["strategy"] == "improve"
    assert item["config"]["mode"] == "blank"
    assert item["source"] == "user"


def test_queue_add_appends_in_order(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        queue_group(["add", "first"])
        queue_group(["add", "second"])
    capsys.readouterr()
    items = json.loads(_queue_file(run_dir).read_text(encoding="utf-8"))
    assert [i["strategy"] for i in items] == ["first", "second"]


def test_queue_add_dangling_set_is_usage_error(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        assert queue_group(["add", "improve", "--set"]) == 1
        assert queue_group(["add", "improve", "--set", "no-equals"]) == 1
        assert queue_group(["add"]) == 1
    assert not _queue_file(run_dir).exists()


def test_queue_list_shows_position_strategy_config_source(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        queue_group(["add", "improve", "--set", "mode=blank"])
        queue_group(["add", "fresh_approach"])
        capsys.readouterr()
        rc = queue_group(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "pos" in lines[0] and "strategy" in lines[0] and "source" in lines[0]
    assert lines[1].startswith("1") and "improve" in lines[1]
    assert "mode=blank" in lines[1] and "user" in lines[1]
    assert lines[2].startswith("2") and "fresh_approach" in lines[2]


def test_queue_list_empty(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = queue_group(["list"])
    assert rc == 0
    assert "Queue is empty." in capsys.readouterr().out


def test_queue_clear(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        queue_group(["add", "improve"])
        queue_group(["add", "improve"])
        capsys.readouterr()
        rc = queue_group(["clear"])
        assert rc == 0
        assert "Cleared 2 queued items." in capsys.readouterr().out
        # File preserved as [] (the read_next convention), not unlinked.
        assert _queue_file(run_dir).read_text(encoding="utf-8") == "[]"
        assert read_next(run_dir) is None

        rc = queue_group(["clear"])
        assert rc == 0
        assert "Queue is empty." in capsys.readouterr().out


def test_queue_help_and_unknown(capsys):
    assert queue_group([]) == 0
    assert queue_group(["-h"]) == 0
    assert queue_group(["frobnicate"]) == 1
    assert "Unknown queue subcommand" in capsys.readouterr().out


def test_corrupt_queue_is_never_reported_empty_or_clobbered(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    qf = _queue_file(run_dir)
    qf.write_text("{not json", encoding="utf-8")
    with _in_dir(run_dir):
        rc = queue_group(["list"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "unreadable" in out
        assert "Queue is empty." not in out

        rc = queue_group(["add", "improve"])
        assert rc == 1
        assert "Cannot add" in capsys.readouterr().out
    # The corrupt file survives both reads untouched.
    assert qf.read_text(encoding="utf-8") == "{not json"

    # The optimizer-side pop warns and treats it as empty, no clobber.
    assert read_next(run_dir) is None
    assert "WARNING" in capsys.readouterr().out
    assert qf.read_text(encoding="utf-8") == "{not json"

    with _in_dir(run_dir):
        queue_group(["clear"])
    assert qf.read_text(encoding="utf-8") == "[]"


def test_read_next_guards_top_level_dict(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _queue_file(run_dir).write_text('{"strategy": "improve"}', encoding="utf-8")
    assert read_next(run_dir) is None
    assert "WARNING" in capsys.readouterr().out
    assert _queue_file(run_dir).read_text(encoding="utf-8") == \
        '{"strategy": "improve"}'


def test_malformed_elements_skipped_with_warning(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    _queue_file(run_dir).write_text(
        json.dumps(["oops", {"strategy": "improve", "config": {},
                    "source": "user"}, 7]),
        encoding="utf-8")
    with _in_dir(run_dir):
        rc = queue_group(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("WARNING: skipping malformed queue item") == 2
    assert "improve" in out

    item = read_next(run_dir)
    assert item["strategy"] == "improve"


QUEUE_WORKER = """
import sys
from pathlib import Path
from groundhog.tools import queue

root = Path(sys.argv[1])
role = sys.argv[2]
n = int(sys.argv[3])

if role.startswith("add:"):
    name = role.split(":", 1)[1]
    for i in range(n):
        queue.add(root, name + "-" + str(i))
else:
    got = []
    while len(got) < n:
        item = queue.read_next(root)
        if item is not None:
            got.append(item["strategy"])
    print("\\n".join(got))
"""

N_ITEMS = 50


def test_concurrent_adders_and_consumer_lose_nothing(tmp_path):
    """Two adder processes and one consumer interleave on queue.json.
    Every queued item must be consumed exactly once (a lost or duplicated
    item hangs or corrupts the consumer's tally)."""
    (tmp_path / "queue.json").write_text("[]", encoding="utf-8")
    script = tmp_path / "worker.py"
    script.write_text(QUEUE_WORKER, encoding="utf-8")

    def spawn(role, n):
        return subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), role, str(n)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    procs = [spawn("add:a", N_ITEMS), spawn("add:b", N_ITEMS),
             spawn("consume", 2 * N_ITEMS)]
    outputs = []
    for p in procs:
        out, err = p.communicate(timeout=180)
        assert p.returncode == 0, err
        outputs.append(out)

    consumed = outputs[2].split()
    expected = {f"{w}-{i}" for w in ("a", "b") for i in range(N_ITEMS)}
    assert len(consumed) == 2 * N_ITEMS
    assert set(consumed) == expected
    # Per-adder FIFO order survives the interleaving.
    for name in ("a", "b"):
        ours = [c for c in consumed if c.startswith(name)]
        assert ours == [f"{name}-{i}" for i in range(N_ITEMS)]
    assert json.loads(
        (tmp_path / "queue.json").read_text(encoding="utf-8")) == []


def test_add_returns_position_under_the_lock(tmp_path):
    assert queue_add(tmp_path, "first") == 1
    assert queue_add(tmp_path, "second") == 2
