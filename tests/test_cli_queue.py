"""Tests for `groundhog queue add/list/clear` — the CLI over tools/queue.py.

Reuses the tmp run-dir harness from test_cli_attempt: a contract-shaped
task.py whose build_toolkit() roots the store (and so queue.json) at the
run dir.
"""

import json

from groundhog.cli import queue_group
from groundhog.tools.queue import read_next

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
