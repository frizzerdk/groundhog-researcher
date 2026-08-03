"""Tests for the commit-without-result guard and `--note-score`.

A no-eval commit with no recorded result.json is legal but loud: a WARNING
plus a ``no_recorded_result`` metadata flag, so downstream consumers know
the attempt is unscored by construction rather than by accident.
"""

import json

from groundhog import rundir
from groundhog.cli import attempt_group

from test_cli_attempt import _in_dir, _write_run_dir


def _open_ws(run_dir, capsys):
    attempt_group(["new"])
    out = capsys.readouterr().out
    wsid = [l for l in out.splitlines()
            if l.startswith("Opened workspace")][0].split()[-1]
    loaded = rundir.load_run(run_dir=run_dir)
    ws_path = [ip.path for ip in loaded.history.list_in_progress()
               if ip.workspace_id == wsid][0]
    (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n",
                                         encoding="utf-8")
    (ws_path / "core_direction.md").write_text("constant baseline\n",
                                               encoding="utf-8")
    return wsid, ws_path


def test_commit_without_result_warns_and_flags(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, _ = _open_ws(run_dir, capsys)
        rc = attempt_group(["commit", wsid])
        assert rc == 0
        out = capsys.readouterr().out
        assert "WARNING: committing with no recorded evaluation" in out
        assert "unscored" in out

        attempt = rundir.load_run(run_dir=run_dir).history.get("1")
        assert attempt.metadata["no_recorded_result"] is True


def test_commit_with_eval_does_not_warn(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, _ = _open_ws(run_dir, capsys)
        rc = attempt_group(["commit", wsid, "--eval"])
        assert rc == 0
        assert "WARNING" not in capsys.readouterr().out

        attempt = rundir.load_run(run_dir=run_dir).history.get("1")
        assert "no_recorded_result" not in attempt.metadata


def test_commit_with_preexisting_result_does_not_warn(tmp_path, capsys):
    """A result.json already recorded in the workspace (e.g. by a separate
    eval step) counts as a recorded evaluation."""
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, ws_path = _open_ws(run_dir, capsys)
        (ws_path / "result.json").write_text(json.dumps({
            "completed": True,
            "failed_stage": None,
            "stages": {"evaluate": {"metrics": {"value": 50.0, "target": 50.0},
                                    "errors": {}, "warnings": {}}},
        }), encoding="utf-8")

        rc = attempt_group(["commit", wsid])
        assert rc == 0
        assert "WARNING" not in capsys.readouterr().out

        attempt = rundir.load_run(run_dir=run_dir).history.get("1")
        assert "no_recorded_result" not in attempt.metadata
        assert attempt.result.completed


def test_note_score_writes_display_note(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, _ = _open_ws(run_dir, capsys)
        rc = attempt_group(["commit", wsid, "--note-score", "0.75"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "note[score] = 0.7500" in out

        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note("1", "score") == "0.7500"


def test_note_score_rejects_non_float(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, _ = _open_ws(run_dir, capsys)
        rc = attempt_group(["commit", wsid, "--note-score", "high"])
        assert rc == 1
        assert "--note-score must be a float" in capsys.readouterr().out
        # Workspace untouched — still open.
        loaded = rundir.load_run(run_dir=run_dir)
        assert any(ip.workspace_id == wsid
                   for ip in loaded.history.list_in_progress())


def test_note_score_with_eval_overrides_cached_note(tmp_path, capsys):
    """--eval caches the computed score note; an explicit --note-score is
    the user's display verdict and wins."""
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        wsid, _ = _open_ws(run_dir, capsys)
        rc = attempt_group(["commit", wsid, "--eval", "--note-score", "0.5"])
        assert rc == 0
        capsys.readouterr()
        history = rundir.load_run(run_dir=run_dir).history
        assert history.get_note("1", "score") == "0.5000"
