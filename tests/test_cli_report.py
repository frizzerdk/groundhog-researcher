"""Tests for `groundhog report` and the tools/report read-side aggregations."""

from groundhog import rundir
from groundhog.cli import attempt_group, cmd_report
from groundhog.tools import report as report_mod

from test_cli_attempt import _in_dir, _write_run_dir


def _commit(run_dir, capsys, value, direction, fail=False):
    attempt_group(["new", "--fresh", "--no-seed"])
    out = capsys.readouterr().out
    wsid = [l for l in out.splitlines()
            if l.startswith("Opened workspace")][0].split()[-1]
    loaded = rundir.load_run(run_dir=run_dir)
    ws_path = [ip.path for ip in loaded.history.list_in_progress()
               if ip.workspace_id == wsid][0]
    body = ("def solve():\n    raise ValueError('boom')\n" if fail
            else f"def solve():\n    return {value}\n")
    (ws_path / "solution.py").write_text(body, encoding="utf-8")
    (ws_path / "core_direction.md").write_text(direction + "\n",
                                               encoding="utf-8")
    attempt_group(["commit", wsid, "--eval"])
    capsys.readouterr()


def test_report_writes_default_path_with_sections(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        _commit(run_dir, capsys, "40.0", "constant baseline")
        _commit(run_dir, capsys, "50.0", "second family")
        _commit(run_dir, capsys, None, "third family", fail=True)

        rc = cmd_report([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Wrote" in out

    report_file = run_dir / "reports" / "state.md"
    assert report_file.exists()
    text = report_file.read_text(encoding="utf-8")
    assert "# Run state: TinyTask" in text
    assert "## Summary" in text
    assert "attempts: 3 (2 done, 1 failed)" in text
    assert "best: 2 score=1.0000" in text
    assert "## Families" in text
    assert "constant baseline" in text and "second family" in text
    assert "## Recent attempts" in text
    assert "## Score trajectory" in text
    assert "## Open questions" in text
    assert "attempt 3 failed" in text
    # No LLM on the toolkit -> data-only, no narrative section.
    assert "## State of the run" not in text


def test_report_out_flag_and_stdout(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        _commit(run_dir, capsys, "50.0", "constant baseline")

        rc = cmd_report(["--out", "sub/custom.md"])
        assert rc == 0
        capsys.readouterr()
        assert (run_dir / "sub" / "custom.md").exists()

        rc = cmd_report(["--out", "-"])
        assert rc == 0
        assert "# Run state: TinyTask" in capsys.readouterr().out


def test_report_empty_run(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = cmd_report(["--out", "-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "attempts: 0" in out
    assert "No scored attempts yet." in out


def test_report_llm_narrative(tmp_path, capsys):
    """A toolkit-carried LLM adds the narrative section; the fake registry
    proves the default tier is asked exactly once."""
    run_dir = _write_run_dir(tmp_path)
    task_py = run_dir / "task.py"
    task_py.write_text(task_py.read_text(encoding="utf-8") + '''

_original_build_toolkit = build_toolkit

def build_toolkit():
    tk = _original_build_toolkit()

    class _FakeResponse:
        text = "The run is young: one family, one perfect score."

    class _FakeBackend:
        def generate(self, prompt, system_prompt=""):
            assert "Run data" in prompt
            return _FakeResponse()

    class _FakeRegistry:
        def __init__(self):
            self.calls = []
        def get(self, tier):
            self.calls.append(tier)
            assert tier == "default"
            return _FakeBackend()

    tk.llm = _FakeRegistry()
    return tk
''', encoding="utf-8")
    with _in_dir(run_dir):
        _commit(run_dir, capsys, "50.0", "constant baseline")
        rc = cmd_report(["--out", "-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## State of the run" in out
    assert "The run is young" in out


def test_report_llm_failure_degrades_to_data_only(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    task_py = run_dir / "task.py"
    task_py.write_text(task_py.read_text(encoding="utf-8") + '''

_original_build_toolkit = build_toolkit

def build_toolkit():
    tk = _original_build_toolkit()

    class _BoomRegistry:
        def get(self, tier):
            raise RuntimeError("no backend")

    tk.llm = _BoomRegistry()
    return tk
''', encoding="utf-8")
    with _in_dir(run_dir):
        _commit(run_dir, capsys, "50.0", "constant baseline")
        rc = cmd_report([])
    assert rc == 0
    assert "LLM narrative unavailable" in capsys.readouterr().out
    text = (run_dir / "reports" / "state.md").read_text(encoding="utf-8")
    assert "## State of the run" not in text
    assert "## Summary" in text


def test_sparkline_shapes():
    assert report_mod.sparkline([]) == ""
    flat = report_mod.sparkline([1.0, 1.0, 1.0])
    assert len(flat) == 3 and len(set(flat)) == 1
    rising = report_mod.sparkline([0.0, 0.5, 1.0])
    assert rising[0] == " " and rising[-1] == "@"
    # Long series are downsampled to the width.
    long = report_mod.sparkline([float(i) for i in range(500)], width=60)
    assert len(long) == 60
    # ASCII only (Windows console safety).
    assert all(ord(c) < 128 for c in rising + long)


def test_open_questions_cover_gate_failures(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        # Fresh commit without a direction -> gate failure.
        attempt_group(["new", "--fresh", "--no-seed"])
        out = capsys.readouterr().out
        wsid = [l for l in out.splitlines()
                if l.startswith("Opened workspace")][0].split()[-1]
        loaded = rundir.load_run(run_dir=run_dir)
        ws_path = [ip.path for ip in loaded.history.list_in_progress()
                   if ip.workspace_id == wsid][0]
        (ws_path / "solution.py").write_text("def solve():\n    return 50.0\n",
                                             encoding="utf-8")
        attempt_group(["commit", wsid, "--eval"])
        capsys.readouterr()

        rc = cmd_report(["--out", "-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gate-failed" in out
    assert "core_direction.md" in out
