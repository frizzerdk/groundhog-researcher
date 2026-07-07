"""Tests for the `groundhog strategy` group: list / show / run.

Same harness as test_cli_attempt: each test builds a tmp run dir with a
contract-shaped ``task.py`` (here also defining a module-level mock
strategy), then calls the CLI handlers directly with arg lists and asserts
exit codes + stdout. No LLM anywhere.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from groundhog.cli import strategy_group
from groundhog.histories.folder import FolderAttemptHistory


_TASK_BODY = '''
from dataclasses import dataclass
from pathlib import Path

from groundhog import Task, Data, Context, Evaluator, EvalStage, StageResult
from groundhog.base.strategy import Strategy, StrategyConfig, param


class TinyData(Data):
    def get_train(self): return {"target": 50.0}
    def get_test(self): return {"target": 50.0}


class TinyContext(Context):
    def get_brief(self): return "Write solve() returning a float near 50."
    def get_extended(self): return "def solve() -> float"


def _read(code_or_path):
    if isinstance(code_or_path, (str, bytes)):
        return code_or_path
    return (Path(code_or_path) / "solution.py").read_text(encoding="utf-8")


def _score(result):
    if result.errors:
        return -1.0
    m = result.metrics
    if "value" not in m:
        return 0.0
    return max(0.0, 1.0 - abs(m["value"] - m["target"]) / 100.0)


class TinyEvaluator(Evaluator):
    def evaluate(self, code_or_path, data):
        code = _read(code_or_path)
        ns = {}
        try:
            exec(code, ns)
            value = float(ns["solve"]())
        except Exception as e:
            return StageResult(errors={"crash": str(e)})
        target = data.get_test()["target"]
        return StageResult(metrics={"value": value, "target": target})

    def get_stages(self, data):
        return [
            EvalStage("evaluate", "full",
                      lambda cp, d=data: self.evaluate(cp, d),
                      scorer=_score),
        ]


task = Task(data=TinyData(), context=TinyContext(), evaluator=TinyEvaluator(),
            name="TinyTask")


@dataclass
class EchoConfig(StrategyConfig):
    value: float = param(50.0, "The value solve() returns")
    retries: int = param(3, "Unused knob, pins int coercion")


class EchoStrategy(Strategy):
    """Writes solve() returning a configured value. No LLM."""

    Config = EchoConfig

    def __call__(self, toolkit, config=None):
        from groundhog.utils.direction import write_direction
        cfg = self._resolve_config(config)
        ws = toolkit.history.workspace()
        (ws.path / "solution.py").write_text(
            f"def solve():\\n    return {cfg.value}", encoding="utf-8")
        seq = len(toolkit.history.list(only_done=False))
        write_direction(ws.path, f"echo direction {seq}")
        result = toolkit.task.evaluate(ws.path)
        toolkit.finalize(ws, result, strategy=self.name)
        return {}


def build_toolkit():
    from groundhog import FolderAttemptHistory, assemble_toolkit
    here = Path(__file__).parent
    return assemble_toolkit(task, history=FolderAttemptHistory(here), path=here)
'''


def _write_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task.py").write_text(_TASK_BODY, encoding="utf-8")
    return run_dir


@contextmanager
def _in_dir(path):
    saved = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(saved)


# --- list -------------------------------------------------------------------

def test_strategy_list_includes_builtins_and_task_module(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "improve" in out
    assert "builtin" in out
    assert "echo" in out
    assert "task" in out


def test_strategy_list_works_outside_a_run_dir(tmp_path, capsys):
    with _in_dir(tmp_path):
        rc = strategy_group(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "improve" in out
    assert "echo" not in out


# --- show -------------------------------------------------------------------

def test_strategy_show_prints_params_table(tmp_path, capsys):
    with _in_dir(tmp_path):
        rc = strategy_group(["show", "improve"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "improve" in out
    assert "max_retries" in out
    assert "default=3" in out
    assert "Max retry attempts" in out


def test_strategy_show_json_is_machine_readable(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["show", "echo", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["name"] == "echo"
    assert data["source"] == "task"
    assert data["doc"] == "Writes solve() returning a configured value. No LLM."
    assert data["params"]["value"] == {
        "type": "float",
        "default": 50.0,
        "description": "The value solve() returns",
    }
    assert data["params"]["retries"]["type"] == "int"


def test_strategy_show_unknown_name_errors(tmp_path, capsys):
    with _in_dir(tmp_path):
        rc = strategy_group(["show", "nope"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No strategy named" in out


# --- run --------------------------------------------------------------------

def test_strategy_run_executes_n_times_with_attribution(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["run", "echo", "-n", "2"])
    out = capsys.readouterr().out
    assert rc == 0

    history = FolderAttemptHistory(run_dir)
    attempts = history.list()
    assert len(attempts) == 2
    for a in attempts:
        assert a.metadata["strategy"] == "echo"
        # The standard finish cached the score note per run.
        assert history.get_note(a.id, "score") == "1.0000"
    assert out.count("1.0000") == 2


def test_strategy_run_set_overrides_config_with_coercion(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["run", "echo", "--set", "value=75"])
    capsys.readouterr()
    assert rc == 0

    history = FolderAttemptHistory(run_dir)
    attempts = history.list()
    assert len(attempts) == 1
    assert "return 75.0" in attempts[0].code
    assert history.get_note(attempts[0].id, "score") == "0.7500"


def test_strategy_run_unknown_set_key_errors(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["run", "echo", "--set", "nope=1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Unknown config key" in out
    assert FolderAttemptHistory(run_dir).list(only_done=False) == []


def test_strategy_run_bad_coercion_errors(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["run", "echo", "--set", "value=abc"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "--set value" in out


def test_strategy_run_unknown_strategy_errors(tmp_path, capsys):
    run_dir = _write_run_dir(tmp_path)
    with _in_dir(run_dir):
        rc = strategy_group(["run", "nope"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No strategy named" in out
