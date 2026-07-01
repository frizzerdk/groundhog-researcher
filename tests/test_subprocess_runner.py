"""Subprocess runner: isolated execution with a print-safe return channel.

Regression provenance: run_code used to return the child's result as pickle
over stdout, so any print() in evaluated code corrupted the unpickle (audit
2026-07-01, bug #3). The result now travels through a dedicated temp file.
"""

import pytest

from groundhog.utils.subprocess_runner import run_code


def test_returns_entry_point_value():
    code = "def run(x):\n    return x * 2\n"
    assert run_code(code, "run", args=(21,)) == 42


def test_user_prints_do_not_corrupt_result():
    """LLM solutions print constantly — the result must survive it."""
    code = (
        "def run(x):\n"
        "    print('epoch 1: loss 0.5')\n"
        "    print('epoch 2: loss 0.3')\n"
        "    return {'score': x + 1}\n"
    )
    assert run_code(code, "run", args=(1,)) == {"score": 2}


def test_error_in_user_code_raises_runtime_error_with_detail():
    code = "def run():\n    raise ValueError('boom')\n"
    with pytest.raises(RuntimeError, match="boom"):
        run_code(code, "run")


def test_error_message_includes_stdout_tail():
    """Prints before a crash are surfaced for debugging, not swallowed."""
    code = (
        "def run():\n"
        "    print('got to step 3')\n"
        "    raise ValueError('boom')\n"
    )
    with pytest.raises(RuntimeError, match="got to step 3"):
        run_code(code, "run")


def test_kwargs_and_imports():
    code = "def run(a, b=0):\n    return math.floor(a + b)\n"
    result = run_code(code, "run", args=(1.5,), kwargs={"b": 2.0}, imports={"math": "math"})
    assert result == 3
