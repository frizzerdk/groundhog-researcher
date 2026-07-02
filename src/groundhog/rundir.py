"""Run-dir loader — import a task.py and call its ``build_toolkit()``.

The run-dir contract: a run dir is a folder whose ``task.py`` exposes

    def build_toolkit() -> Toolkit:
        tk = assemble_toolkit(task, ...)   # construct + configure the bench
        tk.llm = auto_registry()
        return tk

``build_toolkit()`` constructs and configures — it never runs anything. The
loader imports the module under a non-``__main__`` name and calls that one
function; the returned toolkit IS the run's real bench (its ``history`` is
the store the run actually uses — no reconstruction, no guessing).

The old loader monkeypatched ``SimpleOptimizer.run/status`` to no-ops and
scanned module globals for an optimizer, so that "unguarded" scripts could be
imported without firing a run. That machinery is gone: the contract makes the
module import-safe by construction (heavy work is ``__main__``-guarded; Data
is lazy by convention).

The one deliberate side effect kept: the import and the ``build_toolkit()``
call run with ``run_dir`` as the working directory and on ``sys.path``, so
relative store paths (``GitAttemptHistory(".")``) and sibling imports
(``from mock_strategy import ...``) resolve against the run dir, not
whatever directory the CLI happened to be launched from.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from groundhog.base.types import Task
from groundhog.base.toolkit import Toolkit
from groundhog.base.attempt_history import AttemptHistory

# Unique-ish counter so repeated loads don't collide in sys.modules (we use a
# throwaway name and never leave the module registered permanently anyway).
_LOAD_COUNTER = itertools.count()


@dataclass
class LoadedRun:
    """Everything a consumer needs from a run dir — the bench, never a run."""
    task: Task
    toolkit: Toolkit
    history: AttemptHistory
    module: object
    run_dir: Path


def find_task_py(run_dir: Optional[Path] = None) -> Path:
    """Locate the run dir's ``task.py``.

    If ``run_dir`` is given, use ``run_dir/task.py`` (must exist). Otherwise
    search the cwd, then walk up to ~5 parent dirs. Raise ``FileNotFoundError``
    with a clear message if none is found.
    """
    if run_dir is not None:
        candidate = Path(run_dir) / "task.py"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"No task.py in {Path(run_dir).resolve()} — "
            f"run from a task folder or pass --run-dir."
        )

    here = Path.cwd().resolve()
    for d in [here, *here.parents][:6]:
        candidate = d / "task.py"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No task.py found in {here} or its parents — "
        f"run from a task folder (one created by `groundhog init`) "
        f"or pass --run-dir DIR."
    )


def load_run(run_dir: Optional[Path] = None) -> LoadedRun:
    """Import a run dir's task.py, call its ``build_toolkit()``, return the bench.

    Raises a clear error if the module doesn't define the contract.
    """
    task_py = find_task_py(run_dir)
    resolved_run_dir = task_py.parent

    saved_cwd = os.getcwd()
    inserted = str(resolved_run_dir)
    mod_name = f"groundhog_runtask_{next(_LOAD_COUNTER)}"
    try:
        os.chdir(resolved_run_dir)
        sys.path.insert(0, inserted)

        spec = importlib.util.spec_from_file_location(mod_name, str(task_py))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load a module spec from {task_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)

            hook = getattr(module, "build_toolkit", None)
            if not callable(hook):
                raise RuntimeError(
                    f"{task_py} does not define `def build_toolkit() -> Toolkit` — "
                    f"the run-dir contract. Define it beside your task:\n"
                    f"    def build_toolkit():\n"
                    f"        tk = assemble_toolkit(task, ...)\n"
                    f"        tk.llm = auto_registry()\n"
                    f"        return tk"
                )
            toolkit = hook()
            if not isinstance(toolkit, Toolkit):
                raise RuntimeError(
                    f"{task_py}: build_toolkit() returned {type(toolkit).__name__}, "
                    f"expected a Toolkit (return the result of assemble_toolkit)."
                )
        finally:
            sys.modules.pop(mod_name, None)
    finally:
        os.chdir(saved_cwd)
        try:
            sys.path.remove(inserted)
        except ValueError:
            pass

    return LoadedRun(
        task=toolkit.task,
        toolkit=toolkit,
        history=toolkit.history,
        module=module,
        run_dir=resolved_run_dir,
    )
