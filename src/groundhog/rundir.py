"""Run-dir loader — import a task.py and hand back its task + toolkit + history.

A "run dir" is a folder with a ``task.py`` that defines a task (and usually
builds a ``SimpleOptimizer``). The CLI's ``attempt``/``eval`` commands need the
*task* and an *attempt history*, but NOT an actual optimization run.

The hard part is that task.py is user code with two shapes:

  * GUARDED — scaffold templates (``templates/basic.py`` etc.) put
    ``optimizer.run()`` under ``if __name__ == "__main__":``. Importing under a
    non-``__main__`` name is already safe.
  * UNGUARDED — real run scripts (e.g. ``C:/repo/groundhog-runs/task.py``) build
    the optimizer and call ``optimizer.run(n)`` at module top level, reading
    ``sys.argv``. Importing that as-is would kick off a real optimization.

``load_run`` sandboxes the import so BOTH shapes load without running anything:
it chdir's into the run dir, neutralizes ``sys.argv``, and monkeypatches
``SimpleOptimizer.run``/``status`` to no-ops for the duration of the import. All
of that is restored in ``finally`` — even on ``SystemExit`` or exception.
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
from groundhog.optimizers.simple import SimpleOptimizer

# Unique-ish counter so repeated loads don't collide in sys.modules (we use a
# throwaway name and never leave the module registered permanently anyway).
_LOAD_COUNTER = itertools.count()


@dataclass
class LoadedRun:
    """Everything the CLI needs from a run dir, with the optimizer never run."""
    task: Task
    toolkit: Toolkit
    history: AttemptHistory
    optimizer: Optional[SimpleOptimizer]
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


def history_for(run_dir: Path) -> AttemptHistory:
    """Pick the attempt-history backend for ``run_dir``.

    Git if a bare store already exists at ``run_dir/attempts/.git``, else the
    plain folder backend. Both are rooted at the run dir.
    """
    run_dir = Path(run_dir)
    if (run_dir / "attempts" / ".git").exists():
        from groundhog.histories.git import GitAttemptHistory
        return GitAttemptHistory(run_dir)
    from groundhog.histories.folder import FolderAttemptHistory
    return FolderAttemptHistory(run_dir)


def build_toolkit(task: Task, run_dir: Path, through: Optional[str] = None) -> Toolkit:
    """Build a no-LLM toolkit for a task by constructing a SimpleOptimizer.

    We only want ``optimizer.toolkit`` (task, history, learnings, logging,
    get_prior, through) — never ``.run()``. The backend is chosen by
    ``history_for``. No LLM is installed: eval / history / learnings reads need
    none.
    """
    run_dir = Path(run_dir)
    history = history_for(run_dir)
    optimizer = SimpleOptimizer(task, history=history, path=run_dir, through=through)
    return optimizer.toolkit


def load_run(run_dir: Optional[Path] = None, through: Optional[str] = None) -> LoadedRun:
    """Import a run dir's task.py and return its task + toolkit + history.

    Robust to guarded and unguarded task.py (see module docstring). The
    optimizer is never run. ``through`` only matters when task.py has no
    optimizer and we build a toolkit ourselves.
    """
    task_py = find_task_py(run_dir)
    resolved_run_dir = task_py.parent

    module = _import_sandboxed(task_py, resolved_run_dir)

    optimizer = _find_optimizer(module)
    task = _find_task(module, optimizer)

    if optimizer is not None:
        toolkit = optimizer.toolkit
        history = toolkit.history
    else:
        toolkit = build_toolkit(task, resolved_run_dir, through=through)
        history = toolkit.history

    return LoadedRun(
        task=task,
        toolkit=toolkit,
        history=history,
        optimizer=optimizer,
        module=module,
        run_dir=resolved_run_dir,
    )


def _import_sandboxed(task_py: Path, run_dir: Path):
    """Exec task.py under a non-``__main__`` name with side effects neutralized.

    Saves and restores cwd, sys.path, sys.argv, and SimpleOptimizer.run/status
    in a try/finally so an unguarded task.py can't fire a real run or print
    status at import time — and so we leave the process exactly as we found it,
    even if the import raises or calls ``sys.exit``.
    """
    saved_cwd = os.getcwd()
    saved_argv = sys.argv
    saved_run = SimpleOptimizer.run
    saved_status = SimpleOptimizer.status
    inserted_path = str(run_dir)
    mod_name = f"groundhog_runtask_{next(_LOAD_COUNTER)}"

    try:
        os.chdir(run_dir)
        sys.path.insert(0, inserted_path)
        sys.argv = ["task.py"]
        SimpleOptimizer.run = lambda self, *a, **k: None
        SimpleOptimizer.status = lambda self, *a, **k: None

        spec = importlib.util.spec_from_file_location(mod_name, str(task_py))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load a module spec from {task_py}")
        module = importlib.util.module_from_spec(spec)
        # Register temporarily so dataclasses / pickling inside task.py resolve,
        # but always remove it afterward — we don't want it lingering globally.
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # An unguarded task.py that calls sys.exit() (e.g. `raise SystemExit`
            # on an unknown backend arg) still gives us a usable module object —
            # whatever was defined before the exit is on it.
            pass
        finally:
            sys.modules.pop(mod_name, None)
        return module
    finally:
        os.chdir(saved_cwd)
        try:
            sys.path.remove(inserted_path)
        except ValueError:
            pass
        sys.argv = saved_argv
        SimpleOptimizer.run = saved_run
        SimpleOptimizer.status = saved_status


def _find_optimizer(module) -> Optional[SimpleOptimizer]:
    """First SimpleOptimizer instance in the module's globals, or None."""
    for value in vars(module).values():
        if isinstance(value, SimpleOptimizer):
            return value
    return None


def _find_task(module, optimizer: Optional[SimpleOptimizer]) -> Task:
    """Resolve the task: optimizer.task, then a module-level Task instance,
    then the first Task subclass instantiated no-arg (guarded)."""
    if optimizer is not None and getattr(optimizer, "task", None) is not None:
        return optimizer.task

    # A module-level `task = Task(...)` instance.
    for value in vars(module).values():
        if isinstance(value, Task):
            return value

    # A Task SUBCLASS defined in the module — instantiate no-arg. Guard
    # exceptions: e.g. MNISTTask downloads data on __init__.
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, Task) and value is not Task:
            try:
                return value()
            except Exception as e:  # noqa: BLE001 — surface a clear, actionable error
                raise RuntimeError(
                    f"Found Task subclass {value.__name__} but could not "
                    f"instantiate it with no arguments: {e}. Define a "
                    f"module-level `task = {value.__name__}(...)` in task.py "
                    f"instead so the loader can use it directly."
                ) from e

    raise RuntimeError(
        "No task found in task.py — define a module-level `task = Task(...)`, "
        "a Task subclass, or an `optimizer = SimpleOptimizer(task, ...)`."
    )
