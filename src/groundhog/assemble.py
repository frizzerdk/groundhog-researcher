"""assemble_toolkit — the one place a complete, standalone Toolkit is built.

The Toolkit is the universal interface: it HOLDS everything useful (task,
history, learnings, backends, tools, selection policy) and every consumer —
the optimizer, agents, notebooks, the CLI, tests — reaches one step deeper
for the verb it needs (``tk.task.evaluate()``, ``tk.history.best()``).

This factory assembles that bench without an optimizer anywhere in sight.
``SimpleOptimizer`` is one consumer: it receives a toolkit (or builds one
here for the legacy task-first signature), owns the strategy schedule and
the ``run()`` loop, and tunes prior selection by supplying a
``SelectionPolicy`` — data on the toolkit, never a rewritten function.

Run-dir contract: a ``task.py`` exposes ``def build_toolkit() -> Toolkit``
that calls this factory and configures the bench (e.g. ``tk.llm = ...``);
``__main__`` hands the finished toolkit to an optimizer and runs it.
"""

import random
from pathlib import Path
from typing import Callable, Optional

from groundhog.base.types import Task
from groundhog.base.toolkit import Toolkit
from groundhog.base.attempt_history import AttemptHistory
from groundhog.base.learnings import Learnings
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.learnings.markdown import MarkdownLearnings
from groundhog.tools.attempt_log import AttemptLog
from groundhog.tools.attempt_logger import MarkdownAttemptLogger
from groundhog.tools.log import StrategyLog
from groundhog.utils.selection import SelectionPolicy, default_prior_selector


def assemble_toolkit(
    task: Task,
    *,
    history: Optional[AttemptHistory] = None,
    learnings: Optional[Learnings] = None,
    path: Optional[Path] = None,
    through: Optional[str] = None,
    agent_through: Optional[str] = None,
    seed: int = 42,
    selection: Optional[SelectionPolicy] = None,
    agent_tools: Optional[Callable[[Toolkit], list]] = None,
) -> Toolkit:
    """Assemble a complete Toolkit for ``task``.

    Everything a consumer needs is installed here, exactly once:
    history/learnings stores (folder/markdown defaults rooted at ``path``),
    logging, seeded rng, agent-backend discovery, default agent tools, and
    the default prior selector reading ``toolkit.selection``.

    ``agent_tools`` is the task.py module hook — ``def agent_tools(toolkit)``
    returning a list of AgentTools. It is called LAST, against the fully
    assembled toolkit, so tools may close over ``toolkit.history`` /
    ``.task`` / ``.path``; its tools shadow same-named framework defaults
    (logged). Pass the FUNCTION, never its result.

    The caller configures the bench afterwards (``tk.llm = auto_registry()``,
    custom tools) and only then hands it to a consumer.
    """
    path = Path(path) if path else Path(".")
    tk = Toolkit(task=task, history=history or FolderAttemptHistory(path), path=path)
    tk.learnings = learnings or MarkdownLearnings(path)
    tk.log = StrategyLog()

    # Per-attempt event stream: strategies emit events through attempt_logger,
    # which fans out to attemptlog.jsonl/.md and the live console renderer
    # (AttemptLog — auto-disables ANSI/heartbeat on non-TTY so CI logs stay clean).
    console = AttemptLog()
    tk.attempt_log = console
    tk.attempt_logger = MarkdownAttemptLogger(console=console)

    if through:
        tk.through = through
    if agent_through:
        tk.agent_through = agent_through

    # Agent backends (optional — strategies check hasattr). Discovery is an
    # explicit act of assembly now, not a constructor side effect.
    from groundhog.backends.discover import auto_agent_registry
    agent_registry = auto_agent_registry()
    if agent_registry:
        tk.agent = agent_registry

    # Seeded rng + selection policy + the default prior selector. The
    # selector is a standing capability: it reads tk.selection (data) on
    # every call, so consumers tune selection by replacing the policy,
    # never by rewriting get_prior.
    tk.rng = random.Random(seed)
    tk.selection = selection or SelectionPolicy()
    tk.get_prior = default_prior_selector

    # The attempt pointer: a stable handle at "the attempt in flight".
    # Strategies bracket their attempt lifetime with it; the CLI points it at
    # any attempt; build-time tools close over it and read at invoke time.
    # (tk.workspace is the namespace alias — same object.)
    from groundhog.base.workspace_handle import WorkspaceHandle
    tk.ws = WorkspaceHandle(tk.history)
    tk.workspace = tk.ws

    # Agent tools LAST, so the task hook sees the finished bench: framework
    # defaults merged with the task.py hook's tools (task wins on a name
    # collision, and the shadow is logged). Set exactly once — no override
    # print fires.
    from groundhog.agents.tools import (
        build_default_agent_tools, collect_task_tools, _merge_agent_tools,
    )
    defaults = build_default_agent_tools(tk)
    custom = collect_task_tools(agent_tools, tk)
    tk.agent_tools = _merge_agent_tools(defaults, custom, layer="task", log=tk.log)

    return tk
