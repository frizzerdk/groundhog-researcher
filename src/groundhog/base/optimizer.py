"""Optimizer — the running environment for iterative function optimization.

The optimizer π(F_init, E, D, C) → F* is not just the loop — it's the entire
running environment. It takes in a Task (E, D, C) and from that point owns
everything: assembling the toolkit, running strategies, tracking attempts,
accumulating learnings, and providing a place for human guidance to land.

The optimizer:
- Takes the Task's evaluation stages and exposes them as callable tools
- Provides persistent storage for attempt history and learnings
- Assembles a Toolkit from Task capabilities, infrastructure, and custom tools
- Accepts human guidance — steering, overrides, queue entries
- Manages configuration that can evolve as the run progresses

The loop:
    Learnings inform which Strategy runs and how
    → Strategy receives the Toolkit and executes
    → Strategy produces a candidate (in a Workspace)
    → Attempt is recorded in Attempt History (code + raw results + artifacts)
    → Scorer interprets results → rankings, improvement detection
    → Outcomes feed back into Learnings
    → Repeat
"""

from abc import ABC, abstractmethod

from groundhog.base.types import Task
from groundhog.base.attempt_history import AttemptHistory


class Optimizer(ABC):
    """Interface for an optimizer.

    Implementations decide how to select strategies, build the toolkit,
    manage the loop, and handle configuration. The optimizer codes against
    AttemptHistory and Strategy interfaces, not specific backends.
    """

    task: Task
    history: AttemptHistory

    @abstractmethod
    def run(self, n: int = 10): ...
