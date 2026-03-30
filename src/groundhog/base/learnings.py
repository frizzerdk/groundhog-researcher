"""Learnings — knowledge about the task accumulated during optimization.

Vault: Learnings.md — not *what was tried*, but *what was learned about trying*.

Strategies add observations, read accumulated knowledge.
Implementations: markdown file, database, LLM-curated, etc.
"""

from abc import ABC, abstractmethod


class Learnings(ABC):
    """Interface for knowledge storage.

    Strategies call add() to record observations and get() to read them.
    edit() allows refining existing knowledge.
    """

    @abstractmethod
    def add(self, text: str): ...

    @abstractmethod
    def get(self, last: int = 0, random: int = 0) -> str:
        """Read accumulated learnings.

        Args:
            last: number of most recent entries to include (0 = all)
            random: number of random older entries to include (0 = none)
        """
        ...

    @abstractmethod
    def edit(self, search: str, replace: str): ...
