"""Acceptance — what counts as progress.

Vault: Acceptance Process.md — emerges from scorer + optimizer behavior.

Two operations:
- is_improvement: is result A better than result B?
- gate: does this result pass the minimum threshold relative to the prior?
"""

from abc import ABC, abstractmethod
from typing import List

from groundhog.base.types import EvaluationResult, EvalStage


class Acceptance(ABC):
    """Interface for determining what counts as progress."""

    @abstractmethod
    def is_improvement(self, result_a: EvaluationResult, result_b: EvaluationResult,
                       stages: List[EvalStage]) -> bool: ...

    @abstractmethod
    def gate(self, result: EvaluationResult, prior_result: EvaluationResult,
             stages: List[EvalStage], threshold: float = 0.9) -> bool: ...
