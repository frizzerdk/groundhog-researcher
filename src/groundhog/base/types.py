"""Core data types for groundhog."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class StageResult:
    """Output from a single evaluation stage.

    Score is the evaluator's judgment (useful in the moment, not persisted).
    Metrics are raw measurements — can be anything (scalars, timeseries, etc.).
    The stage's scorer is designed to interpret whatever metrics are here.
    Artifacts are non-numeric outputs for inspection (plots, logs, etc.).
    Errors is a list — consumers decide how to handle them.
    """
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Any] = field(default_factory=dict)
    warnings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete output from evaluating a candidate across stages."""
    stages: Dict[str, StageResult] = field(default_factory=dict)
    completed: bool = True
    failed_stage: Optional[str] = None


def _default_score(result: StageResult) -> float:
    """Default scoring: return result.score."""
    return result.score


@dataclass(frozen=True)
class EvalStage:
    """One level of evaluation fidelity.

    Stages are ordered cheapest-first in a list. The last stage is ground truth.
    Each stage can define its own scorer to interpret its metrics.
    """
    name: str
    description: str
    call: Callable  # (code_or_path) -> StageResult — accepts str or Path
    scorer: Callable[[StageResult], float] = _default_score

    def score(self, result: StageResult) -> float:
        """Score a result using this stage's scorer."""
        return self.scorer(result)


class Data(ABC):
    """Provides train and test data splits."""

    @abstractmethod
    def get_train(self) -> Any: ...

    @abstractmethod
    def get_test(self) -> Any: ...


class Context(ABC):
    """Provides context/instructions for code generation."""

    @abstractmethod
    def get_brief(self) -> str: ...

    @abstractmethod
    def get_extended(self) -> str: ...

    def get(self) -> str:
        return f"{self.get_brief()}\n\n{self.get_extended()}"

    def get_scoring(self) -> str:
        """Optional: describe the scoring function for the agent prompt.
        Override to explain how score is computed, what matters most, etc."""
        return ""


class Evaluator(ABC):
    """Evaluates code: E(code_or_path, D) -> StageResult.

    Receives either a code string or a Path to a workspace directory.
    If Path, read whatever files you need (e.g. path / "solution.py").
    If string, use it directly. This lets simple evaluators exec code
    while complex evaluators inspect workspace artifacts.
    """

    @abstractmethod
    def evaluate(self, code_or_path: Union[str, Path], data: 'Data') -> StageResult: ...

    def get_stages(self, data: 'Data') -> List[EvalStage]:
        """Return evaluation stages, cheapest first. Override to add cheaper stages."""
        return [
            EvalStage("evaluate", "Full evaluation",
                      lambda code_or_path, d=data: self.evaluate(code_or_path, d))
        ]

    def eval_stages(self, data: 'Data', through=None) -> List[EvalStage]:
        """Return stage list, optionally sliced by name or index."""
        stages = self.get_stages(data)
        if through is None:
            return stages
        if isinstance(through, int):
            idx = through if through >= 0 else len(stages) + through
            idx = max(0, min(idx, len(stages) - 1))
            return stages[:idx + 1]
        if isinstance(through, str):
            for i, stage in enumerate(stages):
                if stage.name == through:
                    return stages[:i + 1]
        return stages

    def run(self, code_or_path: Union[str, Path], data: 'Data', through=None) -> EvaluationResult:
        """Evaluate by cascading through stages. Stops on error."""
        stages = self.eval_stages(data, through=through)
        eval_result = EvaluationResult()
        for stage in stages:
            stage_result = stage.call(code_or_path)
            eval_result.stages[stage.name] = stage_result
            if stage_result.errors:
                eval_result.completed = False
                eval_result.failed_stage = stage.name
                break
        return eval_result


class Task:
    """Composition of Data + Context + Evaluator. The problem definition."""

    def __init__(self, data: 'Data', context: 'Context', evaluator: 'Evaluator', name: str = None):
        self.data = data
        self.context = context
        self.evaluator = evaluator
        self.name = name or self.__class__.__name__

    def evaluate(self, code_or_path: Union[str, Path], through=None) -> EvaluationResult:
        """Evaluate against this task. Accepts code string or workspace Path."""
        return self.evaluator.run(code_or_path, self.data, through=through)
