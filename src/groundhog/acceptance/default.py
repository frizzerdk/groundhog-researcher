"""Default acceptance — compare highest common stage scorer."""

from typing import List

from groundhog.base.types import EvaluationResult, EvalStage
from groundhog.base.acceptance import Acceptance


class DefaultAcceptance(Acceptance):
    """Compares highest common completed stage. Gate is relative to parent score."""

    def _find_common_stage(self, result_a: EvaluationResult, result_b: EvaluationResult,
                           stages: List[EvalStage]) -> EvalStage:
        for stage in reversed(stages):
            if stage.name in result_a.stages and stage.name in result_b.stages:
                a_stage = result_a.stages[stage.name]
                b_stage = result_b.stages[stage.name]
                if not a_stage.errors and not b_stage.errors:
                    return stage
        return None

    def is_improvement(self, result_a: EvaluationResult, result_b: EvaluationResult,
                       stages: List[EvalStage]) -> bool:
        stage = self._find_common_stage(result_a, result_b, stages)
        if stage is None:
            return False
        return stage.score(result_a.stages[stage.name]) > stage.score(result_b.stages[stage.name])

    def gate(self, result: EvaluationResult, prior_result: EvaluationResult,
             stages: List[EvalStage], threshold: float = 0.9) -> bool:
        stage = self._find_common_stage(result, prior_result, stages)
        if stage is None:
            return False
        score = stage.score(result.stages[stage.name])
        prior_score = stage.score(prior_result.stages[stage.name])
        return score >= threshold * prior_score
