"""CrossPollinate strategy — improve one solution by drawing ideas from another.

Vault: Strategy — Types of Action.md (Cross-pollinate)

Selects a parent to improve and an inspiration from a different trunk.
Keeps the parent's core approach but incorporates techniques from the inspiration.
"""

from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.attempt_logger import (
    AssistantEvent, MarkdownAttemptLogger, SystemEvent, UserEvent, eval_event,
)
from groundhog.utils.codegen import extract_code, build_prompt
from groundhog.utils.selection import get_trunk_leaders


@dataclass
class CrossPollinateConfig(StrategyConfig):
    """Configuration for the CrossPollinate strategy."""
    max_retries: int = param(3, "Max retry attempts when evaluation fails")
    learnings_last: int = param(20, "Most recent learnings to include")
    learnings_random: int = param(10, "Random older learnings for diversity")


class CrossPollinate(Strategy):
    """Improve one solution by drawing ideas from a different approach.

    Composed method pattern:
        init → select parent + inspiration → workspace → generate → evaluate → commit
    """

    Config = CrossPollinateConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        prior, inspiration = self._select_pair(toolkit)
        if prior is None or inspiration is None:
            return {"skipped": "no alternative trunk for cross-pollination"}
        prior_score = self._score_result(prior.result, toolkit)
        insp_score = self._score_result(inspiration.result, toolkit)
        self.log.start(f"--- CrossPollinate | parent=#{prior.number} ({prior_score:.3f}) | inspiration=#{inspiration.number} ({insp_score:.3f})")
        ws = self._start_workspace(toolkit, prior)
        self.logger.attempt_start(ws.path)
        self._prepare_workspace(toolkit, ws, prior)
        self.log.inline("generating... ")
        self._do_work(toolkit, ws, prior, inspiration)
        self.log.tock()
        self.log.inline("evaluating... ")
        result = self._evaluate_with_retries(toolkit, ws)
        self.log.tock()
        # Soft-gate: re-copy parent's core direction so the borrowed ideas
        # don't drift the family identity.
        if hasattr(prior, "path"):
            from groundhog.utils.direction import (
                enforce_inherited_direction,
                inherited_direction_changed,
            )
            direction_changed = inherited_direction_changed(ws.path, prior.path)
            enforce_inherited_direction(ws.path, prior.path)
        metadata = {
            "strategy": "cross_pollinate",
            "prior": prior.number,
            "inspiration": inspiration.number,
            "cost": round(self.logger.total_cost(), 6),
        }
        if hasattr(prior, "path") and direction_changed:
            metadata["direction_restored"] = True
        # Flag if we landed on byte-identical code to either source.
        if self._is_duplicate_solution(ws, prior) or self._is_duplicate_solution(ws, inspiration):
            metadata["non_promotable"] = True
            metadata["non_promotable_reason"] = "solution.py is byte-identical to parent or inspiration"
        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata=metadata)
        attempt = ws.commit(success=result.completed)
        return self._build_log(attempt, prior, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.logger = getattr(toolkit, 'attempt_logger', None) or MarkdownAttemptLogger()
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()

    @staticmethod
    def _is_duplicate_solution(ws, other) -> bool:
        """True iff ws/solution.py == other/solution.py byte-for-byte."""
        if other is None or not hasattr(other, "path"):
            return False
        ours = ws.path / "solution.py"
        theirs = other.path / "solution.py"
        if not ours.exists() or not theirs.exists():
            return False
        try:
            return ours.read_bytes() == theirs.read_bytes()
        except OSError:
            return False

    # --- Selection ---

    def _select_pair(self, toolkit):
        """Select parent (to improve) and inspiration (from different trunk)."""
        # Parent: use toolkit.get_prior or fall back to best
        if hasattr(toolkit, 'get_prior'):
            prior = toolkit.get_prior(toolkit)
        else:
            stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
            prior = toolkit.history.best(stages[-1].score)

        if prior is None:
            return None, None

        # Inspiration: best from a different trunk
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        leaders = get_trunk_leaders(toolkit.history, stages[-1].score, exclude=prior.number)
        # Filter out failed attempts
        leaders = [a for a in leaders if a.result.completed and self._score_result(a.result, toolkit) > 0]
        if not leaders:
            return prior, None

        # Pick the highest-scoring leader that isn't the parent
        inspiration = max(leaders, key=lambda a: self._score_result(a.result, toolkit))
        return prior, inspiration

    # --- Workspace ---

    def _start_workspace(self, toolkit, prior):
        return toolkit.history.workspace(parent=prior.number)

    def _prepare_workspace(self, toolkit, ws, prior):
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")
        (ws.path / "solution.py").write_text(prior.code, encoding="utf-8")
        # Cross-pollinate stays in the parent's family: inherit the core
        # direction; borrow ideas from the inspiration's solution / artifacts
        # but keep the algorithmic backbone.
        if hasattr(prior, "path"):
            from groundhog.utils.direction import inherit_direction
            inherit_direction(prior.path, ws.path)

    # --- Core work ---

    def _do_work(self, toolkit, ws, prior, inspiration):
        if not hasattr(toolkit, 'llm'):
            return

        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None

        prompt_parts = [
            "Improve the base approach by incorporating ideas from the inspiration.",
            "Keep the base approach's core algorithm but adapt techniques that could help.",
            f"\n## Task\n{toolkit.task.context.get()}",
        ]

        if learnings:
            prompt_parts.append(f"\n## Learnings\n{learnings}")

        prompt_parts.append(f"\n## Base approach (keep this core algorithm)\n```python\n{prior.code}\n```")
        prompt_parts.append(f"\n## Inspiration (draw useful ideas from this)\n```python\n{inspiration.code}\n```")
        prompt_parts.append("\nUse SEARCH/REPLACE blocks to modify the base approach.")

        prompt = "\n\n".join(prompt_parts)

        system_prompt = """You are an expert programmer combining ideas from two different solutions.
The base approach is the one to improve — keep its core algorithm.
The inspiration has different techniques — adapt what could help.
Output SEARCH/REPLACE blocks modifying the base approach."""

        self.logger.log(UserEvent(content=prompt))
        self.logger.log(SystemEvent(content=system_prompt))

        response = toolkit.llm.get("high").generate(prompt=prompt, system_prompt=system_prompt)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage))

        new_code, diff = extract_code(response.text, prior.code)
        if new_code:
            self.log.inline(f"{diff.method} ({diff.blocks} blocks)... " if diff.blocks else f"{diff.method}... ")
            (ws.path / "solution.py").write_text(new_code, encoding="utf-8")
        else:
            self.log.inline("no changes... ")

    # --- Evaluation with retries ---

    def _evaluate_with_retries(self, toolkit, ws):
        for attempt_num in range(self.cfg.max_retries + 1):
            if not (ws.path / "solution.py").exists():
                (ws.path / "solution.py").write_text("# no code generated", encoding="utf-8")
            result = toolkit.task.evaluate(ws.path, through=self.through)
            self.logger.log(eval_event(result, self._score_result(result, toolkit)))

            if result.completed:
                return result

            if attempt_num < self.cfg.max_retries and hasattr(toolkit, 'llm'):
                error_stage = result.stages[result.failed_stage]
                code = (ws.path / "solution.py").read_text(encoding="utf-8")
                self.log.inline(f"retry {attempt_num + 1}... ")
                self._retry_fix(toolkit, ws, code, error_stage, attempt_num + 1)

        return result

    def _retry_fix(self, toolkit, ws, broken_code, error_stage, retry_num):
        error_context = f"Attempt {retry_num} failed with errors: {error_stage.errors}"
        prompt = build_prompt(
            context=toolkit.task.context.get(),
            prior_code=broken_code,
            mode="diff",
        )
        prompt += f"\n\nERROR — PLEASE FIX:\n{error_context}"
        system_prompt = "The code has errors. Fix them using SEARCH/REPLACE blocks."

        self.logger.log(UserEvent(content=prompt, data={"label": f"Retry {retry_num}"}))
        self.logger.log(SystemEvent(content=system_prompt))

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage, data={"label": f"Retry {retry_num}"}))

        fixed_code, _ = extract_code(response.text, broken_code)
        if fixed_code:
            (ws.path / "solution.py").write_text(fixed_code, encoding="utf-8")

    # --- Scoring ---

    def _score_result(self, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        return stages[-1].score(final_result) if final_result else -1.0

    # --- Logging ---

    def _build_log(self, attempt, prior, result, toolkit):
        score = self._score_result(result, toolkit)
        return {
            "attempt": attempt.number,
            "prior": prior.number,
            "score": round(score, 4),
            "strategy": "cross_pollinate",
        }
