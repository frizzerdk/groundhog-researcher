"""CrossPollinate strategy — improve one solution by drawing ideas from another.

Vault: Strategy — Types of Action.md (Cross-pollinate)

Selects a parent to improve and an inspiration from a different trunk.
Keeps the parent's core approach but incorporates techniques from the inspiration.
"""

from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.conversation_log import conversation_log
from groundhog.utils.codegen import extract_code, parse_diff, apply_diff, build_prompt
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
        self._prepare_workspace(toolkit, ws, prior)
        self.log.inline("generating... ")
        self._do_work(toolkit, ws, prior, inspiration)
        self.log.tock()
        self.log.inline("evaluating... ")
        result = self._evaluate_with_retries(toolkit, ws)
        self.log.tock()
        attempt = ws.commit(result, metadata={
            "strategy": "cross_pollinate",
            "prior": prior.number,
            "inspiration": inspiration.number,
            "cost": round(self.cost, 6),
        })
        return self._build_log(attempt, prior, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.log_conversation = toolkit.conversation_log if hasattr(toolkit, 'conversation_log') else conversation_log
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

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
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get())
        (ws.path / "solution.py").write_text(prior.code)

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

        self.log_conversation(ws.path, prompt, role="User")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("high").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response)

        new_code = self._apply_response(response.text, prior.code)
        (ws.path / "solution.py").write_text(new_code)

    def _apply_response(self, response_text, prior_code):
        diffs = parse_diff(response_text)
        if diffs:
            try:
                result = apply_diff(prior_code, diffs)
                self.log.inline(f"diff ({len(diffs)} blocks)... ")
                return result
            except ValueError:
                self.log.inline("diff failed, ")
        extracted = extract_code(response_text)
        if extracted and extracted != response_text.strip():
            self.log.inline("full rewrite... ")
            return extracted
        self.log.inline("no changes... ")
        return prior_code

    # --- Evaluation with retries ---

    def _evaluate_with_retries(self, toolkit, ws):
        for attempt_num in range(self.cfg.max_retries + 1):
            code_path = ws.path / "solution.py"
            if not code_path.exists():
                return toolkit.task.evaluate("# no code generated", through=self.through)
            code = code_path.read_text()
            result = toolkit.task.evaluate(code, through=self.through)

            if result.completed:
                return result

            if attempt_num < self.cfg.max_retries and hasattr(toolkit, 'llm'):
                error_stage = result.stages[result.failed_stage]
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

        self.log_conversation(ws.path, prompt, role="User", label=f"Retry {retry_num}")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response, label=f"Retry {retry_num}")

        new_code = self._apply_response(response.text, broken_code)
        (ws.path / "solution.py").write_text(new_code)

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
