"""Fresh approach strategy — generate code from scratch via LLM.

Vault: Strategy — Types of Action.md (Explore)

Two modes:
  blank    — no prior code, no learnings. Pure exploration from task context.
  different — shows top approaches from each trunk, asks for something fundamentally new.
"""

from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.conversation_log import conversation_log
from groundhog.utils.codegen import extract_code, build_prompt
from groundhog.utils.selection import get_trunk_leaders


@dataclass
class FreshApproachConfig(StrategyConfig):
    """Configuration for the FreshApproach strategy."""
    mode: str = param("different", "blank=no context, different=find new approach from existing trunks")
    max_retries: int = param(3, "Max retry attempts when evaluation fails")
    learnings_last: int = param(20, "Most recent learnings to include in prompt")
    learnings_random: int = param(10, "Random older learnings to include for diversity")


class FreshApproach(Strategy):
    """Generate code from scratch. No prior needed.

    Composed method pattern:
        init → workspace → prepare → generate full code → evaluate → commit
    """

    Config = FreshApproachConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        if not hasattr(toolkit, 'llm'):
            return {"skipped": "no LLM available"}
        self.log.start(f"--- FreshApproach ({self.cfg.mode})")
        ws = self._start_workspace(toolkit)
        self._prepare_workspace(toolkit, ws)
        self.log.inline("generating... ")
        self._do_work(toolkit, ws)
        self.log.tock()
        self.log.inline("approach... ")
        self._generate_approach(toolkit, ws)
        self.log.tock()
        self.log.inline("evaluating... ")
        result = self._evaluate_with_retries(toolkit, ws)
        self.log.tock()
        attempt = ws.commit(result, metadata={
            "strategy": "fresh_approach",
            "mode": self.cfg.mode,
            "cost": round(self.cost, 6),
        })
        return self._build_log(attempt, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.log_conversation = toolkit.conversation_log if hasattr(toolkit, 'conversation_log') else conversation_log
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

    # --- Workspace ---

    def _start_workspace(self, toolkit):
        return toolkit.history.workspace(parent=None)

    def _prepare_workspace(self, toolkit, ws):
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get())
        if self.cfg.mode != "blank" and hasattr(toolkit, 'learnings'):
            learnings_text = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random)
            if learnings_text:
                (ws.path / "learnings.md").write_text(learnings_text)

    # --- Core work ---

    def _do_work(self, toolkit, ws):
        if self.cfg.mode == "blank":
            self._do_blank(toolkit, ws)
        else:
            self._do_different(toolkit, ws)

    def _do_blank(self, toolkit, ws):
        """Pure exploration — just task context, no prior code or learnings."""
        prompt = build_prompt(
            context=toolkit.task.context.get(),
            mode="full",
        )
        system_prompt = "You are an expert programmer. Write complete, runnable code."

        self.log_conversation(ws.path, prompt, role="User")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("high").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response)

        code = extract_code(response.text)
        if code:
            (ws.path / "solution.py").write_text(code)

    def _do_different(self, toolkit, ws):
        """Find a fundamentally different approach from existing trunks."""
        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None

        # Get successful leaders from each trunk
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        leaders = [a for a in get_trunk_leaders(toolkit.history, stages[-1].score)
                   if a.result.completed]

        prompt_parts = [
            "Write a complete, runnable solution using a FUNDAMENTALLY DIFFERENT approach.",
            f"\n## Task\n{toolkit.task.context.get()}",
        ]

        if learnings:
            prompt_parts.append(f"\n## Learnings\n{learnings}")

        if leaders:
            prompt_parts.append("\n## Existing approaches (DO NOT reuse these core algorithms)")
            for leader in leaders:
                score = self._score_result(leader.result, toolkit)
                prompt_parts.append(f"\n### Approach (score: {score:.4f})\n```python\n{leader.code}\n```")
            prompt_parts.append("\nYour solution must use a different algorithm family than all of the above.")

        prompt = "\n\n".join(prompt_parts)

        system_prompt = """You are an expert programmer exploring a fundamentally new approach.
The existing approaches are shown so you can AVOID them — use a completely different algorithm.
Write complete, runnable code in a ```python block."""

        self.log_conversation(ws.path, prompt, role="User")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("high").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response)

        code = extract_code(response.text)
        if code:
            (ws.path / "solution.py").write_text(code)

    # --- Approach description ---

    def _generate_approach(self, toolkit, ws):
        """Ask LLM to describe the approach in 2-3 sentences. Written as attempt metadata."""
        code_path = ws.path / "solution.py"
        if not code_path.exists() or not hasattr(toolkit, 'llm'):
            return

        code = code_path.read_text()
        prompt = (
            f"Describe the core approach of this code in 2-3 sentences. "
            f"Focus on the algorithm and technique, not implementation details.\n\n"
            f"```python\n{code}\n```"
        )
        response = toolkit.llm.get("default").generate(
            prompt=prompt,
            system_prompt="Write a brief, factual description of the algorithm. No preamble."
        )
        self.cost += response.cost
        (ws.path / "approach.md").write_text(response.text.strip())

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

    def _apply_response(self, response_text, prior_code):
        from groundhog.utils.codegen import parse_diff, apply_diff
        diffs = parse_diff(response_text)
        if diffs:
            try:
                return apply_diff(prior_code, diffs)
            except ValueError:
                pass
        extracted = extract_code(response_text)
        if extracted and extracted != response_text.strip():
            return extracted
        return prior_code

    # --- Scoring ---

    def _score_result(self, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        return stages[-1].score(final_result) if final_result else -1.0

    # --- Logging ---

    def _build_log(self, attempt, result, toolkit):
        score = self._score_result(result, toolkit)
        return {
            "attempt": attempt.number,
            "score": round(score, 4),
            "strategy": "fresh_approach",
            "mode": self.cfg.mode,
        }
