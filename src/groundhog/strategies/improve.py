"""Improve strategy — refine existing code via LLM-generated diffs.

Vault: Strategy — Types of Action.md (Refine)

Takes the best prior attempt, shows its evaluation results to the LLM,
asks for improvements via SEARCH/REPLACE blocks, evaluates, and retries
on error with error feedback.
"""

from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.attempt_logger import (
    AssistantEvent, MarkdownAttemptLogger, SystemEvent, UserEvent, eval_event,
)
from groundhog.utils.codegen import extract_code, build_prompt


@dataclass
class ImproveConfig(StrategyConfig):
    """Configuration for the Improve strategy."""
    max_retries: int = param(3, "Max retry attempts when evaluation fails")
    learnings_last: int = param(20, "Most recent learnings to include in prompt")
    learnings_random: int = param(10, "Random older learnings to include for diversity")


class Improve(Strategy):
    """Refine existing code via LLM-generated diffs.

    Composed method pattern:
        init ->select prior ->workspace ->prepare ->generate ->evaluate+retry ->commit
    """

    Config = ImproveConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        prior = self._select_prior(toolkit)
        if prior is None:
            return {"skipped": "no prior to improve"}
        prior_score = self._score_result(prior.result, toolkit)
        learnings_count = toolkit.learnings.count() if hasattr(toolkit, 'learnings') else 0
        self.log.start(f"--- Improve | prior=#{prior.id} ({prior_score:.3f}) | retries={self.cfg.max_retries} | learnings={learnings_count}")
        ws = self._start_workspace(toolkit, prior)
        self.logger.attempt_start(ws.path)
        self._prepare_workspace(toolkit, ws, prior)
        self.log.inline("generating... ")
        self._do_work(toolkit, ws, prior)
        self.log.tock()
        self.log.inline("evaluating... ")
        result = self._evaluate_with_retries(toolkit, ws, prior)
        self.log.tock()
        self.log.inline("learnings... ")
        self._record_learnings(toolkit, ws, prior, result)
        self.log.tock()
        # Soft-gate: re-copy parent's core direction so a stray rewrite during
        # the LLM session can't fork the family. Backend-agnostic (works on git).
        from groundhog.utils.direction import (
            inherit_direction_from_attempt,
            inherited_direction_changed_from,
        )
        direction_changed = inherited_direction_changed_from(ws.path, prior)
        inherit_direction_from_attempt(prior, ws.path)
        metadata = {"strategy": "improve", "prior": prior.id,
                    "cost": round(self.logger.total_cost(), 6)}
        if direction_changed:
            metadata["direction_restored"] = True
        if self._is_duplicate_solution(ws, prior):
            metadata["non_promotable"] = True
            metadata["non_promotable_reason"] = "solution.py is byte-identical to parent"
        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata=metadata)
        from groundhog.utils.direction import workspace_name
        ws.name = workspace_name(ws.path)
        attempt = ws.commit(success=result.completed)
        return self._build_log(attempt, prior, result, toolkit)

    # --- Init ---

    def _init(self, toolkit, config):
        """Setup from toolkit and config."""
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.logger = getattr(toolkit, 'attempt_logger', None) or MarkdownAttemptLogger()
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()

    # --- Selection ---

    def _select_prior(self, toolkit):
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return toolkit.history.best(stages[-1].score)

    # --- Workspace setup ---

    def _start_workspace(self, toolkit, prior):
        return toolkit.history.workspace(parent=prior.id)

    def _prepare_workspace(self, toolkit, ws, prior):
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")
        (ws.path / "solution.py").write_text(prior.code, encoding="utf-8")
        # Inherit the family's core direction from the parent (backend-agnostic;
        # legacy approach.md is migrated forward to core_direction.md).
        from groundhog.utils.direction import inherit_direction_from_attempt
        inherit_direction_from_attempt(prior, ws.path)
        # Learnings are included in the prompt via build_prompt(learnings=...),
        # and logged in the attempt log — no need to duplicate as a file.

    # --- Prior results context ---

    def _build_prior_results_context(self, prior):
        result = prior.result
        if not result.completed:
            return f"Prior attempt FAILED at stage '{result.failed_stage}'."
        lines = ["Previous attempt results:"]
        for stage_name, stage_result in result.stages.items():
            if stage_result.errors:
                lines.append(f"  {stage_name}: FAILED — {stage_result.errors}")
            else:
                metrics = {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in stage_result.metrics.items()}
                lines.append(f"  {stage_name}: {metrics}")
        return "\n".join(lines)

    # --- Core work ---

    def _do_work(self, toolkit, ws, prior):
        prior_code = prior.code
        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None
        prior_results = self._build_prior_results_context(prior)

        prompt = build_prompt(
            context=toolkit.task.context.get(),
            prior_code=prior_code,
            learnings=learnings,
            mode="diff",
        )
        prompt += f"\n\n{prior_results}"

        # Include the family's core direction if available (legacy approach.md
        # is read as a fallback by the helper).
        from groundhog.utils.direction import read_direction
        direction_text = read_direction(ws.path)
        if direction_text and direction_text.strip():
            prompt += (
                "\n\n## Core direction (preserve — the algorithmic invariant)\n"
                f"{direction_text.strip()}"
            )

        if not hasattr(toolkit, 'llm'):
            return

        # System prompt for single-call code improvement.
        # Keep generic — task-specific guidance belongs in the task context,
        # run-specific observations belong in learnings.
        # Only put rules here that apply to ANY improve call on ANY task.
        system_prompt = """You are an expert programmer improving code iteratively. Each change is evaluated automatically.

Rules:
1. Analyze first: what approach is being used, what has been tried, what hasn't been tried, and what is the biggest limitation? Then propose a change.
2. Focus on one problem at a time. A complete rewrite to fix one limitation is fine — changing multiple unrelated things is not.
3. Read the learnings — don't repeat approaches that already failed.
4. Simpler is better. Removing unhelpful code is a valid improvement.
5. Be bold and creative. Prefer untried approaches over tweaking what exists. Large rewrites are welcome — just address one problem per iteration.

Output your changes as SEARCH/REPLACE blocks."""
        self.logger.log(UserEvent(content=prompt))
        self.logger.log(SystemEvent(content=system_prompt))

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage))

        new_code, diff = extract_code(response.text, prior_code)
        if new_code:
            self.log.inline(f"{diff.method} ({diff.blocks} blocks)... " if diff.blocks else f"{diff.method}... ")
            (ws.path / "solution.py").write_text(new_code, encoding="utf-8")
        else:
            self.log.inline("no changes... ")

    # --- Evaluation with retries ---

    def _evaluate_with_retries(self, toolkit, ws, prior):
        for attempt_num in range(self.cfg.max_retries + 1):
            result = toolkit.task.evaluate(ws.path, through=self.through)
            self.logger.log(eval_event(result, self._score_result(result, toolkit)))

            if result.completed:
                return result

            if attempt_num < self.cfg.max_retries and hasattr(toolkit, 'llm'):
                error_stage = result.stages[result.failed_stage]
                code = (ws.path / "solution.py").read_text(encoding="utf-8")
                self.log.inline(f"retry {attempt_num + 1}... ")
                self._retry_fix(toolkit, ws, prior, code, error_stage, attempt_num + 1)

        return result

    def _retry_fix(self, toolkit, ws, prior, broken_code, error_stage, retry_num):
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

        fixed_code, diff = extract_code(response.text, broken_code)
        if fixed_code:
            (ws.path / "solution.py").write_text(fixed_code, encoding="utf-8")

    # --- Learnings ---

    def _record_learnings(self, toolkit, ws, prior, result):
        if not hasattr(toolkit, 'learnings') or not hasattr(toolkit, 'llm'):
            return

        prior_score = self._score_result(prior.result, toolkit)
        new_score = self._score_result(result, toolkit)
        new_code = (ws.path / "solution.py").read_text(encoding="utf-8") if (ws.path / "solution.py").exists() else ""

        prompt = (
            "Compare these two attempts at the same coding task.\n\n"
            f"PRIOR CODE (score {prior_score:.4f}):\n```\n{prior.code}\n```\n\n"
            f"NEW CODE (score {new_score:.4f}):\n```\n{new_code}\n```\n\n"
            f"The new attempt {'improved' if new_score > prior_score else 'regressed' if new_score < prior_score else 'matched'} "
            f"({prior_score:.4f} ->{new_score:.4f}).\n\n"
            "Write 1-2 directive learnings, one line each, in the form:\n"
            "[tried X] -> [because/observed Y] -> [next time do Z]\n"
            "Be specific about techniques and numbers "
            "(e.g. 'raised augmentation 5x->20x -> val acc dropped 0.91->0.87 -> keep augmentation <=10x'). "
            "One actionable line, not a paragraph. These notes guide future attempts."
        )
        system_prompt = "You are a concise research assistant. Write directive, actionable learnings — one line each."

        self.logger.log(UserEvent(content=prompt, data={"label": "Learnings"}))
        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.logger.log(AssistantEvent(content=response.text, role=response.model,
                                       cost=response.cost, usage=response.usage, data={"label": "Learnings"}))

        toolkit.learnings.add(response.text)

    @staticmethod
    def _is_duplicate_solution(ws, prior) -> bool:
        """True iff committed solution.py equals the parent's (backend-agnostic)."""
        from groundhog.utils.direction import solution_matches_attempt
        return solution_matches_attempt(ws.path, prior)

    def _score_result(self, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        return stages[-1].score(final_result) if final_result else -1.0

    # --- Logging ---

    def _build_log(self, attempt, prior, result, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        final_name = stages[-1].name
        final_result = result.stages.get(final_name)
        score = stages[-1].score(final_result) if final_result else -1.0
        return {
            "attempt": attempt.id,
            "prior": prior.id,
            "score": round(score, 4),
            "strategy": "improve",
        }
