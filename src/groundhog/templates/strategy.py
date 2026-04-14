"""
Custom strategy template.

A STRATEGY is any action that moves the optimizer's state forward. It owns the
full loop: select what to work on, generate code, evaluate it, record the result.

REQUIRED:
  - A Config dataclass declaring configurable parameters with param()
  - __call__(self, toolkit, config=None) implementing the composed method pattern

THE COMPOSED METHOD PATTERN:
  __call__ should read like a story — each step is a clearly named method.
  Implementation details live in the step methods, not in __call__.

TOOLKIT CAPABILITIES (check with hasattr before using):
  toolkit.task          — the Task (data, context, evaluator)
  toolkit.history       — AttemptHistory (list, best, workspace, derive_trunks)
  toolkit.learnings     — Learnings (add, get, edit, count)
  toolkit.llm           — BackendRegistry (.get("default"), .get("high"), etc.)
  toolkit.log           — StrategyLog (start, inline, tock, info, end)
  toolkit.rng           — seeded Random instance
  toolkit.get_prior     — callable(toolkit) -> Attempt (pluggable prior selection)
  toolkit.through       — stage name to evaluate through (e.g. "evaluate")

EVALUATION:
  result = toolkit.task.evaluate(ws.path, through=self.through)
  Accepts a workspace Path (reads solution.py etc.) or a code string.

WORKSPACE PATTERN:
  ws = toolkit.history.workspace(parent=prior.number)  # or parent=None for fresh
  (ws.path / "solution.py").write_text(code, encoding="utf-8")
  write_result(ws.path, result, metadata={...})         # serialize results
  attempt = ws.commit(success=result.completed)          # finalize
  # or ws.abort() to discard without recording

APPROACH FILE:
  If starting a new trunk (parent=None), generate and write approach.md.
  If building on a prior, copy approach.md from the parent attempt.

LLM TIERS:
  toolkit.llm.get("default")  — standard generation
  toolkit.llm.get("high")     — complex reasoning, cross-pollination
  toolkit.llm.get("cheap")    — bulk/fast operations
  Missing tiers fall back to "default".

CONVERSATION LOGGING:
  self.log_conversation(ws.path, prompt, role="User")           # log a prompt
  self.log_conversation(ws.path, response)                      # log an LLMResponse
  self.log_conversation(ws.path, response, label="Learnings")   # tagged segment

RETURN VALUE:
  Return a Dict[str, Any] for logging/debug. Nothing depends on it.

RUN: uv run python task.py 10
"""

from dataclasses import dataclass

from groundhog import Strategy, StrategyConfig, param
from groundhog.tools.conversation_log import conversation_log
from groundhog.utils.codegen import extract_code, build_prompt


# ==========================================================================
# CONFIG — declare all configurable parameters here
# ==========================================================================

@dataclass
class MyStrategyConfig(StrategyConfig):
    """Each field uses param(default, "description").

    Introspectable: MyStrategy.Config().describe()
    Resolution: class defaults -> constructor -> call-time override
    """
    max_retries: int = param(3, "Max retry attempts when evaluation fails")
    learnings_last: int = param(20, "Most recent learnings to include in prompt")
    learnings_random: int = param(10, "Random older learnings to sample for diversity")


# ==========================================================================
# STRATEGY — the main class
# ==========================================================================

class MyStrategy(Strategy):
    """One-line description of what this strategy does.

    Composed method:
        init -> select prior -> workspace -> prepare -> work -> evaluate -> commit
    """

    Config = MyStrategyConfig

    # --- The story ---

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        prior = self._select_prior(toolkit)
        self.log.start(f"--- MyStrategy | prior=#{prior.number if prior else 'none'}")
        ws = self._start_workspace(toolkit, prior)
        self._prepare_workspace(toolkit, ws, prior)
        self.log.inline("generating... ")
        self._do_work(toolkit, ws, prior)
        self.log.tock()
        self.log.inline("evaluating... ")
        result = self._evaluate_with_retries(toolkit, ws)
        self.log.tock()
        from groundhog.utils.results import write_result
        write_result(ws.path, result, metadata={
            "strategy": "my_strategy",
            "prior": prior.number if prior else None,
            "cost": round(self.cost, 6),
        })
        attempt = ws.commit(success=result.completed)
        return {"attempt": attempt.number, "strategy": "my_strategy"}

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.log_conversation = toolkit.conversation_log if hasattr(toolkit, 'conversation_log') else conversation_log
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

    # --- Selection ---

    def _select_prior(self, toolkit):
        """Select the attempt to build from. Returns None for a fresh start."""
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return toolkit.history.best(stages[-1].score)

    # --- Workspace ---

    def _start_workspace(self, toolkit, prior):
        return toolkit.history.workspace(parent=prior.number if prior else None)

    def _prepare_workspace(self, toolkit, ws, prior):
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get(), encoding="utf-8")
        if prior:
            (ws.path / "solution.py").write_text(prior.code, encoding="utf-8")
            # Copy approach from parent
            if hasattr(prior, 'path') and (prior.path / "approach.md").exists():
                approach_text = (prior.path / "approach.md").read_text(encoding="utf-8")
                (ws.path / "approach.md").write_text(approach_text, encoding="utf-8")
        # Learnings are included in the prompt via build_prompt(learnings=...),
        # and logged in conversation.json — no need to duplicate as a file.

    # --- Core work (REPLACE THIS WITH YOUR LOGIC) ---

    def _do_work(self, toolkit, ws, prior):
        """Generate or modify code. This is where your strategy's logic goes."""
        if not hasattr(toolkit, 'llm'):
            return

        learnings = toolkit.learnings.get(last=self.cfg.learnings_last, random=self.cfg.learnings_random) if hasattr(toolkit, 'learnings') else None
        prompt = build_prompt(
            context=toolkit.task.context.get(),
            prior_code=prior.code if prior else None,
            learnings=learnings,
            mode="diff" if prior else "full",
        )

        system_prompt = "You are an expert programmer."
        self.log_conversation(ws.path, prompt, role="User")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response)

        new_code, diff = extract_code(response.text, prior.code if prior else "")
        if new_code:
            (ws.path / "solution.py").write_text(new_code, encoding="utf-8")

    # --- Evaluation with retries ---

    def _evaluate_with_retries(self, toolkit, ws):
        for attempt_num in range(self.cfg.max_retries + 1):
            if not (ws.path / "solution.py").exists():
                (ws.path / "solution.py").write_text("# no code generated", encoding="utf-8")
            result = toolkit.task.evaluate(ws.path, through=self.through)

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
        prompt = build_prompt(context=toolkit.task.context.get(), prior_code=broken_code, mode="diff")
        prompt += f"\n\nERROR — PLEASE FIX:\n{error_context}"
        system_prompt = "The code has errors. Fix them using SEARCH/REPLACE blocks."

        self.log_conversation(ws.path, prompt, role="User", label=f"Retry {retry_num}")
        self.log_conversation(ws.path, system_prompt, role="System")

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        self.log_conversation(ws.path, response, label=f"Retry {retry_num}")

        fixed_code, _ = extract_code(response.text, broken_code)
        if fixed_code:
            (ws.path / "solution.py").write_text(fixed_code, encoding="utf-8")
