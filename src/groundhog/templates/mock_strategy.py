"""Mock strategy for testing. Randomly generates code with a fixed number.

Follows the general strategy pattern that real strategies should use:
    select prior → start workspace → prepare context → do work → evaluate → commit/abort
"""

import random

from groundhog import StageResult


class MockStrategy:
    """Generates `def solve(): return <random_float>`. No LLM needed.

    Follows the full workspace pattern as a template for real strategies.
    """

    def __call__(self, toolkit, config=None):
        prior = self._select_prior(toolkit)
        ws = self._start_workspace(toolkit, prior)
        self._prepare_workspace(toolkit, ws, prior)
        self._do_work(toolkit, ws)
        result = self._evaluate(toolkit, ws)
        attempt = ws.commit(result)
        return self._log(attempt, prior, result, toolkit)

    # --- Selection ---

    def _select_prior(self, toolkit):
        """Select the attempt to build from."""
        if hasattr(toolkit, 'get_prior'):
            return toolkit.get_prior(toolkit)
        return self._default_get_prior(toolkit)

    def _default_get_prior(self, toolkit):
        stages = toolkit.task.evaluator.get_stages(toolkit.task.data)
        return toolkit.history.best(stages[-1].score)

    # --- Workspace setup ---

    def _start_workspace(self, toolkit, prior):
        return toolkit.history.workspace(parent=prior.number if prior else None)

    def _prepare_workspace(self, toolkit, ws, prior):
        """Write context files to workspace before doing work."""
        # Write task context so an agent/LLM could read it
        (ws.path / "TASK_CONTEXT.md").write_text(toolkit.task.context.get())

        # Write prior code if available
        if prior:
            (ws.path / "solution.py").write_text(prior.code)

    # --- Core work ---

    def _do_work(self, toolkit, ws):
        """Generate code and write to workspace. (Mock: random number)"""
        rng = toolkit.rng if hasattr(toolkit, 'rng') else random.Random()
        value = rng.uniform(0, 100)
        code = f"def solve():\n    return {value}"
        (ws.path / "solution.py").write_text(code)

    # --- Evaluation ---

    def _evaluate(self, toolkit, ws):
        """Read code from workspace and evaluate."""
        code = (ws.path / "solution.py").read_text()
        through = toolkit.through if hasattr(toolkit, 'through') else None
        return toolkit.task.evaluate(code, through=through)

    # --- Logging ---

    def _log(self, attempt, prior, result, toolkit):
        stages = toolkit.task.evaluator.get_stages(toolkit.task.data)
        final_result = result.stages.get(stages[-1].name)
        score = stages[-1].score(final_result) if final_result else -1.0
        return {
            "attempt": attempt.number,
            "prior": prior.number if prior else None,
            "score": round(score, 4),
        }
