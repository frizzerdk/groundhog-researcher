"""Fresh agent strategy — design from scratch in agent mode (no prior).

Same iterative loop as :class:`AgentStrategy` (tools, eval, learnings, fix,
reflect) but the agent always starts from an empty workspace. Useful for
opening a new family/trunk via the queue without anchoring to the live-
best attempt.

Direction handling at commit:
    1. If the agent wrote ``work/core_direction.md`` (or legacy
       ``work/approach.md``), it is promoted to attempt root.
    2. Otherwise, the strategy asks ``toolkit.llm`` to summarise the
       resulting ``solution.py`` into a 1-2 line core direction so every
       fresh attempt establishes a family identity descendants can
       inherit.

This is distinct from :class:`FreshApproach` (a single-shot LLM strategy):
``FreshAgentStrategy`` runs the full agent loop against a blank workspace.
"""

from groundhog.strategies.agent import AgentStrategy
from groundhog.utils.direction import (
    find_direction_path,
    promote_workspace_direction,
    write_direction,
)


_DIRECTION_PROMPT = (
    "Describe this code's CORE DIRECTION in 1-2 short lines: the "
    "algorithmic invariant a follow-up attempt should preserve "
    '(e.g. "CNN architecture", "rollout-based search", '
    '"rule-based with hand-tuned weights"). Narrow on purpose — '
    "not a description of the whole implementation, just the "
    "backbone that defines the family.\n\n"
    "```python\n{code}\n```"
)

_DIRECTION_SYSTEM_PROMPT = (
    "Write the core direction as 1-2 lines. No preamble, no code, "
    "no markdown headers."
)


class FreshAgentStrategy(AgentStrategy):
    """AgentStrategy that always starts from a blank workspace (no prior)."""

    def _select_prior(self, toolkit):
        return None

    def _prepare_workspace(self, toolkit, ws, prior):
        """Standard prep, plus seed an initial ``core_direction.md`` if the
        config provided one (e.g. queued by PlanApproaches)."""
        super()._prepare_workspace(toolkit, ws, prior)
        initial = (self.cfg.core_direction or self.cfg.initial_direction or "").strip()
        if initial:
            from groundhog.utils.direction import write_direction
            write_direction(ws.path, initial)

    def _finalize(self, ws, result, prior):
        """Ensure a direction exists before the fresh-direction gate runs."""
        promote_workspace_direction(ws.path)
        self._ensure_direction(ws)
        super()._finalize(ws, result, prior)

    def _ensure_direction(self, ws):
        """If no direction is recorded after promotion, ask the LLM to
        summarise ``solution.py`` into one. Best-effort — failures are
        logged but don't block commit."""
        if find_direction_path(ws.path) is not None:
            return  # Agent already wrote one; promotion handled it.
        code_path = ws.path / "solution.py"
        if not code_path.exists():
            return
        toolkit = getattr(self, "_toolkit", None)
        if toolkit is None or not hasattr(toolkit, "llm"):
            return
        try:
            code = code_path.read_text(encoding="utf-8")
            response = toolkit.llm.get("default").generate(
                prompt=_DIRECTION_PROMPT.format(code=code),
                system_prompt=_DIRECTION_SYSTEM_PROMPT,
            )
            self.cost += response.cost
            write_direction(ws.path, response.text)
        except Exception as e:
            # Direction generation is best-effort; not having one is
            # acceptable for legacy/fallback paths.
            self.log.inline(f"(direction-gen failed: {type(e).__name__}) ")
