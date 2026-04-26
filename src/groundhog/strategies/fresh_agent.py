"""Fresh agent strategy — design from scratch in agent mode (no prior).

Same iterative loop as :class:`AgentStrategy` (tools, eval, learnings, fix,
reflect) but the agent always starts from an empty workspace. Useful for
opening a new trunk via the queue without anchoring to the live-best
attempt.

This is distinct from :class:`FreshApproach` (a single-shot LLM strategy):
``FreshAgentStrategy`` runs the full agent loop against a blank workspace.
"""

from groundhog.strategies.agent import AgentStrategy


class FreshAgentStrategy(AgentStrategy):
    """AgentStrategy that always starts from a blank workspace (no prior)."""

    def _select_prior(self, toolkit):
        return None
