"""CrossPollinateAgent — cross-family agent strategy.

Same iterative agent loop as :class:`AgentStrategy`, but the agent is
explicitly given an *inspiration* attempt from a **different** direction
family alongside its parent. The agent is expected to keep the parent's
core direction (the family's algorithmic invariant) while borrowing
ideas — features, preprocessing, hyperparameters, tricks — from the
inspiration.

Mechanism: the inspiration is not the prior. Refine-style inheritance
applies as usual (parent's ``core_direction.md`` is inherited and
re-enforced at commit). The inspiration's files are reachable via the
existing ``get-prior-file`` tool with the inspiration's attempt id.

Vault: Strategy — Types of Action.md (Combine), Implementation Details/
Family Identity.md.
"""

from __future__ import annotations

from groundhog.strategies.agent import AgentStrategy
from groundhog.utils.selection import get_trunk_leaders


_INSPIRATION_PROMPT = """\

## Cross-pollination context

This is a CROSS-POLLINATE session. You have a parent (which defines
your family's core direction — preserve that) AND an INSPIRATION from
a different direction family.

Inspiration: attempt {insp_num} (family: {insp_family}).
Score: {insp_score:.4f}.

Use the `get-prior-file` tool with `attempt={insp_num}` to read its
solution.py, artifacts, or learnings. Look for transferable ideas:
features, preprocessing, post-processing, hyperparameters, library
choices, tricks. Do NOT switch to the inspiration's core direction —
your family's algorithmic backbone stays.

Goal: improve the parent by borrowing ideas, not by replacing it.
"""


class CrossPollinateAgent(AgentStrategy):
    """Agent strategy that selects an inspiration from a different family.

    The selected prior remains the family anchor (refine-style inheritance);
    the inspiration is exposed via cross-family context in the explore
    prompt and reachable via the standard prior-reader tools.
    """

    def __call__(self, toolkit, config=None):
        # Initialize so we have access to scorer / log; defer the parent
        # __call__ until after we've located the inspiration so we can
        # weave it into the prompt-building path.
        self._init(toolkit, config)
        if not hasattr(toolkit, "agent"):
            return {"skipped": "no agent backend available"}
        if not hasattr(toolkit, "history"):
            return {"skipped": "no history on toolkit"}

        prior = self._select_prior(toolkit)
        if prior is None:
            return {"skipped": "no prior available — cross-pollinate needs a parent"}

        inspiration = self._select_inspiration(toolkit, prior)
        if inspiration is None:
            return {"skipped": "no different-family attempt to draw inspiration from"}

        # Stash for prompt-building hook; cleared after commit.
        self._fixed_prior = prior
        self._inspiration = inspiration
        try:
            return super().__call__(toolkit, config)
        finally:
            self._fixed_prior = None
            self._inspiration = None

    def _select_prior(self, toolkit):
        fixed = getattr(self, "_fixed_prior", None)
        if fixed is not None:
            return fixed
        return super()._select_prior(toolkit)

    # --- Selection ---

    def _select_inspiration(self, toolkit, prior):
        """Pick the best-scoring trunk leader whose family differs from the
        parent's. Returns ``None`` if no other family exists."""
        from groundhog.utils.direction import read_direction, normalize_direction

        prior_text = read_direction(prior.path) if hasattr(prior, "path") else None
        prior_family = normalize_direction(prior_text) if prior_text else None

        stages = toolkit.task.evaluator.eval_stages(
            toolkit.task.data, through=self.through
        )
        scorer = stages[-1].score
        candidates = []
        for leader in get_trunk_leaders(toolkit.history, scorer, exclude=prior.number):
            if not leader.result.completed or leader.metadata.get("non_promotable"):
                continue
            if self._score_attempt(leader, scorer) <= 0:
                continue
            text = read_direction(leader.path) if hasattr(leader, "path") else None
            family = normalize_direction(text) if text else None
            # Different family iff keys differ (sentinel None counts as
            # "no family", which is treated as different from any real
            # family — useful for legacy attempts).
            if family != prior_family:
                candidates.append(leader)
        if not candidates:
            return None
        # Highest score among different-family leaders.
        return max(candidates, key=lambda a: self._score_attempt(a, scorer))

    @staticmethod
    def _score_attempt(attempt, scorer):
        result = attempt.result
        if not result.completed:
            return -1.0
        last = list(result.stages.values())[-1]
        return scorer(last)

    def _prior_tool_options(self, toolkit, ws, prior):
        """Expose inspiration candidates while hiding the parent family."""
        from groundhog.utils.direction import read_direction

        parent_direction = read_direction(prior.path) if hasattr(prior, "path") else None
        return {"scope": "all", "exclude_direction": parent_direction}

    # --- Prompt augmentation ---

    def _build_approach_context(self, ws):
        """Standard core-direction context plus an inspiration block."""
        base = super()._build_approach_context(ws)
        insp = getattr(self, "_inspiration", None)
        if insp is None:
            return base

        from groundhog.utils.direction import read_direction, direction_title
        insp_text = read_direction(insp.path) if hasattr(insp, "path") else None
        insp_family = direction_title(insp_text or "")
        # Reuse the strategy's scorer-derived score helper.
        try:
            stages = self._toolkit.task.evaluator.eval_stages(
                self._toolkit.task.data, through=self.through
            )
            insp_score = self._score_attempt(insp, stages[-1].score)
        except Exception:
            insp_score = float("nan")

        return base + _INSPIRATION_PROMPT.format(
            insp_num=insp.number,
            insp_family=insp_family,
            insp_score=insp_score,
        )

    # --- Metadata ---

    def _build_metadata(self, prior):
        meta = super()._build_metadata(prior)
        meta["strategy"] = "cross_pollinate_agent"
        if getattr(self, "_inspiration", None) is not None:
            meta["inspiration"] = self._inspiration.number
        return meta
