"""PlanApproaches strategy — propose new direction families.

Vault: Strategy — Types of Action.md (Plan); ported and simplified from
``EvaluatableExperiments/src/optimizers/strategies/plan_approaches.py``.

Reads task context + current direction families, asks the LLM to propose
N fundamentally-different core directions, and queues a fresh-agent run
for each (with the proposed direction text seeded as the workspace's
``core_direction.md``). Does NOT create an attempt itself.

Useful when the optimizer has converged on one or two families and you
want to deliberately plant new ones.
"""

import json
import re
from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.queue import add as queue_add


@dataclass
class PlanApproachesConfig(StrategyConfig):
    """Configuration for the PlanApproaches strategy."""
    n_directions: int = param(3, "How many distinct core directions to propose")
    fresh_strategy_name: str = param(
        "fresh_agent",
        "Name (in optimizer's strategy registry) of the strategy to queue per "
        "proposed direction. Default: fresh_agent.",
    )
    guidance_prefix: str = param(
        "",
        "Optional text prepended to the agent's guidance for each queued run.",
    )
    tier: str = param(
        "default",
        "Backend tier passed to the queued fresh-agent runs (default/high/budget).",
    )


_PROMPT = """\
Propose {n} fundamentally-different CORE DIRECTIONS for this task.

Each direction is the algorithmic invariant a follow-up attempt should
preserve — narrow on purpose. Examples: "CNN architecture",
"rollout-based search", "rule-based with hand-tuned weights",
"genetic algorithm with tournament selection". NOT a full implementation
plan, NOT parameter tuning, NOT a sub-component — the algorithmic
backbone that defines a family of attempts.

## Task context
{task_context}

## Direction families currently in history
{families_summary}

Output a JSON array (and nothing else) with exactly {n} entries:

[
  {{
    "name": "short-hyphenated-name",
    "direction": "1-2 line description of the algorithmic invariant",
    "guidance": "Multi-sentence rationale: why this direction is worth \
trying, what it might explore that current families haven't, key \
implementation considerations."
  }},
  ...
]

Rules:
- Each direction must be fundamentally different from the others (different
  algorithmic backbone, not just different parameters).
- Each direction must NOT duplicate a family already in history.
- The "direction" field becomes the core_direction.md content; keep it
  short (1-2 lines).
- The "guidance" field is appended to the agent's prompt; richer (a
  paragraph or two).
- Names: short, hyphenated, descriptive. No spaces.
"""


class PlanApproaches(Strategy):
    """Plan and queue fresh-agent runs for N new direction families.

    Composed method pattern:
        init -> gather context -> LLM proposes directions -> queue runs
    """

    Config = PlanApproachesConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        if not hasattr(toolkit, "llm"):
            return {"skipped": "no LLM available"}
        if not hasattr(toolkit, "task"):
            return {"skipped": "no task on toolkit"}

        self.log.start(
            f"--- PlanApproaches | proposing {self.cfg.n_directions} directions"
        )

        task_context = toolkit.task.context.get()
        families_summary = self._summarize_families(toolkit)

        self.log.inline("planning... ")
        proposals = self._propose_directions(toolkit, task_context, families_summary)
        self.log.tock()

        if not proposals:
            self.log.info("no directions proposed")
            return {"strategy": "plan_approaches", "queued": 0, "cost": round(self.cost, 6)}

        # Queue one fresh-agent run per proposal.
        queue_path = getattr(toolkit, "path", None)
        if queue_path is None:
            # Best-effort: optimizer's path is normally exposed; fall back to
            # toolkit.history's parent if available.
            history = getattr(toolkit, "history", None)
            base = getattr(history, "base_path", None)
            queue_path = base.parent if base is not None else None
        if queue_path is None:
            self.log.info("could not resolve queue path; nothing queued")
            return {"strategy": "plan_approaches", "queued": 0, "cost": round(self.cost, 6)}

        queued = 0
        for proposal in proposals:
            direction = (proposal.get("direction") or "").strip()
            if not direction:
                continue
            guidance = (proposal.get("guidance") or "").strip()
            if self.cfg.guidance_prefix:
                guidance = f"{self.cfg.guidance_prefix.strip()}\n\n{guidance}".strip()
            queue_add(
                queue_path,
                self.cfg.fresh_strategy_name,
                config={
                    "core_direction": direction,
                    "guidance": guidance,
                    "tier": self.cfg.tier,
                    "name": (proposal.get("name") or "").strip(),
                },
                source="plan_approaches",
            )
            queued += 1

        self.log.info(f"queued {queued} fresh-direction runs")
        return {
            "strategy": "plan_approaches",
            "queued": queued,
            "directions": [p.get("name", "?") for p in proposals],
            "cost": round(self.cost, 6),
        }

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.log = toolkit.log if hasattr(toolkit, "log") else StrategyLog()
        self.cost = 0.0

    # --- Context gathering ---

    def _summarize_families(self, toolkit):
        history = getattr(toolkit, "history", None)
        if history is None or not hasattr(history, "derive_families"):
            return "(none — no derive_families available)"
        families = history.derive_families()
        if not families:
            return "(none — empty history)"
        from groundhog.utils.direction import read_direction_from_attempt, direction_title
        lines = []
        for members in families:
            # Backend-agnostic read (the old hasattr('path') guard blanked
            # directions on the git backend — same class as audit bug #4).
            sample = read_direction_from_attempt(members[0])
            title = direction_title(sample or "")
            lines.append(f"  - [{len(members)} attempts] {title}")
        return "\n".join(lines)

    # --- LLM call ---

    def _propose_directions(self, toolkit, task_context, families_summary):
        prompt = _PROMPT.format(
            n=self.cfg.n_directions,
            task_context=task_context,
            families_summary=families_summary,
        )
        system_prompt = (
            "You propose distinct algorithmic directions for an optimizer. "
            "Output ONLY a JSON array, no preamble, no markdown fences."
        )

        response = toolkit.llm.get("default").generate(
            prompt=prompt, system_prompt=system_prompt
        )
        self.cost += response.cost
        return self._parse_proposals(response.text)

    @staticmethod
    def _parse_proposals(text):
        """Best-effort JSON-array extraction. LLMs sometimes wrap output in
        markdown fences or add preamble; strip those, then ``json.loads``."""
        if not text:
            return []
        text = text.strip()
        # Strip optional ```json ... ``` fence.
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Find the first '[' and the last ']' as a fallback.
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)]
