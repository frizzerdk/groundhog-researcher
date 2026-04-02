"""Analyse strategy — compress and reformat learnings.

Vault: Strategy — Types of Action.md (Analyse)

Reads history + current learnings, asks LLM to compress: remove duplicates,
identify patterns, note what's been tried vs untried. Replaces learnings
with the compressed version. Does NOT generate code or create an attempt.
"""

from dataclasses import dataclass

from groundhog.base.strategy import Strategy, StrategyConfig, param
from groundhog.tools.conversation_log import conversation_log


@dataclass
class AnalyseConfig(StrategyConfig):
    """Configuration for the Analyse strategy."""
    max_attempts: int = param(20, "How many recent attempts to summarize")


class Analyse(Strategy):
    """Compress and reformat learnings based on recent attempt history.

    Composed method pattern:
        init ->gather context ->ask LLM to compress ->replace learnings
    """

    Config = AnalyseConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        if not hasattr(toolkit, 'learnings') or not hasattr(toolkit, 'llm'):
            return {"skipped": "no learnings or LLM available"}

        self.log.start(f"--- Analyse | compressing learnings")

        current_learnings = toolkit.learnings.get()
        entries_before = toolkit.learnings.count()

        if entries_before == 0:
            return {"skipped": "no learnings to analyse"}

        # Gather recent attempt summaries
        attempts_summary = self._summarize_attempts(toolkit)

        # Ask LLM to compress
        self.log.inline("compressing... ")
        compressed = self._compress(toolkit, current_learnings, attempts_summary)
        self.log.tock()

        if compressed:
            # Replace learnings entirely
            toolkit.learnings._path.write_text(compressed.strip() + "\n", encoding="utf-8")
            entries_after = toolkit.learnings.count()
            self.log.info(f"learnings: {entries_before} ->{entries_after} entries")
        else:
            entries_after = entries_before
            self.log.info("no compression produced")

        return {
            "strategy": "analyse",
            "entries_before": entries_before,
            "entries_after": entries_after,
            "cost": round(self.cost, 6),
        }

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.log_conversation = toolkit.conversation_log if hasattr(toolkit, 'conversation_log') else conversation_log
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

    # --- Context gathering ---

    def _summarize_attempts(self, toolkit):
        attempts = toolkit.history.list()
        recent = attempts[-self.cfg.max_attempts:]
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        scorer = stages[-1].score

        lines = []
        for a in recent:
            result = a.result
            if not result.completed:
                lines.append(f"  #{a.number} (parent={a.parent}): FAILED at {result.failed_stage}")
                continue
            last = list(result.stages.values())[-1]
            score = scorer(last)
            strategy = a.metadata.get("strategy", "?")
            lines.append(f"  #{a.number} (parent={a.parent}, {strategy}): score={score:.4f}")

        return "\n".join(lines)

    # --- Compression ---

    def _compress(self, toolkit, current_learnings, attempts_summary):
        prompt = f"""Here are the accumulated learnings from an optimization run:

## Current learnings
{current_learnings}

## Recent attempts
{attempts_summary}

Compress the learnings into a clean, non-redundant summary:
- Remove duplicate observations
- Group related findings
- Note what approaches have been tried and their outcomes
- Flag what has NOT been tried or what resources are underutilized
- Keep specific, actionable insights — drop vague ones
- Use the --- separator between entries

Output only the compressed learnings, nothing else."""

        system_prompt = "You are a research assistant compressing experimental notes. Be concise and specific."

        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost

        return response.text
