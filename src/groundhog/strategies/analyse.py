"""Analyse strategy — compress learnings and refresh the run report.

Vault: Strategy — Types of Action.md (Analyse)

Reads history + current learnings, asks the LLM to compress them (remove
duplicates, group findings, flag what's untried), and refreshes a
"state of the run" report — a periodic hand-off document (narrative header
+ data sections) that a campaign orchestrator reads between waves.

Analyse never generates code and never creates an attempt: the report is a
VIEW of the tree (overwritten in place), not a record. Schedule it by
weight in the optimizer's rotation — nothing else changes:

    strategies=[(Analyse(), 1), (Improve(), 9)]   # report every ~10 attempts
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from groundhog.base.strategy import Strategy, StrategyConfig, param


def _md_cell(text) -> str:
    """Escape pipes so free text (family names, strategy labels) cannot
    shred a markdown table row."""
    return str(text).replace("|", "\\|")


@dataclass
class AnalyseConfig(StrategyConfig):
    """Configuration for the Analyse strategy."""
    max_attempts: int = param(20, "How many recent attempts to summarize")
    write_report: bool = param(True, "Refresh the state-of-the-run report file")
    report_path: str = param("reports/state.md",
                             "Report location, relative to the run root")
    report_last_k: int = param(10, "Attempts shown in the report's recent table")
    report_top_learnings: int = param(8, "Learnings shown in the report")


class Analyse(Strategy):
    """Compress learnings and refresh the run report from recent history.

    Composed method pattern:
        init -> gather context -> compress learnings -> write report
    """

    Config = AnalyseConfig

    def __call__(self, toolkit, config=None):
        self._init(toolkit, config)
        if not hasattr(toolkit, 'learnings') or not hasattr(toolkit, 'llm'):
            return {"skipped": "no learnings or LLM available"}

        self.log.start("--- Analyse | compressing learnings")

        current_learnings = toolkit.learnings.get()
        entries_before = toolkit.learnings.count()

        attempts_summary = self._summarize_attempts(toolkit)

        entries_after = entries_before
        if entries_before > 0:
            self.log.inline("compressing... ")
            compressed = self._compress(toolkit, current_learnings, attempts_summary)
            self.log.tock()
            if compressed:
                toolkit.learnings._path.write_text(compressed.strip() + "\n", encoding="utf-8")
                entries_after = toolkit.learnings.count()
                self.log.info(f"learnings: {entries_before} -> {entries_after} entries")
            else:
                self.log.info("no compression produced")

        result = {
            "strategy": self.name,
            "entries_before": entries_before,
            "entries_after": entries_after,
        }

        if self.cfg.write_report:
            report_path = self._write_report(toolkit)
            self.log.info(f"report: {report_path}")
            result["report"] = str(report_path)

        result["cost"] = round(self.cost, 6)
        if self.cost:
            # Analyse commits no attempt, so this spend attaches to no
            # record — say it out loud; the optimizer folds it into the
            # run total from the returned dict.
            self.log.info(f"analyse cost: ${self.cost:.4f}")
        return result

    # --- Init ---

    def _init(self, toolkit, config):
        from groundhog.tools.log import StrategyLog
        self.cfg = self._resolve_config(config)
        self.through = getattr(toolkit, 'through', None)
        self.log = toolkit.log if hasattr(toolkit, 'log') else StrategyLog()
        self.cost = 0.0

    # --- Context gathering ---

    def _scorer(self, toolkit):
        stages = toolkit.task.evaluator.eval_stages(toolkit.task.data, through=self.through)
        return stages[-1].score

    def _score_of(self, attempt, scorer):
        """Last-stage score for an attempt, or None when unscoreable."""
        try:
            result = attempt.result
        except (OSError, ValueError):
            return None
        if not result.completed:
            return None
        stages = list(result.stages.values())
        if not stages:
            return None
        try:
            return scorer(stages[-1])
        except Exception:
            return None

    def _summarize_attempts(self, toolkit):
        attempts = toolkit.history.list()
        recent = attempts[-self.cfg.max_attempts:]
        scorer = self._scorer(toolkit)

        lines = []
        for a in recent:
            strategy = (a.metadata or {}).get("strategy", "?")
            score = self._score_of(a, scorer)
            if score is not None:
                lines.append(f"  #{a.id} (parent={a.parent}, {strategy}): score={score:.4f}")
            else:
                lines.append(f"  #{a.id} (parent={a.parent}, {strategy}): "
                             f"{self._status_label(a)}")

        return "\n".join(lines)

    @staticmethod
    def _status_label(attempt):
        """Row label for an unscoreable attempt — by STATUS, not by score:
        an in-progress leftover or a no-eval commit is not a failure."""
        if attempt.status == "fail":
            return "FAILED"
        if attempt.status == "in-progress":
            return "IN-PROGRESS"
        return "no score"

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

        return response.text or ""

    # --- Report ---
    # Aggregations below intentionally stay local: their shapes differ from
    # the read layer in utils/queries.py (score-based failed counts, tuple
    # best/latest) and the report renderer/narrator depend on them.

    def _write_report(self, toolkit):
        scorer = self._scorer(toolkit)
        attempts = toolkit.history.list(only_done=False)

        summary = self._report_summary(attempts, scorer)
        families = self._report_families(attempts, scorer)
        narrative = self._narrate(toolkit, summary, families)

        body = self._render_report(toolkit, narrative, summary, families, attempts, scorer)

        path = self._resolve_report_path(toolkit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def _resolve_report_path(self, toolkit):
        root = getattr(toolkit, 'path', None)
        if root is None:
            base = getattr(toolkit.history, 'base_path', None)
            root = Path(base).parent if base is not None else Path.cwd()
        rel = Path(self.cfg.report_path)
        return rel if rel.is_absolute() else Path(root) / rel

    def _report_summary(self, attempts, scorer):
        scores = []
        completed = 0
        for a in attempts:
            s = self._score_of(a, scorer)
            if s is not None:
                completed += 1
                scores.append((a.id, s))
        # Failed by STATUS: an in-progress leftover or a no-eval commit is
        # unscoreable but not a failure.
        failed = sum(1 for a in attempts if a.status == "fail")
        best = max(scores, key=lambda t: t[1]) if scores else None
        latest = scores[-1] if scores else None
        return {
            "total": len(attempts),
            "completed": completed,
            "failed": failed,
            "best": best,
            "latest": latest,
        }

    def _report_families(self, attempts, scorer):
        from groundhog.utils.direction import (
            direction_title, normalize_direction, read_direction_from_attempt,
        )
        groups = {}
        for a in attempts:
            text = read_direction_from_attempt(a)
            key = normalize_direction(text) if text else None
            groups.setdefault(key, {"text": text, "members": []})["members"].append(a)

        out = []
        for group in groups.values():
            best = None
            for a in group["members"]:
                s = self._score_of(a, scorer)
                if s is not None and (best is None or s > best):
                    best = s
            out.append({
                "name": direction_title(group["text"] or "") or "(no direction)",
                "members": len(group["members"]),
                "best": best,
            })
        return out

    def _narrate(self, toolkit, summary, families):
        best = f"{summary['best'][1]:.4f} (#{summary['best'][0]})" if summary["best"] else "none"
        latest = f"{summary['latest'][1]:.4f} (#{summary['latest'][0]})" if summary["latest"] else "none"
        fam_rows = []
        for f in families:
            best_s = f"{f['best']:.4f}" if f["best"] is not None else "none"
            fam_rows.append(f"- {f['name']}: {f['members']} attempts, best {best_s}")
        fam_lines = "\n".join(fam_rows) or "- (none yet)"

        prompt = f"""You are writing the header of a "state of the run" report for an
ongoing optimization campaign. Someone picking up the run reads this to
orient before the next wave of work.

## Numbers
- Attempts: {summary['total']} ({summary['completed']} scored, {summary['failed']} failed)
- Best score: {best}
- Latest score: {latest}

## Families (approaches tried)
{fam_lines}

Write 5-10 lines of prose covering: the trajectory so far, what's working,
what's stuck or underexplored, and the recommended next moves. Be specific
and concrete. Output only the narrative, no heading."""

        system_prompt = "You are a research lead writing a concise state-of-the-run briefing."
        response = toolkit.llm.get("default").generate(prompt=prompt, system_prompt=system_prompt)
        self.cost += response.cost
        return (response.text or "").strip()

    def _render_report(self, toolkit, narrative, summary, families, attempts, scorer):
        lines = ["# State of the run", "", narrative, "", "## Summary", ""]

        best = summary["best"]
        latest = summary["latest"]
        lines.append(f"- Attempts: {summary['total']} "
                     f"({summary['completed']} scored, {summary['failed']} failed)")
        lines.append(f"- Best score: {best[1]:.4f} (#{best[0]})" if best else "- Best score: none")
        lines.append(f"- Latest score: {latest[1]:.4f} (#{latest[0]})" if latest else "- Latest score: none")
        lines.append(f"- Families: {len(families)}")
        lines.append("")

        lines += ["## Families", "", "| Family | Attempts | Best score |", "| --- | --- | --- |"]
        for f in families:
            best_s = f"{f['best']:.4f}" if f["best"] is not None else "-"
            lines.append(f"| {_md_cell(f['name'])} | {f['members']} | {best_s} |")
        lines.append("")

        k = self.cfg.report_last_k
        lines += [f"## Last {k} attempts", "", "| # | Parent | Strategy | Score |",
                  "| --- | --- | --- | --- |"]
        for a in attempts[-k:]:
            s = self._score_of(a, scorer)
            score_s = f"{s:.4f}" if s is not None else self._status_label(a)
            strategy = (a.metadata or {}).get("strategy", "?")
            lines.append(f"| {a.id} | {a.parent} | {_md_cell(strategy)} | {score_s} |")
        lines.append("")

        learnings = toolkit.learnings.get(last=self.cfg.report_top_learnings).strip()
        lines += ["## Top learnings", "", learnings or "_(none yet)_", ""]

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"_generated by analyse at {stamp}_")
        return "\n".join(lines) + "\n"
