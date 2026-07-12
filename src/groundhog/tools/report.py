"""Run-state report — a markdown snapshot of the attempt store.

Read-side only: everything is derived from the history + scorer at render
time via the read layer (utils/queries), nothing is persisted (the score
NOTE cache is not consulted).
"""

from __future__ import annotations

import json
from typing import List, Optional

from groundhog.utils import queries as _queries


def gather(history, scorer) -> dict:
    """All the data the report renders, as plain dicts/lists."""
    rows = _queries.attempt_table(history, scorer)
    fams = _queries.families(history, scorer)
    summary = _queries.run_summary(history, scorer)
    return {
        "summary": summary,
        "families": fams,
        "rows": rows,
        "open_questions": _open_questions(history, rows),
    }


def render_markdown(task_name: str, data: dict,
                    narrative: Optional[str] = None) -> str:
    summary, fams, rows = data["summary"], data["families"], data["rows"]
    lines = [f"# Run state: {task_name}", ""]

    if narrative:
        lines += ["## State of the run", "", narrative.strip(), ""]

    best = summary.get("best")
    lines += [
        "## Summary", "",
        f"- attempts: {summary['n_attempts']} "
        f"({summary['n_done']} done, {summary['n_failed']} failed)",
        f"- best: " + (f"{best['id']} score={best['score']:.4f} {best['name']}"
                       if best else "none"),
        f"- families: {summary['n_families']}",
        f"- total cost: ${summary['total_cost']:.4f}",
        "",
    ]

    lines += ["## Families", ""]
    if fams:
        lines += ["| family | n | best | best score |",
                  "| --- | --- | --- | --- |"]
        for f in fams:
            score = (f"{f['best_score']:.4f}"
                     if f.get("best_score") is not None else "-")
            lines.append(f"| {f['family_name']} | {len(f['members'])} "
                         f"| {_short(f.get('best_id'))} | {score} |")
    else:
        lines.append("No attempts yet.")
    lines.append("")

    lines += ["## Recent attempts", ""]
    recent = rows[-10:]
    if recent:
        lines += ["| id | status | score | strategy | name |",
                  "| --- | --- | --- | --- | --- |"]
        for r in reversed(recent):
            score = f"{r['score']:.4f}" if r.get("score") is not None else "-"
            lines.append(f"| {_short(r['id'])} | {r['status']} | {score} "
                         f"| {r.get('strategy') or '-'} | {r.get('name') or ''} |")
    else:
        lines.append("No attempts yet.")
    lines.append("")

    lines += ["## Score trajectory", ""]
    trajectory = [s for _, s in summary.get("score_trajectory", [])]
    if trajectory:
        lines += ["```",
                  sparkline(trajectory),
                  f"min {min(trajectory):.4f} | max {max(trajectory):.4f} "
                  f"| {len(trajectory)} scored attempts",
                  "```"]
    else:
        lines.append("No scored attempts yet.")
    lines.append("")

    lines += ["## Open questions", ""]
    if data["open_questions"]:
        for q in data["open_questions"]:
            lines.append(f"- {q}")
    else:
        lines.append("None - recent attempts committed clean.")
    lines.append("")

    return "\n".join(lines)


# ASCII on purpose: the report may be catted to a Windows console, where
# unicode block characters depend on the codepage.
_SPARK_LEVELS = " .:-=+*#%@"


def sparkline(values: List[float], width: int = 60) -> str:
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _SPARK_LEVELS[len(_SPARK_LEVELS) // 2] * len(values)
    span = hi - lo
    top = len(_SPARK_LEVELS) - 1
    return "".join(_SPARK_LEVELS[round((v - lo) / span * top)] for v in values)


def narrative(llm, data: dict) -> Optional[str]:
    """One cheap LLM pass turning the data into a short 'state of the run'.

    Best-effort: any failure (no default tier, backend error, empty text)
    degrades to the data-only report, never an exception.
    """
    try:
        backend = llm.get("default") if hasattr(llm, "get") else llm
        compact = {
            "summary": data["summary"],
            "families": data["families"],
            "recent": data["rows"][-10:],
            "open_questions": data["open_questions"],
        }
        prompt = (
            "You are summarizing the state of an iterative code-optimization "
            "run for its operator.\n\nRun data (JSON):\n"
            + json.dumps(compact, indent=1, default=str)
            + "\n\nWrite 5-10 short lines of plain prose: where the best "
            "score stands and which family produced it, which families look "
            "active vs stalled, notable recent failures, and the most "
            "promising next move. No headers, no code blocks."
        )
        text = backend.generate(prompt).text.strip()
        return text or None
    except Exception:  # noqa: BLE001 — the narrative is optional garnish
        return None


def _open_questions(history, rows, recent: int = 10) -> List[str]:
    """Attempts a human should look at: recent failures and flagged commits."""
    out = []
    for r in rows[-recent:]:
        attempt = history.get(r["id"])
        metadata = _metadata_of(attempt) if attempt else {}
        if metadata.get("gate_failure"):
            out.append(f"attempt {_short(r['id'])} gate-failed: "
                       f"{metadata['gate_failure']}")
        elif r["status"] == "fail":
            out.append(f"attempt {_short(r['id'])} failed "
                       f"(strategy: {metadata.get('strategy') or '?'})")
        elif metadata.get("non_promotable"):
            out.append(f"attempt {_short(r['id'])} flagged non-promotable: "
                       f"{metadata.get('non_promotable_reason', '')}")
        elif metadata.get("no_recorded_result"):
            out.append(f"attempt {_short(r['id'])} committed without a "
                       f"recorded evaluation")
    return out


def _metadata_of(attempt) -> dict:
    try:
        return attempt.metadata or {}
    except Exception:  # noqa: BLE001
        return {}


def _short(value, n=8):
    if value is None:
        return "-"
    s = str(value)
    return s[:n] if len(s) > n else s
