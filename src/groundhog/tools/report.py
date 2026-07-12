"""Run-state report — a markdown snapshot of the attempt store.

Read-side only: everything is derived from the history + scorer at render
time, nothing is persisted (the score NOTE cache is not consulted). The
aggregations below are vendored minimal versions of the read layer on
feat/read-layer-queries (utils/queries.py); the defensive import prefers
that module when present, so the two converge at merge.
"""

from __future__ import annotations

import json
from typing import List, Optional

try:
    from groundhog.utils import queries as _queries
except ImportError:
    _queries = None


def gather(history, scorer) -> dict:
    """All the data the report renders, as plain dicts/lists."""
    if _queries is not None:
        rows = _queries.attempt_table(history, scorer)
        fams = _queries.families(history, scorer)
        summary = _queries.run_summary(history, scorer)
    else:
        rows = _attempt_rows(history, scorer)
        fams = _families(history, rows)
        summary = _summary(rows, fams)
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


# --- vendored read-layer aggregations (see module docstring) ---------------


def _attempt_rows(history, scorer) -> List[dict]:
    rows = []
    for a in sorted(history.list(only_done=False),
                    key=lambda a: (a.created_at, a.id)):
        if a.status not in ("done", "fail"):
            continue
        metadata = _metadata_of(a)
        rows.append({
            "id": a.id,
            "parent": a.parent,
            "status": a.status,
            "score": _score_of(a, scorer),
            "name": a.name,
            "created_at": a.created_at,
            "strategy": metadata.get("strategy"),
            "cost": _as_float(metadata.get("cost")),
        })
    return rows


def _families(history, rows) -> List[dict]:
    from groundhog.utils.direction import (
        direction_title,
        normalize_direction,
        read_direction_from_attempt,
    )

    by_id = {r["id"]: r for r in rows}
    groups: dict = {}
    for members in history.derive_families():
        text = read_direction_from_attempt(members[0])
        key = normalize_direction(text) if text else None
        groups[key] = {"text": text, "members": members}

    out = []
    for group in groups.values():
        members = group["members"]
        best_id, best_score = None, None
        for a in members:
            row = by_id.get(a.id)
            s = row["score"] if row else None
            if s is not None and (best_score is None or s > best_score):
                best_id, best_score = a.id, s
        out.append({
            "family_name": direction_title(group["text"] or ""),
            "root_id": members[0].id,
            "members": [a.id for a in members],
            "best_id": best_id,
            "best_score": best_score,
        })
    return out


def _summary(rows, fams) -> dict:
    scored = [r for r in rows if r["score"] is not None]
    trajectory = []
    best_so_far = None
    for r in scored:
        best_so_far = (r["score"] if best_so_far is None
                       else max(best_so_far, r["score"]))
        trajectory.append((r["created_at"], best_so_far))
    best = max(scored, key=lambda r: r["score"]) if scored else None
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    return {
        "n_attempts": len(rows),
        "n_done": sum(1 for r in rows if r["status"] == "done"),
        "n_failed": sum(1 for r in rows if r["status"] == "fail"),
        "best": ({"id": best["id"], "score": best["score"], "name": best["name"]}
                 if best else None),
        "n_families": len(fams),
        "total_cost": round(sum(costs), 6),
        "score_trajectory": trajectory,
    }


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


def _score_of(attempt, scorer) -> Optional[float]:
    if scorer is None:
        return None
    try:
        result = attempt.result
    except Exception:  # noqa: BLE001
        return None
    if not result.completed or not result.stages:
        return None
    try:
        return float(scorer(list(result.stages.values())[-1]))
    except Exception:  # noqa: BLE001
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short(value, n=8):
    if value is None:
        return "-"
    s = str(value)
    return s[:n] if len(s) > n else s
