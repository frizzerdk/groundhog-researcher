"""Read layer over the attempt store — queries, never state.

Pure read-side functions over any ``AttemptHistory`` (folder, git, or a
user backend) plus an optional scorer. Every view a consumer wants —
dashboard, LLM report, agent search, learnings lens — is a presentation
of these queries; nothing here writes, caches, or persists. Scores are
computed on demand from raw metrics (the "raw results, never scores"
principle), so re-querying with a different scorer reinterprets the
whole run without touching the store.

All functions return plain JSON-serializable dicts/lists so a CLI, an
HTTP layer, or a notebook can consume them unchanged. ``created_at`` is
the raw unix-epoch float (the canonical sort key) — formatting is the
presenter's job. A ``score`` of ``None`` means "no score": a failed
attempt, a missing/unreadable result, no scorer given, or a scorer that
raised.

Sub-results CONVENTION (not core schema)
-----------------------------------------
Some evaluators measure per-cell / per-case tables (e.g. accuracy per
grid cell). The attempt record has no dedicated slot for these — they
ride in ``StageResult.metrics`` like any other raw measurement. By
convention, ``sub_results`` surfaces a stage metric as a sub-result
table when it is a dict stored under a key ending in ``_by_cell`` or
under the key ``cells``. Evaluators that follow the convention get
their tables surfaced in every consumer for free; nothing requires it,
and unknown shapes are simply not surfaced.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from groundhog.base.attempt_history import Attempt, AttemptHistory
from groundhog.base.types import EvaluationResult, StageResult

Scorer = Callable[[StageResult], float]


def safe_result(attempt: Attempt) -> Optional[EvaluationResult]:
    """The attempt's result, or ``None`` for the sanctioned no-result state.

    An attempt committed without ``--eval`` has no ``result.json``: the
    folder backend raises ``OSError``, the git backend hands back an empty
    (stage-less) result. Both normalize to ``None``, as does a corrupt
    ``result.json`` (``ValueError``).
    """
    try:
        result = attempt.result
    except (OSError, ValueError):
        return None
    return result if result.stages else None


def safe_code(attempt: Attempt) -> Optional[str]:
    """The attempt's solution code, or ``None`` when absent.

    A failed attempt may lack ``solution.py``: the folder backend raises
    ``OSError``, the git backend hands back ``""``. Both normalize to
    ``None``.
    """
    try:
        code = attempt.code
    except OSError:
        return None
    return code or None


def attempt_table(history: AttemptHistory, scorer: Optional[Scorer] = None,
                  *, only_done: bool = False) -> List[dict]:
    """One row per committed attempt, sorted by ``(created_at, id)``.

    Row: ``{id, parent, status, score, name, created_at, strategy, cost}``.
    ``strategy`` and ``cost`` come from the attempt's metadata; ``score``
    is the scorer applied to the last recorded stage (``None`` when
    unscoreable). ``only_done=False`` includes failed attempts — nothing
    is discarded, and a family map without its failures misleads. Open
    (in-progress) workspaces are never rows: only committed attempts are
    facts of the run.
    """
    return [_row(a, scorer) for a in _committed(history, only_done)]


def families(history: AttemptHistory, scorer: Optional[Scorer] = None,
             *, only_done: bool = False) -> List[dict]:
    """Group attempts by root direction (family identity), oldest family first.

    Family: ``{family_name, root_id, members, best_id, best_score,
    latest_activity}``. Identity is the normalized ``core_direction.md``
    content (same key as ``AttemptHistory.derive_families``); the display
    name is the direction's first line. Attempts with no direction land
    in the ``"(no direction)"`` sentinel family.
    """
    from groundhog.utils.direction import (
        direction_title,
        normalize_direction,
        read_direction_from_attempt,
    )

    groups: dict = {}
    for a in _committed(history, only_done):
        text = read_direction_from_attempt(a)
        key = normalize_direction(text) if text else None
        groups.setdefault(key, {"text": text, "members": []})["members"].append(a)

    # Attempts were iterated oldest-first, so dict insertion order already
    # puts the oldest family (the seed) first.
    out = []
    for group in groups.values():
        members = group["members"]
        best_id, best_score = None, None
        for a in members:
            s = _score_of(a, scorer)
            if s is not None and (best_score is None or s > best_score):
                best_id, best_score = a.id, s
        out.append({
            "family_name": direction_title(group["text"] or ""),
            "root_id": members[0].id,
            "members": [a.id for a in members],
            "best_id": best_id,
            "best_score": best_score,
            "latest_activity": max(a.created_at for a in members),
        })
    return out


def lineage(history: AttemptHistory, attempt_id: str,
            scorer: Optional[Scorer] = None) -> List[dict]:
    """Root-to-attempt chain as table rows (see ``attempt_table``).

    Returns ``[]`` for an unknown attempt id.
    """
    attempt = history.get(attempt_id)
    if attempt is None:
        return []
    return [_row(a, scorer) for a in history.lineage(attempt)]


def run_summary(history: AttemptHistory, scorer: Optional[Scorer] = None) -> dict:
    """Whole-run totals: counts, best, families, cost, score trajectory.

    ``score_trajectory`` is ``[(created_at, best_so_far), ...]`` — one
    entry per scored attempt in creation order, so plotting it shows the
    run's progress curve under the scorer.
    """
    rows = attempt_table(history, scorer)
    scored = [r for r in rows if r["score"] is not None]

    trajectory = []
    best_so_far = None
    for r in scored:
        best_so_far = r["score"] if best_so_far is None else max(best_so_far, r["score"])
        trajectory.append((r["created_at"], best_so_far))

    best = max(scored, key=lambda r: r["score"]) if scored else None
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    return {
        "n_attempts": len(rows),
        "n_done": sum(1 for r in rows if r["status"] == "done"),
        "n_failed": sum(1 for r in rows if r["status"] == "fail"),
        "best": ({"id": best["id"], "score": best["score"], "name": best["name"]}
                 if best else None),
        "n_families": len(families(history)),
        "total_cost": round(sum(costs), 6),
        "score_trajectory": trajectory,
    }


def sub_results(history: AttemptHistory, attempt_id: str) -> dict:
    """Per-cell tables an evaluator recorded by convention (see module doc).

    Returns ``{stage_name: {metric_key: table_dict}}`` for dict-valued
    metrics under a ``*_by_cell`` key or the ``cells`` key; ``{}`` when
    the attempt is unknown, has no readable result, or follows no
    convention.
    """
    attempt = history.get(attempt_id)
    if attempt is None:
        return {}
    result = _result_of(attempt)
    if result is None:
        return {}
    out: dict = {}
    for stage_name, stage in result.stages.items():
        tables = {
            key: value for key, value in stage.metrics.items()
            if isinstance(value, dict) and (key == "cells" or key.endswith("_by_cell"))
        }
        if tables:
            out[stage_name] = tables
    return out


def attempt_detail(history: AttemptHistory, attempt_id: str,
                   scorer: Optional[Scorer] = None) -> Optional[dict]:
    """Everything one attempt exposes read-side, in one dict.

    The table row plus ``metadata``, per-stage ``stages`` (score, metrics,
    errors, warnings), ``files``, ``lineage`` (ids root-to-here), and
    ``sub_results``. ``None`` for an unknown id.
    """
    attempt = history.get(attempt_id)
    if attempt is None:
        return None
    detail = _row(attempt, scorer)
    detail["metadata"] = _metadata_of(attempt)
    stages = {}
    result = _result_of(attempt)
    if result is not None:
        for name, stage in result.stages.items():
            stages[name] = {
                "score": _apply(scorer, stage),
                "metrics": stage.metrics,
                "errors": stage.errors,
                "warnings": stage.warnings,
            }
    detail["stages"] = stages
    detail["files"] = attempt.list_files()
    detail["lineage"] = [a.id for a in history.lineage(attempt)]
    detail["sub_results"] = sub_results(history, attempt_id)
    return detail


def _committed(history: AttemptHistory, only_done: bool) -> List[Attempt]:
    # The folder backend's list(only_done=False) includes open workspaces;
    # a summary that counts them reports phantoms mid-run.
    attempts = [a for a in history.list(only_done=only_done)
                if a.status in ("done", "fail")]
    return sorted(attempts, key=lambda a: (a.created_at, a.id))


def _row(attempt: Attempt, scorer: Optional[Scorer]) -> dict:
    metadata = _metadata_of(attempt)
    return {
        "id": attempt.id,
        "parent": attempt.parent,
        "status": attempt.status,
        "score": _score_of(attempt, scorer),
        "name": attempt.name,
        "created_at": attempt.created_at,
        "strategy": metadata.get("strategy"),
        "cost": _as_float(metadata.get("cost")),
    }


def _metadata_of(attempt: Attempt) -> dict:
    try:
        return attempt.metadata or {}
    except Exception:
        return {}


def _result_of(attempt: Attempt):
    try:
        return attempt.result
    except Exception:
        return None


def _score_of(attempt: Attempt, scorer: Optional[Scorer]) -> Optional[float]:
    if scorer is None:
        return None
    result = _result_of(attempt)
    if result is None or not result.completed or not result.stages:
        return None
    return _apply(scorer, list(result.stages.values())[-1])


def _apply(scorer: Optional[Scorer], stage: StageResult) -> Optional[float]:
    if scorer is None:
        return None
    try:
        return float(scorer(stage))
    except Exception:
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
