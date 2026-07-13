"""The standard finish — one helper for the 95% way to end an attempt.

``finalize_attempt`` composes the finish every producer normally wants:
surface the direction, evaluate the legitimacy gates, apply the standard
responses, write the record, commit, and cache the score note. It is a
CONVENTION, not a contract: nothing requires a strategy to call it, the
Strategy contract never mentions it, and every piece it composes stays
public (``utils.gates``, ``utils.direction``, ``utils.results``) so a
strategy that wants a different finish assembles its own from the same
parts.

The helper owns the commit deliberately: the score note is keyed by the
commit id, so a finish that stopped short of committing could not cache
it. Everything before the commit runs exactly as
``AgentStrategy._finalize`` historically did.

Standard responses to gate violations (see utils/gates.py):
    fail  → recorded, never blocked: metadata["gate_failure"] is set and
            the result is marked failed, so the attempt commits as a
            failed entry the family map remembers and selection skips.
    flag  → metadata only: direction_restored / non_promotable.
"""

from __future__ import annotations

from typing import Optional

from groundhog.utils.direction import (
    mark_result_failed,
    promote_workspace_direction,
    restore_inherited_direction,
    workspace_name,
)
from groundhog.utils.gates import (
    DIRECTION_MODIFIED,
    evaluate_gates,
    gate_metadata,
)


def finalize_attempt(
    toolkit,
    ws,
    result,
    prior=None,
    *,
    metadata: Optional[dict] = None,
    strategy: str = "manual",
    cost: float = 0.0,
    name: Optional[str] = None,
):
    """Run the standard finish on ``ws`` and return the committed Attempt.

    Args:
        toolkit: the bench — used for ``history`` (gates + score note) and
            ``task`` (the score note's scorer). Tolerates missing pieces:
            without a history the duplicate gate and score note are
            skipped, never guessed.
        ws: the open workspace to finish.
        result: the EvaluationResult of record. A fail-gate violation
            mutates it (marked failed) before it is written.
        prior: the parent Attempt, or None for a fresh attempt.
        metadata: the caller's metadata dict (a strategy passes its own,
            e.g. with its real cost). When omitted, a standard one is
            built from ``strategy`` and ``cost``.
        strategy: producer label for the built metadata ("manual",
            "session", "agent", ...). Ignored when ``metadata`` is given.
        cost: cost for the built metadata. Ignored when ``metadata`` is
            given.
        name: explicit display name; otherwise derived from the finished
            core_direction.md's first line.
    """
    history = getattr(toolkit, "history", None)
    if metadata is None:
        metadata = {
            "strategy": strategy,
            "prior": prior.id if prior else None,
            "cost": round(cost, 6),
        }

    # The metadata pass-through: a wrapping strategy (e.g. ABTest) sets
    # toolkit._extra_attempt_metadata around an inner strategy's call to
    # stamp attribution onto whatever it commits, without touching the
    # inner strategy at all.
    extra = getattr(toolkit, "_extra_attempt_metadata", None)
    if extra:
        metadata = {**metadata, **extra}

    # Mutation first (fresh only): surface the agent-written direction so
    # the gates judge the post-promote state.
    if prior is None:
        promote_workspace_direction(ws.path)

    violations = evaluate_gates(
        ws.path, prior, history=history, exclude=[ws.display_id]
    )
    metadata.update(gate_metadata(violations))
    for v in violations:
        if v.severity == "fail":
            mark_result_failed(result, "core_direction", v.message)

    # Mutation second (inherited only): restore the parent's FULL
    # direction whenever it differs — directions are immutable, and the
    # restore keeps families from forking mid-session.
    if prior is not None and any(v.gate == DIRECTION_MODIFIED for v in violations):
        restore_inherited_direction(ws.path, prior)

    from groundhog.utils.results import write_result
    write_result(ws.path, result, metadata=metadata)

    # Display name: the caller's slug if provided, else a slug of the
    # finalized core direction's first line.
    ws.name = workspace_name(ws.path, explicit=name)

    attempt = ws.commit(success=result.completed)

    # Cache the score as a mutable note beside the record (git: a real git
    # note; folder: notes.json). Read-side stays canonical — this is a
    # low-effort cache, refreshed whenever the record is scored again.
    _cache_score_note(toolkit, history, attempt, result)

    return attempt


def _cache_score_note(toolkit, history, attempt, result) -> None:
    """Best-effort: never let a cache write fail a finished attempt."""
    if history is None or not hasattr(history, "set_note"):
        return
    try:
        value = "fail"
        if result.completed:
            value = f"{_score_result(toolkit, result):.4f}"
        # Pass the attempt OBJECT, not its id: id lookup resolves through
        # history.get(), which only sees done attempts — a failed attempt's
        # note would silently vanish (folder backend).
        history.set_note(attempt, "score", value)
    except Exception:  # noqa: BLE001
        # Broad on purpose: this runs AFTER the commit, and the scorer is
        # user code (arbitrary exceptions) while git note writes can hit
        # ref-lock contention (GitError). An escape here would bubble into
        # the strategy's except-path, which aborts the workspace — deleting
        # the attempt that just committed.
        pass


def _score_result(toolkit, result) -> float:
    """Score via the task's live stage scorer, falling back through stages."""
    task = getattr(toolkit, "task", None)
    if task is None:
        raise ValueError("no task on toolkit")
    through = getattr(toolkit, "through", None)
    stages = task.evaluator.eval_stages(task.data, through=through)
    for stage in reversed(stages):
        stage_result = result.stages.get(stage.name)
        if stage_result is not None:
            return stage.score(stage_result)
    return -1.0
