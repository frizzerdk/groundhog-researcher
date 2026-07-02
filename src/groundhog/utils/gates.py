"""Legitimacy gates — pure facts about a workspace, never actions.

A gate answers one question: is this workspace a legitimate entry for
the store? It reads and reports; it never fails an attempt, commits
anything, or touches a file. Whoever asked — a strategy's finish, the
CLI's commit, an agent pressing check-gates mid-work — decides what to
do with the violations. Mutations (promote, restore, mark-failed) stay
with the caller: the standard responses live in
``groundhog.utils.direction``, and composing them is convention, not
contract.

Severities:
    fail — the standard finish records the attempt as failed (recorded,
           never blocked): a fresh attempt with no core_direction.md,
           or one duplicating an existing family.
    flag — recorded in metadata, the attempt stays done: solution
           byte-identical to the parent (non-promotable), or an
           inherited direction that was modified mid-session (it is
           restored at finish).

These are legitimacy gates only. Performance acceptance is deliberately
read-side (scorer + selection); a run that wants a performance gate
composes its own violation and response around this kit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from groundhog.utils.direction import (
    direction_exists,
    inherited_direction_changed_from,
    read_direction,
    solution_matches_attempt,
)

#: Gate identifiers, stable strings consumers may branch on.
DIRECTION_MISSING = "direction-missing"
DIRECTION_DUPLICATE = "direction-duplicate"
DIRECTION_MODIFIED = "direction-modified"
SOLUTION_IDENTICAL = "solution-identical"

# Messages are load-bearing: the strategy writes them into
# metadata["gate_failure"] / result errors, and existing tests pin them.
_MESSAGES = {
    DIRECTION_MISSING: "fresh attempt did not create core_direction.md",
    DIRECTION_DUPLICATE: "fresh attempt duplicated an existing core direction",
    DIRECTION_MODIFIED: (
        "inherited core_direction.md was modified; the parent's direction "
        "is restored at finish"
    ),
    SOLUTION_IDENTICAL: "solution.py is byte-identical to parent",
}


@dataclass(frozen=True)
class GateViolation:
    """One fact the gates found. ``severity`` is "fail" or "flag"."""

    gate: str
    severity: str
    message: str


def evaluate_gates(
    ws_dir: Path | str,
    parent=None,
    *,
    history=None,
    exclude: Iterable[str] = (),
) -> list[GateViolation]:
    """Evaluate every legitimacy gate against a workspace. Pure.

    Args:
        ws_dir: the workspace directory (root or mid-work — a direction
            still sitting in ``work/`` counts as present, because the
            standard finish promotes it).
        parent: the parent Attempt, or ``None`` for a fresh attempt.
            Fresh attempts face the direction gates; children face the
            modified-direction and identical-solution facts.
        history: AttemptHistory for the duplicate check (fresh only).
            ``None`` skips that gate rather than guessing.
        exclude: attempt ids ignored by the duplicate check (pass the
            workspace's own display id when it is already visible in
            history).

    Returns violations in a stable order: fail gates first.
    """
    violations: list[GateViolation] = []

    if parent is None:
        direction = read_direction(ws_dir)
        if not direction:
            violations.append(_violation(DIRECTION_MISSING, "fail"))
        elif direction_exists(
            history, direction, exclude=exclude, only_done=False
        ):
            violations.append(_violation(DIRECTION_DUPLICATE, "fail"))
    else:
        if inherited_direction_changed_from(ws_dir, parent):
            violations.append(_violation(DIRECTION_MODIFIED, "flag"))

    if solution_matches_attempt(ws_dir, parent):
        violations.append(_violation(SOLUTION_IDENTICAL, "flag"))

    return violations


def _violation(gate: str, severity: str) -> GateViolation:
    return GateViolation(gate=gate, severity=severity, message=_MESSAGES[gate])


class GateKit:
    """The bench binding: ``toolkit.gates.evaluate(ws, parent)``.

    A thin convenience over :func:`evaluate_gates` that reads the
    history off the toolkit at call time and derives ``exclude`` from
    the workspace's display id. Accepts a workspace object (``.path``)
    or a plain directory path.
    """

    def __init__(self, toolkit):
        self._toolkit = toolkit

    def evaluate(
        self,
        ws,
        parent=None,
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> list[GateViolation]:
        ws_dir = getattr(ws, "path", ws)
        if exclude is None:
            display_id = getattr(ws, "display_id", None)
            exclude = [display_id] if display_id else []
        return evaluate_gates(
            ws_dir,
            parent,
            history=getattr(self._toolkit, "history", None),
            exclude=exclude,
        )
