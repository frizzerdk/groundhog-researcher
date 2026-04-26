"""Prior selection — potential-weighted random across trunk leaders.

Vault: Strategy — Selection.md, Selection_PotentialScore.md

Selects which attempt to build from next. High-scoring trunks get picked more,
but short/unexplored trunks get an exploration bonus. Long, well-explored
trunks get deprioritized.
"""

import math
import random as rand_module
from typing import Callable, List, Optional

from groundhog.base.attempt_history import Attempt, AttemptHistory
from groundhog.base.types import StageResult


def select_prior(
    history: AttemptHistory,
    scorer: Callable[[StageResult], float],
    rng: rand_module.Random = None,
    *,
    trunk_weight: float = 0.3,
    direction_weight: float = 0.5,
    direction_decay: float = 0.1,
    exclude_non_promotable: bool = True,
    direction_bonus: Optional[float] = None,
    skip_non_promotable: Optional[bool] = None,
) -> Optional[Attempt]:
    """Select a prior using potential-weighted random across trunk leaders.

    Returns the best attempt from a probabilistically chosen trunk.
    High-scoring trunks are favored; short trunks get an unexplored bonus;
    rarer direction families get an additional bonus so successful families
    don't crowd out exploration.

    Args:
        history: attempt history to sample from.
        scorer: per-stage scorer; the last stage's score drives the score
            component.
        rng: random source. Defaults to a fresh instance.
        trunk_weight: weight on the per-trunk unexplored bonus.
        direction_weight: weight on the per-direction-family unexplored
            bonus. The "no-direction" sentinel family is skipped from this
            bonus so legacy/un-attributed attempts don't dominate.
        direction_decay: exponential decay applied as a direction family grows.
        exclude_non_promotable: skip attempts whose ``result.json``
            metadata sets ``non_promotable=True`` (e.g. duplicate
            solutions flagged by Phase 5).
        direction_bonus: compatibility alias for ``direction_weight``.
        skip_non_promotable: compatibility alias for ``exclude_non_promotable``.

    Formula:
        potential = score_norm
                  + trunk_weight     * exp(-0.1 * (trunk_len - 1))
                  + direction_weight * exp(-direction_decay * (family_size - 1))
    """
    if direction_bonus is not None:
        direction_weight = direction_bonus
    if skip_non_promotable is not None:
        exclude_non_promotable = skip_non_promotable

    trunks = history.derive_trunks(scorer)
    if not trunks:
        return None

    rng = rng or rand_module.Random()
    # Pick each trunk's "effective leader": the highest-scoring attempt that
    # passes the filters (positive score, not non-promotable). This lets a
    # trunk whose tip is flagged still contribute its best earlier attempt
    # rather than getting dropped entirely.
    leaders = []
    for trunk in trunks:
        candidate = None
        candidate_score = -1.0
        for a in trunk:
            if _score(a, scorer) <= 0:
                continue
            if exclude_non_promotable and _is_non_promotable(a):
                continue
            s = _score(a, scorer)
            if s > candidate_score:
                candidate = a
                candidate_score = s
        if candidate is not None:
            leaders.append((candidate, len(trunk)))
    if not leaders:
        return None

    global_best = max(_score(a, scorer) for a, _ in leaders)
    family_sizes = _family_sizes(history)

    potentials = []
    for attempt, trunk_len in leaders:
        score = _score(attempt, scorer)
        score_component = max(score / global_best, 0.01) if global_best > 0 else 0.01
        unexplored_trunk = math.exp(-0.1 * (trunk_len - 1))

        # Direction-family bonus. Sentinel ("no direction") is skipped so
        # legacy attempts don't get rewarded for being un-classified.
        family_key = _attempt_family_key(attempt)
        if family_key is None:
            unexplored_family = 0.0
        else:
            family_size = family_sizes.get(family_key, 1)
            unexplored_family = math.exp(-direction_decay * (family_size - 1))

        potential = (
            score_component
            + trunk_weight * unexplored_trunk
            + direction_weight * unexplored_family
        )
        potentials.append(max(potential ** 2, 0.001))

    selected = rng.choices([a for a, _ in leaders], weights=potentials, k=1)[0]
    return selected


def _attempt_family_key(attempt: Attempt) -> Optional[str]:
    """Normalized core_direction.md text, or None for the sentinel family."""
    if not hasattr(attempt, "path"):
        return None
    from groundhog.utils.direction import read_direction, normalize_direction
    text = read_direction(attempt.path)
    return normalize_direction(text) if text else None


def _family_sizes(history: AttemptHistory) -> dict:
    """Map family-key -> number of attempts in that family.

    The sentinel (``None``) family is included for completeness but
    callers typically skip it from bonus calculation.
    """
    if not hasattr(history, "derive_families"):
        return {}
    sizes = {}
    for members in history.derive_families():
        if not members:
            continue
        key = _attempt_family_key(members[0])
        sizes[key] = len(members)
    return sizes


def _is_non_promotable(attempt: Attempt) -> bool:
    """Read the ``non_promotable`` flag from the attempt's metadata."""
    try:
        return bool(attempt.metadata.get("non_promotable", False))
    except Exception:
        return False


def get_trunk_leaders(history: AttemptHistory, scorer: Callable[[StageResult], float],
                      exclude: Optional[int] = None) -> List[Attempt]:
    """Get the best attempt from each trunk, optionally excluding one trunk."""
    trunks = history.derive_trunks(scorer)
    leaders = []
    for trunk in trunks:
        leader = trunk[-1]
        if exclude is not None and leader.number == exclude:
            continue
        leaders.append(leader)
    return leaders


def _score(attempt: Attempt, scorer) -> float:
    result = attempt.result
    if not result.completed:
        return -1.0
    last = list(result.stages.values())[-1]
    return scorer(last)
